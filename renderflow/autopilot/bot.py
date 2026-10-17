"""Autopilot daemon: the daily scheduler + the Telegram remote control.

Run alongside the API and the Celery worker (all on the worker PC):

    .venv/bin/python -m renderflow.autopilot.bot

Under RENDERFLOW_CELERY_EAGER (the Windows PC: no Redis, no worker process)
there is no separate daemon — the API runs `run_loop` in a background
thread, because only the API process owns the in-process job queue.

One loop does both jobs: long-poll Telegram (~25s), then check whether
today's episode is due. Nothing slow happens here — starting an episode
reserves its row and dispatches a Celery task, so the bot stays responsive
while a render runs for an hour.

Without TELEGRAM_BOT_TOKEN the loop still runs the schedule (sleeping
instead of polling) and review happens in the dashboard's Autopilot tab.

Commands (only from TELEGRAM_CHAT_ID):
  /status            what today's episode is doing
  /today             start today's episode now (ignores the run hour)
  /approve [id]      upload the latest (or given) ready episode
  /thumb [id]        regenerate its thumbnail
  /newtopic [id]     skip it and make a different topic
  /skip [id]         skip it, no upload today
  /pause, /resume    turn the daily schedule off/on
  /trends            today's raw trend list (no AI call)
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from renderflow.autopilot import core
from renderflow.autopilot.jobs import ActionError, perform_action, start_new_episode
from renderflow.config import Settings
from renderflow.db import AutopilotChannel, Episode, init_db, new_session
from renderflow.providers import build_messenger, build_trend_sources
from renderflow.providers.messaging.telegram import escape

log = logging.getLogger("renderflow.autopilot.bot")

HELP = (
    "<b>RenderFlow Autopilot</b>\n"
    "/status — what today's video is doing\n"
    "/today — make today's video now\n"
    "/approve — upload the ready video\n"
    "/thumb — new thumbnail\n"
    "/newtopic — different topic instead\n"
    "/skip — no video today\n"
    "/pause · /resume — daily schedule off/on\n"
    "/trends — today's trending searches"
)

STATUS_LABELS = {
    "planning": "🧭 picking a topic and writing the script",
    "generating": "⚙️ rendering",
    "ready": "👀 waiting for your approval",
    "uploading": "⬆️ uploading to YouTube",
    "uploaded": "✅ uploaded",
    "skipped": "⏭ skipped",
    "failed": "⚠️ failed",
}


def the_channel(session: Session) -> AutopilotChannel | None:
    """The bot serves one channel (the user runs a single channel)."""
    return session.query(AutopilotChannel).order_by(AutopilotChannel.id).first()


def latest_episode(session: Session, channel: AutopilotChannel) -> Episode | None:
    return (
        session.query(Episode)
        .filter(Episode.channel_id == channel.id)
        .order_by(Episode.created_at.desc())
        .first()
    )


def status_text(session: Session, channel: AutopilotChannel) -> str:
    episode = latest_episode(session, channel)
    schedule = (
        f"Daily at {channel.run_hour:02d}:00" if channel.enabled else "Schedule paused"
    )
    if episode is None:
        return f"<b>{escape(channel.name)}</b> · {schedule}\nNo episodes yet. Send /today to make one."
    line = STATUS_LABELS.get(episode.status, episode.status)
    text = (
        f"<b>{escape(channel.name)}</b> · {schedule}\n\n"
        f"Episode #{episode.id} ({episode.day}): {line}\n"
    )
    if episode.title:
        text += f"<b>{escape(episode.title)}</b>\n"
    if episode.niche:
        text += f"Niche: {escape(episode.niche)}\n"
    if episode.youtube_url:
        text += f"{episode.youtube_url}\n"
    if episode.status == "failed" and episode.error:
        text += f"\n{escape(episode.error[-400:])}"
    return text


def _resolve_episode(session: Session, channel: AutopilotChannel, arg: str | None) -> Episode:
    if arg:
        try:
            episode = session.get(Episode, int(arg.lstrip("#")))
        except ValueError:
            raise ActionError(f"not an episode number: {arg}")
        if episode is None or episode.channel_id != channel.id:
            raise ActionError(f"no episode #{arg}")
        return episode
    episode = latest_episode(session, channel)
    if episode is None:
        raise ActionError("no episodes yet — send /today")
    return episode


def handle_command(session: Session, channel: AutopilotChannel, text: str) -> str:
    """Text command or button callback ("approve:12") -> reply text."""
    text = text.strip()
    if ":" in text and not text.startswith("/"):
        action, _, arg = text.partition(":")
    else:
        parts = text.split()
        action = parts[0].lstrip("/").split("@")[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else None
    try:
        if action in ("start", "help", ""):
            return HELP
        if action == "status":
            return status_text(session, channel)
        if action == "today":
            episode = start_new_episode(session, channel)
            return f"🧭 Starting episode #{episode.id} — scanning today's trends…"
        if action == "pause":
            channel.enabled = False
            session.commit()
            return "⏸ Daily schedule paused. /resume to turn it back on."
        if action == "resume":
            channel.enabled = True
            session.commit()
            return f"▶️ Daily schedule on — next video at {channel.run_hour:02d}:00."
        if action == "trends":
            trends = core.gather_trends(build_trend_sources(Settings.load()), Settings.load().trends_geo)
            if not trends:
                return "No trend data right now."
            lines = [f"• {escape(t.title)} {escape(t.traffic)}" for t in trends[:20]]
            return "<b>Trending today</b>\n" + "\n".join(lines)
        if action in ("approve", "skip", "thumb", "newtopic"):
            episode = _resolve_episode(session, channel, arg)
            return perform_action(session, episode, action)
        return "I don't know that one.\n\n" + HELP
    except ActionError as exc:
        session.rollback()
        return f"⚠️ {escape(str(exc))}"


def handle_event(session: Session, messenger: Any, event: dict[str, Any], allowed_chat: str) -> None:
    if not allowed_chat:
        # Pairing: tell whoever writes what to put in .env, act on nothing.
        messenger.send_text(
            f"Your chat id is <code>{event['chat_id']}</code>.\n"
            "Put <code>TELEGRAM_CHAT_ID=" + event["chat_id"] + "</code> in RenderFlow's .env "
            "and restart the bot.",
            chat_id=event["chat_id"],
        )
        return
    if event["chat_id"] != allowed_chat:
        log.warning("ignoring telegram message from unknown chat %s", event["chat_id"])
        return
    channel = the_channel(session)
    if channel is None:
        messenger.send_text(
            "No channel set up yet — open the dashboard's Autopilot tab once to create it."
        )
        return
    reply = handle_command(session, channel, event["text"])
    if event["kind"] == "callback" and event.get("callback_id"):
        messenger.answer_callback(event["callback_id"])
    messenger.send_text(reply)


def tick_schedule(session: Session) -> None:
    for channel in session.query(AutopilotChannel).filter(AutopilotChannel.enabled.is_(True)).all():
        channel.heartbeat_at = time.time()
        session.commit()
        if core.is_due(session, channel) and core.active_episode(session, channel) is None:
            episode = start_new_episode(session, channel)
            log.info("autopilot: started daily episode %s", episode.id)
    # Heartbeat for paused channels too, so the dashboard shows the bot online.
    for channel in session.query(AutopilotChannel).filter(AutopilotChannel.enabled.is_(False)).all():
        channel.heartbeat_at = time.time()
    session.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="RenderFlow autopilot daemon")
    parser.add_argument("--once", action="store_true", help="one schedule check, then exit")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if Settings.load().celery_eager:
        # A second poller would fight the API's for Telegram updates, and
        # its renders would bypass the API's one-at-a-time job queue.
        log.error(
            "RENDERFLOW_CELERY_EAGER is on: the autopilot runs inside the API "
            "process (python -m renderflow.api) — don't start it separately"
        )
        return 1
    init_db()
    return run_loop(once=args.once)


def run_loop(once: bool = False) -> int:
    """The scheduler + Telegram loop; never returns unless `once`."""
    try:
        messenger = build_messenger(Settings.load())
    except ValueError as exc:
        log.warning("%s — running the schedule only (review in the dashboard)", exc)
        messenger = None
    allowed_chat = getattr(messenger, "chat_id", "") if messenger else ""
    if messenger and allowed_chat:
        try:
            messenger.send_text("🟢 Autopilot is online.\n\n" + HELP)
        except Exception as exc:
            log.warning("could not greet on telegram: %s", exc)

    while True:
        session = new_session()
        try:
            tick_schedule(session)
            if once:
                return 0
            if messenger is None:
                time.sleep(30)
                continue
            try:
                events = messenger.poll(timeout=25)
            except Exception as exc:
                log.warning("telegram poll failed: %s", exc)
                time.sleep(10)
                continue
            for event in events:
                try:
                    handle_event(session, messenger, event, allowed_chat)
                except Exception:
                    session.rollback()
                    log.exception("failed handling telegram event %r", event.get("text"))
        except Exception:
            session.rollback()
            log.exception("autopilot loop error")
            time.sleep(10)
        finally:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
