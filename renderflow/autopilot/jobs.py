"""Autopilot Celery tasks + the actions the bot and dashboard share.

Slow work (LLM calls, uploads, preview encodes) always runs in the Celery
worker, never in the bot's polling loop or an API request — same rule as
the pipeline itself. The bot and dashboard only reserve/flip Episode rows
and .delay() these tasks.

Flow: start_episode -> (make_video.py job runs) -> on_pipeline_job_finished
-> package_episode (preview + Telegram review message) -> user taps
Approve -> upload_episode.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from sqlalchemy.orm import Session

from renderflow.autopilot import core
from renderflow.config import Settings
from renderflow.db import AutopilotChannel, Episode, Job, Project, new_session
from renderflow.providers import (
    build_llm,
    build_messenger,
    build_publisher,
    build_trend_sources,
)
from renderflow.providers.base import Messenger
from renderflow.providers.messaging.telegram import escape
from renderflow.tasks import cancel_job, celery_app, run_pipeline

log = logging.getLogger("renderflow.autopilot.jobs")


# Auto-resume waits out provider rate-limit windows (Pexels: 200/hour)
# instead of retrying into the same wall immediately.
AUTO_RESUME_DELAY_SEC = 30 * 60


def enqueue_pipeline(
    session: Session, project: Project, kind: str, argv: list[str], countdown: int = 0
) -> Job:
    """Same contract as api._enqueue: commit the row before queueing."""
    job = Job(project_id=project.id, kind=kind, argv=argv)
    session.add(job)
    session.commit()
    result = (
        run_pipeline.apply_async((job.id,), countdown=countdown)
        if countdown
        else run_pipeline.delay(job.id)
    )
    job.celery_task_id = result.id
    session.commit()
    return job


def messenger_or_none() -> Messenger | None:
    """Telegram is optional — without it, review happens in the dashboard."""
    try:
        messenger = build_messenger(Settings.load())
    except ValueError:
        return None
    return messenger if getattr(messenger, "chat_id", "") else None


def notify(text: str, buttons=None) -> None:
    messenger = messenger_or_none()
    if messenger is None:
        return
    try:
        messenger.send_text(text, buttons)
    except Exception as exc:  # a notification must never fail the job
        log.warning("telegram notify failed: %s", exc)


def _fail(session: Session, episode: Episode, error: str) -> None:
    episode.status = "failed"
    episode.error = error[-2000:]
    episode.updated_at = time.time()
    session.commit()
    notify(f"⚠️ <b>Autopilot episode #{episode.id} failed</b>\n{escape(error[-600:])}")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@celery_app.task(name="renderflow.autopilot.start_episode")
def start_episode(episode_id: int, exclude_topics: list[str] | None = None) -> None:
    session = new_session()
    try:
        episode = session.get(Episode, episode_id)
        if episode is None or episode.status != "planning":
            return
        settings = Settings.load()
        try:
            core.plan_episode(
                session,
                episode,
                build_llm(settings),
                build_trend_sources(settings),
                enqueue_pipeline,
                exclude_topics={t.lower() for t in (exclude_topics or [])},
            )
        except Exception as exc:
            session.rollback()
            episode = session.get(Episode, episode_id)
            _fail(session, episode, f"planning failed: {exc}")
            return
        notify(
            f"🎬 <b>Making today's video</b> (#{episode.id})\n"
            f"<b>{escape(episode.title)}</b>\n"
            f"Niche: {escape(episode.niche)} · trend: {escape(episode.trend)}\n"
            "I'll send it here for approval when the render finishes."
        )
    finally:
        session.close()


def on_pipeline_job_finished(job_id: int) -> None:
    """Called by tasks.run_pipeline after every job. Moves an autopilot
    episode forward when its project's run ends; no-op for normal projects."""
    session = new_session()
    try:
        job = session.get(Job, job_id)
        if job is None:
            return
        episode = (
            session.query(Episode)
            .filter(Episode.project_id == job.project_id, Episode.status == "generating")
            .first()
        )
        if episode is None:
            return
        if job.status == "succeeded":
            paths = core.project_paths(session, episode)
            if paths and (paths.output / "final.mp4").exists():
                package_episode.delay(episode.id)
            return
        if job.status == "cancelled":
            _fail(session, episode, "the render was cancelled")
            return
        # A failed create usually means a provider rate limit left some
        # scenes without assets (make_video stops before render, everything
        # done so far is saved) — resume a couple of times on its own.
        failures = (
            session.query(Job)
            .filter(Job.project_id == job.project_id, Job.status == "failed")
            .count()
        )
        project = session.get(Project, job.project_id)
        if failures <= core.STUCK_RESUME_LIMIT and job.kind in ("create", "resume"):
            paths = core.project_paths(session, episode)
            if paths and paths.scenes_json.exists():
                log.info("autopilot: auto-resuming episode %s (failure %d)", episode.id, failures)
                enqueue_pipeline(
                    session,
                    project,
                    "resume",
                    ["--scenes-file", str(paths.scenes_json)],
                    countdown=AUTO_RESUME_DELAY_SEC,
                )
                return
        _fail(session, episode, job.error or f"pipeline job {job.kind} failed")
    finally:
        session.close()


@celery_app.task(name="renderflow.autopilot.package_episode")
def package_episode(episode_id: int) -> None:
    """Preview clip + review message; episode -> ready."""
    session = new_session()
    try:
        episode = session.get(Episode, episode_id)
        if episode is None or episode.status != "generating":
            return
        paths = core.project_paths(session, episode)
        final = paths.output / "final.mp4"
        thumb = paths.output / "thumbnail.jpg"
        episode.status = "ready"
        episode.error = None
        episode.updated_at = time.time()
        session.commit()

        messenger = messenger_or_none()
        if messenger is None:
            return
        cost = episode.llm_cost + core.project_cost(paths)
        kind_label = "how-to" if episode.kind == "howto" else "fact story"
        caption = (
            f"🎬 <b>Ready for review</b> — episode #{episode.id}\n\n"
            f"<b>{escape(episode.title)}</b>\n"
            f"Niche: {escape(episode.niche)} ({kind_label})\n"
            f"Trend: {escape(episode.trend)}\n"
            f"Length: {core.video_duration_label(final)} · cost ${cost:.2f}\n\n"
            "Watch the preview below, then tap a button."
        )
        buttons = core.episode_buttons(episode)
        try:
            if thumb.exists():
                messenger.send_photo(thumb, caption, buttons)
            else:
                messenger.send_text(caption, buttons)
            seo = (
                f"<b>Title</b>\n{escape(episode.title)}\n\n"
                f"<b>Description</b>\n{escape(episode.description)}\n\n"
                f"<b>Tags</b>\n{escape(', '.join(episode.tags))}"
            )
            messenger.send_text(seo)
            preview = core.build_preview(final, paths.output / "preview.mp4")
            messenger.send_video(preview, f"Preview: first {core.PREVIEW_SECONDS}s of #{episode.id}")
        except Exception as exc:
            log.warning("telegram review message failed for episode %s: %s", episode.id, exc)
    finally:
        session.close()


@celery_app.task(name="renderflow.autopilot.upload_episode")
def upload_episode(episode_id: int) -> None:
    session = new_session()
    try:
        episode = session.get(Episode, episode_id)
        if episode is None or episode.status != "uploading":
            return
        channel = session.get(AutopilotChannel, episode.channel_id)
        paths = core.project_paths(session, episode)
        try:
            publisher = build_publisher(Settings.load())
            result = publisher.upload(
                paths.output / "final.mp4",
                episode.title,
                episode.description,
                episode.tags,
                thumbnail=paths.output / "thumbnail.jpg",
                privacy=channel.privacy,
            )
        except Exception as exc:
            session.rollback()
            episode = session.get(Episode, episode_id)
            _fail(session, episode, f"upload failed: {exc}")
            return
        episode.status = "uploaded"
        episode.youtube_video_id = result.video_id
        episode.youtube_url = result.url
        episode.uploaded_at = time.time()
        episode.updated_at = time.time()
        session.commit()
        privacy = result.meta.get("privacy") or channel.privacy
        lines = [
            f"✅ <b>Uploaded</b> — {escape(episode.title)}",
            result.url,
            f"Privacy: {privacy}",
        ]
        if privacy != "public":
            lines.append(
                "Make it public in YouTube Studio: "
                f"https://studio.youtube.com/video/{result.video_id}/edit"
            )
        if result.meta.get("thumbnail_error"):
            lines.append(
                "⚠️ Custom thumbnail not set (the channel may need phone "
                "verification) — upload output/thumbnail.jpg in Studio."
            )
        notify("\n".join(lines))
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Actions (bot + dashboard)
# ---------------------------------------------------------------------------

ACTIONS = ("approve", "skip", "thumb", "newtopic")


class ActionError(Exception):
    pass


def start_new_episode(session: Session, channel: AutopilotChannel, exclude: list[str] | None = None) -> Episode:
    if core.active_episode(session, channel):
        raise ActionError("an episode is already being made — wait for it, or skip it first")
    episode = core.new_episode(session, channel)
    start_episode.delay(episode.id, exclude or [])
    return episode


def perform_action(session: Session, episode: Episode, action: str) -> str:
    """Apply one review action; returns a short human-readable result."""
    if action == "approve":
        if episode.youtube_video_id:
            raise ActionError(f"already uploaded: {episode.youtube_url}")
        paths = core.project_paths(session, episode)
        retry_upload = episode.status == "failed" and paths and (paths.output / "final.mp4").exists()
        if episode.status != "ready" and not retry_upload:
            raise ActionError(f"episode #{episode.id} is {episode.status}, not ready to upload")
        episode.status = "uploading"
        episode.updated_at = time.time()
        session.commit()
        upload_episode.delay(episode.id)
        return "Uploading to YouTube…"

    if action == "skip":
        if episode.status in ("uploading", "uploaded"):
            raise ActionError(f"episode #{episode.id} is already {episode.status}")
        episode.status = "skipped"
        episode.updated_at = time.time()
        session.commit()
        if episode.project_id is not None:
            # Don't burn hours rendering a video nobody will upload.
            from renderflow.db import active_job

            job = active_job(session, episode.project_id)
            if job is not None:
                cancel_job(session, job)
        return "Skipped. No upload today unless you ask for a new topic."

    if action == "thumb":
        if episode.status != "ready":
            raise ActionError(f"episode #{episode.id} is {episode.status} — a new thumbnail needs a finished video")
        project = session.get(Project, episode.project_id)
        paths = core.project_paths(session, episode)
        episode.status = "generating"
        episode.updated_at = time.time()
        session.commit()
        enqueue_pipeline(
            session, project, "thumbnail", ["--scenes-file", str(paths.scenes_json), "--thumbnail-only"]
        )
        return "Making a new thumbnail — I'll resend the video when it's done."

    if action == "newtopic":
        if episode.status in ("uploading", "uploaded"):
            raise ActionError(f"episode #{episode.id} is already {episode.status}")
        if episode.status in core.EPISODE_ACTIVE_STATUSES:
            raise ActionError("this episode is still being made — skip it first, then ask for a new topic")
        channel = session.get(AutopilotChannel, episode.channel_id)
        if episode.status != "skipped":
            episode.status = "skipped"
            session.commit()
        new = start_new_episode(session, channel, exclude=[episode.topic] if episode.topic else [])
        return f"Picking a different topic (episode #{new.id})…"

    raise ActionError(f"unknown action {action!r}")
