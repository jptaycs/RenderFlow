"""Telegram Bot API messenger — the autopilot's approval remote control.

Setup: talk to @BotFather in Telegram, /newbot, put the token in
TELEGRAM_BOT_TOKEN. Then send your bot any message and run
`python -m renderflow.autopilot.bot --whoami` (or just start the bot: it
replies with your chat id) and put that in TELEGRAM_CHAT_ID.

Security model: the bot only ever acts on updates from TELEGRAM_CHAT_ID —
anyone else who finds the bot gets nothing but their own chat id back (so
pairing works), never a command executed.

Uses long polling (getUpdates), not a webhook: the worker PC sits behind a
home router with no public URL. Bots can upload at most 50 MB per file, so
the autopilot sends a short low-res preview clip, never the full render.
"""

from __future__ import annotations

import html
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.telegram")

API_ROOT = "https://api.telegram.org"
CAPTION_LIMIT = 1024
MESSAGE_LIMIT = 4096


def escape(text: str) -> str:
    return html.escape(text, quote=False)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _markup(buttons: list[list[tuple[str, str]]] | None) -> str | None:
    if not buttons:
        return None
    return json.dumps(
        {
            "inline_keyboard": [
                [{"text": label, "callback_data": data} for label, data in row]
                for row in buttons
            ]
        }
    )


def normalize_update(update: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten a raw Telegram update into {kind, chat_id, text, callback_id}.

    kind is "message" (text typed by the user) or "callback" (an inline
    button tap; text = the button's callback_data)."""
    if "callback_query" in update:
        cq = update["callback_query"]
        message = cq.get("message") or {}
        return {
            "kind": "callback",
            "chat_id": str((message.get("chat") or {}).get("id", "")),
            "text": cq.get("data", ""),
            "callback_id": cq.get("id"),
        }
    message = update.get("message") or update.get("channel_post")
    if message and "text" in message:
        return {
            "kind": "message",
            "chat_id": str(message["chat"]["id"]),
            "text": message["text"],
            "callback_id": None,
        }
    return None


class TelegramMessenger:
    name = "telegram"

    def __init__(self, token: str | None = None, chat_id: str | None = None) -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN") or None
        if not self.token:
            raise ValueError("telegram messenger needs TELEGRAM_BOT_TOKEN in .env")
        self.chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
        self._offset: int | None = None
        self._client = httpx.Client(timeout=httpx.Timeout(30, read=90))

    def _url(self, method: str) -> str:
        return f"{API_ROOT}/bot{self.token}/{method}"

    @retryable(attempts=3, exceptions=(httpx.TransportError,))
    def _call(self, method: str, data: dict[str, Any] | None = None, files=None) -> Any:
        response = self._client.post(self._url(method), data=data, files=files)
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"telegram {method} failed: {payload.get('description')}")
        return payload["result"]

    def _require_chat(self) -> str:
        if not self.chat_id:
            raise RuntimeError(
                "TELEGRAM_CHAT_ID is not set — message your bot, it replies "
                "with your chat id; put that in .env"
            )
        return self.chat_id

    def send_text(
        self,
        text: str,
        buttons: list[list[tuple[str, str]]] | None = None,
        chat_id: str | None = None,
    ) -> None:
        data = {
            "chat_id": chat_id or self._require_chat(),
            "text": _truncate(text, MESSAGE_LIMIT),
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
        if markup := _markup(buttons):
            data["reply_markup"] = markup
        self._call("sendMessage", data)

    def send_photo(
        self,
        photo: Path,
        caption: str,
        buttons: list[list[tuple[str, str]]] | None = None,
    ) -> None:
        data = {
            "chat_id": self._require_chat(),
            "caption": _truncate(caption, CAPTION_LIMIT),
            "parse_mode": "HTML",
        }
        if markup := _markup(buttons):
            data["reply_markup"] = markup
        with photo.open("rb") as fh:
            self._call("sendPhoto", data, files={"photo": (photo.name, fh, "image/jpeg")})

    def send_video(self, video: Path, caption: str) -> None:
        data = {
            "chat_id": self._require_chat(),
            "caption": _truncate(caption, CAPTION_LIMIT),
            "parse_mode": "HTML",
            "supports_streaming": "true",
        }
        with video.open("rb") as fh:
            self._call("sendVideo", data, files={"video": (video.name, fh, "video/mp4")})

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        try:
            self._call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text})
        except Exception as exc:  # cosmetic (clears the button spinner) — never fatal
            log.warning("answerCallbackQuery failed: %s", exc)

    def poll(self, timeout: int = 30) -> list[dict[str, Any]]:
        """Long-poll for new updates; returns normalized events (all chats —
        the caller decides what to do with strangers)."""
        data: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": json.dumps(["message", "callback_query"]),
        }
        if self._offset is not None:
            data["offset"] = self._offset
        updates = self._call("getUpdates", data)
        events = []
        for update in updates:
            self._offset = update["update_id"] + 1
            event = normalize_update(update)
            if event:
                events.append(event)
        return events
