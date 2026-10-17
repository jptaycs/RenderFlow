"""Dashboard endpoints for the Autopilot tab (mounted by renderflow/api.py).

Setup and monitoring live here; day-to-day approval normally happens in
Telegram, but every review action is mirrored here so the loop still works
when Telegram isn't configured.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from renderflow import youtube as youtube_module
from renderflow.auth import current_user
from renderflow.autopilot import core
from renderflow.config import Settings
from renderflow.db import Episode, Project, User, get_db

router = APIRouter(prefix="/api/autopilot")

BOT_ONLINE_SEC = 120


def _jobs():
    # Imported lazily: jobs imports renderflow.tasks (Celery app), and tests
    # swap these functions out via monkeypatch on this module attribute.
    from renderflow.autopilot import jobs

    return jobs


def _connections(channel_name: str) -> dict[str, bool]:
    settings = Settings.load()  # load_dotenv populates os.environ with .env values
    return {
        "llm": bool(
            os.environ.get("OPENAI_API_KEY" if settings.llm_provider == "openai" else "ANTHROPIC_API_KEY")
        ),
        "telegram": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "telegramChat": bool(os.environ.get("TELEGRAM_CHAT_ID")),
        "youtube": youtube_module.is_connected(channel_name, settings.separate_youtube_channels),
        "youtubeTrending": bool(os.environ.get("YOUTUBE_API_KEY")),
    }


def _episode_view(session: Session, episode: Episode) -> dict[str, Any]:
    project = session.get(Project, episode.project_id) if episode.project_id else None
    paths = core.project_paths(session, episode)
    return {
        "id": episode.id,
        "day": episode.day,
        "status": episode.status,
        "niche": episode.niche,
        "kind": episode.kind,
        "trend": episode.trend,
        "topic": episode.topic,
        "angle": episode.angle,
        "title": episode.title,
        "description": episode.description,
        "tags": episode.tags,
        "candidates": episode.candidates,
        "projectSlug": project.slug if project else None,
        "cost": round(episode.llm_cost + core.project_cost(paths), 4),
        "youtubeUrl": episode.youtube_url,
        "youtubeVideoId": episode.youtube_video_id,
        "error": episode.error,
        "createdAt": episode.created_at,
        "uploadedAt": episode.uploaded_at,
    }


@router.get("")
def autopilot_state(
    user: User = Depends(current_user), session: Session = Depends(get_db)
) -> dict[str, Any]:
    channel = core.get_or_create_channel(session, user)
    episodes = (
        session.query(Episode)
        .filter(Episode.channel_id == channel.id)
        .order_by(Episode.created_at.desc())
        .limit(30)
        .all()
    )
    return {
        "channel": {
            "name": channel.name,
            "enabled": channel.enabled,
            "runHour": channel.run_hour,
            "lengthMinutes": channel.length_minutes,
            "privacy": channel.privacy,
            "niches": channel.niches or list(core.DEFAULT_NICHES),
        },
        "botOnline": bool(
            channel.heartbeat_at and time.time() - channel.heartbeat_at < BOT_ONLINE_SEC
        ),
        "connections": _connections(channel.name),
        "episodes": [_episode_view(session, e) for e in episodes],
        "today": core.today(),
    }


class Niche(BaseModel):
    name: str
    kind: str
    description: str = ""


class ChannelUpdate(BaseModel):
    name: str
    enabled: bool
    runHour: int
    # Ignored: every episode is a Short (see core's module docstring). Still
    # accepted so an open dashboard tab from before the change can save.
    lengthMinutes: float | None = None
    privacy: str
    niches: list[Niche]


@router.put("/channel")
def update_channel(
    body: ChannelUpdate,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    name = " ".join(body.name.split())
    if not name:
        raise HTTPException(422, "channel name must not be empty")
    if not 0 <= body.runHour <= 23:
        raise HTTPException(422, "run hour must be 0-23")
    if body.privacy not in ("private", "unlisted", "public"):
        raise HTTPException(422, "privacy must be private, unlisted, or public")
    niches = []
    for n in body.niches:
        nm = " ".join(n.name.split())
        if not nm:
            continue
        if n.kind not in core.KINDS:
            raise HTTPException(422, f"niche {nm!r}: kind must be 'fact' or 'howto'")
        niches.append({"name": nm, "kind": n.kind, "description": n.description.strip()})
    if len(niches) < 2:
        raise HTTPException(422, "keep at least 2 niches — the picker never repeats yesterday's")
    channel = core.get_or_create_channel(session, user)
    channel.name = name
    channel.enabled = body.enabled
    channel.run_hour = body.runHour
    channel.privacy = body.privacy
    channel.niches = niches
    return {"ok": "saved"}


@router.post("/run-now", status_code=202)
def run_now(
    user: User = Depends(current_user), session: Session = Depends(get_db)
) -> dict[str, int]:
    channel = core.get_or_create_channel(session, user)
    session.commit()
    jobs = _jobs()
    try:
        episode = jobs.start_new_episode(session, channel)
    except jobs.ActionError as exc:
        raise HTTPException(409, str(exc))
    return {"episodeId": episode.id}


@router.post("/episodes/{episode_id}/{action}")
def episode_action(
    episode_id: int,
    action: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    jobs = _jobs()
    if action not in jobs.ACTIONS:
        raise HTTPException(404, f"unknown action {action!r}")
    channel = core.get_or_create_channel(session, user)
    episode = session.get(Episode, episode_id)
    if episode is None or episode.channel_id != channel.id:
        raise HTTPException(404, f"no episode {episode_id}")
    try:
        message = jobs.perform_action(session, episode, action)
    except jobs.ActionError as exc:
        raise HTTPException(409, str(exc))
    return {"message": message}
