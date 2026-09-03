"""YouTube Data API v3 publishing — upload the finished video + thumbnail.

One-time OAuth setup: scripts/setup_youtube.py (installed-app flow — opens
your browser once; a refresh token is then stored in .youtube_token.json
and silently refreshed on every future call, no re-consent needed). See
CLAUDE.md for the full Google Cloud Console walkthrough.

Single-channel dev shortcut, not a per-user "Connect YouTube" flow: the
token file is shared by whoever runs the dashboard/worker on this host,
matching how every other paid-provider credential in this app works (one
key per host, not per user). See CLAUDE.md for the path to a real per-user
OAuth flow if RenderFlow ever needs one.

Never called from the API request handler directly — publish_youtube.py
is spawned as a subprocess by tasks.py, same as make_video.py, since a
multi-hundred-MB upload can take minutes (the "never render synchronously
inside a request" rule applies here just as much as to the pipeline).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("renderflow.youtube")

TOKEN_PATH = Path(".youtube_token.json")
CLIENT_SECRET_PATH = Path(".youtube_client_secret.json")
# youtube.upload alone covers both videos.insert and thumbnails.set —
# verified against the live API docs 2026-09 rather than assumed; no need
# for the broader (and riskier, full read/write) `youtube` scope.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
# "Education" — fits trivia/documentary content. Overridable per call.
DEFAULT_CATEGORY_ID = "27"


class YouTubeNotConnected(RuntimeError):
    """No valid token file — run scripts/setup_youtube.py first."""


def is_connected() -> bool:
    return TOKEN_PATH.exists()


def _credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not TOKEN_PATH.exists():
        raise YouTubeNotConnected(
            "no YouTube credentials — run `python scripts/setup_youtube.py` "
            "once (see CLAUDE.md for the one-time Google Cloud OAuth setup)"
        )
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return creds


def _client():
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=_credentials())


def channel_title() -> str:
    """The connected channel's display name — used by setup_youtube.py to
    confirm which account got authorized, and available for a dashboard
    "Connected as X" indicator later.

    Needs a broader scope than SCOPES actually requests: `channels.list`
    403s with "insufficient authentication scopes" under plain
    `youtube.upload` (confirmed live 2026-09) — it needs `youtube.readonly`
    or the full `youtube` scope. Deliberately NOT widening SCOPES for this:
    the upload feature itself (videos.insert + thumbnails.set) only needs
    `youtube.upload`, and this is a nice-to-have confirmation, not a
    dependency — widening scope for it would ask users to grant more
    access than the app needs. Callers (setup_youtube.py) must treat this
    as best-effort and keep working if it raises.
    """
    response = _client().channels().list(part="snippet", mine=True).execute()
    items = response.get("items", [])
    if not items:
        raise YouTubeNotConnected(
            "authorized, but no YouTube channel exists on this Google account"
        )
    return items[0]["snippet"]["title"]


def upload_video(
    video_path: Path,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    privacy_status: str = "public",
    contains_synthetic_media: bool = True,
    made_for_kids: bool = False,
    thumbnail_path: Path | None = None,
    category_id: str = DEFAULT_CATEGORY_ID,
) -> dict[str, Any]:
    """Upload video_path, optionally set a custom thumbnail, return
    {"video_id": ..., "url": "https://youtu.be/..."}.

    `contains_synthetic_media` sets `status.containsSyntheticMedia` — the
    YouTube Data API's disclosure flag for altered/synthetic content, added
    2024-10-30. Defaults to True since every RenderFlow video is an
    AI-narrated voice over AI-generated visuals; only pass False if you've
    deliberately decided a specific upload doesn't need it.
    """
    from googleapiclient.http import MediaFileUpload

    youtube = _client()
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": made_for_kids,
            "containsSyntheticMedia": contains_synthetic_media,
        },
    }
    media = MediaFileUpload(
        str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4"
    )
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info("upload progress: %d%%", int(status.progress() * 100))
    video_id = response["id"]
    log.info("uploaded video %s (%r)", video_id, title)

    if thumbnail_path is not None and thumbnail_path.exists():
        thumb_media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
        youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
        log.info("set custom thumbnail for %s", video_id)

    return {"video_id": video_id, "url": f"https://youtu.be/{video_id}"}
