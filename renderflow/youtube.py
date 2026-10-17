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

# Anchored to the repo root (same pattern as tasks.py's REPO_ROOT), not a
# bare relative path — api.py calls is_connected() in-process (see the
# module docstring above; that one check is a local file-existence read,
# not an external API call, so it doesn't need the subprocess indirection
# publish_youtube.py uses for the actual upload). A relative path would
# silently resolve against whatever the API/worker process's cwd happens
# to be instead of always the repo root. Found in a full-app scan 2026-09.
REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN_PATH = REPO_ROOT / ".youtube_token.json"
CLIENT_SECRET_PATH = REPO_ROOT / ".youtube_client_secret.json"
# youtube.upload alone covers both videos.insert and thumbnails.set —
# verified against the live API docs 2026-09 rather than assumed; no need
# for the broader (and riskier, full read/write) `youtube` scope.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
# "Education" — fits trivia/documentary content. Overridable per call.
DEFAULT_CATEGORY_ID = "27"


class YouTubeNotConnected(RuntimeError):
    """No valid token file — run scripts/setup_youtube.py first."""


def channel_token_path(channel_name: str) -> Path:
    """Where a content channel's *own* YouTube login is stored (added
    2026-10, client request: a separate YouTube channel for the How-To
    Hacks content channel). `scripts/setup_youtube.py --channel NAME`
    writes it; `.youtube_token.json` stays the main account's."""
    from renderflow.storage import slugify

    return REPO_ROOT / f".youtube_token.{slugify(channel_name)}.json"


def resolve_token_path(
    channel_name: str | None = None, separate_channels: tuple[str, ...] = ()
) -> Path | None:
    """The token a video from `channel_name` must be uploaded with.

    A channel with its own token file always uses it. A channel listed in
    `separate_channels` (RENDERFLOW_SEPARATE_YOUTUBE_CHANNELS) but not yet
    connected resolves to None, never to the main account — falling back
    would publish its videos to the wrong YouTube channel. Everything else
    uses the main `.youtube_token.json`, exactly as before.
    """
    if channel_name:
        own = channel_token_path(channel_name)
        if own.exists():
            return own
        if channel_name in separate_channels:
            return None
    return TOKEN_PATH


def is_connected(
    channel_name: str | None = None, separate_channels: tuple[str, ...] = ()
) -> bool:
    token = resolve_token_path(channel_name, separate_channels)
    return token is not None and token.exists()


def _credentials(token_path: Path | None = None):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = token_path or TOKEN_PATH
    if not token_path.exists():
        raise YouTubeNotConnected(
            f"no YouTube credentials at {token_path.name} — run "
            "`python scripts/setup_youtube.py` once (add `--channel NAME` "
            "for a separate channel; see CLAUDE.md for the one-time Google "
            "Cloud OAuth setup)"
        )
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
    return creds


def _client(token_path: Path | None = None):
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=_credentials(token_path))


def channel_title(token_path: Path | None = None) -> str:
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
    response = _client(token_path).channels().list(part="snippet", mine=True).execute()
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
    token_path: Path | None = None,
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

    youtube = _client(token_path)
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
        # Best-effort, never fails the whole publish (added 2026-09, client
        # report: "the shorts has no thumbnail when its in youtube
        # studio"). Custom thumbnails via thumbnails.set are documented as
        # unreliable/unsupported for Shorts specifically (YouTube's own
        # Jan-2025 custom-Shorts-cover feature is gated to the YouTube
        # Partner Program and only editable through Studio's own UI, not
        # confirmed as an API capability at all — see the CLAUDE.md note)
        # and even for a regular upload it needs a phone-verified channel;
        # before this fix, either failure mode raised here and the video
        # — already successfully uploaded a moment earlier — never got its
        # youtube.json written, so the dashboard showed the publish as
        # failed even though it had actually succeeded on YouTube.
        try:
            thumb_media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
            youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
            log.info("set custom thumbnail for %s", video_id)
        except Exception:
            log.warning(
                "could not set a custom thumbnail for %s (video itself "
                "still published) — this is expected for Shorts on most "
                "channels, see CLAUDE.md", video_id, exc_info=True,
            )

    return {"video_id": video_id, "url": f"https://youtu.be/{video_id}"}
