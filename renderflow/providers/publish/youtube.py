"""YouTube Data API v3 uploader (OAuth refresh token, plain httpx).

Uses the same token files as the dashboard's Publish button
(renderflow/youtube.py, written by scripts/setup_youtube.py): google-auth's
Credentials JSON carries the client_id, client_secret and refresh_token this
adapter refreshes with. Which file a channel uploads with is decided by
renderflow.youtube.resolve_token_path.

Things learned from YouTube's rules that shape this adapter:
- Every upload sets status.containsSyntheticMedia=true — the "altered or
  synthetic content" disclosure YouTube requires for realistic AI people.
  The host is an AI avatar, so this is never optional here.
- Uploads from an API project that hasn't passed Google's compliance audit
  are locked to PRIVATE regardless of the privacyStatus requested. The
  default privacy is therefore "private"; the user flips it public in
  YouTube Studio until the audit is approved.
- Custom thumbnails need a phone-verified channel; thumbnail failure is
  logged and returned in meta, never fatal to the upload.
- Quota: videos.insert = 1600 units of the 10,000/day default — fine for
  one video a day.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from renderflow.providers.base import PublishResult
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.youtube")

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
THUMBNAIL_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"
# "Education" — the closest standard category for facts + how-to videos.
DEFAULT_CATEGORY = "27"
TITLE_LIMIT = 100
DESCRIPTION_LIMIT = 4900  # API limit is 5000 bytes; leave headroom
TAGS_LIMIT = 480  # API limit is 500 chars incl. separators/quotes


def default_token_file() -> Path:
    from renderflow.youtube import TOKEN_PATH

    return TOKEN_PATH


def _clean(text: str) -> str:
    # The API rejects "<" and ">" in titles and descriptions.
    return text.replace("<", "").replace(">", "")


def clean_title(title: str) -> str:
    title = " ".join(_clean(title).split())
    return title if len(title) <= TITLE_LIMIT else title[: TITLE_LIMIT - 1].rstrip() + "…"


def clean_description(description: str) -> str:
    description = _clean(description)
    encoded = description.encode()
    if len(encoded) <= DESCRIPTION_LIMIT:
        return description
    return encoded[:DESCRIPTION_LIMIT].decode(errors="ignore").rstrip() + "…"


def clean_tags(tags: list[str]) -> list[str]:
    kept: list[str] = []
    used = 0
    for tag in tags:
        tag = " ".join(_clean(tag).replace(",", " ").split())
        if not tag or tag.lower() in (t.lower() for t in kept):
            continue
        cost = len(tag) + (2 if " " in tag else 0) + 1  # quotes + comma
        if used + cost > TAGS_LIMIT:
            break
        kept.append(tag)
        used += cost
    return kept


def build_metadata(
    title: str,
    description: str,
    tags: list[str],
    privacy: str = "private",
    category_id: str = DEFAULT_CATEGORY,
    publish_at: str | None = None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "privacyStatus": privacy,
        "selfDeclaredMadeForKids": False,
        "containsSyntheticMedia": True,
        "embeddable": True,
    }
    if publish_at:
        # Scheduled publishing requires privacyStatus=private.
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at
    return {
        "snippet": {
            "title": clean_title(title),
            "description": clean_description(description),
            "tags": clean_tags(tags),
            "categoryId": category_id,
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": status,
    }


class YouTubePublisher:
    name = "youtube"

    def __init__(self, token_file: Path | None = None) -> None:
        self.token_file = token_file or default_token_file()
        if not self.token_file.exists():
            raise ValueError(
                f"YouTube is not connected — run scripts/setup_youtube.py "
                f"(expected {self.token_file})"
            )
        self.creds = json.loads(self.token_file.read_text())
        self._access_token: str | None = None
        self._expires_at = 0.0

    def _token(self) -> str:
        if self._access_token and time.time() < self._expires_at - 60:
            return self._access_token
        response = httpx.post(
            TOKEN_URL,
            data={
                "client_id": self.creds["client_id"],
                "client_secret": self.creds["client_secret"],
                "refresh_token": self.creds["refresh_token"],
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"YouTube token refresh failed ({response.status_code}): "
                f"{response.text[:300]} — re-run scripts/setup_youtube.py"
            )
        data = response.json()
        self._access_token = data["access_token"]
        self._expires_at = time.time() + float(data.get("expires_in", 3600))
        return self._access_token

    def channel_title(self) -> str | None:
        response = httpx.get(
            CHANNELS_URL,
            params={"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=30,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        return items[0]["snippet"]["title"] if items else None

    @retryable(attempts=3, base_delay=10.0, exceptions=(httpx.TransportError,))
    def _upload_video(self, video: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        size = video.stat().st_size
        start = httpx.post(
            UPLOAD_URL,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "Authorization": f"Bearer {self._token()}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(size),
            },
            content=json.dumps(metadata),
            timeout=60,
        )
        if start.status_code != 200:
            raise RuntimeError(
                f"YouTube upload init failed ({start.status_code}): {start.text[:500]}"
            )
        location = start.headers["Location"]
        with video.open("rb") as fh:
            done = httpx.put(
                location,
                headers={
                    "Authorization": f"Bearer {self._token()}",
                    "Content-Type": "video/mp4",
                    "Content-Length": str(size),
                },
                content=fh,
                timeout=httpx.Timeout(60, write=None, read=600),
            )
        if done.status_code not in (200, 201):
            raise RuntimeError(
                f"YouTube upload failed ({done.status_code}): {done.text[:500]}"
            )
        return done.json()

    def _set_thumbnail(self, video_id: str, thumbnail: Path) -> str | None:
        """Returns an error string on failure (non-fatal), None on success."""
        try:
            response = httpx.post(
                THUMBNAIL_URL,
                params={"videoId": video_id},
                headers={
                    "Authorization": f"Bearer {self._token()}",
                    "Content-Type": "image/jpeg",
                },
                content=thumbnail.read_bytes(),
                timeout=120,
            )
        except httpx.HTTPError as exc:
            return str(exc)
        if response.status_code != 200:
            return f"{response.status_code}: {response.text[:300]}"
        return None

    def upload(
        self,
        video: Path,
        title: str,
        description: str,
        tags: list[str],
        thumbnail: Path | None = None,
        **params: Any,
    ) -> PublishResult:
        metadata = build_metadata(
            title,
            description,
            tags,
            privacy=params.get("privacy", "private"),
            publish_at=params.get("publish_at"),
        )
        result = self._upload_video(video, metadata)
        video_id = result["id"]
        meta: dict[str, Any] = {
            "privacy": result.get("status", {}).get("privacyStatus"),
            "upload_status": result.get("status", {}).get("uploadStatus"),
        }
        if thumbnail is not None and thumbnail.exists():
            error = self._set_thumbnail(video_id, thumbnail)
            meta["thumbnail_error"] = error
            if error:
                log.warning("YouTube thumbnail not set for %s: %s", video_id, error)
        log.info("uploaded %s to YouTube as %s (%s)", video.name, video_id, meta["privacy"])
        return PublishResult(
            video_id=video_id,
            url=f"https://youtu.be/{video_id}",
            provider=self.name,
            meta=meta,
        )
