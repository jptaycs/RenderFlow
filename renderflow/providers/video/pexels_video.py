"""Pexels free stock-video search — real B-roll footage, $0.

Same API key, license terms, and adapter patterns as the photo provider
(providers/image/pexels.py): keyword extraction from the pipeline's prompt
shapes (shared pexels_query module), request throttling, per-query dedup so
regenerates return a different clip, random pick from the top results, and
facey alt-text deprioritized (no pixel-level face check on video — too
heavy; this reorder is the mitigation).

File-variant choice: the smallest rendition at least 1920px wide (the
renderer crops to 1920x1080), else the widest available — stock originals
can be 4K, no need to download those.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import httpx

from renderflow.providers.base import GeneratedAsset
from renderflow.providers.pexels_query import (
    FACEY_ALT_RE,
    candidate_queries,
    search_query,
)
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.pexels_video")

API_URL = "https://api.pexels.com/videos/search"
PER_PAGE = 20
TOP_POOL = 8
MIN_INTERVAL = 0.5
# Don't bother with clips shorter than this — sub-3s loops look stuttery.
MIN_CLIP_SEC = 3.0
MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024


class PexelsVideo:
    name = "pexels-video"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY") or None
        if not self.api_key:
            raise ValueError(
                "pexels-video provider needs PEXELS_API_KEY in .env "
                "(free key from pexels.com/api)"
            )
        self._used_ids: dict[str, set[int]] = {}
        self._next_allowed = 0.0

    def _throttle(self) -> None:
        wait = self._next_allowed - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next_allowed = time.monotonic() + MIN_INTERVAL

    def _search(self, query: str) -> list[dict[str, Any]]:
        self._throttle()
        response = httpx.get(
            API_URL,
            params={
                "query": query,
                "orientation": "landscape",
                "size": "medium",
                "per_page": PER_PAGE,
            },
            headers={"Authorization": self.api_key},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("videos", [])

    @staticmethod
    def _pick_file(video: dict[str, Any]) -> dict[str, Any] | None:
        """Smallest mp4 rendition ≥1920 wide, else the widest one."""
        files = [
            f for f in video.get("video_files", [])
            if (f.get("file_type") or "").startswith("video/mp4") and f.get("width")
        ]
        if not files:
            return None
        big_enough = [f for f in files if f["width"] >= 1920]
        if big_enough:
            return min(big_enough, key=lambda f: f["width"])
        return max(files, key=lambda f: f["width"])

    @retryable(attempts=4, base_delay=5.0, max_delay=30.0, exceptions=(httpx.HTTPError,))
    def find_clip(
        self, prompt: str, min_duration_sec: float, **params: Any
    ) -> GeneratedAsset:
        query = search_query(prompt)
        videos: list[dict[str, Any]] = []
        used_query = query
        for candidate in candidate_queries(query):
            log.info("searching pexels videos for %r", candidate)
            videos = self._search(candidate)
            if videos:
                used_query = candidate
                break
        min_len = max(min_duration_sec, MIN_CLIP_SEC)
        usable = [
            v for v in videos
            if v.get("duration", 0) >= min_len and self._pick_file(v) is not None
        ]
        # Shorter clips loop in the renderer — accept them before giving up.
        if not usable:
            usable = [v for v in videos if self._pick_file(v) is not None]
        if not usable:
            raise ValueError(f"pexels found no usable videos for {query!r}")

        used = self._used_ids.setdefault(query, set())
        fresh = [v for v in usable if v["id"] not in used] or usable
        fresh.sort(
            key=lambda v: bool(
                FACEY_ALT_RE.search((v.get("alt") or "") + " " + (v.get("url") or ""))
            )
        )
        video = random.choice(fresh[:TOP_POOL])
        used.add(video["id"])
        file = self._pick_file(video)

        self._throttle()
        payload = httpx.get(file["link"], timeout=300.0, follow_redirects=True)
        payload.raise_for_status()
        if len(payload.content) > MAX_DOWNLOAD_BYTES:
            raise httpx.HTTPError(
                f"pexels video too large ({len(payload.content)} bytes)"
            )
        return GeneratedAsset(
            data=payload.content,
            provider=self.name,
            params={
                "query": used_query,
                "video_id": video["id"],
                "width": file["width"],
                "duration": video.get("duration"),
            },
            cost=0.0,
            meta={
                "videographer": (video.get("user") or {}).get("name"),
                "video_url": video.get("url"),
            },
        )
