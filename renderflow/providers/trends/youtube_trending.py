"""YouTube "most popular" chart via the YouTube Data API v3.

Needs YOUTUBE_API_KEY (a plain API key from the same Google Cloud project
as the upload OAuth client; videos.list costs 1 quota unit of the 10,000
free daily units). Optional: without a key this source is skipped and the
picker runs on Google Trends alone.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from renderflow.providers.base import TrendItem
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.youtube_trending")

API_URL = "https://www.googleapis.com/youtube/v3/videos"


def _views_label(count: str | None) -> str:
    if not count:
        return ""
    n = int(count)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M views"
    if n >= 1_000:
        return f"{n // 1_000}K views"
    return f"{n} views"


def parse_response(data: dict) -> list[TrendItem]:
    items: list[TrendItem] = []
    for video in data.get("items", []):
        snippet = video.get("snippet", {})
        title = (snippet.get("title") or "").strip()
        if not title:
            continue
        description = (snippet.get("description") or "").strip().split("\n")[0]
        context = [c for c in (snippet.get("channelTitle"), description[:200]) if c]
        items.append(
            TrendItem(
                title=title,
                source="youtube-trending",
                traffic=_views_label(video.get("statistics", {}).get("viewCount")),
                context=context,
                url=f"https://www.youtube.com/watch?v={video.get('id', '')}",
            )
        )
    return items


class YouTubeTrending:
    name = "youtube-trending"

    def __init__(self, api_key: str | None = None, region: str = "US") -> None:
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY") or None
        if not self.api_key:
            raise ValueError("youtube-trending needs YOUTUBE_API_KEY in .env")
        self.region = region

    @retryable(attempts=3, exceptions=(httpx.HTTPError,))
    def fetch(self, **params: Any) -> list[TrendItem]:
        response = httpx.get(
            API_URL,
            params={
                "part": "snippet,statistics",
                "chart": "mostPopular",
                "regionCode": params.get("geo", self.region),
                "maxResults": 50,
                "key": self.api_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        items = parse_response(response.json())
        log.info("youtube trending: %d videos for region=%s", len(items), self.region)
        return items
