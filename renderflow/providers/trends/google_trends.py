"""Google Trends "Trending now" daily RSS feed — keyless, free.

The feed (trends.google.com/trending/rss?geo=US) lists today's breakout
searches with an approximate search volume and a few related news
headlines. There is no official free Trends API; this RSS feed is the
stable public surface (the unofficial pytrends scraper breaks regularly).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from renderflow.providers.base import TrendItem
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.google_trends")

FEED_URL = "https://trends.google.com/trending/rss"


def _local(tag: str) -> str:
    """Tag name without its XML namespace ({ns}approx_traffic -> approx_traffic)."""
    return tag.rsplit("}", 1)[-1]


def parse_feed(xml_text: str) -> list[TrendItem]:
    root = ET.fromstring(xml_text)
    items: list[TrendItem] = []
    for item in root.iter("item"):
        title = traffic = link = ""
        context: list[str] = []
        for child in item:
            name = _local(child.tag)
            if name == "title":
                title = (child.text or "").strip()
            elif name == "approx_traffic":
                traffic = (child.text or "").strip()
            elif name == "link":
                link = (child.text or "").strip()
            elif name == "news_item":
                for sub in child:
                    if _local(sub.tag) == "news_item_title" and sub.text:
                        context.append(sub.text.strip())
        if title:
            items.append(
                TrendItem(
                    title=title,
                    source="google-trends",
                    traffic=traffic,
                    context=context[:3],
                    url=link,
                )
            )
    return items


class GoogleTrendsRSS:
    name = "google-trends"

    def __init__(self, geo: str = "US") -> None:
        self.geo = geo

    @retryable(attempts=3, exceptions=(httpx.HTTPError,))
    def fetch(self, **params: Any) -> list[TrendItem]:
        response = httpx.get(
            FEED_URL,
            params={"geo": params.get("geo", self.geo)},
            headers={"User-Agent": "Mozilla/5.0 (RenderFlow trend scan)"},
            timeout=30,
            follow_redirects=True,
        )
        response.raise_for_status()
        items = parse_feed(response.text)
        log.info("google trends: %d items for geo=%s", len(items), self.geo)
        return items
