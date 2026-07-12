"""Pexels free stock-photo search — real photos, not AI generation.

Free API key from pexels.com/api (200 requests/hour, 20k/month); the
license allows commercial use without attribution. Search wants 2-4
keywords, not the long photo-caption prompts the pipeline builds, so
this adapter extracts a short query from the known prompt shapes
(_local_image_prompt, _thumbnail_prompt) with a stopword fallback.

Repeat calls with the same prompt (the face-check retry) return a
*different* photo — there is no seed to vary, so the adapter excludes
already-returned photo ids per query. A manual regenerate runs in a
fresh subprocess where that memory is empty; picking randomly from the
top results (instead of always result #0) keeps regenerates from
returning the identical photo across processes.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Any

import httpx

from renderflow.providers.base import GeneratedAsset
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.pexels")

API_URL = "https://api.pexels.com/v1/search"
PER_PAGE = 30
# Pick randomly among this many top-ranked results — variety without
# drifting into the low-relevance tail of the result list.
TOP_POOL = 10
MIN_INTERVAL = 0.5

# Matches the prompt builders in pipeline/script.py and pipeline/assets.py.
_SCENE_TOPIC_RE = re.compile(r"photograph about (.+?) —")
_SCENE_EXCERPT_RE = re.compile(r"described here:\s*(.+?)\s+Keep ")
_THUMBNAIL_TOPIC_RE = re.compile(r"dramatic photograph of\s*(.+?)[.,]")
# api.py::_vary_prompt appends this on manual regenerate — search junk.
_VARIATION_MARKER = " Try this take: "

_STOPWORDS = frozenset(
    """a an the and or but of in on at to for with from by about into over
    under is are was were be been being it its this that these those his her
    their there here as if then than so not no never only even still very
    while when where which who whom what how all any each every some most
    more much they them he she we you i had has have will would could should
    photograph photo image picture shot wide medium documentary b-roll
    realistic cinematic dramatic lighting composition frame text words
    letters typography face faces people person camera portrait visible
    absolutely never recognizably present directly""".split()
)

# Stock search returns plenty of face-forward shots; scene backgrounds
# must avoid them (see assets._generate_face_free). Photos whose alt text
# looks face-y go to the back of the candidate list — facecheck.py still
# has the final say on the actual pixels.
_FACEY_ALT_RE = re.compile(r"\b(face|portrait|selfie|headshot|smiling)\b", re.IGNORECASE)


def _content_words(text: str, limit: int) -> list[str]:
    words = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
        word = raw.lower()
        if word in _STOPWORDS or word in words:
            continue
        words.append(word)
        if len(words) >= limit:
            break
    return words


def _search_query(prompt: str) -> str:
    """Reduce a pipeline image prompt to a short stock-search query."""
    base = prompt.split(_VARIATION_MARKER)[0]
    topic_match = _SCENE_TOPIC_RE.search(base) or _THUMBNAIL_TOPIC_RE.search(base)
    if topic_match:
        topic_words = _content_words(topic_match.group(1), 3)
        excerpt = _SCENE_EXCERPT_RE.search(base)
        extra = _content_words(excerpt.group(1), 6) if excerpt else []
        words = topic_words + [w for w in extra if w not in topic_words]
        return " ".join(words[:6])
    return " ".join(_content_words(base, 6))


def _candidate_queries(query: str) -> list[str]:
    """The query, then progressively shorter fallbacks for empty results."""
    words = query.split()
    candidates = []
    while words:
        candidates.append(" ".join(words))
        words = words[:-1] if len(words) > 2 else []
    return candidates or [query]


class PexelsImage:
    name = "pexels"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY") or None
        if not self.api_key:
            raise ValueError(
                "pexels image provider needs PEXELS_API_KEY in .env "
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
                "size": "large",
                "per_page": PER_PAGE,
            },
            headers={"Authorization": self.api_key},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("photos", [])

    @retryable(attempts=4, base_delay=5.0, max_delay=30.0, exceptions=(httpx.HTTPError,))
    def generate(
        self, prompt: str, negative_prompt: str | None = None, **params: Any
    ) -> GeneratedAsset:
        # negative_prompt is meaningless for search; the alt-text reorder
        # below plus the pipeline's face-check cover its main job.
        query = _search_query(prompt)
        photos: list[dict[str, Any]] = []
        used_query = query
        for candidate in _candidate_queries(query):
            log.info("searching pexels for %r", candidate)
            photos = self._search(candidate)
            if photos:
                used_query = candidate
                break
        if not photos:
            raise ValueError(f"pexels found no photos for {query!r}")

        used = self._used_ids.setdefault(query, set())
        fresh = [p for p in photos if p["id"] not in used] or photos
        fresh.sort(key=lambda p: bool(_FACEY_ALT_RE.search(p.get("alt") or "")))
        photo = random.choice(fresh[:TOP_POOL])
        used.add(photo["id"])

        self._throttle()
        image = httpx.get(
            photo["src"]["original"], timeout=120.0, follow_redirects=True
        )
        image.raise_for_status()
        if not image.content.startswith((b"\x89PNG", b"\xff\xd8", b"RIFF")):
            raise httpx.HTTPError(
                f"pexels returned non-image payload ({len(image.content)} bytes)"
            )
        return GeneratedAsset(
            data=image.content,
            provider=self.name,
            params={"query": used_query, "photo_id": photo["id"]},
            cost=0.0,
            meta={
                "photographer": photo.get("photographer"),
                "photo_url": photo.get("url"),
                "alt": photo.get("alt"),
            },
        )
