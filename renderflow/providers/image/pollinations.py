"""Pollinations.ai free image generation (Flux-based, no API key).

Works keyless, but anonymous requests are capped at 1024x576 regardless of
the requested size. A free registered token (auth.pollinations.ai) sent as
a Bearer header lifts the cap — set POLLINATIONS_TOKEN in .env.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any
from urllib.parse import quote

import httpx

from renderflow.providers.base import GeneratedAsset
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.pollinations")

API_BASE = "https://image.pollinations.ai/prompt"
# Published rate limits: anonymous = 1 request / 15 s, registered = 1 / 5 s.
# Bursting past them earns sticky 429s, so space requests out ourselves.
MIN_INTERVAL_ANON = 15.0
MIN_INTERVAL_TOKEN = 5.0


class PollinationsImage:
    name = "pollinations"

    def __init__(self, model: str = "flux", token: str | None = None) -> None:
        self.model = model
        self.token = token or os.environ.get("POLLINATIONS_TOKEN") or None
        self._next_allowed = 0.0

    def _throttle(self) -> None:
        wait = self._next_allowed - time.monotonic()
        if wait > 0:
            log.info("pollinations rate limit: waiting %.1fs", wait)
            time.sleep(wait)
        interval = MIN_INTERVAL_TOKEN if self.token else MIN_INTERVAL_ANON
        self._next_allowed = time.monotonic() + interval

    @retryable(attempts=5, base_delay=20.0, max_delay=60.0, exceptions=(httpx.HTTPError,))
    def generate(
        self, prompt: str, negative_prompt: str | None = None, **params: Any
    ) -> GeneratedAsset:
        # Pollinations has no separate negative_prompt input; fold it into the
        # prompt so it still influences generation.
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}. Avoid: {negative_prompt}"
        seed = params.pop("seed", random.randint(0, 2**31))
        query: dict[str, Any] = {
            "width": 1920,
            "height": 1080,
            "model": self.model,
            "nologo": "true",
            "private": "true",
            "seed": seed,
            **params,
        }
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._throttle()
        log.info(
            "generating image via pollinations (%s, %s)",
            self.model,
            "registered" if self.token else "anonymous",
        )
        response = httpx.get(
            f"{API_BASE}/{quote(full_prompt)}",
            params=query,
            headers=headers,
            timeout=300.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        if not response.content.startswith((b"\x89PNG", b"\xff\xd8", b"RIFF")):
            raise httpx.HTTPError(
                f"pollinations returned non-image payload ({len(response.content)} bytes)"
            )
        return GeneratedAsset(
            data=response.content,
            provider=self.name,
            params={"prompt": full_prompt, **query},
            cost=0.0,
        )
