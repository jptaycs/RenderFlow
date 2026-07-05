"""Pollinations.ai free image generation (Flux-based, no API key)."""

from __future__ import annotations

import logging
import random
from typing import Any
from urllib.parse import quote

import httpx

from renderflow.providers.base import GeneratedAsset
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.pollinations")

API_BASE = "https://image.pollinations.ai/prompt"


class PollinationsImage:
    name = "pollinations"

    def __init__(self, model: str = "flux") -> None:
        self.model = model

    @retryable(attempts=4, base_delay=5.0, exceptions=(httpx.HTTPError,))
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
        log.info("generating image via pollinations (%s)", self.model)
        response = httpx.get(
            f"{API_BASE}/{quote(full_prompt)}",
            params=query,
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
