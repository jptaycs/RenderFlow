"""ElevenLabs text-to-speech provider."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from renderflow.providers.base import GeneratedAsset
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.elevenlabs")

API_BASE = "https://api.elevenlabs.io/v1"
# Approximate Creator-tier rate; keep in sync with the plan in use.
COST_PER_CHAR = 0.00018


class ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(
        self, api_key: str | None = None, model_id: str = "eleven_multilingual_v2"
    ) -> None:
        self.api_key = api_key or os.environ["ELEVENLABS_API_KEY"]
        self.model_id = model_id

    @retryable(attempts=3, exceptions=(httpx.HTTPError,))
    def synthesize(self, text: str, voice: str, **params: Any) -> GeneratedAsset:
        log.info("synthesizing %d chars with voice %s", len(text), voice)
        response = httpx.post(
            f"{API_BASE}/text-to-speech/{voice}",
            headers={"xi-api-key": self.api_key},
            params={"output_format": "mp3_44100_128"},
            json={"text": text, "model_id": self.model_id, **params},
            timeout=120.0,
        )
        response.raise_for_status()
        return GeneratedAsset(
            data=response.content,
            provider=self.name,
            params={"voice": voice, "model_id": self.model_id, **params},
            cost=len(text) * COST_PER_CHAR,
            meta={"characters": len(text), "format": "mp3_44100_128"},
        )
