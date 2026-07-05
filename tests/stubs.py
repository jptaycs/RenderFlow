"""Stub providers for tests — no network, deterministic output."""

from __future__ import annotations

import json
from typing import Any

from renderflow.providers.base import GeneratedAsset, LLMResult

CANNED_SCRIPT = {
    "title": "Test Video",
    "style": "documentary",
    "scenes": [
        {
            "id": i,
            "duration_estimate_sec": 15,
            "narration": f"Narration for scene {i}.",
            "image_prompt": f"Cinematic still {i}",
            "negative_prompt": "text, watermark",
            "motion": {"effect": "zoom_in", "intensity": 0.08},
        }
        for i in (1, 2)
    ],
}


class StubLLM:
    name = "stub-llm"

    def complete(self, system: str, prompt: str, **params: Any) -> LLMResult:
        return LLMResult(text=json.dumps(CANNED_SCRIPT), provider=self.name, cost=0.01)


class StubImage:
    name = "stub-image"

    def generate(
        self, prompt: str, negative_prompt: str | None = None, **params: Any
    ) -> GeneratedAsset:
        return GeneratedAsset(
            data=b"fake-png", provider=self.name, params={"prompt": prompt}, cost=0.003
        )


class StubTTS:
    name = "stub-tts"

    def synthesize(self, text: str, voice: str, **params: Any) -> GeneratedAsset:
        return GeneratedAsset(
            data=b"fake-mp3", provider=self.name, params={"voice": voice}, cost=0.002
        )
