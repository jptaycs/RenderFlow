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


class StubVideo:
    name = "stub-video"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def find_clip(
        self, prompt: str, min_duration_sec: float, **params: Any
    ) -> GeneratedAsset:
        self.calls.append(prompt)
        if self.fail:
            raise ValueError("stub video failure")
        return GeneratedAsset(
            data=b"fake-broll-mp4",
            provider=self.name,
            params={"prompt": prompt, "min_duration": min_duration_sec},
            cost=0.0,
            meta={"videographer": "Stub", "video_url": "https://example.com/v/1"},
        )


class StubAvatar:
    name = "stub-avatar"

    def generate_clip(
        self, avatar_image, voice_audio, script_text: str, **params: Any
    ) -> GeneratedAsset:
        return GeneratedAsset(
            data=b"fake-mp4",
            provider=self.name,
            params={
                "avatar_image": str(avatar_image),
                "voice_audio": str(voice_audio),
                "script_chars": len(script_text),
            },
            cost=0.004,
            meta={"format": "mp4"},
        )
