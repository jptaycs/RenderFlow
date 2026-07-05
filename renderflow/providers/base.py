"""Provider abstraction layer.

No module calls an external API directly — all AI calls go through these
interfaces. Switching providers is a config change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class GeneratedAsset:
    """One generated artifact plus everything needed to reproduce it."""

    data: bytes
    provider: str
    params: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    """Text completion plus its cost — cost tracking from day one."""

    text: str
    provider: str
    cost: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ImageProvider(Protocol):
    name: str

    def generate(
        self, prompt: str, negative_prompt: str | None = None, **params: Any
    ) -> GeneratedAsset: ...


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    def synthesize(self, text: str, voice: str, **params: Any) -> GeneratedAsset: ...


@runtime_checkable
class AvatarProvider(Protocol):
    name: str

    def generate_clip(
        self,
        avatar_image: Path,
        voice_audio: Path,
        script_text: str,
        **params: Any,
    ) -> GeneratedAsset: ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str, **params: Any) -> LLMResult: ...
