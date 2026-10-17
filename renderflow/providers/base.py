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
class VideoProvider(Protocol):
    """Stock-video search (B-roll) — returns a downloaded clip, not
    generation (the no-AI-video-generation cost rule stands)."""

    name: str

    def find_clip(
        self, prompt: str, min_duration_sec: float, **params: Any
    ) -> GeneratedAsset: ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str, **params: Any) -> LLMResult: ...


# ---------------------------------------------------------------------------
# Autopilot (renderflow/autopilot/): trend discovery, publishing, messaging.
# ---------------------------------------------------------------------------


@dataclass
class TrendItem:
    """One trending search/video, normalized across trend sources."""

    title: str
    source: str  # e.g. "google-trends", "youtube-trending"
    # Rough popularity signal as the source reports it ("200K+" searches,
    # view count, ...) — for the LLM picker and the dashboard, not math.
    traffic: str = ""
    # Related headlines / video description snippets: context for finding
    # an evergreen angle without the picker inventing what the trend is.
    context: list[str] = field(default_factory=list)
    url: str = ""


@runtime_checkable
class TrendProvider(Protocol):
    name: str

    def fetch(self, **params: Any) -> list[TrendItem]: ...


@dataclass
class PublishResult:
    video_id: str
    url: str
    provider: str
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VideoPublisher(Protocol):
    name: str

    def upload(
        self,
        video: Path,
        title: str,
        description: str,
        tags: list[str],
        thumbnail: Path | None = None,
        **params: Any,
    ) -> PublishResult: ...


@runtime_checkable
class Messenger(Protocol):
    """Approval channel (Telegram today). Buttons are (label, callback_data)
    rows; callback_data comes back verbatim when a button is tapped."""

    name: str

    def send_text(
        self, text: str, buttons: list[list[tuple[str, str]]] | None = None
    ) -> None: ...

    def send_photo(
        self,
        photo: Path,
        caption: str,
        buttons: list[list[tuple[str, str]]] | None = None,
    ) -> None: ...

    def send_video(self, video: Path, caption: str) -> None: ...

    def poll(self, timeout: int = 30) -> list[dict[str, Any]]: ...
