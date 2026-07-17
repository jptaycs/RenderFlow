"""Provider registry — config selects the active provider per category."""

from __future__ import annotations

from renderflow.config import Settings
from renderflow.providers.base import (
    AvatarProvider,
    ImageProvider,
    LLMProvider,
    TTSProvider,
    VideoProvider,
)


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "claude":
        from renderflow.providers.llm.claude import ClaudeLLM

        return ClaudeLLM(model=settings.llm_model)
    raise ValueError(f"unknown LLM provider: {settings.llm_provider}")


def build_image(settings: Settings, name: str | None = None) -> ImageProvider:
    """Build an image provider — `name` overrides the configured default
    (used for the thumbnail's per-image provider split)."""
    name = name or settings.image_provider
    if name == "flux-replicate":
        from renderflow.providers.image.flux_replicate import FluxReplicate

        return FluxReplicate()
    if name == "pollinations":
        from renderflow.providers.image.pollinations import PollinationsImage

        return PollinationsImage()
    if name == "pexels":
        from renderflow.providers.image.pexels import PexelsImage

        return PexelsImage()
    raise ValueError(f"unknown image provider: {name}")


def build_tts(settings: Settings) -> TTSProvider:
    if settings.tts_provider == "elevenlabs":
        from renderflow.providers.tts.elevenlabs_tts import ElevenLabsTTS

        return ElevenLabsTTS()
    if settings.tts_provider == "piper":
        from renderflow.providers.tts.piper_tts import PiperTTS

        return PiperTTS()
    if settings.tts_provider == "kokoro":
        from renderflow.providers.tts.kokoro_tts import KokoroTTS

        return KokoroTTS()
    raise ValueError(f"unknown TTS provider: {settings.tts_provider}")


def build_broll(settings: Settings) -> VideoProvider | None:
    """Stock-video B-roll provider, or None when disabled (the default).

    B-roll is optional end to end: None simply means every scene renders
    from its still image, exactly as before the feature existed."""
    if not settings.broll_provider:
        return None
    if settings.broll_provider == "pexels-video":
        from renderflow.providers.video.pexels_video import PexelsVideo

        return PexelsVideo()
    raise ValueError(f"unknown b-roll provider: {settings.broll_provider}")


def build_avatar(settings: Settings) -> AvatarProvider:
    if settings.avatar_provider == "ffmpeg-still":
        from renderflow.providers.avatar.ffmpeg_still import FFMpegStillAvatar

        return FFMpegStillAvatar()
    if settings.avatar_provider == "sadtalker-replicate":
        from renderflow.providers.avatar.sadtalker_replicate import SadTalkerReplicate

        return SadTalkerReplicate()
    if settings.avatar_provider == "memo-hf":
        from renderflow.providers.avatar.memo_hf import MemoHFAvatar

        return MemoHFAvatar()
    if settings.avatar_provider == "wav2lip-local":
        from renderflow.providers.avatar.wav2lip_local import Wav2LipLocal

        return Wav2LipLocal()
    raise ValueError(f"unknown avatar provider: {settings.avatar_provider}")
