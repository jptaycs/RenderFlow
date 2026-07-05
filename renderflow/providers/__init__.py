"""Provider registry — config selects the active provider per category."""

from __future__ import annotations

from renderflow.config import Settings
from renderflow.providers.base import ImageProvider, LLMProvider, TTSProvider


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "claude":
        from renderflow.providers.llm.claude import ClaudeLLM

        return ClaudeLLM(model=settings.llm_model)
    raise ValueError(f"unknown LLM provider: {settings.llm_provider}")


def build_image(settings: Settings) -> ImageProvider:
    if settings.image_provider == "flux-replicate":
        from renderflow.providers.image.flux_replicate import FluxReplicate

        return FluxReplicate()
    if settings.image_provider == "pollinations":
        from renderflow.providers.image.pollinations import PollinationsImage

        return PollinationsImage()
    raise ValueError(f"unknown image provider: {settings.image_provider}")


def build_tts(settings: Settings) -> TTSProvider:
    if settings.tts_provider == "elevenlabs":
        from renderflow.providers.tts.elevenlabs_tts import ElevenLabsTTS

        return ElevenLabsTTS()
    if settings.tts_provider == "piper":
        from renderflow.providers.tts.piper_tts import PiperTTS

        return PiperTTS()
    raise ValueError(f"unknown TTS provider: {settings.tts_provider}")
