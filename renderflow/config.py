"""Environment-driven configuration. Never hardcode API keys."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    image_provider: str
    tts_provider: str
    avatar_provider: str
    llm_model: str
    tts_voice: str
    tts_length_scale: float
    avatar_image: Path | None
    projects_dir: Path

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            llm_provider=os.getenv("RENDERFLOW_LLM_PROVIDER", "claude"),
            image_provider=os.getenv("RENDERFLOW_IMAGE_PROVIDER", "flux-replicate"),
            tts_provider=os.getenv("RENDERFLOW_TTS_PROVIDER", "elevenlabs"),
            avatar_provider=os.getenv("RENDERFLOW_AVATAR_PROVIDER", "ffmpeg-still"),
            llm_model=os.getenv("RENDERFLOW_LLM_MODEL", "claude-opus-4-8"),
            tts_voice=os.getenv("RENDERFLOW_TTS_VOICE", "21m00Tcm4TlvDq8ikWAM"),
            tts_length_scale=float(os.getenv("RENDERFLOW_TTS_LENGTH_SCALE", "1.25")),
            avatar_image=(
                Path(value)
                if (value := os.getenv("RENDERFLOW_AVATAR_IMAGE"))
                else None
            ),
            projects_dir=Path(os.getenv("RENDERFLOW_PROJECTS_DIR", "projects")),
        )
