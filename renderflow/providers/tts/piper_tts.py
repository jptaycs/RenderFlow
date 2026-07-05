"""Piper local TTS — free, MIT-licensed voices, runs offline.

Voice models are downloaded once with:
    python -m piper.download_voices en_US-lessac-medium --data-dir .voices
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Any

from renderflow.providers.base import GeneratedAsset

log = logging.getLogger("renderflow.providers.piper")

DEFAULT_VOICES_DIR = Path(".voices")


class PiperTTS:
    name = "piper"

    def __init__(self, voices_dir: Path | None = None) -> None:
        self.voices_dir = voices_dir or DEFAULT_VOICES_DIR
        self._voices: dict[str, Any] = {}

    def _load(self, voice: str) -> Any:
        if voice not in self._voices:
            from piper import PiperVoice

            model = self.voices_dir / f"{voice}.onnx"
            if not model.exists():
                raise FileNotFoundError(
                    f"piper voice model not found: {model} — download with "
                    f"`python -m piper.download_voices {voice} --data-dir {self.voices_dir}`"
                )
            self._voices[voice] = PiperVoice.load(str(model))
        return self._voices[voice]

    def synthesize(self, text: str, voice: str, **params: Any) -> GeneratedAsset:
        log.info("piper synthesizing %d chars with %s", len(text), voice)
        piper_voice = self._load(voice)
        length_scale = params.pop("length_scale", None)
        syn_config = None
        if length_scale is not None:
            from piper.config import SynthesisConfig

            syn_config = SynthesisConfig(length_scale=float(length_scale))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            piper_voice.synthesize_wav(text, wav, syn_config=syn_config)
        return GeneratedAsset(
            data=buf.getvalue(),
            provider=self.name,
            params={"voice": voice, "length_scale": length_scale, **params},
            cost=0.0,
            meta={
                "format": "wav",
                "characters": len(text),
                "length_scale": length_scale,
            },
        )
