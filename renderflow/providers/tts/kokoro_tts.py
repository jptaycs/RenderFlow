"""Kokoro-82M local TTS — free, Apache-2.0, far more natural than Piper.

Model files are downloaded once with:
    python scripts/setup_kokoro.py          (~340 MB into .kokoro/)

Good documentary voices: am_onyx (deep US male), bm_george (older British
male), am_fenrir (energetic US male). Speed 1.0 is natural pace.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

from renderflow.providers.base import GeneratedAsset
from renderflow.providers.tts.piper_tts import _split_sentences

log = logging.getLogger("renderflow.providers.kokoro")

DEFAULT_MODEL_DIR = Path(".kokoro")
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


class KokoroTTS:
    name = "kokoro"

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(
            model_dir
            or os.environ.get("RENDERFLOW_KOKORO_DIR", "")
            or DEFAULT_MODEL_DIR
        )
        self.model_path = self.model_dir / MODEL_FILE
        self.voices_path = self.model_dir / VOICES_FILE
        if not self.model_path.exists() or not self.voices_path.exists():
            raise ValueError(
                f"kokoro model files missing in {self.model_dir} — "
                "run scripts/setup_kokoro.py first"
            )
        self._engine: Any = None

    def _load(self) -> Any:
        if self._engine is None:
            from kokoro_onnx import Kokoro

            log.info("loading kokoro model (%s)", self.model_path.name)
            self._engine = Kokoro(str(self.model_path), str(self.voices_path))
        return self._engine

    def synthesize(self, text: str, voice: str, **params: Any) -> GeneratedAsset:
        import numpy as np
        import soundfile as sf

        speed = float(params.get("speed", 1.0))
        pause_sec = float(params.get("sentence_pause_sec", 0.35))
        engine = self._load()

        chunks: list[np.ndarray] = []
        sample_rate = 24000
        sentences = _split_sentences(text)
        for index, sentence in enumerate(sentences):
            samples, sample_rate = engine.create(
                sentence, voice=voice, speed=speed, lang="en-us"
            )
            chunks.append(np.asarray(samples, dtype=np.float32))
            if index < len(sentences) - 1 and pause_sec > 0:
                # An ellipsis earns a longer, dramatic pause (Piper parity).
                factor = 2.0 if sentence.rstrip().endswith("…") else 1.0
                chunks.append(
                    np.zeros(int(pause_sec * factor * sample_rate), dtype=np.float32)
                )

        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        log.info(
            "kokoro synthesized %.1fs (voice=%s, speed=%.2f)",
            len(audio) / sample_rate, voice, speed,
        )
        return GeneratedAsset(
            data=buffer.getvalue(),
            provider=self.name,
            params={"voice": voice, "speed": speed, "sentence_pause_sec": pause_sec},
            cost=0.0,
            meta={"format": "wav", "sample_rate": sample_rate},
        )
