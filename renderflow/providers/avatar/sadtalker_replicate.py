"""Real lip-synced talking avatar via SadTalker on Replicate.

One portrait image + narration audio -> video where the presenter's lips
follow the audio and the head moves naturally (blinks, small pose changes).
Compute is billed per second, so cost scales with narration length; a short
clip is roughly $0.08-0.15.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
import replicate

from renderflow.providers.avatar.postprocess import ensure_wav, overlay_disclosure
from renderflow.providers.base import GeneratedAsset
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.sadtalker")

MODEL = "cjwbw/sadtalker"
# Pinned for reproducibility; latest as of 2026-07.
VERSION = "a519cc0cfebaaeade068b23899165a11ec76aaa1d2b313d40d214f204ec957a3"
# The model runs on an Nvidia A100 (80GB); keep in sync with
# https://replicate.com/pricing. Cost is computed from measured predict_time;
# this fallback matches Replicate's "typical run" figure.
A100_COST_PER_SECOND = 0.0014
FALLBACK_COST_PER_RUN = 0.084


class SadTalkerReplicate:
    name = "sadtalker-replicate"

    def __init__(
        self,
        version: str = VERSION,
        preprocess: str = "full",
        still_mode: bool = False,
        use_enhancer: bool = True,
        size_of_image: int = 512,
        expression_scale: float = 1.0,
    ) -> None:
        if not os.getenv("REPLICATE_API_TOKEN"):
            raise ValueError(
                "REPLICATE_API_TOKEN is required for the sadtalker-replicate "
                "avatar provider — set it in .env, or set "
                "RENDERFLOW_AVATAR_PROVIDER=ffmpeg-still for the free placeholder"
            )
        self.version = version
        self.preprocess = preprocess
        self.still_mode = still_mode
        self.use_enhancer = use_enhancer
        self.size_of_image = size_of_image
        self.expression_scale = expression_scale

    @retryable(attempts=2, base_delay=10.0)
    def generate_clip(
        self,
        avatar_image: Path,
        voice_audio: Path,
        script_text: str,
        **params: Any,
    ) -> GeneratedAsset:
        disclosure = params.get("disclosure") or "AI-generated host"
        inputs: dict[str, Any] = {
            "preprocess": self.preprocess,
            "still_mode": self.still_mode,
            "use_enhancer": self.use_enhancer,
            "use_eyeblink": True,
            "size_of_image": self.size_of_image,
            "expression_scale": self.expression_scale,
        }
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            audio = ensure_wav(voice_audio, work)
            log.info(
                "generating lip-sync clip via %s (%d chars narration)",
                MODEL, len(script_text),
            )
            with open(avatar_image, "rb") as image_f, open(audio, "rb") as audio_f:
                prediction = replicate.predictions.create(
                    version=self.version,
                    input={"source_image": image_f, "driven_audio": audio_f, **inputs},
                )
                prediction.wait()
            if prediction.status != "succeeded":
                raise RuntimeError(
                    f"sadtalker prediction {prediction.status}: {prediction.error}"
                )
            output = prediction.output
            url = output[0] if isinstance(output, list) else output
            response = httpx.get(url, timeout=600.0, follow_redirects=True)
            response.raise_for_status()
            data = overlay_disclosure(response.content, disclosure, work)

        predict_time = (prediction.metrics or {}).get("predict_time")
        cost = (
            predict_time * A100_COST_PER_SECOND
            if predict_time
            else FALLBACK_COST_PER_RUN
        )
        return GeneratedAsset(
            data=data,
            provider=self.name,
            params={
                "model": MODEL,
                "version": self.version,
                "avatar_image": str(avatar_image),
                "voice_audio": str(voice_audio),
                "script_chars": len(script_text),
                "disclosure": disclosure,
                **inputs,
            },
            cost=cost,
            meta={"format": "mp4", "predict_time": predict_time},
        )

