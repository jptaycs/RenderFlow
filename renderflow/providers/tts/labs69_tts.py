"""69labs text-to-speech (job-queue REST API).

Paid alternative alongside kokoro/piper/elevenlabs — enable with
RENDERFLOW_TTS_PROVIDER=labs69 (needs LABS69_API_KEY). `voice`
(settings.tts_voice) is passed straight through as `voiceId`; which wire
voice-provider that ID belongs to (elevenlabs/edgetts/minimax, per the
69labs docs) is a separate setting since a bare ID doesn't say which
backend it's for — RENDERFLOW_LABS69_VOICE_PROVIDER, default "elevenlabs".

No length_scale/sentence_pause handling here, matching elevenlabs_tts.py:
make_video.py only builds those tts_params for piper/kokoro (see the
`elif settings.tts_provider == "kokoro"` branch there) — cloud TTS
providers are expected to produce natural pacing on their own.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from renderflow.providers.base import GeneratedAsset
from renderflow.providers.labs69_client import (
    Labs69Client,
    cost_from_status,
    require_completed,
)

log = logging.getLogger("renderflow.providers.labs69_tts")


class Labs69TTS:
    name = "labs69"

    def __init__(
        self,
        api_key: str | None = None,
        voice_provider: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.client = Labs69Client(api_key=api_key)
        self.voice_provider = voice_provider or os.environ.get(
            "RENDERFLOW_LABS69_VOICE_PROVIDER", "elevenlabs"
        )
        self.model_id = model_id or os.environ.get("RENDERFLOW_LABS69_TTS_MODEL") or None

    def synthesize(self, text: str, voice: str, **params: Any) -> GeneratedAsset:
        body: dict[str, Any] = {
            "text": text,
            "voiceId": voice,
            "voiceProvider": self.voice_provider,
        }
        if self.model_id:
            body["modelId"] = self.model_id
        body.update(params)

        log.info(
            "submitting %d chars to 69labs tts (%s voice %s)",
            len(text), self.voice_provider, voice,
        )
        submitted = self.client.submit("/tts/generate", body)
        job_id = submitted["id"]
        status = self.client.poll_until_terminal(
            f"/tts/status/{job_id}", interval=3.0, timeout=300.0
        )
        require_completed(status, context="69labs TTS job")
        data = self.client.download(f"/tts/download/{job_id}")

        return GeneratedAsset(
            data=data,
            provider=self.name,
            params={
                "voiceId": voice,
                "voiceProvider": self.voice_provider,
                "modelId": self.model_id,
                "job_id": job_id,
            },
            cost=cost_from_status(status),
            # 69labs doesn't report an audio outputMetadata format in the
            # docs' examples; every voiceProvider it wraps (elevenlabs,
            # edgetts, minimax) defaults to mp3 output.
            meta={"characters": len(text), "format": "mp3"},
        )
