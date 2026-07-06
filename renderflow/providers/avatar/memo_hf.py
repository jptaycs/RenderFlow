"""Free lip-synced talking avatar via the MEMO Space on Hugging Face ZeroGPU.

One portrait image + narration audio -> video with lip sync, blinks, and
expressive head motion. Costs $0.00 but is best-effort: calls queue behind
other users, ZeroGPU enforces a per-user GPU quota (a free HF_TOKEN raises
it), and public Spaces can change or go down without notice. The paid
sadtalker-replicate provider is the reliable fallback.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from renderflow.providers.avatar.postprocess import ensure_wav, overlay_disclosure
from renderflow.providers.base import GeneratedAsset
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.memo_hf")

SPACE = "fffiloni/MEMO"


class MemoHFAvatar:
    name = "memo-hf"

    def __init__(self, space: str = SPACE, seed: int = 0) -> None:
        self.space = space
        self.seed = seed
        self.hf_token = os.getenv("HF_TOKEN") or None
        self._client: Any = None

    def _connect(self) -> Any:
        if self._client is None:
            import httpx
            from gradio_client import Client

            log.info("connecting to Space %s", self.space)
            self._client = Client(
                self.space,
                token=self.hf_token,
                verbose=False,
                # Generation takes minutes on the shared queue; never cut off
                # a slow read, only a dead connection.
                httpx_kwargs={"timeout": httpx.Timeout(30.0, read=None)},
            )
        return self._client

    @retryable(attempts=2, base_delay=30.0)
    def generate_clip(
        self,
        avatar_image: Path,
        voice_audio: Path,
        script_text: str,
        **params: Any,
    ) -> GeneratedAsset:
        from gradio_client import handle_file

        disclosure = params.get("disclosure") or "AI-generated host"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            audio = ensure_wav(voice_audio, work)
            log.info(
                "generating lip-sync clip via %s (%d chars narration) — "
                "free ZeroGPU queue, this can take several minutes",
                self.space, len(script_text),
            )
            result = self._connect().predict(
                input_video=handle_file(str(avatar_image)),
                input_audio=handle_file(str(audio)),
                seed=self.seed,
                api_name="/generate",
            )
            clip_path = Path(result["video"] if isinstance(result, dict) else result)
            data = overlay_disclosure(clip_path.read_bytes(), disclosure, work)
        return GeneratedAsset(
            data=data,
            provider=self.name,
            params={
                "space": self.space,
                "seed": self.seed,
                "avatar_image": str(avatar_image),
                "voice_audio": str(voice_audio),
                "script_chars": len(script_text),
                "disclosure": disclosure,
            },
            cost=0.0,
            meta={"format": "mp4"},
        )

