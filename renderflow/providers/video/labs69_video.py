"""69labs AI video generation as B-roll (job-queue REST API).

Enable with RENDERFLOW_BROLL_PROVIDER=labs69 (needs LABS69_API_KEY). This
DELIBERATELY reverses the earlier "no AI video generation, stock footage
only" cost-control decision recorded in CLAUDE.md's "What NOT to do" — the
user explicitly asked for real AI-generated B-roll despite the added
per-clip cost, 2026-09. pexels-video (free stock search) is unaffected and
stays the $0 default when RENDERFLOW_BROLL_PROVIDER is left unset.

Unlike pexels-video, this is generation, not search: the scene's full
`image_prompt` (already a face-free, topic-anchored photo caption — see
pipeline/script.py) is used directly as the video prompt instead of being
reduced to the 2-4 search keywords a stock search needs. Valid
`duration`/`aspectRatio` values are model-specific and change over time,
so they're read from GET /videos/models at call time (cached for the life
of the provider instance) rather than hardcoded, per the API docs'
"fetch at runtime, don't hardcode ids or costs" guidance.
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

log = logging.getLogger("renderflow.providers.labs69_video")

# RenderFlow renders 1920x1080; used only if the resolved model actually
# lists 16:9 as a valid aspectRatio (see _pick_aspect_ratio).
DEFAULT_ASPECT_RATIO = "16:9"


class Labs69Video:
    name = "labs69"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.client = Labs69Client(api_key=api_key)
        self.model = model or os.environ.get("RENDERFLOW_LABS69_VIDEO_MODEL") or None
        self._model_info: dict[str, Any] | None = None
        self._model_info_fetched = False

    def _lookup_model(self) -> dict[str, Any] | None:
        """Best-effort model-catalog lookup for the active model's valid
        durations/aspectRatios. Fetched once and cached; a failure here
        just means generation falls back to omitting those optional
        params rather than blocking B-roll entirely (B-roll is optional
        end to end — see generate_broll)."""
        if self._model_info_fetched:
            return self._model_info
        self._model_info_fetched = True
        try:
            catalog = self.client.get_json("/videos/models")
        except Exception:
            log.warning(
                "could not fetch 69labs /videos/models, using request defaults",
                exc_info=True,
            )
            return None
        models = catalog.get("models") or []
        target = self.model or catalog.get("defaultModelId")
        self._model_info = next((m for m in models if m.get("id") == target), None)
        return self._model_info

    @staticmethod
    def _pick_duration(
        model_info: dict[str, Any] | None, min_duration_sec: float
    ) -> str | None:
        """Smallest valid duration that covers the scene, else the longest
        available (render._render_broll_clip loops a too-short clip and
        trims a too-long one, so either direction is safe)."""
        durations = (model_info or {}).get("durations")
        if not durations:
            return None
        parsed = sorted((float(d), d) for d in durations)
        for value, raw in parsed:
            if value >= min_duration_sec:
                return raw
        return parsed[-1][1]

    @staticmethod
    def _pick_aspect_ratio(model_info: dict[str, Any] | None) -> str | None:
        ratios = (model_info or {}).get("aspectRatios")
        if not ratios:
            return None
        values = {r.get("value") for r in ratios if isinstance(r, dict)}
        return DEFAULT_ASPECT_RATIO if DEFAULT_ASPECT_RATIO in values else None

    def find_clip(
        self,
        prompt: str,
        min_duration_sec: float,
        negative_prompt: str | None = None,
        **params: Any,
    ) -> GeneratedAsset:
        model_info = self._lookup_model()
        full_prompt = f"{prompt}. Avoid: {negative_prompt}" if negative_prompt else prompt
        body: dict[str, Any] = {
            "prompt": full_prompt,
            # render._render_broll_clip never maps the clip's own audio
            # track, so stripping it server-side just saves download size.
            "mute": True,
        }
        if self.model:
            body["model"] = self.model
        aspect_ratio = self._pick_aspect_ratio(model_info)
        if aspect_ratio:
            body["aspectRatio"] = aspect_ratio
        duration = self._pick_duration(model_info, min_duration_sec)
        if duration:
            body["duration"] = duration
        body.update(params)

        log.info(
            "submitting b-roll video job to 69labs (%s, target >=%.1fs)",
            self.model or "default model", min_duration_sec,
        )
        submitted = self.client.submit("/videos/generate", body)
        job_id = submitted["id"]
        status = self.client.poll_until_terminal(
            f"/videos/status/{job_id}", interval=8.0, timeout=900.0
        )
        require_completed(status, context="69labs video job")
        data = self.client.download(f"/videos/download/{job_id}")

        meta = status.get("outputMetadata") or {}
        return GeneratedAsset(
            data=data,
            provider=self.name,
            params={
                "prompt": full_prompt,
                "model": self.model,
                "duration": duration,
                "aspectRatio": aspect_ratio,
                "job_id": job_id,
            },
            cost=cost_from_status(status),
            meta={"duration_sec": meta.get("durationSeconds")},
        )
