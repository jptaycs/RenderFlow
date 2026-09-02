"""69labs image generation (job-queue REST API).

Paid alternative alongside pollinations/flux-replicate/pexels — enable
with RENDERFLOW_IMAGE_PROVIDER=labs69 (needs LABS69_API_KEY). Submits
POST /images/generate, polls GET /images/status/{id}, downloads
GET /images/download/{id}. Model IDs, aspect ratios, and resolutions are
account/plan-specific and can change — GET /images/models is the source
of truth; this adapter defaults to nano-banana-2 (the API's own default)
and lets RENDERFLOW_LABS69_IMAGE_MODEL / _RESOLUTION override it rather
than hardcoding a model this account might not have access to.
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

log = logging.getLogger("renderflow.providers.labs69_image")

# RenderFlow renders 1920x1080 — every model in the current catalog lists
# 16:9 among its aspectRatios, so this is a safe constant default (same
# pattern as flux_replicate.py hardcoding "16:9").
ASPECT_RATIO = "16:9"


class Labs69Image:
    name = "labs69"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        resolution: str | None = None,
    ) -> None:
        self.client = Labs69Client(api_key=api_key)
        self.model = model or os.environ.get("RENDERFLOW_LABS69_IMAGE_MODEL") or None
        # Only some models accept `resolution` — omit unless configured, so
        # accounts on a model without resolutions never get a 400 for it.
        self.resolution = resolution or os.environ.get(
            "RENDERFLOW_LABS69_IMAGE_RESOLUTION"
        )

    def generate(
        self, prompt: str, negative_prompt: str | None = None, **params: Any
    ) -> GeneratedAsset:
        # No separate negative_prompt input in the API — fold it into the
        # prompt text, same as the pollinations adapter.
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}. Avoid: {negative_prompt}"
        body: dict[str, Any] = {"prompt": full_prompt, "aspectRatio": ASPECT_RATIO}
        if self.model:
            body["model"] = self.model
        if self.resolution:
            body["resolution"] = self.resolution
        body.update(params)

        log.info("submitting image job to 69labs (%s)", self.model or "default model")
        submitted = self.client.submit("/images/generate", body)
        job_id = submitted["id"]
        status = self.client.poll_until_terminal(f"/images/status/{job_id}")
        require_completed(status, context="69labs image job")
        data = self.client.download(f"/images/download/{job_id}")

        meta = status.get("outputMetadata") or {}
        # assets.py always writes scene images to a hardcoded `.png` path
        # regardless of asset.meta["format"] (same as pollinations/flux) —
        # this just records what 69labs actually reported for provenance.
        return GeneratedAsset(
            data=data,
            provider=self.name,
            params={
                "prompt": full_prompt,
                "model": self.model,
                "resolution": self.resolution,
                "aspectRatio": ASPECT_RATIO,
                "job_id": job_id,
            },
            cost=cost_from_status(status),
            meta={"format": meta.get("format", "png")},
        )
