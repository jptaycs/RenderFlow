"""Flux image generation via Replicate."""

from __future__ import annotations

import logging
from typing import Any

import replicate

from renderflow.providers.base import GeneratedAsset
from renderflow.retry import retryable

log = logging.getLogger("renderflow.providers.flux")

# Replicate bills flux-schnell per image; keep in sync with their pricing page.
COST_PER_IMAGE = 0.003


class FluxReplicate:
    name = "flux-replicate"

    def __init__(self, model: str = "black-forest-labs/flux-schnell") -> None:
        self.model = model

    @retryable(attempts=3)
    def generate(
        self, prompt: str, negative_prompt: str | None = None, **params: Any
    ) -> GeneratedAsset:
        # flux-schnell has no negative_prompt input; recorded in params for
        # provenance and used by providers that support it.
        inputs: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "output_format": "png",
            "num_outputs": 1,
            **params,
        }
        log.info("generating image via %s", self.model)
        output = replicate.run(self.model, input=inputs)
        first = output[0] if isinstance(output, list) else output
        data = first.read()
        return GeneratedAsset(
            data=data,
            provider=self.name,
            params={"model": self.model, "negative_prompt": negative_prompt, **inputs},
            cost=COST_PER_IMAGE,
        )
