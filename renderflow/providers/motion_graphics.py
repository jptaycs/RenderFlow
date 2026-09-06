"""69labs Motion Graphics: templated Remotion renders (title cards, kinetic
typography, data overlays, ...) via the same job-queue REST pattern as the
other labs69 adapters (image/tts/video) — POST /motion-graphics/render, GET
/motion-graphics/status/{id}, GET /motion-graphics/download/{id}.

Deliberately outside the ImageProvider/TTSProvider/etc. Protocol registry
(providers/base.py): there's exactly one implementation (69labs — no free
or alternate backend for templated motion graphics exists) and it's an
opt-in branding-card upgrade, not a swappable per-scene asset generator —
a Protocol layer here would be ceremony with no real abstraction behind
it, same reasoning as the youtube.py exception documented in CLAUDE.md's
Hard Rules.

Request/response shape verified live 2026-09 against the real API (69labs
has no public/indexed docs to read instead, and GET /motion-graphics/models
doesn't exist — the catalog lives under GET /models -> "motionGraphics"):
the render body is {"templateId": str, "props": {...control values...}} —
a flat top-level body, or "controls"/"inputs" wrappers, both 400 with
"Unrecognized keys: ...". Confirmed live: submitting
{templateId: "kinetic-title-card", props: {title, subtitle, footer,
aspectRatio: "9:16", durationSeconds: 4}} returned a real 1080x1920,
4-second rendered MP4 with those exact field values burned in.
"""

from __future__ import annotations

import logging
from typing import Any

from renderflow.providers.base import GeneratedAsset
from renderflow.providers.labs69_client import (
    Labs69Client,
    cost_from_status,
    require_completed,
)

log = logging.getLogger("renderflow.providers.labs69_motion_graphics")

# Only one of 219 available templates used for v1 — a text-only animated
# title/subtitle/footer card, close enough in shape to the Pillow card it
# replaces (title/subtitle/channel name) to need no per-card template
# selection logic yet. GET /models -> motionGraphics.templates lists every
# other option (data counters, comparison layouts, map animations, ...) if
# this is ever expanded beyond intro/outro cards.
DEFAULT_TEMPLATE_ID = "kinetic-title-card"
# kinetic-title-card's own duration control range (GET /models), reused
# here to clamp a caller-supplied narration-derived duration into what the
# template will actually accept rather than risking a 400 on an
# unusually long/short card.
MIN_DURATION_SEC = 3
MAX_DURATION_SEC = 30


def _aspect_ratio_for(width: int, height: int) -> str:
    if width == height:
        return "1:1"
    return "9:16" if height > width else "16:9"


class Labs69MotionGraphics:
    name = "labs69"

    def __init__(self, api_key: str | None = None) -> None:
        self.client = Labs69Client(api_key=api_key)

    def render_card(
        self,
        *,
        title: str,
        subtitle: str = "",
        footer: str = "",
        width: int,
        height: int,
        duration_seconds: float,
        palette: dict[str, str] | None = None,
        template_id: str = DEFAULT_TEMPLATE_ID,
    ) -> GeneratedAsset:
        props: dict[str, Any] = {
            "title": title,
            "subtitle": subtitle,
            "footer": footer,
            "aspectRatio": _aspect_ratio_for(width, height),
            "durationSeconds": max(
                MIN_DURATION_SEC, min(MAX_DURATION_SEC, round(duration_seconds))
            ),
        }
        if palette:
            props["palette"] = palette
        body = {"templateId": template_id, "props": props}

        log.info("submitting motion-graphics render (%s)", template_id)
        submitted = self.client.submit("/motion-graphics/render", body)
        job_id = submitted["id"]
        status = self.client.poll_until_terminal(
            f"/motion-graphics/status/{job_id}", interval=4.0, timeout=180.0
        )
        require_completed(status, context="69labs motion-graphics job")
        data = self.client.download(f"/motion-graphics/download/{job_id}")

        meta = status.get("outputMetadata") or {}
        return GeneratedAsset(
            data=data,
            provider=self.name,
            params={"templateId": template_id, "props": props, "job_id": job_id},
            cost=cost_from_status(status),
            meta={"format": meta.get("format", "mp4")},
        )
