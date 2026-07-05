"""Generate per-scene image and voice assets, tracking state and cost."""

from __future__ import annotations

import logging

from renderflow.providers.base import ImageProvider, TTSProvider
from renderflow.schema import AssetRef, AssetStatus, ScenePlan
from renderflow.storage import ProjectPaths, save_plan

log = logging.getLogger("renderflow.pipeline.assets")


def _skip(ref: AssetRef) -> bool:
    return ref.status is AssetStatus.COMPLETED and ref.path is not None


def generate_images(
    plan: ScenePlan, provider: ImageProvider, paths: ProjectPaths
) -> None:
    for scene in plan.scenes:
        ref = scene.assets.image
        if _skip(ref):
            continue
        ref.advance(AssetStatus.RUNNING)
        save_plan(plan, paths)
        try:
            asset = provider.generate(scene.image_prompt, scene.negative_prompt)
        except Exception:
            ref.advance(AssetStatus.FAILED)
            save_plan(plan, paths)
            raise
        out = paths.images / f"scene_{scene.id:03d}.png"
        out.write_bytes(asset.data)
        ref.path = str(out)
        ref.provider = asset.provider
        ref.cost = asset.cost
        ref.advance(AssetStatus.COMPLETED)
        save_plan(plan, paths)
        log.info("scene %d image done (%s)", scene.id, out.name)


def generate_voice(
    plan: ScenePlan, provider: TTSProvider, voice: str, paths: ProjectPaths
) -> None:
    for scene in plan.scenes:
        ref = scene.assets.voice
        if _skip(ref):
            continue
        ref.advance(AssetStatus.RUNNING)
        save_plan(plan, paths)
        try:
            asset = provider.synthesize(scene.narration, voice)
        except Exception:
            ref.advance(AssetStatus.FAILED)
            save_plan(plan, paths)
            raise
        ext = asset.meta.get("format", "mp3").split("_")[0]
        out = paths.voice / f"scene_{scene.id:03d}.{ext}"
        out.write_bytes(asset.data)
        ref.path = str(out)
        ref.provider = asset.provider
        ref.cost = asset.cost
        ref.advance(AssetStatus.COMPLETED)
        save_plan(plan, paths)
        log.info("scene %d voice done (%s)", scene.id, out.name)
