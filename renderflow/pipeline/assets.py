"""Generate per-scene image and voice assets, tracking state and cost."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from renderflow.providers.base import AvatarProvider, ImageProvider, TTSProvider
from renderflow.schema import AssetRef, AssetStatus, Scene, ScenePlan
from renderflow.storage import ProjectPaths, save_plan

log = logging.getLogger("renderflow.pipeline.assets")


def _skip(ref: AssetRef) -> bool:
    return ref.status is AssetStatus.COMPLETED and ref.path is not None


def generate_images(
    plan: ScenePlan,
    provider: ImageProvider,
    paths: ProjectPaths,
    avatar_image: Path | None = None,
) -> None:
    for scene in plan.scenes:
        ref = scene.assets.image
        if not _skip(ref):
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

        avatar_ref = scene.assets.avatar_image
        if scene.type != "talking_avatar" or _skip(avatar_ref):
            continue
        avatar_ref.advance(AssetStatus.RUNNING)
        save_plan(plan, paths)
        avatar_out = paths.images / f"scene_{scene.id:03d}_avatar{_image_ext(avatar_image)}"
        if avatar_image is not None:
            shutil.copyfile(avatar_image, avatar_out)
            avatar_ref.path = str(avatar_out)
            avatar_ref.provider = "local-file"
            avatar_ref.cost = 0.0
            avatar_ref.advance(AssetStatus.COMPLETED)
            save_plan(plan, paths)
            log.info("scene %d avatar image copied (%s)", scene.id, avatar_out.name)
            continue
        try:
            avatar_asset = provider.generate(_avatar_image_prompt(scene), scene.negative_prompt)
        except Exception:
            avatar_ref.advance(AssetStatus.FAILED)
            save_plan(plan, paths)
            raise
        avatar_out = paths.images / f"scene_{scene.id:03d}_avatar.png"
        avatar_out.write_bytes(avatar_asset.data)
        avatar_ref.path = str(avatar_out)
        avatar_ref.provider = avatar_asset.provider
        avatar_ref.cost = avatar_asset.cost
        avatar_ref.advance(AssetStatus.COMPLETED)
        save_plan(plan, paths)
        log.info("scene %d avatar image done (%s)", scene.id, avatar_out.name)


def generate_voice(
    plan: ScenePlan,
    provider: TTSProvider,
    voice: str,
    paths: ProjectPaths,
    **tts_params,
) -> None:
    for scene in plan.scenes:
        ref = scene.assets.voice
        if _skip(ref):
            continue
        ref.advance(AssetStatus.RUNNING)
        save_plan(plan, paths)
        try:
            asset = provider.synthesize(scene.narration, voice, **tts_params)
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


def generate_avatar_clips(
    plan: ScenePlan, provider: AvatarProvider, paths: ProjectPaths
) -> None:
    for scene in plan.scenes:
        if scene.type != "talking_avatar":
            continue

        ref = scene.assets.avatar_clip
        if _skip(ref):
            continue
        avatar_image_path = scene.assets.avatar_image.path or scene.assets.image.path
        if not avatar_image_path or not scene.assets.voice.path:
            raise ValueError(
                f"scene {scene.id} needs completed avatar image and voice before avatar"
            )

        ref.advance(AssetStatus.RUNNING)
        save_plan(plan, paths)
        try:
            asset = provider.generate_clip(
                Path(avatar_image_path),
                Path(scene.assets.voice.path),
                scene.narration,
                disclosure=scene.avatar.disclosure if scene.avatar else None,
                avatar=scene.avatar.model_dump() if scene.avatar else None,
            )
        except Exception:
            ref.advance(AssetStatus.FAILED)
            save_plan(plan, paths)
            raise

        ext = asset.meta.get("format", "mp4").split("_")[0]
        out = paths.avatar / f"scene_{scene.id:03d}.{ext}"
        out.write_bytes(asset.data)
        ref.path = str(out)
        ref.provider = asset.provider
        ref.cost = asset.cost
        ref.advance(AssetStatus.COMPLETED)
        save_plan(plan, paths)
        log.info("scene %d avatar clip done (%s)", scene.id, out.name)


def _avatar_image_prompt(scene: Scene) -> str:
    if not scene.avatar:
        return (
            "Cinematic photorealistic documentary host portrait, middle-aged "
            "male presenter, plain dark shirt, neutral studio lighting, no text"
        )
    parts = [
        "Cinematic photorealistic documentary host portrait",
        scene.avatar.description,
    ]
    if scene.avatar.background:
        parts.append(f"background: {scene.avatar.background}")
    parts.append("waist-up composition, looking at camera, no visible text")
    return ", ".join(parts)


def _image_ext(path: Path | None) -> str:
    if path is None:
        return ".png"
    return path.suffix if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".png"
