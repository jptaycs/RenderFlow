"""Generate per-scene image and voice assets, tracking state and cost."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from renderflow.pipeline import facecheck
from renderflow.pipeline.script import (
    scene_is_avatar_solo,
    scene_is_visual_only,
    topic_from_title,
)
from renderflow.pipeline.text_normalize import normalize_for_speech
from renderflow.providers.base import (
    AvatarProvider,
    GeneratedAsset,
    ImageProvider,
    TTSProvider,
    VideoProvider,
)
from renderflow.schema import AssetRef, AssetStatus, Scene, ScenePlan
from renderflow.storage import ProjectPaths, save_plan

log = logging.getLogger("renderflow.pipeline.assets")

# Scene backgrounds and the thumbnail topic image must never show a face
# (the host's own portrait/reaction shot are the deliberate exceptions —
# never wrap those calls in this). Prompt wording alone doesn't reliably
# stop it (see facecheck.py), so retry with a fresh generation instead.
MAX_FACE_RETRIES = 3


def _generate_face_free(
    provider: ImageProvider, prompt: str, negative_prompt: str | None, label: str
) -> GeneratedAsset:
    asset = provider.generate(prompt, negative_prompt)
    for attempt in range(2, MAX_FACE_RETRIES + 1):
        if not facecheck.has_prominent_face(asset.data):
            return asset
        log.info("%s had a prominent face, retrying (%d/%d)", label, attempt, MAX_FACE_RETRIES)
        asset = provider.generate(prompt, negative_prompt)
    if facecheck.has_prominent_face(asset.data):
        log.warning("%s still has a face after %d attempts, keeping it", label, MAX_FACE_RETRIES)
    return asset


def _skip(ref: AssetRef) -> bool:
    return ref.status is AssetStatus.COMPLETED and ref.path is not None


def _start(ref: AssetRef) -> None:
    """Move an asset into RUNNING, routing failed assets through RETRYING.

    An asset already marked RUNNING was orphaned by a crashed/killed run
    (nothing else may run concurrently on a project) — fail it first so it
    can legally re-enter RUNNING.
    """
    if ref.status is AssetStatus.RUNNING:
        ref.advance(AssetStatus.FAILED)
    if ref.status is AssetStatus.FAILED:
        ref.advance(AssetStatus.RETRYING)
    ref.advance(AssetStatus.RUNNING)


def generate_images(
    plan: ScenePlan,
    provider: ImageProvider,
    paths: ProjectPaths,
    avatar_image: Path | None = None,
) -> None:
    """One failed scene must not stop the rest of the batch.

    A single stubborn provider error (rate limit, timeout) used to raise and
    kill the whole run before any other scene was even attempted — costly on
    a 50+ scene video. Failures are logged and left as FAILED for per-scene
    retry from the dashboard; generation continues with the next scene.
    """
    for scene in plan.scenes:
        ref = scene.assets.image
        # Solo-layout scenes (see scene_is_avatar_solo) render the avatar
        # full-screen with no background visual — skip generating an image
        # that render.py will never use, saving cost/time and rate-limit
        # budget on ~1/3 of scenes.
        is_solo = scene.type == "talking_avatar" and scene_is_avatar_solo(scene)
        if not is_solo and not _skip(ref):
            _start(ref)
            save_plan(plan, paths)
            try:
                asset = _generate_face_free(
                    provider, scene.image_prompt, scene.negative_prompt,
                    f"scene {scene.id} image",
                )
            except Exception:
                log.warning("scene %d image failed, continuing", scene.id, exc_info=True)
                ref.advance(AssetStatus.FAILED)
                save_plan(plan, paths)
                continue
            out = paths.images / f"scene_{scene.id:03d}.png"
            out.write_bytes(asset.data)
            ref.path = str(out)
            ref.provider = asset.provider
            ref.cost = asset.cost
            ref.advance(AssetStatus.COMPLETED)
            save_plan(plan, paths)
            log.info("scene %d image done (%s)", scene.id, out.name)

        avatar_ref = scene.assets.avatar_image
        # Visual-only scenes (see scene_is_visual_only) never show the
        # avatar at all — skip the portrait too, not just the background.
        if scene.type != "talking_avatar" or scene_is_visual_only(scene) or _skip(avatar_ref):
            continue
        _start(avatar_ref)
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
            log.warning("scene %d avatar image failed, continuing", scene.id, exc_info=True)
            avatar_ref.advance(AssetStatus.FAILED)
            save_plan(plan, paths)
            continue
        avatar_out = paths.images / f"scene_{scene.id:03d}_avatar.png"
        avatar_out.write_bytes(avatar_asset.data)
        avatar_ref.path = str(avatar_out)
        avatar_ref.provider = avatar_asset.provider
        avatar_ref.cost = avatar_asset.cost
        avatar_ref.advance(AssetStatus.COMPLETED)
        save_plan(plan, paths)
        log.info("scene %d avatar image done (%s)", scene.id, avatar_out.name)


def generate_broll(
    plan: ScenePlan,
    provider: VideoProvider,
    paths: ProjectPaths,
) -> None:
    """Optional stock-video B-roll for scenes rendered full-frame from a
    still (plain narration scenes, and visual-only avatar scenes).

    OPTIONAL by design: a failure is logged and marked FAILED but never
    blocks the render — render.py falls back to the still image, and
    completeness checks (api._refs, make_video._incomplete_scenes)
    deliberately ignore this asset. Same per-scene continue-on-failure
    pattern as generate_images.
    """
    for scene in plan.scenes:
        eligible = scene.type == "narration" or (
            scene.type == "talking_avatar" and scene_is_visual_only(scene)
        )
        if not eligible or scene.broll_mode != "auto":
            continue
        ref = scene.assets.broll
        if _skip(ref):
            continue
        _start(ref)
        save_plan(plan, paths)
        try:
            asset = provider.find_clip(
                scene.image_prompt, min_duration_sec=scene.duration_estimate_sec
            )
        except Exception:
            log.warning("scene %d b-roll failed, continuing", scene.id, exc_info=True)
            ref.advance(AssetStatus.FAILED)
            save_plan(plan, paths)
            continue
        out = paths.broll / f"scene_{scene.id:03d}.mp4"
        out.write_bytes(asset.data)
        ref.path = str(out)
        ref.provider = asset.provider
        ref.cost = asset.cost
        ref.advance(AssetStatus.COMPLETED)
        save_plan(plan, paths)
        log.info("scene %d b-roll done (%s)", scene.id, out.name)


def generate_voice(
    plan: ScenePlan,
    provider: TTSProvider,
    voice: str,
    paths: ProjectPaths,
    **tts_params,
) -> None:
    """One failed scene must not stop the rest of the batch (see generate_images)."""
    for scene in plan.scenes:
        ref = scene.assets.voice
        if _skip(ref):
            continue
        _start(ref)
        save_plan(plan, paths)
        try:
            asset = provider.synthesize(
                normalize_for_speech(scene.narration), voice, **tts_params
            )
        except Exception:
            log.warning("scene %d voice failed, continuing", scene.id, exc_info=True)
            ref.advance(AssetStatus.FAILED)
            save_plan(plan, paths)
            continue
        ext = asset.meta.get("format", "mp3").split("_")[0]
        out = paths.voice / f"scene_{scene.id:03d}.{ext}"
        out.write_bytes(asset.data)
        ref.path = str(out)
        ref.provider = asset.provider
        ref.cost = asset.cost
        ref.advance(AssetStatus.COMPLETED)
        save_plan(plan, paths)
        log.info("scene %d voice done (%s)", scene.id, out.name)


def generate_subtitles(plan: ScenePlan, paths: ProjectPaths) -> None:
    """Caption PNGs + timing per scene, synced to the real audio duration.

    Must run after voice (and avatar clips, for talking_avatar scenes) since
    it needs their actual duration via ffprobe — never duration_estimate_sec.
    SubtitleRef has no state-machine transitions (unlike AssetRef); status is
    set directly.
    """
    from renderflow.pipeline.render import probe_duration
    from renderflow.pipeline.subtitles import write_scene_subtitles

    for scene in plan.scenes:
        ref = scene.assets.subtitle
        if ref.status is AssetStatus.COMPLETED and ref.path:
            continue
        audio_path = (
            scene.assets.avatar_clip.path
            if scene.type == "talking_avatar" and not scene_is_visual_only(scene)
            else scene.assets.voice.path
        )
        if not audio_path:
            continue
        ref.status = AssetStatus.RUNNING
        save_plan(plan, paths)
        duration = probe_duration(Path(audio_path))
        meta_path = write_scene_subtitles(scene, duration, paths)
        ref.path = str(meta_path)
        ref.status = AssetStatus.COMPLETED
        save_plan(plan, paths)
        log.info("scene %d subtitles done (%s)", scene.id, meta_path.name)


def generate_avatar_clips(
    plan: ScenePlan, provider: AvatarProvider, paths: ProjectPaths
) -> None:
    """One failed scene must not stop the rest of the batch (see generate_images)."""
    for scene in plan.scenes:
        # Visual-only scenes (see scene_is_visual_only) never show the
        # avatar, so no lip-synced clip is needed for them at all.
        if scene.type != "talking_avatar" or scene_is_visual_only(scene):
            continue

        ref = scene.assets.avatar_clip
        if _skip(ref):
            continue
        avatar_image_path = scene.assets.avatar_image.path or scene.assets.image.path
        if not avatar_image_path or not scene.assets.voice.path:
            log.warning(
                "scene %d missing avatar image or voice, skipping avatar clip for now",
                scene.id,
            )
            continue

        _start(ref)
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
            log.warning("scene %d avatar clip failed, continuing", scene.id, exc_info=True)
            ref.advance(AssetStatus.FAILED)
            save_plan(plan, paths)
            continue

        ext = asset.meta.get("format", "mp4").split("_")[0]
        out = paths.avatar / f"scene_{scene.id:03d}.{ext}"
        out.write_bytes(asset.data)
        ref.path = str(out)
        ref.provider = asset.provider
        ref.cost = asset.cost
        ref.advance(AssetStatus.COMPLETED)
        save_plan(plan, paths)
        log.info("scene %d avatar clip done (%s)", scene.id, out.name)


def generate_thumbnail(
    plan: ScenePlan,
    provider: ImageProvider,
    paths: ProjectPaths,
    reaction_provider: ImageProvider | None = None,
) -> None:
    """Generate the clickbait thumbnail: topic background + host reaction face.

    Both images are generated under the same `plan.thumbnail` asset (one
    retry unit — thumbnails are cheap and non-critical, no need for finer
    per-image tracking), but may come from *different providers*: the
    background wants AI generation (dramatic saturated clickbait
    composition stock search rarely has), while the reaction face can be a
    real stock photo (`reaction_provider`, defaults to `provider`). True
    identity-preserving image editing of the real avatar photo
    (Pollinations' `kontext` image-to-image mode) needs the source photo
    hosted at a public URL, which a local file isn't — so the reaction
    face is a fresh portrait, not a pixel-edit of the actual avatar photo.
    """
    reaction_provider = reaction_provider or provider
    ref = plan.thumbnail
    if _skip(ref):
        return
    _start(ref)
    save_plan(plan, paths)
    try:
        bg_asset = _generate_face_free(
            provider,
            _thumbnail_prompt(plan),
            "text, words, letters, watermark, logo, cartoon, illustration, "
            "3d render, low quality, blurry",
            "thumbnail background",
        )
        # The reaction face is a deliberate exception — it's supposed to
        # have a face — so it's a plain generate() call, never wrapped.
        reaction_asset = reaction_provider.generate(
            _thumbnail_reaction_prompt(plan),
            "text, words, letters, watermark, logo, cartoon, illustration, "
            "3d render, CGI, low quality, blurry, multiple people, deformed "
            "hands, extra fingers, calm expression, neutral expression",
        )
    except Exception:
        ref.advance(AssetStatus.FAILED)
        save_plan(plan, paths)
        raise
    bg_out = paths.output / "thumbnail_src.png"
    bg_out.write_bytes(bg_asset.data)
    reaction_out = paths.output / "thumbnail_reaction.png"
    reaction_out.write_bytes(reaction_asset.data)
    ref.path = str(bg_out)
    ref.provider = (
        bg_asset.provider
        if reaction_asset.provider == bg_asset.provider
        else f"{bg_asset.provider}+{reaction_asset.provider}"
    )
    ref.cost = (bg_asset.cost or 0.0) + (reaction_asset.cost or 0.0)
    ref.advance(AssetStatus.COMPLETED)
    save_plan(plan, paths)
    log.info("thumbnail image + reaction face done (%s)", bg_out.name)


def _thumbnail_reaction_prompt(plan: ScenePlan) -> str:
    avatar = next((s.avatar for s in plan.scenes if s.avatar), None)
    base = avatar.description if avatar else "a documentary host, middle-aged, plain clothing"
    # Strip mood/expression wording from the normal on-screen description —
    # it was written for calm narration shots and directly contradicts the
    # exaggerated reaction we want here.
    for phrase in (
        "calm serious expression", "neutral studio lighting",
        "cinematic portrait lighting",
    ):
        base = base.replace(phrase, "").replace("  ", " ").strip(", ")
    return (
        f"Photorealistic close-up reaction portrait of this person: {base}. "
        "Exaggerated shocked and amazed expression — wide eyes, eyebrows "
        "raised high, mouth open in surprise, leaning toward the camera, "
        "dramatic YouTube thumbnail reaction shot, vivid dramatic lighting, "
        "high contrast, sharp focus on the face, no text or lettering."
    )


def _thumbnail_prompt(plan: ScenePlan) -> str:
    # Topic-literal on purpose: the host portrait is composited on the left
    # afterwards (render_thumbnail), so the subject sits right, no people.
    topic = topic_from_title(plan.title)
    return (
        f"Viral YouTube thumbnail background: a dramatic photograph of "
        f"{topic}. One instantly recognizable {topic} scene, huge in the "
        "frame and positioned toward the right side. No people, no faces. "
        "Extreme contrast, vivid saturated colors, dramatic cinematic "
        "lighting, photorealistic, absolutely no text, no words, no letters, "
        "no typography."
    )


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
