"""Walking skeleton: topic → finished MP4, synchronously.

Usage:
    python make_video.py --topic "The history of Amish farming" --length 3

    # Start from an existing scenes.json (skips the LLM call entirely):
    python make_video.py --scenes-file scenes.json --slug client-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from renderflow.config import Settings
from renderflow.pipeline.assets import (
    generate_avatar_clips,
    generate_broll,
    generate_images,
    generate_subtitles,
    generate_thumbnail,
    generate_voice,
)
from renderflow.pipeline.render import render_thumbnail, render_video
from renderflow.pipeline.script import (
    generate_script,
    scene_is_avatar_solo,
    scene_is_visual_only,
    script_markdown,
    split_script,
    split_script_local,
)
from renderflow.providers import (
    build_avatar,
    build_broll,
    build_image,
    build_llm,
    build_tts,
)
from renderflow.providers.base import ImageProvider
from renderflow.schema import AssetRef, ScenePlan
from renderflow.storage import ProjectPaths, save_plan, slugify


def _incomplete_scenes(plan: ScenePlan) -> list[int]:
    """Scene ids still missing an image/voice/avatar clip after generation.

    Generation now continues past a per-scene failure (a stubborn provider
    error must not kill a 50+ scene batch), so render can no longer assume
    every asset landed — this is the clean stop instead of an AssertionError
    mid-render.
    """
    missing = []
    for scene in plan.scenes:
        # Visual-only scenes (see scene_is_visual_only) never get an avatar
        # clip — they need only voice + background image, same as a plain
        # narration scene.
        visual_only = scene.type == "talking_avatar" and scene_is_visual_only(scene)
        needs_avatar = scene.type == "talking_avatar" and not visual_only
        solo = needs_avatar and scene_is_avatar_solo(scene)
        ok = bool(scene.assets.voice.path) and (solo or bool(scene.assets.image.path))
        if needs_avatar:
            ok = ok and bool(scene.assets.avatar_clip.path)
        if not ok:
            missing.append(scene.id)
    return missing


def _thumbnail_image_providers(
    settings: Settings, image: ImageProvider
) -> tuple[ImageProvider, ImageProvider]:
    """Resolve the thumbnail's (background, reaction) image providers.

    Each is its own env var ("" = same as the scene image provider) so the
    background can be AI-generated (dramatic clickbait composition) while
    the reaction face comes from stock search, independent of what scenes
    use. Matching names reuse the same instance — provider-internal state
    (rate-limit throttle, Pexels used-photo dedup) stays shared.
    """
    built: dict[str, ImageProvider] = {settings.image_provider: image}

    def resolve(name: str) -> ImageProvider:
        name = name or settings.image_provider
        if name not in built:
            built[name] = build_image(settings, name)
        return built[name]

    return (
        resolve(settings.thumbnail_bg_provider),
        resolve(settings.thumbnail_reaction_provider),
    )


def _regenerate_thumbnail(
    plan: ScenePlan,
    paths: ProjectPaths,
    background: ImageProvider,
    reaction: ImageProvider,
    avatar_image: Path | None,
) -> int:
    """Regenerate only the project thumbnail, leaving everything else alone.

    The thumbnail isn't part of final.mp4, but generate_thumbnail's
    save_plan calls bump scenes.json's mtime — which api.py's final_ready
    staleness check compares against final.mp4. Capture whether the render
    was fresh before touching anything and re-stamp it after, so a
    thumbnail-only change never makes a valid render look stale (same
    lesson as the layout-migration mtime restore in storage.py).
    """
    final = paths.output / "final.mp4"
    final_was_fresh = (
        final.exists()
        and final.stat().st_mtime >= paths.scenes_json.stat().st_mtime
    )
    # COMPLETED is terminal in the asset state machine — a regenerate is a
    # fresh asset, not a transition.
    plan.thumbnail = AssetRef()
    try:
        generate_thumbnail(plan, background, paths, reaction_provider=reaction)
        # Only after generation succeeded: render_thumbnail skips when
        # thumbnail.jpg exists (resume behavior), so it must go — but a
        # failed generation above keeps the old thumbnail downloadable.
        (paths.output / "thumbnail.jpg").unlink(missing_ok=True)
        render_thumbnail(plan, paths, avatar_image=avatar_image)
    finally:
        if final_was_fresh and final.exists():
            os.utime(final)
    print(f"\nDone: {paths.output / 'thumbnail.jpg'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a video from a topic")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--topic")
    source.add_argument(
        "--script-file", type=Path, help="client-provided script to split into scenes"
    )
    source.add_argument(
        "--scenes-file", type=Path, help="existing scenes.json (skips the LLM step)"
    )
    parser.add_argument("--length", type=float, default=3.0, help="target minutes")
    parser.add_argument("--style", default="documentary")
    parser.add_argument("--slug", help="project directory name (default: from title)")
    parser.add_argument(
        "--title", help="video title (default: LLM-chosen or inferred from the script)"
    )
    parser.add_argument(
        "--llm-split",
        action="store_true",
        help="use the configured LLM to split --script-file instead of the free local splitter",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help=(
            "generate/regenerate assets only, skip the final FFmpeg render — "
            "for regenerating a single scene without re-rendering the whole "
            "video; run again without this flag (or hit Resume) to render"
        ),
    )
    parser.add_argument(
        "--projects-dir",
        type=Path,
        help=(
            "override the projects root for this run (used by the worker to "
            "point at per-user directories). Must be a CLI flag, not an env "
            "var: Settings.load() runs load_dotenv(override=True), which "
            "clobbers any RENDERFLOW_PROJECTS_DIR inherited from the parent "
            "process with the .env value"
        ),
    )
    parser.add_argument(
        "--thumbnail-only",
        action="store_true",
        help=(
            "regenerate only the project thumbnail (background + reaction "
            "face) and stop — scenes and the final render are untouched; "
            "requires --scenes-file"
        ),
    )
    args = parser.parse_args()
    if args.thumbnail_only and not args.scenes_file:
        parser.error("--thumbnail-only requires --scenes-file")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    settings = Settings.load()
    avatar_image = settings.avatar_image
    if avatar_image is not None and not avatar_image.exists():
        raise FileNotFoundError(f"avatar image not found: {avatar_image}")
    image = build_image(settings)
    tts = build_tts(settings)

    if args.scenes_file:
        print(f"[1/4] Loading scene plan from {args.scenes_file}")
        plan = ScenePlan.model_validate_json(args.scenes_file.read_text())
        script_cost = 0.0
    elif args.script_file:
        script_text = args.script_file.read_text()
        if args.llm_split:
            llm = build_llm(settings)
            print(f"[1/4] Splitting client script {args.script_file} into scenes")
            plan, script_result = split_script(llm, script_text, args.style)
        else:
            print(
                f"[1/4] Locally splitting client script "
                f"{args.script_file} into scenes"
            )
            plan, script_result = split_script_local(
                script_text, args.style, topic_hint=args.title
            )
        script_cost = script_result.cost or 0.0
    else:
        llm = build_llm(settings)
        print(f"[1/4] Generating script for: {args.topic}")
        plan, script_result = generate_script(llm, args.topic, args.length, args.style)
        script_cost = script_result.cost or 0.0

    if args.title:
        plan.title = args.title

    projects_dir = args.projects_dir or settings.projects_dir
    paths = ProjectPaths.create(projects_dir, args.slug or slugify(plan.title))
    if args.thumbnail_only:
        # Before save_plan — the freshness capture inside must see
        # scenes.json's pre-run mtime.
        print("Regenerating project thumbnail")
        thumb_bg, thumb_reaction = _thumbnail_image_providers(settings, image)
        return _regenerate_thumbnail(plan, paths, thumb_bg, thumb_reaction, avatar_image)
    save_plan(plan, paths)
    (paths.script / "script.md").write_text(script_markdown(plan))
    print(f"      {len(plan.scenes)} scenes → {paths.scenes_json}")

    print(f"[2/4] Generating {len(plan.scenes)} images ({image.name})")
    generate_images(plan, image, paths, avatar_image=avatar_image)

    broll = build_broll(settings)
    if broll is not None:
        # Optional stock-video B-roll for full-frame scenes — failures fall
        # back to the still image and never block the render.
        print(f"      Fetching stock B-roll ({broll.name})")
        generate_broll(plan, broll, paths)

    print(f"[3/4] Generating {len(plan.scenes)} voice clips ({tts.name})")
    tts_params = {}
    if settings.tts_provider == "piper":
        tts_params["length_scale"] = settings.tts_length_scale
        tts_params["sentence_pause_sec"] = settings.tts_sentence_pause
    elif settings.tts_provider == "kokoro":
        # Kokoro speed is the inverse of Piper's length_scale (1.0 = natural).
        tts_params["speed"] = 1.0 / settings.tts_length_scale
        tts_params["sentence_pause_sec"] = settings.tts_sentence_pause
    generate_voice(plan, tts, settings.tts_voice, paths, **tts_params)

    avatar_scene_count = sum(scene.type == "talking_avatar" for scene in plan.scenes)
    if avatar_scene_count:
        avatar = build_avatar(settings)
        print(f"[4/5] Generating {avatar_scene_count} avatar clips ({avatar.name})")
        generate_avatar_clips(plan, avatar, paths)
        render_step = "[5/5]"
    else:
        render_step = "[4/4]"

    print("      Generating scene captions")
    generate_subtitles(plan, paths)

    thumb_bg, thumb_reaction = _thumbnail_image_providers(settings, image)
    generate_thumbnail(plan, thumb_bg, paths, reaction_provider=thumb_reaction)
    render_thumbnail(plan, paths, avatar_image=avatar_image)

    missing = _incomplete_scenes(plan)
    if missing:
        print(
            f"\nStopped before rendering: {len(missing)} scene(s) still need assets "
            f"({', '.join(str(n) for n in missing)}).\n"
            "This is expected after a partial provider failure (e.g. a rate limit) — "
            "everything that succeeded is saved. Re-run the same --slug (or hit "
            "Resume in the dashboard) to retry only what's missing."
        )
        return 1

    total = script_cost + plan.total_asset_cost()
    (paths.logs / "costs.json").write_text(
        json.dumps(
            {"script": script_cost, "assets": plan.total_asset_cost(), "total": total},
            indent=2,
        )
    )

    if args.skip_render:
        print(
            "\nAssets done, render skipped (--skip-render). Run again without "
            "the flag (or hit Resume in the dashboard) to render the final MP4."
        )
        print(f"Total cost so far: ${total:.4f} (script ${script_cost:.4f})")
        return 0

    print(f"{render_step} Rendering with FFmpeg")
    final = render_video(plan, paths)

    print(f"\nDone: {final}")
    print(f"Total cost: ${total:.4f} (script ${script_cost:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
