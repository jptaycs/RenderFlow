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
import sys
from pathlib import Path

from renderflow.config import Settings
from renderflow.pipeline.assets import (
    generate_avatar_clips,
    generate_images,
    generate_voice,
)
from renderflow.pipeline.render import render_thumbnail, render_video
from renderflow.pipeline.script import (
    generate_script,
    script_markdown,
    split_script,
    split_script_local,
)
from renderflow.providers import build_avatar, build_image, build_llm, build_tts
from renderflow.schema import ScenePlan
from renderflow.storage import ProjectPaths, save_plan, slugify


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
    args = parser.parse_args()

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
            plan, script_result = split_script_local(script_text, args.style)
        script_cost = script_result.cost or 0.0
    else:
        llm = build_llm(settings)
        print(f"[1/4] Generating script for: {args.topic}")
        plan, script_result = generate_script(llm, args.topic, args.length, args.style)
        script_cost = script_result.cost or 0.0

    if args.title:
        plan.title = args.title

    paths = ProjectPaths.create(settings.projects_dir, args.slug or slugify(plan.title))
    save_plan(plan, paths)
    (paths.script / "script.md").write_text(script_markdown(plan))
    print(f"      {len(plan.scenes)} scenes → {paths.scenes_json}")

    print(f"[2/4] Generating {len(plan.scenes)} images ({image.name})")
    generate_images(plan, image, paths, avatar_image=avatar_image)

    print(f"[3/4] Generating {len(plan.scenes)} voice clips ({tts.name})")
    tts_params = {}
    if settings.tts_provider == "piper":
        tts_params["length_scale"] = settings.tts_length_scale
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

    render_thumbnail(plan, paths)

    print(f"{render_step} Rendering with FFmpeg")
    final = render_video(plan, paths)

    total = script_cost + plan.total_asset_cost()
    (paths.logs / "costs.json").write_text(
        json.dumps(
            {"script": script_cost, "assets": plan.total_asset_cost(), "total": total},
            indent=2,
        )
    )
    print(f"\nDone: {final}")
    print(f"Total cost: ${total:.4f} (script ${script_cost:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
