"""Upload a finished project video to YouTube.

Usage:
    python publish_youtube.py --slug <slug> --projects-dir <dir> \
        --title "..." [--description "..."] [--tags "a,b,c"] \
        [--privacy public|unlisted|private] [--no-synthetic-disclosure]

Spawned as a subprocess by tasks.py (job kind "youtube_publish"), exactly
like make_video.py — a multi-hundred-MB upload can take minutes, so this
never runs inside an API request. Requires the one-time OAuth setup in
scripts/setup_youtube.py.

Writes the result to <project>/youtube.json (schema.YouTubePublish) —
read back by api.py's project view. Pipeline-adjacent scripts never touch
the DB directly (see CLAUDE.md), same separation make_video.py keeps.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Same Windows console-encoding fix as make_video.py — see that file for
# why this has to happen before anything prints.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from renderflow.schema import YouTubePublish
from renderflow.storage import ProjectPaths, save_youtube_publish
from renderflow.youtube import upload_video


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Publish a finished video to YouTube")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--projects-dir", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--tags", default="", help="comma-separated")
    parser.add_argument(
        "--privacy", default="public", choices=["public", "unlisted", "private"]
    )
    parser.add_argument(
        "--no-synthetic-disclosure",
        action="store_true",
        help="omit status.containsSyntheticMedia — only for content that "
        "genuinely isn't AI-altered/generated",
    )
    args = parser.parse_args()

    paths = ProjectPaths.create(args.projects_dir, args.slug)
    video_path = paths.output / "final.mp4"
    thumb_path = paths.output / "thumbnail.jpg"
    if not video_path.exists():
        print(f"no final.mp4 at {video_path} — render the video before publishing")
        return 1

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    contains_synthetic_media = not args.no_synthetic_disclosure
    print(f"Uploading {video_path.name} to YouTube as {args.privacy}...")
    result = upload_video(
        video_path,
        title=args.title,
        description=args.description,
        tags=tags,
        privacy_status=args.privacy,
        contains_synthetic_media=contains_synthetic_media,
        thumbnail_path=thumb_path if thumb_path.exists() else None,
    )

    save_youtube_publish(
        YouTubePublish(
            video_id=result["video_id"],
            url=result["url"],
            privacy_status=args.privacy,
            contains_synthetic_media=contains_synthetic_media,
            published_at=time.time(),
        ),
        paths,
    )
    print(f"Published: {result['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
