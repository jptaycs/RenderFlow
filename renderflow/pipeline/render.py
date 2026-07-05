"""FFmpeg composition: still + zoompan + narration per scene, then concat.

Scene duration is derived from the generated voice audio length, not the
LLM's estimate.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path

from renderflow.schema import Scene, ScenePlan
from renderflow.storage import ProjectPaths

log = logging.getLogger("renderflow.pipeline.render")

FPS = 30
WIDTH, HEIGHT = 1920, 1080
# Upscale before zoompan so sub-pixel motion doesn't jitter.
PRESCALE_W, PRESCALE_H = 2560, 1440


class RenderError(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(
            f"command failed ({proc.returncode}): {' '.join(cmd[:4])}...\n{proc.stderr[-2000:]}"
        )


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RenderError(f"ffprobe failed for {path}: {proc.stderr}")
    return float(proc.stdout.strip())


def _zoompan_expr(scene: Scene, frames: int) -> str:
    i = max(scene.motion.intensity, 0.02)
    center_x = "iw/2-(iw/zoom/2)"
    center_y = "ih/2-(ih/zoom/2)"
    match scene.motion.effect:
        case "zoom_in":
            z, x, y = f"1+{i}*on/{frames}", center_x, center_y
        case "zoom_out":
            z, x, y = f"1+{i}-{i}*on/{frames}", center_x, center_y
        case "pan_left":
            z = f"{1 + i}"
            x = f"(iw-iw/zoom)*(1-on/{frames})"
            y = center_y
        case "pan_right":
            z = f"{1 + i}"
            x = f"(iw-iw/zoom)*on/{frames}"
            y = center_y
    return (
        f"scale={PRESCALE_W}:{PRESCALE_H}:force_original_aspect_ratio=increase,"
        f"crop={PRESCALE_W}:{PRESCALE_H},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )


def render_scene_clip(scene: Scene, out: Path) -> Path:
    if scene.type == "talking_avatar" and scene.assets.avatar_clip.path:
        avatar_clip = Path(scene.assets.avatar_clip.path)
        if avatar_clip.resolve() != out.resolve():
            shutil.copyfile(avatar_clip, out)
        return out

    assert scene.assets.image.path and scene.assets.voice.path
    image = Path(scene.assets.image.path)
    audio = Path(scene.assets.voice.path)
    duration = probe_duration(audio)
    frames = math.ceil(duration * FPS)
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(image),
            "-i", str(audio),
            "-filter_complex", f"[0:v]{_zoompan_expr(scene, frames)}[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out),
        ]
    )
    return out


def render_video(plan: ScenePlan, paths: ProjectPaths) -> Path:
    clips: list[Path] = []
    for scene in plan.scenes:
        clip = paths.output / f"clip_{scene.id:03d}.mp4"
        log.info("rendering scene %d", scene.id)
        clips.append(render_scene_clip(scene, clip))

    concat_list = paths.output / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{clip.resolve()}'\n" for clip in clips)
    )
    final = paths.output / "final.mp4"
    _run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-movflags", "+faststart",
            str(final),
        ]
    )
    log.info("final video: %s", final)
    return final
