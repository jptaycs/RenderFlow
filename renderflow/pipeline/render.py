"""FFmpeg composition: still + zoompan + narration per scene, then concat.

Scene duration is derived from the generated voice audio length, not the
LLM's estimate.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from pathlib import Path

from renderflow.pipeline import parallax
from renderflow.schema import Scene, ScenePlan
from renderflow.storage import ProjectPaths

log = logging.getLogger("renderflow.pipeline.render")

FPS = 30
WIDTH, HEIGHT = 1920, 1080
AVATAR_W = 768
VISUAL_W = WIDTH - AVATAR_W
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


def _parallax_visual(
    scene: Scene, width: int, height: int, duration: float, out: Path
) -> Path | None:
    """Depth-parallax visual for the scene, or None to use zoompan instead.

    RENDERFLOW_MOTION=zoompan opts out; missing deps or a depth failure fall
    back silently (parallax logs the reason once).
    """
    if os.getenv("RENDERFLOW_MOTION", "parallax") == "zoompan":
        return None
    if not scene.assets.image.path or not parallax.available():
        return None
    tmp = out.with_name(f"visual_{scene.id:03d}.mp4")
    ok = parallax.render_parallax_clip(
        Path(scene.assets.image.path), tmp, duration, width, height,
        effect=scene.motion.effect, intensity=scene.motion.intensity,
    )
    return tmp if ok else None


def render_scene_clip(scene: Scene, out: Path) -> Path:
    if scene.type == "talking_avatar" and scene.assets.avatar_clip.path:
        assert scene.assets.image.path
        return render_avatar_split_clip(
            scene, Path(scene.assets.avatar_clip.path), Path(scene.assets.image.path), out
        )

    assert scene.assets.image.path and scene.assets.voice.path
    image = Path(scene.assets.image.path)
    audio = Path(scene.assets.voice.path)
    duration = probe_duration(audio)

    visual = _parallax_visual(scene, WIDTH, HEIGHT, duration, out)
    if visual is not None:
        _run(
            [
                "ffmpeg", "-y",
                "-i", str(visual),
                "-i", str(audio),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-t", f"{duration:.3f}",
                "-shortest",
                str(out),
            ]
        )
        visual.unlink(missing_ok=True)
        return out

    frames = math.ceil(duration * FPS)
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(image),
            "-i", str(audio),
            "-filter_complex", f"[0:v]{_zoompan_expr(scene, frames)}[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{duration:.3f}",
            "-shortest",
            str(out),
        ]
    )
    return out


def render_avatar_split_clip(
    scene: Scene, avatar_clip: Path, visual_image: Path, out: Path
) -> Path:
    duration = probe_duration(avatar_clip)

    visual = _parallax_visual(scene, VISUAL_W, HEIGHT, duration, out)
    if visual is not None:
        right_input = ["-i", str(visual)]
        right_filter = f"[1:v]fps={FPS},format=yuv420p[right];"
    else:
        frames = math.ceil(duration * FPS)
        right_zoom = _zoompan_expr(scene, frames).replace(
            f"s={WIDTH}x{HEIGHT}", f"s={VISUAL_W}x{HEIGHT}"
        )
        right_input = ["-loop", "1", "-i", str(visual_image)]
        right_filter = f"[1:v]{right_zoom},format=yuv420p[right];"

    filter_complex = (
        f"[0:v]scale={AVATAR_W}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={AVATAR_W}:{HEIGHT},format=yuv420p[left];"
        + right_filter +
        "[left][right]hstack=inputs=2[v]"
    )
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(avatar_clip),
            *right_input,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{duration:.3f}",
            "-shortest",
            str(out),
        ]
    )
    if visual is not None:
        visual.unlink(missing_ok=True)
    return out


def render_thumbnail(
    plan: ScenePlan, paths: ProjectPaths, avatar_image: Path | None = None
) -> Path | None:
    """Project thumbnail (1280x720 JPEG): topic image + host badge.

    Prefers the generated clickbait image (output/thumbnail_src.png, see
    assets.generate_thumbnail); falls back to the first completed scene image.
    The presenter portrait is composited bottom-left as a circular badge.
    Runs at the end of asset generation; skipped if already present (resume).
    """
    thumb = paths.output / "thumbnail.jpg"
    if thumb.exists():
        return thumb
    generated = paths.output / "thumbnail_src.png"
    source = (
        str(generated)
        if generated.exists()
        else next((s.assets.image.path for s in plan.scenes if s.assets.image.path), None)
    )
    if source is None:
        return None
    _run(
        [
            "ffmpeg", "-y",
            "-i", source,
            "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
            "-frames:v", "1", "-q:v", "3",
            str(thumb),
        ]
    )
    if avatar_image is not None and avatar_image.exists():
        _overlay_host_badge(thumb, avatar_image)
    log.info("thumbnail: %s", thumb)
    return thumb


def _overlay_host_badge(thumb: Path, portrait_path: Path) -> None:
    """Composite the presenter portrait as a ringed circle, bottom-left."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        log.warning("PIL missing — thumbnail rendered without host badge")
        return
    base = Image.open(thumb).convert("RGB")
    portrait = Image.open(portrait_path).convert("RGB")
    side = min(portrait.width, portrait.height)
    portrait = portrait.crop(
        ((portrait.width - side) // 2, 0, (portrait.width + side) // 2, side)
    )
    dia = int(base.height * 0.44)
    ring = max(int(dia * 0.045), 4)
    portrait = portrait.resize((dia, dia))
    x = int(base.width * 0.035)
    y = base.height - dia - int(base.height * 0.07)

    ring_size = (dia + 2 * ring, dia + 2 * ring)
    ring_mask = Image.new("L", ring_size, 0)
    ImageDraw.Draw(ring_mask).ellipse((0, 0) + ring_size, fill=255)
    base.paste(Image.new("RGB", ring_size, (255, 255, 255)), (x - ring, y - ring), ring_mask)

    face_mask = Image.new("L", (dia, dia), 0)
    ImageDraw.Draw(face_mask).ellipse((0, 0, dia, dia), fill=255)
    base.paste(portrait, (x, y), face_mask)
    base.save(thumb, quality=90)


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
    concat_inputs = [arg for clip in clips for arg in ("-i", str(clip))]
    concat_streams = "".join(f"[{i}:v][{i}:a]" for i in range(len(clips)))
    _run(
        [
            "ffmpeg", "-y",
            *concat_inputs,
            "-filter_complex", f"{concat_streams}concat=n={len(clips)}:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            str(final),
        ]
    )
    log.info("final video: %s", final)
    return final
