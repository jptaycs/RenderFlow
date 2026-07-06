"""2.5D parallax motion: depth-aware camera moves over still images.

Instead of a flat Ken Burns zoompan, each still gets a monocular depth map
(Depth-Anything-V2 small, ~50 MB one-time download, runs on CPU in ~1 s) and
frames are warped so near pixels move more than far ones — the camera appears
to travel through the scene. Free and local; if any dependency or the depth
model fails, callers fall back to the plain zoompan path.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

log = logging.getLogger("renderflow.pipeline.parallax")

FPS = 30
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
# Fraction of frame width the nearest pixels drift across a scene. Small on
# purpose: parallax reads as camera motion at 2-3%, as a glitch at 10%.
DRIFT = 0.022
# Extra image around the target crop so warped edges never show.
OVERSCAN = 1.08

_depth_pipeline = None
_unavailable: str | None = None


def available() -> bool:
    if _unavailable is not None:
        return False
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        _mark_unavailable(f"missing dependency: {exc}")
        return False
    return True


def _mark_unavailable(reason: str) -> None:
    global _unavailable
    if _unavailable is None:
        _unavailable = reason
        log.warning("parallax disabled — %s (falling back to zoompan)", reason)


def _depth(image) -> "numpy.ndarray | None":  # noqa: F821
    """HxW float32 depth in [0, 1], 1 = near. None if the model fails."""
    global _depth_pipeline
    import numpy as np
    from PIL import Image

    try:
        if _depth_pipeline is None:
            from transformers import pipeline

            log.info("loading depth model %s", DEPTH_MODEL)
            _depth_pipeline = pipeline("depth-estimation", model=DEPTH_MODEL)
        result = _depth_pipeline(Image.fromarray(image[:, :, ::-1]))
        depth = np.asarray(result["predicted_depth"], dtype=np.float32)
    except Exception as exc:
        _mark_unavailable(f"depth model failed: {exc}")
        return None
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-6:
        return None
    return (depth - lo) / (hi - lo)


def render_parallax_clip(
    image_path: Path,
    out: Path,
    duration: float,
    width: int,
    height: int,
    effect: str = "zoom_in",
    intensity: float = 0.08,
) -> bool:
    """Write a silent H.264 clip of the still with depth-parallax motion.

    Returns False when parallax cannot be produced; the caller must then use
    the zoompan path instead.
    """
    if not available():
        return False
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return False

    # Work on an overscanned canvas so displaced edges stay outside the crop.
    work_w, work_h = int(width * OVERSCAN) // 2 * 2, int(height * OVERSCAN) // 2 * 2
    scale = max(work_w / image.shape[1], work_h / image.shape[0])
    resized = cv2.resize(
        image, (round(image.shape[1] * scale), round(image.shape[0] * scale))
    )
    y0 = (resized.shape[0] - work_h) // 2
    x0 = (resized.shape[1] - work_w) // 2
    canvas = resized[y0 : y0 + work_h, x0 : x0 + work_w]

    depth = _depth(canvas)
    if depth is None:
        return False
    depth = cv2.resize(depth, (work_w, work_h))
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=8)
    depth -= float(depth.mean())  # drift happens around the depth midpoint

    frames = max(int(math.ceil(duration * FPS)), 1)
    drift_px = DRIFT * work_w * (0.5 + min(max(intensity, 0.02), 0.2) / 0.16)
    match effect:
        case "pan_left":
            dir_x, dir_y, zoom_amp = -1.0, 0.0, 0.02
        case "pan_right":
            dir_x, dir_y, zoom_amp = 1.0, 0.0, 0.02
        case "zoom_out":
            dir_x, dir_y, zoom_amp = 0.35, 0.12, -0.05
        case _:  # zoom_in
            dir_x, dir_y, zoom_amp = 0.35, 0.12, 0.05

    xx, yy = np.meshgrid(
        np.arange(work_w, dtype=np.float32), np.arange(work_h, dtype=np.float32)
    )
    cx, cy = work_w / 2.0, work_h / 2.0
    crop_x, crop_y = (work_w - width) // 2, (work_h - height) // 2

    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(FPS),
            "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    try:
        for i in range(frames):
            t = i / max(frames - 1, 1)
            progress = t * t * (3 - 2 * t)  # smoothstep ease-in-out
            offset = (progress - 0.5) * 2 * drift_px
            zoom = 1.0 + zoom_amp * progress
            # Depth-modulated sampling: near pixels (depth > 0) shift more.
            map_x = (xx - cx) / zoom + cx - depth * offset * dir_x
            map_y = (yy - cy) / zoom + cy - depth * offset * dir_y
            frame = cv2.remap(
                canvas, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            crop = frame[crop_y : crop_y + height, crop_x : crop_x + width]
            encoder.stdin.write(crop.tobytes())
    finally:
        encoder.stdin.close()
        encoder.wait()
    if encoder.returncode != 0:
        log.warning("parallax encode failed for %s", image_path.name)
        return False
    return True
