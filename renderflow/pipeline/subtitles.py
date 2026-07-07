"""Bold on-screen captions, synced to each scene's actual audio duration.

No TTS provider here gives word-level timestamps, so a scene's narration is
split into short chunks (~4 words) and time is distributed across the
scene's real audio duration, weighted by each chunk's word count. Chunks are
pre-rendered to transparent PNGs (Pillow); render.py overlays them onto the
scene clip with ffmpeg, timed via `enable='between(t,start,end)'`.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

from renderflow.schema import Scene
from renderflow.storage import ProjectPaths

log = logging.getLogger("renderflow.pipeline.subtitles")

# Matches the full-canvas render resolution (renderflow.pipeline.render.WIDTH/HEIGHT).
CANVAS_W, CANVAS_H = 1920, 1080
CAPTION_BAND_H = 300
FONT_SIZE = 68
STROKE_WIDTH = 7
CHUNK_WORDS = 4
MARGIN_BOTTOM = 90

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    log.warning("no bundled bold font found — captions will use PIL's default font")
    return ImageFont.load_default()


def _chunk_words(narration: str, size: int = CHUNK_WORDS) -> list[str]:
    words = narration.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def build_chunks(narration: str, duration: float) -> list[tuple[str, float, float]]:
    """(text, start_sec, end_sec) for each caption chunk within the scene."""
    texts = _chunk_words(narration)
    if not texts or duration <= 0:
        return []
    weights = [len(t.split()) for t in texts]
    total_words = sum(weights)
    chunks: list[tuple[str, float, float]] = []
    t = 0.0
    for text, weight in zip(texts, weights):
        span = duration * weight / total_words
        chunks.append((text, t, min(t + span, duration)))
        t += span
    return chunks


def render_caption_png(text: str, out: Path) -> None:
    """One transparent PNG, full canvas width, bold white text with a black
    outline — the standard high-contrast caption look for narrated shorts."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (CANVAS_W, CAPTION_BAND_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(FONT_SIZE)
    upper = text.upper()

    max_width = CANVAS_W * 0.88
    lines = [upper]
    if draw.textbbox((0, 0), upper, font=font, stroke_width=STROKE_WIDTH)[2] > max_width:
        wrapped = textwrap.wrap(upper, width=max(len(upper) // 2, 8))
        lines = wrapped or [upper]

    line_boxes = [
        draw.textbbox((0, 0), line, font=font, stroke_width=STROKE_WIDTH) for line in lines
    ]
    line_heights = [b[3] - b[1] for b in line_boxes]
    total_h = sum(line_heights) + (len(lines) - 1) * 10
    y = (CAPTION_BAND_H - total_h) / 2
    for line, box, lh in zip(lines, line_boxes, line_heights):
        lw = box[2] - box[0]
        x = (CANVAS_W - lw) / 2 - box[0]
        draw.text(
            (x, y - box[1]), line, font=font, fill=(255, 255, 255, 255),
            stroke_width=STROKE_WIDTH, stroke_fill=(0, 0, 0, 255),
        )
        y += lh + 10
    image.save(out)


def write_scene_subtitles(scene: Scene, duration: float, paths: ProjectPaths) -> Path:
    """Render this scene's caption PNGs and return the chunk-timing JSON path."""
    chunks = build_chunks(scene.narration, duration)
    entries = []
    for i, (text, start, end) in enumerate(chunks):
        img = paths.subtitles / f"scene_{scene.id:03d}_{i:02d}.png"
        render_caption_png(text, img)
        entries.append({"image": str(img), "start": start, "end": end})
    meta = paths.subtitles / f"scene_{scene.id:03d}.json"
    meta.write_text(json.dumps(entries))
    return meta


def load_scene_subtitles(meta_path: str | Path) -> list[dict]:
    return json.loads(Path(meta_path).read_text())
