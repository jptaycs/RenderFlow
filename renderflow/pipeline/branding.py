"""Intro/outro title cards, rendered with Pillow.

Dark 1920x1080 cards matching the dashboard's look: near-black background,
white bold text, a thin red accent. Reuses the caption font loader
(subtitles._font) so the same bold system font drives captions and cards.
Cards are silent by design — the music bed plays over them (render.py).
"""

from __future__ import annotations

from pathlib import Path

WIDTH, HEIGHT = 1920, 1080
BACKGROUND = (13, 13, 16)
TEXT = (242, 242, 244)
MUTED = (154, 154, 161)
ACCENT = (255, 0, 0)

TITLE_SIZE = 96
SUBTITLE_SIZE = 44
MAX_TEXT_WIDTH = int(WIDTH * 0.82)


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textbbox((0, 0), trial, font=font)[2] > max_width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _card(lines_spec: list[tuple[str, int, tuple]], out: Path) -> Path:
    """Render centered text lines (text, font_size, color) on the dark card,
    with a thin accent rule above the block."""
    from PIL import Image, ImageDraw

    from renderflow.pipeline.subtitles import _font

    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(img)

    # Wrap and measure the whole block first so it can be vertically centered.
    rendered: list[tuple] = []  # (line, font, color, height, gap_below)
    block_height = 0
    for text, size, color in lines_spec:
        if not text:
            continue
        font = _font(size)
        gap = int(size * 0.45)
        for line in _wrap(draw, text, font, MAX_TEXT_WIDTH):
            box = draw.textbbox((0, 0), line, font=font)
            height = box[3] - box[1]
            rendered.append((line, font, color, height, gap))
            block_height += height + gap

    accent_gap = 56
    y = (HEIGHT - block_height - accent_gap) // 2 + accent_gap
    draw.rectangle(
        (WIDTH // 2 - 90, y - accent_gap, WIDTH // 2 + 90, y - accent_gap + 8),
        fill=ACCENT,
    )
    for line, font, color, height, gap in rendered:
        box = draw.textbbox((0, 0), line, font=font)
        x = (WIDTH - (box[2] - box[0])) // 2
        draw.text((x, y - box[1]), line, font=font, fill=color)
        y += height + gap

    img.save(out)
    return out


def build_intro_card(title: str, channel_name: str, out: Path) -> Path:
    lines = [(title, TITLE_SIZE, TEXT)]
    if channel_name:
        lines.append((channel_name.upper(), SUBTITLE_SIZE, MUTED))
    return _card(lines, out)


def build_outro_card(channel_name: str, out: Path) -> Path:
    lines = [
        ("Thanks for watching", TITLE_SIZE, TEXT),
        ("Subscribe for more", SUBTITLE_SIZE, ACCENT),
    ]
    if channel_name:
        lines.append((channel_name.upper(), SUBTITLE_SIZE, MUTED))
    return _card(lines, out)
