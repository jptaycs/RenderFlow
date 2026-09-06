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

# Bumped up from 96/44 (client feedback: "make the texts bigger").
TITLE_SIZE = 132
SUBTITLE_SIZE = 58


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# Same four colors as the Pillow card above, converted for 69labs Motion
# Graphics' "palette" control (background/foreground/accent/secondary hex
# strings) — see pipeline/motion_graphics.py — so a rendered card matches
# this project's existing look instead of the template's own default blue
# palette.
MOTION_GRAPHICS_PALETTE = {
    "background": _hex(BACKGROUND),
    "foreground": _hex(TEXT),
    "accent": _hex(ACCENT),
    "secondary": _hex(MUTED),
}


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


def _card(
    lines_spec: list[tuple[str, int, tuple]], out: Path,
    width: int = WIDTH, height: int = HEIGHT,
) -> Path:
    """Render centered text lines (text, font_size, color) on the dark card,
    with a thin accent rule above the block.

    `width`/`height` (added 2026-09 for the Shorts closing-message card,
    see `render._shorts_outro_clip`) default to the landscape constants so
    every existing call site is unaffected; the max-wrap-width scales
    proportionally rather than staying pinned to the landscape figure —
    otherwise a portrait 1080-wide card would wrap far more aggressively
    than intended relative to its own width.
    """
    from PIL import Image, ImageDraw

    from renderflow.pipeline.subtitles import _font

    img = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(img)
    max_text_width = int(width * 0.82)

    # Wrap and measure the whole block first so it can be vertically centered.
    rendered: list[tuple] = []  # (line, font, color, height, gap_below)
    block_height = 0
    for text, size, color in lines_spec:
        if not text:
            continue
        font = _font(size)
        gap = int(size * 0.45)
        for line in _wrap(draw, text, font, max_text_width):
            box = draw.textbbox((0, 0), line, font=font)
            line_height = box[3] - box[1]
            rendered.append((line, font, color, line_height, gap))
            block_height += line_height + gap

    accent_gap = 56
    y = (height - block_height - accent_gap) // 2 + accent_gap
    draw.rectangle(
        (width // 2 - 90, y - accent_gap, width // 2 + 90, y - accent_gap + 8),
        fill=ACCENT,
    )
    for line, font, color, line_height, gap in rendered:
        box = draw.textbbox((0, 0), line, font=font)
        x = (width - (box[2] - box[0])) // 2
        draw.text((x, y - box[1]), line, font=font, fill=color)
        y += line_height + gap

    img.save(out)
    return out


def build_intro_card(title: str, channel_name: str, out: Path) -> Path:
    lines = [(title, TITLE_SIZE, TEXT)]
    if channel_name:
        lines.append((channel_name.upper(), SUBTITLE_SIZE, MUTED))
    return _card(lines, out)


CTA_LINE = "Let us know in the comments!"


def outro_lines(message: str | None) -> list[tuple[str, int, tuple]]:
    """Pick the two headline lines for the outro card from the full
    narrated outro sentence (assets._outro_line).

    Bug fixed 2026-09: `message` is the *whole* narrated sentence, e.g.
    "<question> Let us know in the comments! <subscribe line>" — this
    used to be dumped onto the card as one dense wrapped line, with a
    second, literally-duplicated "Let us know in the comments!" line
    appended below it regardless of whether a real question was even
    asked (the plain "Thanks for watching! <subscribe>" fallback got the
    same bogus CTA line, with no question to react to). Client-reported
    (indirectly, by asking for the question to be added — it technically
    already was, just unreadable this way): "in shorts add the questions
    at the end of the video for them to leave a comment". Fixed by
    splitting the actual call-to-action phrase off the narrated sentence
    so it renders once, in its own accent line, with just the question
    above it — and falling back to the original clean two-line "Thanks
    for watching" / "Subscribe for more trivia" copy whenever there's no
    real question (LLM unset/failed, or an old project predating this
    feature).
    """
    if message and CTA_LINE in message:
        headline = message.split(CTA_LINE, 1)[0].strip()
        return [
            (headline, SUBTITLE_SIZE, TEXT),
            (CTA_LINE, SUBTITLE_SIZE, ACCENT),
        ]
    return [
        ("Thanks for watching", TITLE_SIZE, TEXT),
        ("Subscribe for more trivia", SUBTITLE_SIZE, ACCENT),
    ]


def build_outro_card(
    channel_name: str, out: Path, message: str | None = None,
    width: int = WIDTH, height: int = HEIGHT,
) -> Path:
    """`message` (added 2026-09, client request: "leave the viewer a
    message") is the actual outro line generate_branding_audio narrated
    — either a topic-specific comment-bait question or the plain
    fallback (see pipeline.script.generate_engagement_question) — shown
    on-screen instead of the old fixed "Thanks for watching" text, so the
    card matches what's actually being said instead of a generic line
    unrelated to it. None (narration disabled, or generation hasn't run
    yet) keeps the original fixed copy. See `outro_lines` for how it's
    turned into the card's headline lines.

    `width`/`height` (added 2026-09, same change as `_card` above) let
    `render._shorts_outro_clip` build a portrait-sized version of this
    same card for Shorts, which have no landscape intro/outro cards at
    all in v1 scope but still get a closing message card of their own.
    """
    lines = outro_lines(message)
    if channel_name:
        lines.append((channel_name.upper(), SUBTITLE_SIZE, MUTED))
    return _card(lines, out, width, height)
