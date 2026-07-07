"""Topic/script → structured scene plan."""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from renderflow.providers.base import LLMProvider, LLMResult
from renderflow.schema import AvatarSpec, GeneratedScript, Motion, Scene, ScenePlan

log = logging.getLogger("renderflow.pipeline.script")

SECONDS_PER_SCENE = 5
LOCAL_AVATAR = AvatarSpec(
    name="David Lester",
    description=(
        "Middle-aged male documentary host, plain dark shirt, calm serious "
        "expression, seated in a modest workshop, cinematic portrait lighting"
    ),
    background="modest rural workshop",
)

SYSTEM_PROMPT = """\
You are the script engine of RenderFlow, an automated YouTube video pipeline.
You produce complete video scripts broken into scenes. Each scene is a still
image with slow pan/zoom motion, narrated by a voiceover.

Rules:
- Narration must read naturally when spoken aloud: no headings, no markdown,
  no stage directions, no "Scene 1:" prefixes.
- Keep scenes short: about 5 seconds each (10-16 spoken words), so the visual
  changes constantly. Never pack several sentences into one scene.
- Each scene's narration should take roughly its duration_estimate_sec to
  speak at a natural pace (~2.5 words per second).
- image_prompt must read like the caption of a real photograph, not digital
  art: a concrete subject in a concrete setting, camera framing (wide shot,
  medium shot, aerial), lens and film feel (e.g. "35mm documentary
  photograph, shallow depth of field"), natural lighting, era-accurate
  details and textures. No text or lettering in the image.
- Avoid a visible human face entirely — never a portrait, never a lone
  face close-up, never a person looking at the camera. If the sentence is
  genuinely about a person, show them from behind, in silhouette, at a
  distance, or cropped to hands/objects only. Every image must combine at
  least two visual elements: what the sentence talks about plus its topic
  context (the setting, objects, actions, or secondary subjects). Wide or
  medium shots only.
- negative_prompt lists things the image model should avoid; always include
  realism killers and face killers (e.g. "text, watermark, cartoon,
  illustration, painting, 3d render, CGI, plastic skin, oversaturated
  colors, deformed hands, low quality, human face, portrait, lone face
  close-up, single face filling the frame, person looking at camera").
- Vary motion between scenes (zoom_in, zoom_out, pan_left, pan_right) with
  intensity between 0.05 and 0.15.
- Scene ids start at 1 and increment by 1.
- The script must have a hook in the first scene and a clear close in the last.
"""


def build_user_prompt(topic: str, length_minutes: float, style: str) -> str:
    target_scenes = max(4, round(length_minutes * 60 / SECONDS_PER_SCENE))
    return (
        f"Write a {length_minutes:g}-minute {style} video script about: {topic}\n\n"
        f"Produce about {target_scenes} scenes of roughly {SECONDS_PER_SCENE} "
        f"seconds each."
    )


def generate_script(
    llm: LLMProvider, topic: str, length_minutes: float, style: str
) -> tuple[ScenePlan, LLMResult]:
    """Returns the validated scene plan and the raw LLM result (for cost)."""
    result = llm.complete(
        SYSTEM_PROMPT,
        build_user_prompt(topic, length_minutes, style),
        json_schema=GeneratedScript.model_json_schema(),
    )
    try:
        generated = GeneratedScript.model_validate_json(result.text)
    except ValidationError:
        log.error("LLM returned JSON that does not match the scene schema")
        raise
    plan = generated.to_plan()
    log.info("generated %d scenes for %r", len(plan.scenes), plan.title)
    return plan, result


SPLIT_SYSTEM_PROMPT = """\
You are the scene-planning engine of RenderFlow, an automated video pipeline.
You receive a finished narration script written by a client. Your job is to
split it into scenes for a stills-with-motion documentary video — you do NOT
rewrite the script.

Rules:
- Preserve the narration text VERBATIM. Every word of the input script must
  appear in exactly one scene's narration, in the original order. Do not add,
  remove, or rephrase anything. Only strip markdown/heading syntax if present.
- Split at natural beats: a new visual idea, location, subject, or argument.
  Aim for 10-16 spoken words per scene (about 5 seconds of speech), so the
  visual changes constantly. Split long sentences at natural clause breaks
  (commas, dashes) instead of merging sentences into one scene.
- duration_estimate_sec = narration word count / 2.5.
- image_prompt must read like the caption of a real photograph, not digital
  art: a concrete subject in a concrete setting, camera framing (wide shot,
  medium shot, aerial), lens and film feel (e.g. "35mm documentary
  photograph, shallow depth of field"), natural lighting, era-accurate
  details and textures. No text or lettering. Keep a consistent
  photographic style across all scenes.
- Avoid a visible human face entirely — never a portrait, never a lone
  face close-up, never a person looking at the camera. If the sentence is
  genuinely about a person, show them from behind, in silhouette, at a
  distance, or cropped to hands/objects only. Every image must combine at
  least two visual elements: what the sentence talks about plus its topic
  context (the setting, objects, actions, or secondary subjects). Wide or
  medium shots only.
- negative_prompt: what the image model should avoid; always include realism
  killers and face killers (e.g. "text, watermark, cartoon, illustration,
  painting, 3d render, CGI, plastic skin, oversaturated colors, deformed
  hands, low quality, human face, portrait, lone face close-up, single
  face filling the frame, person looking at camera").
- Vary motion between scenes (zoom_in, zoom_out, pan_left, pan_right),
  intensity 0.05-0.15.
- Scene ids start at 1 and increment by 1.
- Infer a short video title from the script content.
"""


def split_script(
    llm: LLMProvider, script_text: str, style: str
) -> tuple[ScenePlan, LLMResult]:
    """Split a client-provided script into scenes without rewriting it."""
    result = llm.complete(
        SPLIT_SYSTEM_PROMPT,
        f"Split this {style} script into scenes:\n\n{script_text}",
        json_schema=GeneratedScript.model_json_schema(),
    )
    try:
        generated = GeneratedScript.model_validate_json(result.text)
    except ValidationError:
        log.error("LLM returned JSON that does not match the scene schema")
        raise
    plan = generated.to_plan()
    plan.style = style
    log.info("split script into %d scenes: %r", len(plan.scenes), plan.title)
    return plan, result


def split_script_local(script_text: str, style: str) -> tuple[ScenePlan, LLMResult]:
    """Split a client-provided script into scenes without any external LLM call."""
    cleaned = _strip_script_markup(script_text)
    chunks = _chunk_sentences(_sentences(cleaned))
    title = _infer_title(cleaned)
    motions = ("zoom_in", "pan_right", "zoom_out", "pan_left")
    scenes = [
        Scene(
            id=index,
            type="talking_avatar" if _uses_local_avatar(index) else "narration",
            duration_estimate_sec=max(2.0, len(chunk.split()) / 2.5),
            narration=chunk,
            image_prompt=_local_image_prompt(chunk, style),
            negative_prompt=(
                "human face, face, portrait, person looking at camera, "
                "lone face close-up, single face filling the frame, isolated "
                "headshot, text, watermark, subtitles, captions, "
                "cartoon, illustration, painting, 3d render, CGI, plastic skin, "
                "oversaturated colors, deformed hands, low quality, blurry"
            ),
            avatar=LOCAL_AVATAR if _uses_local_avatar(index) else None,
            motion=Motion(
                effect=motions[(index - 1) % len(motions)],
                intensity=0.08 + (0.01 * ((index - 1) % 4)),
            ),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    plan = ScenePlan(title=title, style=style, scenes=scenes)
    result = LLMResult(
        text=plan.model_dump_json(),
        provider="local-script-splitter",
        cost=0.0,
        meta={"scene_count": len(scenes)},
    )
    log.info("locally split script into %d scenes: %r", len(plan.scenes), plan.title)
    return plan, result


def _uses_local_avatar(scene_index: int) -> bool:
    # The host speaks in every scene — solo or split-screen (see
    # scene_is_avatar_solo), the presenter is always on camera.
    return True


def scene_is_avatar_solo(scene: Scene) -> bool:
    """Solo full-screen avatar vs. split-screen (avatar left + that scene's
    own visual right).

    A per-scene `avatar_layout` override ("solo"/"split") wins outright —
    it's a deliberate user choice, made from the dashboard when the
    generated visual isn't wanted for that beat. The default, "auto", falls
    back to a repeating cycle of 3 (1 solo, then 2 split; scene ids start
    at 1 and increment by 1, so id 1, 4, 7, ... are solo) — most projects
    never need to touch the override at all. Every caller (render.py: which
    layout to draw; assets.py: skip generating an unused background image
    for solo scenes) derives its answer from this single function.
    """
    if scene.avatar_layout == "solo":
        return True
    if scene.avatar_layout == "split":
        return False
    return (scene.id - 1) % 3 == 0


def _strip_script_markup(script_text: str) -> str:
    text = script_text.replace("\ufeff", "")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^\*\*(.+)\*\*$", r"\1", line)
        lines.append(line)
    return " ".join(lines)


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        raise ValueError("script file is empty")
    return re.split(r"(?<=[.!?])\s+", normalized)


def _chunk_sentences(
    sentences: list[str], min_words: int = 6, max_words: int = 16
) -> list[str]:
    # ~2.5 spoken words/second, so max_words=16 keeps scenes around 5 seconds:
    # the visual changes on every beat instead of lingering for half a minute.
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence.split()) > max_words:
            pieces.extend(_split_long_sentence(sentence, max_words))
        else:
            pieces.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for piece in pieces:
        words = len(piece.split())
        if current and current_words >= min_words and current_words + words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(piece)
        current_words += words

    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    """Split an over-long sentence at clause breaks so TTS pauses naturally."""
    clauses = re.split(r"(?<=[,;:—])\s+", sentence)
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for clause in clauses:
        words = len(clause.split())
        if current and current_words + words > max_words:
            pieces.append(" ".join(current))
            current = []
            current_words = 0
        current.append(clause)
        current_words += words
    if current:
        pieces.append(" ".join(current))
    return pieces


def _infer_title(script_text: str) -> str:
    words = re.findall(r"[A-Za-z0-9'$-]+", script_text)
    if not words:
        return "Untitled Script"
    title = " ".join(words[:8]).strip()
    return title[:80]


def _local_image_prompt(narration: str, style: str) -> str:
    excerpt = narration
    if len(excerpt) > 180:
        excerpt = excerpt[:177].rsplit(" ", 1)[0] + "..."
    excerpt = excerpt.rstrip(",;:— ")
    if not excerpt.endswith((".", "!", "?", "…")):
        excerpt += "."
    # The composition rule leads, before the narration content: image models
    # weight earlier prompt tokens more heavily, and for abstract narration
    # (no concrete noun — "he was told he would be independent") Pollinations
    # otherwise defaults to a generic stock-portrait face, which the
    # negative_prompt alone doesn't reliably suppress. Learned 2026-07 on a
    # 59-scene render: ~30% of scenes fell back to a face-forward shot even
    # with the "never a lone face close-up" wording — client feedback was to
    # avoid a visible face altogether, not just avoid it being a close-up.
    return (
        "Wide or medium documentary b-roll photograph — absolutely no "
        "visible human face, never a portrait, never a person looking at "
        "the camera. If people appear, show them from behind, in "
        "silhouette, at a distance, or cropped to hands/objects only — "
        f"never a face. Show the concrete setting, objects, tools, or "
        f"action of this moment: {excerpt} "
        f"Realistic {style} photograph, shot on 35mm film, natural lighting, "
        "shallow depth of field, documentary composition, era-accurate "
        "details, realistic surface textures, muted natural colors, "
        "no visible text."
    )


def script_markdown(plan: ScenePlan) -> str:
    lines = [f"# {plan.title}", "", f"Style: {plan.style}", ""]
    for scene in plan.scenes:
        lines += [f"## Scene {scene.id}", "", scene.narration, ""]
    return "\n".join(lines)
