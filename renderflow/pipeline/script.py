"""Topic/script → structured scene plan."""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import ValidationError

from renderflow.providers.base import LLMProvider, LLMResult
from renderflow.schema import (
    AvatarSpec,
    GeneratedScript,
    GeneratedTopicIdea,
    Motion,
    Scene,
    ScenePlan,
)

log = logging.getLogger("renderflow.pipeline.script")

SECONDS_PER_SCENE = 5
LOCAL_AVATAR = AvatarSpec(
    name="John Doe",
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


def _target_scene_count(length_minutes: float) -> int:
    return max(4, round(length_minutes * 60 / SECONDS_PER_SCENE))


def build_user_prompt(topic: str, length_minutes: float, style: str) -> str:
    target_scenes = _target_scene_count(length_minutes)
    return (
        f"Write a {length_minutes:g}-minute {style} video script about: {topic}\n\n"
        f"Produce about {target_scenes} scenes of roughly {SECONDS_PER_SCENE} "
        f"seconds each."
    )


# Rough per-scene JSON budget once every GeneratedScene field is counted:
# narration (~10-16 words), a full photographic image_prompt (~60-100 words,
# by far the biggest field), a negative_prompt list, plus JSON punctuation
# overhead. Deliberately generous — max_tokens is a ceiling the model is
# billed against only for what it actually emits, never for headroom, so
# overshooting costs nothing while undershooting truncates the *entire*
# response (all scenes lost, not just the last one, since it's one JSON
# document) — see ClaudeLLM.complete's max_tokens truncation check.
_TOKENS_PER_SCENE_ESTIMATE = 400
_SCRIPT_MAX_TOKENS_FLOOR = 16000
# Sonnet 5 / Opus 5's hard output-token ceiling (128K) — see claude-api skill.
_SCRIPT_MAX_TOKENS_CEILING = 128000


def _script_max_tokens(target_scenes: int) -> int:
    """Added 2026-09 alongside the dashboard's 10/12-minute length options.

    The prior flat 16000-token default (ClaudeLLM.complete's own default,
    sized for short 2-3 minute topic videos) would silently truncate any
    script beyond roughly 35-40 scenes (~3 minutes) — the whole point of
    raising the length picker's ceiling would have shipped broken without
    this. A 12-minute video (~144 scenes) needs up to ~60K tokens; this
    scales with it instead of hardcoding a bigger flat number that would
    eventually hit the same wall at some other length.
    """
    return min(
        _SCRIPT_MAX_TOKENS_CEILING,
        max(_SCRIPT_MAX_TOKENS_FLOOR, target_scenes * _TOKENS_PER_SCENE_ESTIMATE + 2000),
    )


def generate_script(
    llm: LLMProvider, topic: str, length_minutes: float, style: str
) -> tuple[ScenePlan, LLMResult]:
    """Returns the validated scene plan and the raw LLM result (for cost)."""
    result = llm.complete(
        SYSTEM_PROMPT,
        build_user_prompt(topic, length_minutes, style),
        json_schema=GeneratedScript.model_json_schema(),
        max_tokens=_script_max_tokens(_target_scene_count(length_minutes)),
    )
    try:
        generated = GeneratedScript.model_validate_json(result.text)
    except ValidationError:
        log.error("LLM returned JSON that does not match the scene schema")
        raise
    plan = generated.to_plan()
    log.info("generated %d scenes for %r", len(plan.scenes), plan.title)
    return plan, result


TOPIC_IDEA_SYSTEM_PROMPT = """\
You invent fresh, surprising short-documentary video ideas for a YouTube
trivia channel ("did you know" style true stories — a war that lasted 38
minutes, a shark older than the United States, that kind of thing).

Rules:
- Invent exactly ONE idea: a punchy, hooky title (under ~9 words — a
  specific factual hook, not a generic clickbait template like "The Truth
  About X") and a short narration script (3-5 sentences, 60-100 words)
  that reads naturally spoken aloud: no headings, no markdown, no scene
  numbers, no "Scene 1:" prefixes.
- The fact must be real, specific, and checkable — no invented statistics
  or made-up numbers.
- Open with the hook itself, not a preamble.
- Never reuse a fact, story, or subject from the "already used" list below
  — not even with a different title or wording. Pick a different subject
  entirely (different era, place, field, and kind of surprise) from every
  title listed, not just a different sentence about the same fact.
"""


def build_topic_idea_prompt(existing_titles: list[str]) -> str:
    if existing_titles:
        taken = "\n".join(f"- {t}" for t in existing_titles)
        return (
            "Invent one new short-documentary video idea. These titles are "
            f"already used — invent something different:\n{taken}"
        )
    return "Invent one new short-documentary video idea."


def generate_topic_idea(
    llm: LLMProvider, existing_titles: list[str]
) -> tuple[GeneratedTopicIdea, LLMResult]:
    """One "surprise me" idea for the dashboard's New Video modal.

    Replaces the old client-side static RANDOM_TOPICS bank (web/index.html,
    kept in git history for prompt calibration — see TOPIC_IDEA_SYSTEM_PROMPT
    above) with a fresh Claude-generated idea on every click: the fixed
    10-entry bank ran out (and started repeating) well before a real user's
    project count did. Small, fast completion (~100-200 output tokens) —
    unlike full script generation (generate_script), which is deliberately
    kept out of the request path (see api.py's create_project), this is
    cheap and quick enough to call directly from a request handler.
    """
    result = llm.complete(
        TOPIC_IDEA_SYSTEM_PROMPT,
        build_topic_idea_prompt(existing_titles),
        json_schema=GeneratedTopicIdea.model_json_schema(),
        max_tokens=500,
    )
    try:
        idea = GeneratedTopicIdea.model_validate_json(result.text)
    except ValidationError:
        log.error("LLM returned JSON that does not match the topic-idea schema")
        raise
    return idea, result


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
    # Same truncation risk as generate_script, scaled off the pasted
    # script's own word count instead of a length_minutes target (there
    # isn't one here — the split preserves whatever length the client
    # already wrote) — ~13 words/scene is the midpoint of the 10-16
    # word/scene split target in SPLIT_SYSTEM_PROMPT.
    estimated_scenes = max(4, round(len(script_text.split()) / 13))
    result = llm.complete(
        SPLIT_SYSTEM_PROMPT,
        f"Split this {style} script into scenes:\n\n{script_text}",
        json_schema=GeneratedScript.model_json_schema(),
        max_tokens=_script_max_tokens(estimated_scenes),
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


VIDEO_PROMPT_SYSTEM_PROMPT = """\
You rewrite a still-photo caption into a prompt for an AI VIDEO generation
model. The input describes a single static frame; your job is to describe
motion happening across a few seconds instead — camera movement (slow
dolly-in, pan, tilt, tracking shot) and/or something changing in the scene
(water moving, light shifting, an object drifting, steam rising) — while
keeping the same subject, setting, and framing as the input.

Rules:
- Keep every constraint from the input in spirit: if it says no visible
  human face, your rewrite must have no visible human face either — never
  add one just because a video can show more than a photo could.
- Describe motion in plain, concrete terms a video model can act on. Two
  or three sentences, no more.
- Do not use the words "video", "clip", "AI", or "generate" — just
  describe what the camera sees and how it moves, like a shot description
  in a nature-documentary shot list.
- No on-screen text, captions, or watermarks.
- Output ONLY the rewritten prompt text — no preamble, no quotes, no
  explanation.
"""


def rewrite_video_prompt(
    llm: LLMProvider, narration: str, image_prompt: str
) -> tuple[str, LLMResult]:
    """Rewrite a still-photo `image_prompt` into a motion-aware prompt for
    AI video generation (`providers/video/labs69_video.py`).

    Motivating gap: `labs69_video.py` originally fed the video model the
    exact same prompt built for a still image — a static-photo caption,
    not a video-generation prompt — wasting a video model's actual ability
    to show movement. One short Claude call per scene, spent right before
    the much more expensive 69labs video call, so it's a small cost for
    real leverage on the payoff of the bigger spend.

    Raises on any LLM failure — deliberately does not swallow exceptions,
    so callers (assets.generate_broll) can catch and fall back to the
    original `image_prompt`; a bad rewrite must never block B-roll, which
    is optional end to end.
    """
    result = llm.complete(
        VIDEO_PROMPT_SYSTEM_PROMPT,
        f"Narration for this moment: {narration}\n\n"
        f"Still-photo prompt to rewrite for video:\n{image_prompt}",
        max_tokens=300,
    )
    return result.text.strip(), result


def split_script_local(
    script_text: str, style: str, topic_hint: str | None = None,
    avatar_enabled: bool = True,
) -> tuple[ScenePlan, LLMResult]:
    """Split a client-provided script into scenes without any external LLM call.

    `topic_hint` is the video's real title when the caller already has one
    (e.g. api.py always passes the user's clickbait title via --title) — it
    drives the topic anchor baked into every scene's image_prompt (see
    _local_image_prompt), which matters here because that override is
    applied to plan.title only *after* this function returns (make_video.py),
    so without it every image prompt would be anchored to the inferred
    8-word title instead of the actual one.

    `avatar_enabled` (Settings.local_avatar_enabled, RENDERFLOW_LOCAL_AVATAR,
    default on) controls whether every scene gets the hardcoded LOCAL_AVATAR
    persona as a talking_avatar scene, or stays plain narration (image/broll
    + voice only, no host on camera) — added 2026-09 because this path
    (script-mode create, e.g. the dashboard's "Random topic" button) was
    the only one that *always* forced an avatar: topic-mode generation
    (generate_script) lets Claude decide per scene and defaults to
    narration, so a script pasted/generated without --llm-split was the
    odd one out, producing a generic-avatar video that looked nothing like
    a topic-mode one from the same providers. Off restores that
    narration-only look end to end with zero other provider changes.
    """
    cleaned = _strip_script_markup(script_text)
    chunks = _chunk_sentences(_sentences(cleaned))
    title = _infer_title(cleaned)
    topic = topic_from_title(topic_hint or title)
    motions = ("zoom_in", "pan_right", "zoom_out", "pan_left")
    scenes = [
        Scene(
            id=index,
            type="talking_avatar" if avatar_enabled and _uses_local_avatar(index) else "narration",
            duration_estimate_sec=max(2.0, len(chunk.split()) / 2.5),
            narration=chunk,
            image_prompt=_local_image_prompt(chunk, style, topic),
            negative_prompt=(
                "human face, face, portrait, person looking at camera, "
                "lone face close-up, single face filling the frame, isolated "
                "headshot, text, watermark, subtitles, captions, "
                "cartoon, illustration, painting, 3d render, CGI, plastic skin, "
                "oversaturated colors, deformed hands, low quality, blurry"
            ),
            avatar=LOCAL_AVATAR if avatar_enabled and _uses_local_avatar(index) else None,
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
    # The host speaks in every scene that has one at all — solo or
    # split-screen (see scene_is_avatar_solo). Whether a scene has one at
    # all is `avatar_enabled` at the split_script_local call site.
    return True


def effective_avatar_layout(scene: Scene) -> Literal["solo", "split", "visual"]:
    """Resolve a scene's `avatar_layout` override (or the "auto" default) to
    one of three concrete rendering modes:

    - "solo": full-screen avatar, no background visual.
    - "split": avatar left, that scene's own background visual right.
    - "visual": background visual only, no avatar shown at all — narration
      plays over the image/parallax visual exactly like a plain
      `type="narration"` scene, even though this scene's `type` is still
      "talking_avatar" (so any avatar assets it happens to already have are
      simply unused, not deleted — same "harmless leftover" philosophy as
      switching solo<->split).

    A per-scene override wins outright — it's a deliberate user choice, made
    from the dashboard when the generated visual (or the avatar itself)
    isn't wanted for that beat. The default, "auto", is always "split":
    every scene gets its own generated visual next to the host, and the
    other two modes are opt-in per scene rather than an automatic cycle
    (changed 2026-07 — the old 1-in-3 auto-solo cycle picked scenes for the
    user before they had a chance to see what visual would have been
    generated). Every caller (render.py: which layout to draw; assets.py:
    skip generating assets a given layout will never use) derives its
    answer from this single function.
    """
    if scene.avatar_layout in ("solo", "split", "visual"):
        return scene.avatar_layout  # type: ignore[return-value]
    return "split"


def scene_is_avatar_solo(scene: Scene) -> bool:
    """Solo full-screen avatar layout — see effective_avatar_layout."""
    return effective_avatar_layout(scene) == "solo"


def scene_is_visual_only(scene: Scene) -> bool:
    """Visual-only layout (no avatar shown) — see effective_avatar_layout."""
    return effective_avatar_layout(scene) == "visual"


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


# Clickbait-template and stop words stripped from titles to find the topic
# noun (e.g. "ants", "solar panels", "mosquitoes"). Feeding the full title
# into the image model makes it render the title as (garbled) text in the
# picture — learned the hard way. Shared with assets.py's thumbnail prompt.
_TITLE_FILLER = frozenset(
    """
    the a an of in on for to and or is are was were it its it's this that
    how why what when who which nobody everybody everyone anyone they you
    your i we truth about tells tell told know knew known should would
    could want wants dont don't wont won't really actually quietly hidden
    untold story secret cost costing money wrong right before after too
    late changes changed everything nothing looked into found
    """.split()
)


def topic_from_title(title: str) -> str:
    words = [
        w for w in re.findall(r"[A-Za-z0-9'-]+", title)
        if w.lower() not in _TITLE_FILLER
    ]
    return " ".join(words) or title


def _local_image_prompt(narration: str, style: str, topic: str) -> str:
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
    #
    # `topic` (the video's main subject, e.g. "ants"/"solar panels", derived
    # from the title via topic_from_title) is repeated on both sides of the
    # excerpt for the same reason: many mid-script sentences refer back to
    # the subject with pronouns ("it", "they", "the whole nest") rather than
    # naming it, so the excerpt alone often gives the image model nothing
    # concrete to draw and it falls back to a generic stock photo of an
    # unrelated person instead of the actual topic. Learned 2026-07: a
    # mosquito-bite script produced scenes of a random person's arm/kitchen
    # instead of anything mosquito-related on the sentences that never said
    # the word "mosquito".
    return (
        f"Wide or medium documentary b-roll photograph about {topic} — "
        "absolutely no visible human face, never a portrait, never a "
        "person looking at the camera. If people appear, show them from "
        "behind, in silhouette, at a distance, or cropped to hands/objects "
        f"only — never a face. The photo must clearly visually relate to "
        f"{topic}: show its concrete setting, objects, tools, or the "
        f"specific action described here: {excerpt} Keep {topic} "
        "recognizably present in the frame even if this line doesn't name "
        f"it directly. Realistic {style} photograph, shot on 35mm film, "
        "natural lighting, shallow depth of field, documentary composition, "
        "era-accurate details, realistic surface textures, muted natural "
        "colors, no visible text."
    )


def script_markdown(plan: ScenePlan) -> str:
    lines = [f"# {plan.title}", "", f"Style: {plan.style}", ""]
    for scene in plan.scenes:
        lines += [f"## Scene {scene.id}", "", scene.narration, ""]
    return "\n".join(lines)
