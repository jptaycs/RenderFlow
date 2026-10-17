"""Autopilot core: trends -> topic -> script + SEO -> RenderFlow project.

Everything here is plain functions over a DB session and injected
providers, so the Celery tasks (jobs.py), the Telegram bot (bot.py), the
dashboard router (api.py), and the tests all share one implementation.

The video itself is made exactly the way the dashboard makes one: the LLM
writes a plain narration script, it is saved as the project's source.txt,
and a `make_video.py --script-file` job is queued — same local splitter,
same host on every scene, same voice/images/captions/music/thumbnail. The
only difference from a dashboard create is no --skip-render: nobody is
sitting at the scene grid, so the run goes straight through to final.mp4
and the review happens in Telegram instead.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

from renderflow.billing import consume_credit, entitlement
from renderflow.config import Settings
from renderflow.db import (
    EPISODE_ACTIVE_STATUSES,
    AutopilotChannel,
    Episode,
    Job,
    Project,
    User,
)
from renderflow.providers.base import LLMProvider, TrendItem, TrendProvider
from renderflow.schema import ProjectPerformance
from renderflow.storage import ProjectPaths, save_performance, slugify

log = logging.getLogger("renderflow.autopilot")

# The default rotation the picker chooses from — editable per channel in
# the dashboard. Two kinds, both aimed at viewers 55+: "fact" stories (the
# channel's existing underrated-facts format) and practical "howto"s.
# Deliberately no health, legal, or investing niches: those are the first
# AI-avatar content YouTube penalizes, and wrong advice there hurts people.
DEFAULT_NICHES: list[dict[str, str]] = [
    {"name": "Forgotten history", "kind": "fact",
     "description": "Surprising true stories from the past most people never learned in school."},
    {"name": "How things really work", "kind": "fact",
     "description": "The hidden mechanics behind everyday things, machines, and systems."},
    {"name": "Strange but true", "kind": "fact",
     "description": "Odd places, events, and phenomena that sound made up but are documented."},
    {"name": "Kitchen & food history", "kind": "fact",
     "description": "Where foods, recipes, and kitchen habits really came from."},
    {"name": "Nature & animals", "kind": "fact",
     "description": "Remarkable animal, plant, and weather facts."},
    {"name": "Fix-it around the house", "kind": "howto",
     "description": "Simple repairs anyone can do: squeaky doors, drafts, drips, sticky drawers."},
    {"name": "Garden & backyard", "kind": "howto",
     "description": "Practical, seasonal gardening and yard tips that actually work."},
    {"name": "Old-timer money savers", "kind": "howto",
     "description": "Household thrift: cutting utility bills, making things last, avoiding waste. Never investing advice."},
    {"name": "Cleaning & home care", "kind": "howto",
     "description": "Old-fashioned cleaning and upkeep tricks using common household items."},
    {"name": "Simple living skills", "kind": "howto",
     "description": "Traditional self-reliance skills: preserving food, mending, wood stoves, power outages."},
]

KINDS = ("fact", "howto")
# Spoken words per minute for the script length target. Narration runs at
# Kokoro speed 1/1.4 (the calm documentary pace older viewers prefer), which
# lands well under a normal 150 wpm read.
WORDS_PER_MINUTE = 120
PREVIEW_SECONDS = 75
STUCK_RESUME_LIMIT = 2


# ---------------------------------------------------------------------------
# LLM schemas (no numeric constraints — unsupported by structured outputs)
# ---------------------------------------------------------------------------


class GeneratedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    niche: str
    kind: str
    trend: str
    topic: str
    angle: str
    working_title: str
    score: int
    reason: str


class GeneratedPick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[GeneratedCandidate]


class GeneratedEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    script: str
    description: str
    tags: list[str]
    hashtags: list[str]


PICK_SYSTEM = """\
You are the programming director of a YouTube channel for viewers aged 55
and over. Every day you look at what is trending and pick ONE video topic
from the channel's niche list.

How to pick:
- Find the evergreen, family-friendly angle inside a trend. A trend is a
  doorway, not the subject: if heating prices trend, the video can be "the
  forgotten 1800s trick that kept houses warm"; if a holiday trends, its
  surprising true origin.
- Prefer the trend with the most search/view traction that has a natural,
  genuinely interesting angle in some niche. Score 0-100 for how strongly
  the topic rides today's interest AND how well it fits viewers over 55.
- If no trend fits a niche well, you may propose an evergreen topic tied to
  the current season or date (trend = "evergreen"), scored lower.
- Never pick: deaths, disasters, crimes, wars in progress, elections or
  politicians, celebrity gossip, sports results, health or medical advice,
  legal advice, investing/crypto/stocks, or anything that exploits fear or
  grief. Household money-saving tips are fine; financial advice is not.
- The topic must be something with real, well-documented facts — no
  rumors, conspiracies, or unverifiable claims.
- Do not repeat or closely resemble a recent topic.
- niche must be copied exactly from the niche list; kind is that niche's
  kind ("fact" or "howto").
- working_title: a curiosity-driven YouTube title the video can honestly
  deliver on (no false claims, no fake urgency).

Return 3-5 candidates, best first.
"""

EPISODE_SYSTEM = """\
You write complete narration scripts for the YouTube channel "{channel}",
watched mostly by people aged 55 and over. The script is read aloud by the
channel's host, an AI-generated presenter, over documentary photos.

Voice and pacing:
- Warm, calm, unhurried, respectful. Plain everyday words, short sentences.
  No slang, no hype, no shouting, no "smash that like button" energy.
- Hook the viewer in the first two sentences with the most surprising
  true detail, then deliver on it.
- The host never claims to be a real person or to have a life story: no
  invented family, farm, hometown, career, age, or "I've done this for 40
  years". Speak as a friendly guide ("here's what most people don't
  know", "let's look at", "you may have noticed").

Accuracy (this matters more than anything):
- Only well-established, verifiable facts. Never invent statistics,
  quotes, names, dates, studies, or prices. When unsure of a specific,
  stay general rather than guess.
- No medical, legal, or financial advice. For how-tos: list what you need,
  give clear steps, mention safety where relevant, and say when to call a
  professional.

Format:
- The script is spoken text only: no headings, no markdown, no bullet
  points, no stage directions, no [music] cues, no scene labels.
- About {words} words (a {minutes:g}-minute video). Fact stories: a story
  arc with several surprising turns. How-tos: why it matters, what you
  need, step by step, common mistakes, a final tip.
- End with one gentle line inviting viewers to subscribe for more
  underrated facts.

SEO (for the same video):
- title: under 70 characters, curiosity-driven, honest — the video must
  deliver exactly what it promises. At most one word in capitals.
- description: first two lines restate the hook with the main search
  keywords; then a 3-5 sentence summary; then "In this video:" followed by
  4-6 short lines starting with "• ". No links, no timestamps.
- tags: 12-20 search phrases people would actually type.
- hashtags: exactly 3, each starting with "#".
"""

DISCLOSURE = (
    "This video is narrated by an AI-generated host. Facts are drawn from "
    "public sources — please double-check anything important before acting on it."
)


# ---------------------------------------------------------------------------
# Channel + scheduling
# ---------------------------------------------------------------------------


def today() -> str:
    return date.today().isoformat()


def get_or_create_channel(session: Session, user: User) -> AutopilotChannel:
    channel = (
        session.query(AutopilotChannel).filter(AutopilotChannel.owner_id == user.id).first()
    )
    if channel is None:
        channel = AutopilotChannel(owner_id=user.id, niches=list(DEFAULT_NICHES))
        session.add(channel)
        session.flush()
    return channel


def episode_for_day(session: Session, channel: AutopilotChannel, day: str) -> Episode | None:
    return (
        session.query(Episode)
        .filter(Episode.channel_id == channel.id, Episode.day == day)
        .order_by(Episode.created_at.desc())
        .first()
    )


def active_episode(session: Session, channel: AutopilotChannel) -> Episode | None:
    return (
        session.query(Episode)
        .filter(Episode.channel_id == channel.id, Episode.status.in_(EPISODE_ACTIVE_STATUSES))
        .order_by(Episode.created_at.desc())
        .first()
    )


def is_due(session: Session, channel: AutopilotChannel, now: datetime | None = None) -> bool:
    """The daily episode should start: enabled, past the run hour, and no
    episode exists for today yet (any status — a skipped day stays skipped;
    "new topic" is an explicit action). Missed hours still run: if the PC
    was off at 6am and boots at 3pm, today's video starts at 3pm."""
    now = now or datetime.now()
    if not channel.enabled or now.hour < channel.run_hour:
        return False
    return episode_for_day(session, channel, now.date().isoformat()) is None


def new_episode(session: Session, channel: AutopilotChannel) -> Episode:
    """Reserve today's episode row *before* any slow work is queued, so the
    scheduler, the bot, and the dashboard can never double-start a day."""
    episode = Episode(channel_id=channel.id, day=today(), status="planning")
    session.add(episode)
    session.commit()
    return episode


def recent_episodes(session: Session, channel: AutopilotChannel, days: int = 60) -> list[Episode]:
    since = (date.today() - timedelta(days=days)).isoformat()
    return (
        session.query(Episode)
        .filter(Episode.channel_id == channel.id, Episode.day >= since, Episode.topic != "")
        .order_by(Episode.created_at.desc())
        .all()
    )


def previous_niche(session: Session, channel: AutopilotChannel, before_day: str) -> str | None:
    """The niche of the last *made* episode before today (skipped/failed
    ones don't count — the viewer never saw them)."""
    row = (
        session.query(Episode)
        .filter(
            Episode.channel_id == channel.id,
            Episode.day < before_day,
            Episode.status.in_(("ready", "uploading", "uploaded")),
        )
        .order_by(Episode.day.desc(), Episode.created_at.desc())
        .first()
    )
    return row.niche if row else None


# ---------------------------------------------------------------------------
# Trends + picking
# ---------------------------------------------------------------------------


def gather_trends(sources: list[TrendProvider], geo: str) -> list[TrendItem]:
    """Fetch every source; one failing source never sinks the day."""
    items: list[TrendItem] = []
    for source in sources:
        try:
            items.extend(source.fetch(geo=geo))
        except Exception as exc:
            log.warning("trend source %s failed: %s", source.name, exc)
    return items


def _trend_lines(trends: list[TrendItem], limit: int = 60) -> str:
    lines = []
    for i, t in enumerate(trends[:limit], start=1):
        extra = f" [{t.traffic}]" if t.traffic else ""
        ctx = f" — {' | '.join(t.context)}" if t.context else ""
        lines.append(f"{i}. ({t.source}) {t.title}{extra}{ctx}")
    return "\n".join(lines) or "(no trend data available today — use evergreen topics)"


def build_pick_prompt(
    niches: list[dict[str, str]],
    trends: list[TrendItem],
    recent_topics: list[str],
    blocked_niche: str | None,
    day: str,
) -> str:
    niche_lines = "\n".join(
        f"- {n['name']} (kind: {n['kind']}): {n.get('description', '')}" for n in niches
    )
    recent = "\n".join(f"- {t}" for t in recent_topics[:40]) or "- (none yet)"
    blocked = (
        f"\nDo NOT use the niche \"{blocked_niche}\" today — it was yesterday's.\n"
        if blocked_niche
        else ""
    )
    return (
        f"Today is {day}.\n\nNiche list:\n{niche_lines}\n{blocked}\n"
        f"Recent topics (don't repeat):\n{recent}\n\n"
        f"Today's trends:\n{_trend_lines(trends)}\n"
    )


def choose_candidate(
    candidates: list[dict[str, Any]],
    niches: list[dict[str, str]],
    blocked_niche: str | None,
    exclude_topics: set[str] = frozenset(),
) -> dict[str, Any] | None:
    """First valid candidate: a known niche, not yesterday's, not excluded.

    Enforced in code, not just the prompt — the "never the same niche twice
    in a row" rule is the user's, and LLMs drift."""
    by_name = {n["name"].lower(): n for n in niches}
    for cand in candidates:
        niche = by_name.get(str(cand.get("niche", "")).lower())
        if niche is None:
            continue
        if blocked_niche and niche["name"].lower() == blocked_niche.lower():
            continue
        if str(cand.get("topic", "")).strip().lower() in exclude_topics:
            continue
        return {**cand, "niche": niche["name"], "kind": niche["kind"]}
    return None


def pick_topic(
    llm: LLMProvider,
    niches: list[dict[str, str]],
    trends: list[TrendItem],
    recent_topics: list[str],
    blocked_niche: str | None,
    day: str,
    exclude_topics: set[str] = frozenset(),
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float]:
    """Returns (chosen candidate or None, all candidates, llm cost)."""
    result = llm.complete(
        PICK_SYSTEM,
        build_pick_prompt(niches, trends, recent_topics, blocked_niche, day),
        json_schema=GeneratedPick.model_json_schema(),
        max_tokens=8000,
    )
    try:
        pick = GeneratedPick.model_validate_json(result.text)
    except ValidationError:
        log.error("topic picker returned JSON that does not match the schema")
        raise
    candidates = sorted(
        (c.model_dump() for c in pick.candidates), key=lambda c: c["score"], reverse=True
    )
    chosen = choose_candidate(candidates, niches, blocked_niche, exclude_topics)
    return chosen, candidates, result.cost or 0.0


# ---------------------------------------------------------------------------
# Script + SEO
# ---------------------------------------------------------------------------


def target_words(length_minutes: float) -> int:
    return max(150, round(length_minutes * WORDS_PER_MINUTE))


def write_episode(
    llm: LLMProvider, channel_name: str, candidate: dict[str, Any], length_minutes: float
) -> tuple[GeneratedEpisode, float]:
    system = EPISODE_SYSTEM.format(
        channel=channel_name, words=target_words(length_minutes), minutes=length_minutes
    )
    kind_label = "how-to" if candidate["kind"] == "howto" else "fact story"
    prompt = (
        f"Niche: {candidate['niche']} ({kind_label})\n"
        f"Topic: {candidate['topic']}\n"
        f"Angle: {candidate['angle']}\n"
        f"Working title: {candidate['working_title']}\n"
        f"Today's trend it connects to: {candidate['trend']}\n"
    )
    result = llm.complete(
        system, prompt, json_schema=GeneratedEpisode.model_json_schema(), max_tokens=16000
    )
    try:
        episode = GeneratedEpisode.model_validate_json(result.text)
    except ValidationError:
        log.error("episode writer returned JSON that does not match the schema")
        raise
    if len(episode.script.split()) < 80:
        raise ValueError("episode writer returned a script that is far too short")
    return episode, result.cost or 0.0


def full_description(generated: GeneratedEpisode) -> str:
    hashtags = " ".join(
        h if h.startswith("#") else f"#{h}" for h in (x.strip().replace(" ", "") for x in generated.hashtags) if h
    )
    parts = [generated.description.strip(), DISCLOSURE]
    if hashtags:
        parts.append(hashtags)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Project creation (mirrors api.create_project, minus HTTP)
# ---------------------------------------------------------------------------


def _unique_slug(session: Session, owner_id: int, title: str) -> str:
    base = slugify(title)
    slug, n = base, 2
    while (
        session.query(Project)
        .filter(Project.owner_id == owner_id, Project.slug == slug)
        .first()
    ):
        slug, n = f"{base}-{n}", n + 1
    return slug


def create_project_for_episode(
    session: Session,
    user: User,
    title: str,
    script: str,
    enqueue: Callable[[Session, Project, str, list[str]], Job],
    channel_name: str,
) -> Project:
    """Same paywall, layout, and job argv as a dashboard create — except no
    --skip-render (see module docstring). --channel-name puts it under the
    channel in the dashboard's switcher and brands the intro/outro cards."""
    ent = entitlement(session, user)
    if ent["kind"] == "blocked" or (ent["kind"] == "subscription" and ent["remaining"] == 0):
        raise RuntimeError("no video credits left on this account — subscribe to keep the autopilot running")
    slug = _unique_slug(session, user.id, title)
    paths = ProjectPaths.create(Settings.load().projects_dir / f"u{user.id}", slug)
    project = Project(
        owner_id=user.id,
        slug=slug,
        title=title,
        dir_path=str(paths.root.resolve()),
        created_at=time.time(),
    )
    session.add(project)
    session.flush()
    consume_credit(session, user)
    source = paths.script / "source.txt"
    source.write_text(script)
    save_performance(ProjectPerformance(created_at=time.time()), paths)
    enqueue(
        session,
        project,
        "create",
        [
            "--script-file", str(source), "--style", "documentary", "--title", title,
            "--format", "landscape", "--channel-name", channel_name,
        ],
    )
    return project


def plan_episode(
    session: Session,
    episode: Episode,
    llm: LLMProvider,
    trend_sources: list[TrendProvider],
    enqueue: Callable[[Session, Project, str, list[str]], Job],
    exclude_topics: set[str] = frozenset(),
) -> Episode:
    """The whole morning: trends -> pick -> script/SEO -> queued render.
    Leaves the episode in "generating"; the pipeline job's completion hook
    (jobs.on_pipeline_job_finished) takes it from there."""
    channel = session.get(AutopilotChannel, episode.channel_id)
    user = session.get(User, channel.owner_id)
    settings = Settings.load()
    niches = channel.niches or list(DEFAULT_NICHES)

    trends = gather_trends(trend_sources, settings.trends_geo)
    recent = [e.topic for e in recent_episodes(session, channel) if e.id != episode.id]
    blocked = previous_niche(session, channel, episode.day)
    chosen, candidates, pick_cost = pick_topic(
        llm, niches, trends, recent, blocked, episode.day, exclude_topics
    )
    episode.candidates = candidates
    episode.llm_cost = pick_cost
    if chosen is None:
        raise RuntimeError("the topic picker returned no usable candidate for today's niches")
    episode.niche = chosen["niche"]
    episode.kind = chosen["kind"]
    episode.trend = chosen["trend"][:512]
    episode.topic = chosen["topic"][:512]
    episode.angle = chosen["angle"]
    episode.updated_at = time.time()
    session.commit()

    generated, write_cost = write_episode(llm, channel.name, chosen, channel.length_minutes)
    episode.llm_cost += write_cost
    episode.title = " ".join(generated.title.split())[:512]
    episode.description = full_description(generated)
    episode.tags = [t.strip() for t in generated.tags if t.strip()]

    project = create_project_for_episode(
        session, user, episode.title, generated.script, enqueue, channel.name
    )
    (Path(project.dir_path) / "logs" / "autopilot.json").write_text(
        json.dumps(
            {
                "episode_id": episode.id,
                "candidate": chosen,
                "llm_cost": episode.llm_cost,
                "title": episode.title,
                "description": episode.description,
                "tags": episode.tags,
            },
            indent=2,
        )
    )
    episode.project_id = project.id
    episode.status = "generating"
    episode.updated_at = time.time()
    session.commit()
    return episode


# ---------------------------------------------------------------------------
# Packaging for review
# ---------------------------------------------------------------------------


def project_paths(session: Session, episode: Episode) -> ProjectPaths | None:
    if episode.project_id is None:
        return None
    project = session.get(Project, episode.project_id)
    return ProjectPaths(root=Path(project.dir_path)) if project else None


def build_preview(final: Path, out: Path, seconds: int = PREVIEW_SECONDS) -> Path:
    """Short 480p clip for Telegram (bots can send at most 50 MB a file)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(final), "-t", str(seconds),
            "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "30", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def video_duration_label(path: Path) -> str:
    try:
        from renderflow.pipeline.render import probe_duration

        total = int(probe_duration(path))
    except Exception:
        return "?"
    return f"{total // 60}:{total % 60:02d}"


def episode_buttons(episode: Episode) -> list[list[tuple[str, str]]]:
    return [
        [("✅ Approve & upload", f"approve:{episode.id}"), ("🖼 New thumbnail", f"thumb:{episode.id}")],
        [("🔁 New topic", f"newtopic:{episode.id}"), ("⏭ Skip today", f"skip:{episode.id}")],
    ]


def project_cost(paths: ProjectPaths | None) -> float:
    if paths is None or not paths.scenes_json.exists():
        return 0.0
    from renderflow.storage import load_plan

    try:
        return load_plan(paths).total_asset_cost()
    except Exception:
        return 0.0
