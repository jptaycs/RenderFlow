"""Dashboard API: auth, per-user project state, job queue, file serving.

The pipeline itself always runs as a `make_video.py` subprocess — never
inside a request handler. Runs are queued as Job rows and executed by the
Celery worker (renderflow/tasks.py); this process only enqueues, cancels,
and reads state. Every project belongs to a User row; all endpoints are
scoped to the signed-in owner.

Run:  .venv/bin/python -m renderflow.api   (serves http://127.0.0.1:8321)
Needs docker compose up -d (Postgres + Redis) and the Celery worker —
see CLAUDE.md Commands.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from renderflow import db
from renderflow import youtube as youtube_module
from renderflow.auth import current_user
from renderflow.auth import router as auth_router
from renderflow.billing import consume_credit, entitlement
from renderflow.billing import router as billing_router
from renderflow.config import Settings
from renderflow.db import Job, Project, User, get_db
from renderflow.pipeline.script import (
    effective_avatar_layout,
    generate_topic_only,
    generate_topic_script,
    scene_is_avatar_solo,
    scene_is_visual_only,
)
from renderflow.providers import build_llm
from renderflow.schema import AssetStatus, ProjectPerformance, Scene, ScenePlan
from renderflow.storage import (
    ProjectPaths,
    load_performance,
    load_plan,
    load_youtube_publish,
    save_performance,
    save_plan,
    slugify,
)
from renderflow.tasks import cancel_job, kill_pipeline_pgid, pid_is_pipeline, run_pipeline

log = logging.getLogger("renderflow.api")

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"

app = FastAPI(title="RenderFlow")
app.include_router(auth_router)
app.include_router(billing_router)


def _created_label(dt: datetime) -> str:
    # "%-d" (no leading zero) is a glibc/macOS strftime extension only —
    # Windows' C runtime raises ValueError on it. `.day` is a plain int on
    # every platform, so build the label without relying on the extension.
    return f"{dt:%b} {dt.day}"


def _projects_dir() -> Path:
    return Settings.load().projects_dir


def _active_job(session: Session, project: Project) -> Job | None:
    return db.active_job(session, project.id)


_EAGER_JOB_QUEUE: "queue.Queue[int]" = queue.Queue()
_eager_worker_started = False
_eager_worker_lock = threading.Lock()


def _eager_worker_loop() -> None:
    """The single persistent worker thread for `RENDERFLOW_CELERY_EAGER=1`
    (added 2026-09) — pulls job ids off `_EAGER_JOB_QUEUE` and runs them
    strictly one at a time, matching the real production Celery worker's
    `--concurrency=1` (see the Commands section of CLAUDE.md).

    Before this, `_enqueue`'s eager branch spawned a brand-new background
    thread per job with no limit at all, so every queued video ran (and
    CPU-contended for FFmpeg encoding, plus the concurrent broll/image/
    voice HTTP calls) at once regardless of how many were already in
    flight. Client-reported (after investigating "why does a Short render
    too long"): four Shorts and several landscape videos were all
    rendering simultaneously on this single dev machine, each one
    crawling because it was sharing CPU with the others instead of
    running at full speed. A single serialized worker (this loop) instead
    of an unbounded thread-per-job gives the same one-at-a-time behavior
    the real deployed worker already has.

    `run_pipeline.run(job_id)` opens its own DB session internally (never
    shares the enqueuing request's `session`), so calling it from this
    one long-lived thread — instead of the caller's request thread, or a
    fresh thread per job — is safe the same way it always was.
    """
    while True:
        job_id = _EAGER_JOB_QUEUE.get()
        try:
            run_pipeline.run(job_id)
        except Exception:
            log.exception("eager-mode pipeline worker: job %d raised", job_id)


def _ensure_eager_worker_started() -> None:
    """Lazily starts the one `_eager_worker_loop` thread, exactly once.

    Double-checked locking (same pattern as `Labs69Video._lookup_model`)
    rather than starting it in `startup()`: a bare `TestClient(app)` (used
    throughout this test suite) doesn't fire FastAPI's startup event
    unless entered as a context manager, so tying worker startup to
    `_enqueue` itself — the thing that actually needs it — works
    regardless of how the app was booted.
    """
    global _eager_worker_started
    if _eager_worker_started:
        return
    with _eager_worker_lock:
        if _eager_worker_started:
            return
        threading.Thread(target=_eager_worker_loop, daemon=True).start()
        _eager_worker_started = True


def _enqueue(session: Session, project: Project, kind: str, argv: list[str]) -> Job:
    """Queue a pipeline run for the Celery worker.

    The Job row must be committed *before* dispatch — the worker (or, in
    eager dev mode, `_eager_worker_loop` below) can start faster than
    this request finishes, and an uncommitted job id would look like a
    stale delivery and be dropped.
    """
    job = Job(project_id=project.id, kind=kind, argv=argv)
    session.add(job)
    session.commit()
    if Settings.load().celery_eager:
        # task_always_eager makes .delay() execute the *entire* pipeline
        # synchronously in the calling thread before returning — correct
        # for a one-off manual CLI-style test, but it means this HTTP
        # request doesn't return until a real render finishes, which can
        # be many minutes. Client-reported: creating a new Short showed
        # "Starting…" and never resolved while an unrelated 11-minute
        # video was still generating — not actually stuck, just blocked
        # behind that other request's full pipeline run on whichever
        # thread picked it up. Dispatching onto the single persistent
        # `_eager_worker_loop` (via a queue, not a thread spawned here)
        # keeps that "request returns immediately" property while also
        # processing jobs one at a time, matching the real worker's
        # `--concurrency=1` — see `_eager_worker_loop`'s docstring.
        # celery_task_id stays unset on this path; cancellation already
        # works off job.pid, not the task id, so nothing else depends on
        # it being set here.
        _ensure_eager_worker_started()
        _EAGER_JOB_QUEUE.put(job.id)
    else:
        result = run_pipeline.delay(job.id)
        job.celery_task_id = result.id
        session.commit()
    return job


# ---------------------------------------------------------------------------
# State assembly
# ---------------------------------------------------------------------------

def _scene_assets(scene: Scene) -> dict[str, str]:
    assets: dict[str, str] = {}
    is_avatar_type = scene.type == "talking_avatar"
    # Solo-layout scenes never generate a background image (see
    # scene_is_avatar_solo) — showing an "Image: pending" chip that can
    # never complete looked like a stuck pipeline step.
    if not (is_avatar_type and scene_is_avatar_solo(scene)):
        assets["image"] = scene.assets.image.status.value
    assets["voice"] = scene.assets.voice.status.value
    # Visual-only scenes (see scene_is_visual_only) never get an avatar clip.
    if is_avatar_type and not scene_is_visual_only(scene):
        assets["avatar"] = scene.assets.avatar_clip.status.value
    return assets


def _file_url(paths: ProjectPaths, slug: str, path: str | Path | None) -> str | None:
    """Map an absolute asset path to its /files URL (project dir only).

    Appends the file's mtime as a cache-busting query param. Scene/thumbnail/
    video filenames are deterministic (scene_002.png, thumbnail.jpg,
    final.mp4) — regenerating a scene or resuming a project overwrites the
    same path, so without this the URL is byte-identical to before and the
    browser keeps showing the stale cached image/video after a "regenerate."
    """
    if not path:
        return None
    file_path = Path(path)
    try:
        rel = file_path.resolve().relative_to(paths.root.resolve())
    except ValueError:
        return None
    try:
        version = int(file_path.stat().st_mtime)
    except OSError:
        version = 0
    return f"/files/{slug}/{rel.as_posix()}?v={version}"


def _scene_thumb(paths: ProjectPaths, slug: str, scene: Scene) -> str | None:
    if scene.assets.image.status is AssetStatus.COMPLETED:
        return _file_url(paths, slug, scene.assets.image.path)
    # Solo-layout scenes have no background image — preview the avatar
    # portrait instead of leaving the card blank.
    if scene.assets.avatar_image.status is AssetStatus.COMPLETED:
        return _file_url(paths, slug, scene.assets.avatar_image.path)
    return None


def _refs(scene: Scene):
    is_avatar_type = scene.type == "talking_avatar"
    # Solo-layout scenes never get a background image (see
    # scene_is_avatar_solo) — counting it here would keep progress stuck
    # below 100% forever.
    if not (is_avatar_type and scene_is_avatar_solo(scene)):
        yield scene.assets.image
    yield scene.assets.voice
    # Visual-only scenes (see scene_is_visual_only) never get avatar assets.
    if is_avatar_type and not scene_is_visual_only(scene):
        yield scene.assets.avatar_image
        yield scene.assets.avatar_clip


def _best_effort_created_at(paths: ProjectPaths) -> float:
    # Projects created before performance.json existed have no recorded
    # creation time. st_birthtime (true creation time) isn't available on
    # every platform, so fall back to the scenes.json mtime — a rough
    # approximation is enough for a "production time" figure on old projects;
    # every project created going forward gets an exact value from
    # create_project() instead.
    try:
        return paths.root.stat().st_birthtime  # type: ignore[attr-defined]
    except AttributeError:
        return paths.scenes_json.stat().st_mtime


def _load_performance_view(paths: ProjectPaths, final_ready: bool, final: Path) -> ProjectPerformance:
    perf = load_performance(paths)
    dirty = False
    if perf.created_at is None:
        perf.created_at = _best_effort_created_at(paths)
        dirty = True
    if final_ready and perf.completed_at is None:
        perf.completed_at = final.stat().st_mtime
        dirty = True
    if dirty:
        save_performance(perf, paths)
    return perf


def _project_view(
    project: Project, plan: ScenePlan, paths: ProjectPaths, job: Job | None
) -> dict[str, Any]:
    slug = project.slug
    refs = [ref for scene in plan.scenes for ref in _refs(scene)]
    total = len(refs)
    done = sum(1 for r in refs if r.status is AssetStatus.COMPLETED)
    any_failed = any(r.status is AssetStatus.FAILED for r in refs)
    all_done = total > 0 and done == total
    final = paths.output / "final.mp4"
    # "Run active" = a queued or running Job row (the worker queue replaced
    # the old in-process pid tracking; job rows survive API restarts by
    # nature, which the run.pid file used to be needed for).
    run = job

    # A final.mp4 left over from an earlier run must not count: the render is
    # done only if every asset is completed AND the video is newer than the
    # last scene-plan change (scenes.json is rewritten on every asset update).
    # Also gate on "not a pipeline run": ffmpeg creates final.mp4 on disk the
    # instant it starts encoding, with a fresh mtime — a still-active render
    # would otherwise look "ready" (and downloadable) while the file is
    # mid-write. A youtube_publish run never touches final.mp4 (it only
    # reads it), so it must NOT hide the download link while uploading —
    # only a pipeline run (create/resume/regenerate/thumbnail) counts here.
    rendering = run is not None and run.kind != "youtube_publish"
    final_ready = (
        not rendering
        and all_done
        and final.exists()
        and final.stat().st_mtime >= paths.scenes_json.stat().st_mtime
    )

    if run and run.kind == "youtube_publish":
        status = "Publishing"
    elif run:
        status = "Rendering" if all_done else "Generating"
    elif final_ready:
        status = "Complete"
    elif any_failed:
        status = "Failed"
    elif done == 0:
        status = "Draft"
    else:
        status = "Paused"

    progress = 100 if final_ready else int((done / total) * 90) if total else 0

    assets_stage = (
        "complete" if all_done
        else "failed" if any_failed and not run
        else "active" if run or done
        else "pending"
    )
    stages = [
        {"name": "Script", "status": "complete"},
        {"name": "Scenes", "status": "complete" if plan.scenes else "pending"},
        {"name": "Assets", "status": assets_stage},
        {
            "name": "Render",
            "status": "complete" if final_ready
            else "active" if run and assets_stage == "complete"
            else "pending",
        },
    ]

    est_sec = sum(s.duration_estimate_sec for s in plan.scenes)
    scenes = [
        {
            "id": s.id,
            "number": s.id,
            "type": s.type,
            "durationSec": s.duration_estimate_sec,
            "narration": s.narration,
            "imagePrompt": s.image_prompt,
            "negativePrompt": s.negative_prompt,
            "provider": s.assets.image.provider or "—",
            "cost": sum(r.cost or 0.0 for r in _refs(s)),
            "assets": _scene_assets(s),
            "thumb": _scene_thumb(paths, slug, s),
            "avatarLayout": s.avatar_layout if s.type == "talking_avatar" else None,
            "effectiveLayout": (
                effective_avatar_layout(s)
                if s.type == "talking_avatar" else None
            ),
            "brollMode": s.broll_mode,
            "hasBroll": s.assets.broll.status is AssetStatus.COMPLETED,
        }
        for s in plan.scenes
    ]

    # Must sum to plan.total_asset_cost() (the displayed total below) — it
    # used to omit B-Roll and the plan-level thumbnail/intro/outro assets
    # entirely, so the per-module breakdown silently undercounted real
    # spend (e.g. every narrated intro/outro card). Found in a full-app
    # scan 2026-09.
    cost_by_category = {
        "Images": sum(
            (s.assets.image.cost or 0.0) + (s.assets.avatar_image.cost or 0.0)
            for s in plan.scenes
        ),
        "Voice": sum(s.assets.voice.cost or 0.0 for s in plan.scenes),
        "Avatar": sum(s.assets.avatar_clip.cost or 0.0 for s in plan.scenes),
        "B-Roll": sum(s.assets.broll.cost or 0.0 for s in plan.scenes),
        "Branding": (
            (plan.thumbnail.cost or 0.0)
            + (plan.intro_audio.cost or 0.0)
            + (plan.outro_audio.cost or 0.0)
            # Motion Graphics cards (added 2026-09) are credit-based on
            # 69labs (see cost_from_status) so this is usually 0.0 in
            # practice — included for the same reason every other
            # optional asset here is: total_asset_cost() sums them, so
            # the breakdown must too or it silently undercounts again.
            + (plan.intro_card_video.cost or 0.0)
            + (plan.outro_card_video.cost or 0.0)
        ),
    }

    cost = plan.total_asset_cost()
    # RenderFlow has no first-class multi-channel model (added 2026-09) —
    # the sidebar's channel switcher (web/index.html) just derives its
    # list of channels from the distinct channelName values across a
    # user's own projects, so this must surface the *resolved* name
    # (the per-project override if one was set at creation, else the
    # global default) rather than the raw possibly-None plan field.
    # None here specifically means "no channel name configured anywhere"
    # (plan override unset AND the global .env default is blank) — the
    # client buckets that as a single "Default" channel.
    settings = Settings.load()
    channel_name = plan.channel_name or settings.channel_name or None
    tts_voice = plan.tts_voice or settings.tts_voice or None
    perf = _load_performance_view(paths, final_ready, final)
    production_time_sec = (
        perf.completed_at - perf.created_at
        if perf.completed_at is not None and perf.created_at is not None
        else None
    )
    profit = perf.revenue_usd - cost if perf.revenue_usd is not None else None
    youtube = load_youtube_publish(paths)

    return {
        "slug": slug,
        "title": plan.title,
        "style": plan.style,
        "format": plan.format,
        "status": status,
        "progress": progress,
        "cost": cost,
        "costByCategory": cost_by_category,
        "estDurationSec": est_sec,
        "createdLabel": _created_label(
            datetime.fromtimestamp(paths.scenes_json.stat().st_mtime)
        ),
        "stages": stages,
        "scenes": scenes,
        "videoUrl": _file_url(paths, slug, paths.output / "final.mp4")
        if final_ready else None,
        "thumbnailUrl": _file_url(paths, slug, paths.output / "thumbnail.jpg")
        if (paths.output / "thumbnail.jpg").exists() else None,
        "running": bool(run),
        "runStartedAt": (run.started_at or run.created_at) if run else None,
        "views": perf.views,
        "watchTimeMinutes": perf.watch_time_minutes,
        "revenueUsd": perf.revenue_usd,
        "notes": perf.notes,
        "profit": profit,
        "productionTimeSec": production_time_sec,
        "createdAt": perf.created_at,
        "channelName": channel_name,
        "ttsVoice": tts_voice,
        "youtube": (
            {
                "url": youtube.url,
                "videoId": youtube.video_id,
                "privacyStatus": youtube.privacy_status,
                "publishedAt": youtube.published_at,
            }
            if youtube
            else None
        ),
        # final_ready gates the button client-side; youtube.is_connected()
        # is cheap (a file-exists check) so it's fine to call on every poll.
        "youtubeConnected": youtube_module.is_connected(),
    }


def _jobs_view(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"running": 0, "retrying": 1, "failed": 2, "pending": 3, "completed": 4}
    jobs: list[dict[str, Any]] = []
    for project in projects:
        run = project["running"]
        started = project.get("runStartedAt")
        elapsed = int(time.time() - started) if run and started else 0
        for scene in project["scenes"]:
            for kind, status in scene["assets"].items():
                if status == "completed" and not run:
                    continue  # keep the feed focused on active/queued work
                jobs.append({
                    "id": f"{project['slug']}/s{scene['id']}/{kind}",
                    "type": kind,
                    "status": status,
                    "provider": scene["provider"] if kind == "image" else "",
                    "projectTitle": project["title"],
                    "elapsedSec": elapsed if status in ("running", "retrying") else 0,
                    "cost": scene["cost"] if status == "completed" else 0.0,
                })
        if run and project["status"] == "Rendering":
            jobs.append({
                "id": f"{project['slug']}/render",
                "type": "render",
                "status": "running",
                "provider": "FFmpeg",
                "projectTitle": project["title"],
                "elapsedSec": elapsed,
                "cost": 0.0,
            })
    jobs.sort(key=lambda j: order.get(j["status"], 5))
    return jobs[:60]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _placeholder_view(session: Session, project: Project) -> dict[str, Any]:
    """View for a project whose create-job hasn't written scenes.json yet.

    Pre-queue, such a directory was simply skipped by the dashboard scan;
    now the Project row exists the instant the user clicks Create, so it
    must render as a card (otherwise a failed create-job would leave an
    invisible project squatting on its slug forever). Same key set as
    _project_view so the frontend never sees missing fields.
    """
    latest = (
        session.query(Job)
        .filter(Job.project_id == project.id)
        .order_by(Job.created_at.desc())
        .first()
    )
    running = latest is not None and latest.status in db.ACTIVE_JOB_STATUSES
    if running:
        status = "Generating"
    elif latest is not None and latest.status == "failed":
        status = "Failed"
    else:
        status = "Draft"
    # scenes.json doesn't exist yet, so plan.format isn't available — read
    # it back from the create job's own argv instead, so the dashboard's
    # Shorts/Landscape tab filter has something to go on immediately after
    # Create, not just once assets finish generating.
    fmt = "landscape"
    if latest is not None and "--format" in latest.argv:
        fmt = latest.argv[latest.argv.index("--format") + 1]
    # Same reasoning as --format above, for the sidebar channel switcher
    # (added 2026-09) — a just-clicked video with a channel override
    # should show up under that channel immediately, not just once
    # scenes.json exists. Falls back to the global default exactly like
    # _project_view's resolution when no override was given.
    settings = Settings.load()
    channel_name = settings.channel_name or None
    tts_voice = settings.tts_voice or None
    if latest is not None and "--channel-name" in latest.argv:
        channel_name = latest.argv[latest.argv.index("--channel-name") + 1]
    if latest is not None and "--tts-voice" in latest.argv:
        tts_voice = latest.argv[latest.argv.index("--tts-voice") + 1]
    return {
        "slug": project.slug,
        "title": project.title,
        "style": "",
        "format": fmt,
        "status": status,
        "progress": 0,
        "cost": 0.0,
        "costByCategory": {"Images": 0.0, "Voice": 0.0, "Avatar": 0.0},
        "estDurationSec": 0,
        "createdLabel": _created_label(datetime.fromtimestamp(project.created_at)),
        "stages": [
            {"name": "Script", "status": "active" if running else "pending"},
            {"name": "Scenes", "status": "pending"},
            {"name": "Assets", "status": "pending"},
            {"name": "Render", "status": "pending"},
        ],
        "scenes": [],
        "videoUrl": None,
        "thumbnailUrl": None,
        "running": running,
        "runStartedAt": (latest.started_at or latest.created_at) if running else None,
        "views": None,
        "watchTimeMinutes": None,
        "revenueUsd": None,
        "notes": "",
        "profit": None,
        "productionTimeSec": None,
        "createdAt": project.created_at,
        "channelName": channel_name,
        "ttsVoice": tts_voice,
    }


@app.get("/api/state")
def get_state(
    user: User = Depends(current_user), session: Session = Depends(get_db)
) -> dict[str, Any]:
    projects: list[dict[str, Any]] = []
    rows = session.query(Project).filter(Project.owner_id == user.id).all()
    for row in rows:
        paths = ProjectPaths(root=Path(row.dir_path))
        if not paths.scenes_json.exists():
            projects.append(_placeholder_view(session, row))
            continue
        try:
            plan = load_plan(paths)
        except (ValueError, json.JSONDecodeError):
            continue  # mid-write or hand-edited; next poll picks it up
        projects.append(_project_view(row, plan, paths, _active_job(session, row)))
    # running-first (active work stays visible), then newest-created —
    # the dashboard's own sort/filter controls (web/index.html's
    # projectsGridHtml, GRID_SORT_OPTIONS) re-sort client-side on top of
    # this, but the raw default itself used to be pure alphabetical by
    # title, which read as arbitrary on a media dashboard where recency
    # matters far more than title. Client-reported: the Shorts page "not
    # organized well".
    projects.sort(key=lambda p: (not p["running"], -(p["createdAt"] or 0)))
    return {
        "projects": projects,
        "jobs": _jobs_view(projects),
        # Polled with the rest of state so the sidebar's credits/plan line
        # is always current (e.g. drops right after a create).
        "billing": entitlement(session, user),
    }


class TopicIdea(BaseModel):
    title: str


class TopicIdeaRequest(BaseModel):
    # Titles already shown to the user this modal session (previous
    # /api/topics/random responses the user hasn't created yet) — see the
    # excludeTitles docstring note below for why this is required, not just
    # the DB's project titles.
    excludeTitles: list[str] = []
    # The channel this new video will belong to (the New Video modal's
    # current channel-name field — see web/index.html's openModal/
    # channelName wiring), added 2026-09 (client request: "the generate
    # topic must be related with the selected channel"). Blank/omitted =
    # no channel scoping, same generic trivia behavior as before — correct
    # for the default/main channel, which has never needed scoping.
    channelName: str | None = None


def _channel_titles(session: Session, user: User, channel_name: str, settings: Settings) -> list[str]:
    """Titles of the user's own projects that resolve to `channel_name`
    (added 2026-09 alongside topic-idea channel scoping) — same resolved-
    name logic as `_project_view`'s `channelName` field (a project's own
    override, else the global default), just without the rest of that
    function's heavier per-project computation, since only the title is
    needed here. A project with no scenes.json yet (a placeholder, still
    queued) has nothing to resolve and is skipped."""
    titles = []
    for project in session.query(Project).filter(Project.owner_id == user.id).all():
        paths = _project_paths(project)
        if not paths.scenes_json.exists():
            continue
        try:
            plan = load_plan(paths)
        except (ValueError, OSError):
            continue
        resolved = plan.channel_name or settings.channel_name or None
        if resolved == channel_name:
            titles.append(project.title)
    return titles


@app.post("/api/topics/random")
def random_topic_idea(
    body: TopicIdeaRequest = TopicIdeaRequest(),
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> TopicIdea:
    """One fresh Claude-generated video **title only** for the New Video
    modal's "🎲 Random topic" button — replaces the old client-side static
    RANDOM_TOPICS bank (see pipeline/script.py::generate_topic_only), which
    was a fixed 10-title array that ran out and started repeating well
    before a real user's project count did.

    **Title-only, split from script generation, added 2026-09** (client
    request: "generate the topic first then below is generate the script
    if user like the topic") — this used to also write the full narration
    in the same call (`generate_topic_idea`), which meant every click paid
    for a full script even though most ideas are clicked past without
    ever being used. The New Video modal now shows just the title with a
    separate "Generate script" action below it (`POST /api/topics/script`)
    that only fires once the user actually wants that specific idea.

    Deliberately a plain synchronous call, not a queued Job like project
    creation — a title-only completion is small and fast regardless of
    the eventual video length, unlike the full --topic script generation
    that stays inside the worker subprocess (see create_project) because
    it also drives image/voice/broll generation on top of the script
    itself.

    `body.excludeTitles` matters because this call is otherwise stateless:
    the *only* exclusion this endpoint used to send Claude was the DB's
    actual project titles, so repeated clicks before ever creating a
    project sent the exact same prompt every time — and with no memory of
    its own past answers, Claude kept converging on the same "obvious"
    fact each click (client-reported: "clicking regenerate topic gives
    the same topic"). The frontend now accumulates every title it's shown
    in this modal session and resends the growing list here.
    """
    existing_titles = [
        row.title
        for row in session.query(Project).filter(Project.owner_id == user.id).all()
    ] + body.excludeTitles
    channel_name = (body.channelName or "").strip() or None
    channel_titles = (
        _channel_titles(session, user, channel_name, Settings.load())
        if channel_name
        else None
    )
    try:
        llm = build_llm(Settings.load())
        idea, _ = generate_topic_only(llm, existing_titles, channel_name, channel_titles)
    except Exception as exc:  # missing/invalid ANTHROPIC_API_KEY, rate limit, etc.
        log.warning("random topic idea generation failed: %s", exc)
        raise HTTPException(
            503, "topic idea generation is unavailable right now"
        ) from exc
    return TopicIdea(title=idea.title)


class TopicScript(BaseModel):
    script: str


class TopicScriptRequest(BaseModel):
    title: str
    # Target video length in minutes — the modal's currently-selected
    # length for landscape, or ~1 for Shorts (see web/index.html's
    # generateTopicScript()). Sizes the generated script's word count to
    # actually match the video it's about to become, instead of always
    # returning a fixed short teaser regardless of target length
    # (client-reported: "the random topic script must have a length the
    # same with the target time like 11 minutes"). Same 1-15 clamp as
    # NewProject's lengthMinutes below.
    lengthMinutes: float = 1.5


@app.post("/api/topics/script")
def random_topic_script(
    body: TopicScriptRequest,
    user: User = Depends(current_user),
) -> TopicScript:
    """Step 2 of the "🎲 Random topic" flow (added 2026-09): the full
    narration script for a title the user has already seen and asked for
    via the New Video modal's "Generate script" button (shown once a
    title comes back from POST /api/topics/random). See
    pipeline/script.py::generate_topic_script.

    Deliberately a plain synchronous call, same reasoning as
    random_topic_idea above — a few seconds to a bit over a minute
    depending on length_minutes, not a queued Job.
    """
    length_minutes = min(max(body.lengthMinutes, 1.0), 15.0)
    try:
        llm = build_llm(Settings.load())
        result, _ = generate_topic_script(llm, body.title, length_minutes)
    except Exception as exc:  # missing/invalid ANTHROPIC_API_KEY, rate limit, etc.
        log.warning("topic script generation failed: %s", exc)
        raise HTTPException(
            503, "script generation is unavailable right now"
        ) from exc
    return TopicScript(script=result.script)


class NewProject(BaseModel):
    title: str
    # Exactly one of these: a client-provided script (split locally into
    # scenes) or a bare topic (Claude writes the full narration + scene
    # plan from scratch via pipeline/script.py::generate_script — needs a
    # live ANTHROPIC_API_KEY). lengthMinutes only applies to topic mode;
    # script mode's length is however long the pasted script naturally is.
    script: str | None = None
    topic: str | None = None
    lengthMinutes: float = 3.0
    style: str = "documentary"
    # "landscape" (default) or "shorts" — see schema.VideoFormat and
    # render.py's format-gating notes for what "shorts" skips (v1 scope).
    format: Literal["landscape", "shorts"] = "landscape"
    # Per-project overrides of RENDERFLOW_CHANNEL_NAME/RENDERFLOW_TTS_VOICE
    # (added 2026-09, client request: a second content "channel" — a
    # distinct branding name and narrator voice for a batch of videos —
    # without flipping the global .env value back and forth between
    # creations). Empty/omitted = use the global setting, same as every
    # project before this feature. See ScenePlan.channel_name/tts_voice's
    # own docstring for why this is persisted on the plan, not just a
    # one-off CLI value.
    channelName: str | None = None
    ttsVoice: str | None = None


@app.post("/api/projects", status_code=201)
def create_project(
    body: NewProject,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    title = " ".join(body.title.split())
    script = (body.script or "").strip()
    topic = (body.topic or "").strip()
    if not title:
        raise HTTPException(422, "title must not be empty")
    if bool(script) == bool(topic):
        raise HTTPException(422, "provide exactly one of script or topic")
    # Generous but bounded — this drives an LLM call and a whole asset
    # generation batch, not just a config knob. Shorts always targets ~1
    # minute regardless of what was submitted — YouTube Shorts tops out
    # around 60s-3min, and the whole point is a short, punchy hook.
    length_minutes = (
        1.0 if body.format == "shorts" else min(max(body.lengthMinutes, 1.0), 15.0)
    )

    # Paywall: trial credits first, then an active subscription's monthly
    # allowance; admins unlimited. 402 tells the frontend to open pricing.
    ent = entitlement(session, user)
    if ent["kind"] == "blocked":
        raise HTTPException(
            402, "your free trial is used up — subscribe to keep creating videos"
        )
    if ent["kind"] == "subscription" and ent["remaining"] == 0:
        raise HTTPException(
            402,
            f"monthly limit reached on the {ent['plan']} plan — "
            "upgrade or wait for the new month",
        )

    slug = slugify(title)
    existing = (
        session.query(Project)
        .filter(Project.owner_id == user.id, Project.slug == slug)
        .first()
    )
    if existing:
        raise HTTPException(409, f"a project titled {title!r} already exists")
    # Per-user namespace: new projects never collide with (or leak) other
    # users' slugs. Adopted legacy projects keep their old flat location via
    # dir_path — this layout only applies to new ones.
    paths = ProjectPaths.create(_projects_dir() / f"u{user.id}", slug)
    project = Project(
        owner_id=user.id,
        slug=slug,
        title=title,
        dir_path=str(paths.root.resolve()),
        created_at=time.time(),
    )
    session.add(project)
    session.flush()
    # One video = one trial credit while unsubscribed (subscription usage is
    # derived from the project count; nothing to write for it). Same
    # transaction as the project row — a failed create can't burn a credit.
    consume_credit(session, user)
    save_performance(ProjectPerformance(created_at=time.time()), paths)
    if topic:
        # No file needed — --topic is a plain CLI value, unlike
        # --script-file. Saved alongside script/scenes.json purely for
        # provenance/debugging (never read back by the pipeline).
        (paths.script / "topic.txt").write_text(topic)
        source_args = [
            "--topic", topic,
            "--length", str(length_minutes),
        ]
    else:
        source = paths.script / "source.txt"
        source.write_text(script)
        source_args = ["--script-file", str(source)]
    # --skip-render: stop after assets so the project lands on "Paused" —
    # the user gets a chance to regenerate scenes or change avatar layouts
    # (now all split-screen by default, see scene_is_avatar_solo) before
    # committing to a multi-minute FFmpeg pass, instead of it rendering
    # immediately with whatever the first generation happened to produce.
    # The dashboard's existing "Resume run" button (shown for Paused
    # projects) does the render pass whenever they're ready.
    # Shorts skip this pause and go straight through to a rendered
    # final.mp4 in the same run: a Short is only ~6-12 scenes generated in
    # one quick batch, so there's little to review mid-way, and stopping
    # at "Paused" just forced an extra manual "Resume run" click for every
    # Short. Client-reported: the shorts process "dont stop after assets
    # in pipeline done". Landscape (potentially 50+ scenes) keeps the
    # pause — that's still worth a review step before a long render.
    create_args = [
        *source_args, "--style", body.style, "--title", title,
        "--format", body.format,
    ]
    if (body.channelName or "").strip():
        create_args += ["--channel-name", body.channelName.strip()]
    if (body.ttsVoice or "").strip():
        create_args += ["--tts-voice", body.ttsVoice.strip()]
    if body.format != "shorts":
        create_args.append("--skip-render")
    _enqueue(session, project, "create", create_args)
    return {"slug": slug}


def _owned_project(session: Session, user: User, slug: str) -> Project:
    """Resolve a slug to the signed-in user's project row.

    404 (not 403) for other users' projects — same response as a
    nonexistent slug, so nothing leaks about what other accounts have."""
    project = (
        session.query(Project)
        .filter(Project.owner_id == user.id, Project.slug == slug)
        .first()
    )
    if project is None:
        raise HTTPException(404, f"no project {slug!r}")
    return project


def _project_paths(project: Project) -> ProjectPaths:
    return ProjectPaths(root=Path(project.dir_path))


@app.delete("/api/projects/{slug}")
def delete_project(
    slug: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    project = _owned_project(session, user, slug)
    # Stop an active run before removing its files out from under it, rather
    # than blocking the delete — the client asked to be able to delete a
    # project while it's still generating, not just after.
    job = _active_job(session, project)
    if job:
        cancel_job(session, job)
    session.query(Job).filter(Job.project_id == project.id).delete()
    session.delete(project)
    shutil.rmtree(project.dir_path, ignore_errors=True)
    return {"deleted": slug}


@app.post("/api/projects/{slug}/cancel")
def cancel_project(
    slug: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    project = _owned_project(session, user, slug)
    job = _active_job(session, project)
    if job is None:
        raise HTTPException(409, "no run in progress")
    cancel_job(session, job)
    return {"slug": slug}


@app.post("/api/projects/{slug}/resume")
def resume_project(
    slug: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    project = _owned_project(session, user, slug)
    if _active_job(session, project):
        raise HTTPException(409, "run already in progress")
    paths = _project_paths(project)
    _enqueue(session, project, "resume", ["--scenes-file", str(paths.scenes_json)])
    return {"slug": slug}


class PerformanceUpdate(BaseModel):
    views: int | None = None
    watchTimeMinutes: float | None = None
    revenueUsd: float | None = None
    notes: str = ""


@app.post("/api/projects/{slug}/performance")
def set_performance(
    slug: str,
    body: PerformanceUpdate,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    """Manual YouTube performance entry — there is no YouTube API integration,
    the dashboard's Revenue form always submits the full set of fields at
    once, so this is a full replace of the user-entered fields (not a partial
    merge) — that's what lets clearing a field back to blank actually stick.
    created_at/completed_at are untouched; the pipeline owns those."""
    paths = _project_paths(_owned_project(session, user, slug))
    perf = load_performance(paths)
    perf.views = body.views
    perf.watch_time_minutes = body.watchTimeMinutes
    perf.revenue_usd = body.revenueUsd
    perf.notes = body.notes
    perf.updated_at = time.time()
    save_performance(perf, paths)
    return {"slug": slug}


# A fresh random seed on the exact same prompt often keeps a similar
# composition (the prompt text drives composition far more than the seed
# does) — so a manual "Regenerate" swaps in a different framing instruction
# too, to actually give a visibly different shot rather than a near-repeat.
_VARIATION_MARKER = " Try this take: "
_REGENERATE_VARIATIONS = (
    "from a different camera angle",
    "as a wider establishing shot",
    "as a closer detail shot",
    "at a different time of day",
    "from a different camera position",
    "with a different composition and framing",
)


def _vary_prompt(prompt: str) -> str:
    # Strip any variation clause appended by an earlier regenerate so
    # repeated clicks don't grow the prompt without bound.
    base = prompt.split(_VARIATION_MARKER)[0].rstrip()
    variation = random.choice(_REGENERATE_VARIATIONS)
    return f"{base}{_VARIATION_MARKER}{variation}."


@app.post("/api/projects/{slug}/scenes/{scene_id}/regenerate")
def regenerate_scene(
    slug: str,
    scene_id: int,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    project = _owned_project(session, user, slug)
    if _active_job(session, project):
        raise HTTPException(409, "run already in progress")
    paths = _project_paths(project)
    plan = load_plan(paths)
    scene = next((s for s in plan.scenes if s.id == scene_id), None)
    if scene is None:
        raise HTTPException(404, f"no scene {scene_id} in {slug!r}")
    for ref in _refs(scene):
        if ref.path:
            Path(ref.path).unlink(missing_ok=True)
    from renderflow.schema import SceneAssets

    if not (scene.type == "talking_avatar" and scene_is_avatar_solo(scene)):
        scene.image_prompt = _vary_prompt(scene.image_prompt)
    scene.assets = SceneAssets()
    (paths.output / "final.mp4").unlink(missing_ok=True)
    save_plan(plan, paths)
    # Skip the final render here — regenerating one scene must not force a
    # multi-minute re-encode of the whole video before the project unlocks
    # for the next regenerate. The dashboard's "Resume run" (already shown
    # once a project has no fresh final.mp4) does the one render pass once
    # the user is done regenerating whatever scenes they wanted to fix.
    _enqueue(
        session,
        project,
        "regenerate",
        ["--scenes-file", str(paths.scenes_json), "--skip-render"],
    )
    return {"slug": slug}


@app.post("/api/projects/{slug}/thumbnail/regenerate")
def regenerate_thumbnail(
    slug: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    """Regenerate only the clickbait thumbnail (background + reaction face).

    Unlike scene regenerate, nothing is reset here — the spawned run's
    --thumbnail-only mode resets the thumbnail asset itself, so there's a
    single writer of the plan. The final render stays valid and
    downloadable: the run re-stamps final.mp4's freshness (see
    make_video._regenerate_thumbnail), and the old thumbnail.jpg is only
    removed after the new images generate successfully, so a failed
    regenerate keeps the previous thumbnail instead of leaving none."""
    project = _owned_project(session, user, slug)
    if _active_job(session, project):
        raise HTTPException(409, "run already in progress")
    paths = _project_paths(project)
    _enqueue(
        session,
        project,
        "thumbnail",
        ["--scenes-file", str(paths.scenes_json), "--thumbnail-only"],
    )
    return {"slug": slug}


class YouTubePublishRequest(BaseModel):
    title: str
    description: str = ""
    tags: list[str] = []
    privacyStatus: str = "public"
    # YouTube's disclosure flag for altered/synthetic content (status.
    # containsSyntheticMedia) — true by default since every RenderFlow
    # video is AI-narrated over AI-generated visuals; see renderflow/
    # youtube.py. The client shows this as a checkbox, not a hidden default.
    containsSyntheticMedia: bool = True


@app.post("/api/projects/{slug}/youtube/publish")
def publish_to_youtube(
    slug: str,
    body: YouTubePublishRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    """Upload the finished final.mp4 (+ thumbnail.jpg) to YouTube via a
    spawned publish_youtube.py subprocess (job kind "youtube_publish") —
    never inside this request; a multi-hundred-MB upload can take minutes,
    same reasoning as never rendering synchronously.

    Requires the one-time OAuth setup (scripts/setup_youtube.py) — 503s
    rather than queueing a job that can only fail immediately.
    """
    if not youtube_module.is_connected():
        raise HTTPException(
            503,
            "YouTube isn't connected yet — run scripts/setup_youtube.py "
            "once (see CLAUDE.md)",
        )
    if body.privacyStatus not in ("public", "unlisted", "private"):
        raise HTTPException(422, "privacyStatus must be public, unlisted, or private")
    project = _owned_project(session, user, slug)
    if _active_job(session, project):
        raise HTTPException(409, "run already in progress")
    paths = _project_paths(project)
    if not (paths.output / "final.mp4").exists():
        raise HTTPException(422, "render the video before publishing")
    title = " ".join(body.title.split())
    if not title:
        raise HTTPException(422, "title must not be empty")

    argv = [
        "--title", title,
        "--description", body.description,
        "--tags", ",".join(t.strip() for t in body.tags if t.strip()),
        "--privacy", body.privacyStatus,
    ]
    if not body.containsSyntheticMedia:
        argv.append("--no-synthetic-disclosure")
    _enqueue(session, project, "youtube_publish", argv)
    return {"slug": slug}


class SceneBrollUpdate(BaseModel):
    mode: str  # "auto" | "off"


@app.post("/api/projects/{slug}/scenes/{scene_id}/broll")
def set_scene_broll(
    slug: str,
    scene_id: int,
    body: SceneBrollUpdate,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    """Per-scene stock-video override: "auto" uses a fetched B-roll clip for
    full-frame scenes when available, "off" forces the still image (e.g. the
    stock clip doesn't fit the narration). Mirrors set_scene_layout: the
    final render is invalidated, and a generation run is queued only when
    turning auto on with no clip fetched yet."""
    if body.mode not in ("auto", "off"):
        raise HTTPException(422, "mode must be 'auto' or 'off'")
    project = _owned_project(session, user, slug)
    if _active_job(session, project):
        raise HTTPException(409, "run already in progress")
    paths = _project_paths(project)
    plan = load_plan(paths)
    scene = next((s for s in plan.scenes if s.id == scene_id), None)
    if scene is None:
        raise HTTPException(404, f"no scene {scene_id} in {slug!r}")

    scene.broll_mode = body.mode
    eligible = scene.type == "narration" or (
        scene.type == "talking_avatar" and scene_is_visual_only(scene)
    )
    needs_generation = (
        body.mode == "auto"
        and eligible
        and bool(Settings.load().broll_provider)
        and scene.assets.broll.status != AssetStatus.COMPLETED
    )
    (paths.output / "final.mp4").unlink(missing_ok=True)
    save_plan(plan, paths)
    if needs_generation:
        _enqueue(
            session,
            project,
            "layout",
            ["--scenes-file", str(paths.scenes_json), "--skip-render"],
        )
    return {"slug": slug}


class SceneLayoutUpdate(BaseModel):
    layout: str  # "auto" | "solo" | "split" | "visual"


@app.post("/api/projects/{slug}/scenes/{scene_id}/layout")
def set_scene_layout(
    slug: str,
    scene_id: int,
    body: SceneLayoutUpdate,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    """Override the avatar layout for one scene: full-screen solo avatar,
    avatar + background visual split-screen, or visual-only (background
    visual with narration audio, no avatar shown at all) — e.g. the user
    doesn't want the generated visual for this beat and would rather the
    host talk full-screen instead, doesn't want the avatar visible at all
    for this beat, or the reverse. Persisted on the scene itself
    (`avatar_layout`); see effective_avatar_layout for how "auto" always
    means split-screen."""
    if body.layout not in ("auto", "solo", "split", "visual"):
        raise HTTPException(422, "layout must be 'auto', 'solo', 'split', or 'visual'")
    project = _owned_project(session, user, slug)
    if _active_job(session, project):
        raise HTTPException(409, "run already in progress")
    paths = _project_paths(project)
    plan = load_plan(paths)
    scene = next((s for s in plan.scenes if s.id == scene_id), None)
    if scene is None:
        raise HTTPException(404, f"no scene {scene_id} in {slug!r}")
    if scene.type != "talking_avatar":
        raise HTTPException(422, "layout override only applies to talking-avatar scenes")

    scene.avatar_layout = body.layout
    # Split/visual need a background visual — generate one if this scene
    # never had one (it was solo up to now). Solo/split need the avatar
    # portrait + lip-synced clip — generate them if this scene never had
    # them (it was visual-only up to now). Whichever assets a layout
    # doesn't need are just left alone, unused, in case they switch back.
    needs_generation = (
        not scene_is_avatar_solo(scene)
        and scene.assets.image.status != AssetStatus.COMPLETED
    ) or (
        not scene_is_visual_only(scene)
        and (
            scene.assets.avatar_image.status != AssetStatus.COMPLETED
            or scene.assets.avatar_clip.status != AssetStatus.COMPLETED
        )
    )
    (paths.output / "final.mp4").unlink(missing_ok=True)
    save_plan(plan, paths)
    if needs_generation:
        _enqueue(
            session,
            project,
            "layout",
            ["--scenes-file", str(paths.scenes_json), "--skip-render"],
        )
    return {"slug": slug}


@app.get("/files/{slug}/{file_path:path}")
def serve_file(
    slug: str,
    file_path: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> FileResponse:
    """Serve project assets (scene thumbs, thumbnail.jpg, final.mp4).

    Replaces the old unauthenticated StaticFiles mount: files are only
    served to the project's owner, and only from inside that project's own
    directory (traversal rejected). The ?v=<mtime> cache-buster on asset
    URLs keeps working — query params are ignored here just as StaticFiles
    ignored them. Browsers send the session cookie on same-origin <img>/<a>
    requests automatically."""
    project = _owned_project(session, user, slug)
    root = Path(project.dir_path).resolve()
    target = (root / file_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/logo.png")
def logo() -> FileResponse:
    return FileResponse(WEB_DIR / "logo.png")


def _recover_orphaned_jobs_eager(session: Session) -> None:
    """Eager-mode counterpart to `tasks.recover_orphaned_jobs` (added
    2026-09). That function only runs on a real Celery worker's
    `worker_ready` signal — which never fires when
    `RENDERFLOW_CELERY_EAGER=1` dispatches pipeline runs on a background
    thread inside *this* process instead (see `_enqueue`). Restarting
    api.py while a job is mid-run (e.g. to pick up a code change) kills
    that background thread: the pipeline subprocess itself keeps running
    independently, but nothing is left alive to write its result back, so
    the Job row is stuck at 'running'/'queued' forever — and
    `_project_view`'s `final_ready` check treats *any* active Job row as
    "still rendering" regardless of what's actually on disk, so a
    genuinely finished video shows as stuck "Generating" indefinitely.

    Caught live 2026-09: three same-session api.py restarts (for unrelated
    code changes) orphaned four Shorts create jobs; all four had actually
    finished — real `final.mp4` on disk, 12-18 minutes after being queued
    — but sat stuck at `status='running'` for 2+ hours until this function
    existed to reconcile them. Client-reported (indirectly): "why the
    queued shorts renders too long."

    Unlike the Celery version (which always marks an orphan 'failed' —
    correct there, since killing a *worker process* really does kill its
    child pipeline subprocess too), this checks actual completion evidence
    first: if the project looks Complete (`_project_view`'s own
    `final_ready` logic, called with `job=None` to ask "if there were no
    active run, would this look done?"), the job is marked 'succeeded'
    instead — the work is real, and marking it 'failed' would hide a
    finished video and could prompt a wasteful, unnecessary re-render.
    Only runs when `settings.celery_eager` — a real Celery worker already
    reconciles its own orphans via `recover_orphaned_jobs`.
    """
    for job in session.query(Job).filter(Job.status.in_(["running", "queued"])).all():
        if job.pid and pid_is_pipeline(job.pid):
            kill_pipeline_pgid(job.pid)
        project = session.get(Project, job.project_id)
        succeeded = False
        if project and job.kind != "youtube_publish":
            paths = ProjectPaths(root=Path(project.dir_path))
            if paths.scenes_json.exists():
                try:
                    plan = load_plan(paths)
                    succeeded = _project_view(project, plan, paths, None)["status"] == "Complete"
                except (ValueError, json.JSONDecodeError):
                    pass  # mid-write or hand-edited; leave it a failure below
        job.status = "succeeded" if succeeded else "failed"
        job.error = None if succeeded else (
            "api server restarted while this job was running — resume to retry"
        )
        job.finished_at = job.finished_at or time.time()
        log.warning(
            "recovered orphaned job %d (%s, project %s) as %s",
            job.id, job.kind, project.slug if project else "?", job.status,
        )
    session.commit()


def _seconds_until_next_run(hour: int, now: datetime | None = None) -> float:
    """Seconds from `now` (real time if omitted) until the next local
    `hour:00:00` — today if that hasn't passed yet, else tomorrow. Pure
    and injectable so the scheduling math is unit-testable without
    actually waiting; the real trigger (`_auto_publish_scheduler_loop`)
    is a thin `time.sleep(...)` wrapper around this, same split as
    `_recover_orphaned_jobs_eager`'s logic-vs-trigger separation."""
    now = now or datetime.now()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _run_auto_publish_batch(session: Session, settings: Settings) -> list[str]:
    """Queue up to `settings.auto_publish_max_per_day` Complete-and-
    unpublished projects for YouTube upload, oldest-finished-first.

    Deliberately reuses `_project_view`'s own `status`/`youtube`
    resolution (same as `_recover_orphaned_jobs_eager` does) rather than
    re-deriving "is this done and unpublished" from the filesystem here —
    this must agree with what the dashboard's own Unpublished filter
    chip (`web/index.html::projectsGridHtml`) shows, or a video could
    auto-publish that the UI still called "Failed"/"Generating", or
    silently skip one the UI clearly shows as Complete.

    Scans every project across every user, same as the manual "Publish
    to YouTube" button already implicitly does — YouTube publishing is
    tied to one shared machine-level OAuth connection regardless of
    which app user clicks the button (see the YouTube publishing note in
    CLAUDE.md), so a scheduled batch has no narrower scope to apply here
    than the existing per-click action already has.

    Returns the slugs it queued (for logging/testing) — never raises for
    a single bad project (a `scenes.json` mid-write, a project missing
    its directory, etc.), same never-block convention as everything else
    in this pipeline; one project's problem must not stop the rest of
    the batch from publishing.
    """
    if not youtube_module.is_connected():
        log.warning("auto-publish: YouTube isn't connected, skipping this run")
        return []

    candidates: list[tuple[float, Project, ScenePlan, ProjectPaths]] = []
    for project in session.query(Project).all():
        try:
            if _active_job(session, project):
                continue  # already publishing, still rendering, etc.
            paths = _project_paths(project)
            if not paths.scenes_json.exists():
                continue
            plan = load_plan(paths)
            view = _project_view(project, plan, paths, None)
            if view["status"] != "Complete":
                continue
            if load_youtube_publish(paths) is not None:
                continue  # already published
            final = paths.output / "final.mp4"
            candidates.append((final.stat().st_mtime, project, plan, paths))
        except (ValueError, json.JSONDecodeError, OSError):
            log.warning("auto-publish: skipping project %s, could not evaluate it", project.slug, exc_info=True)

    candidates.sort(key=lambda c: c[0])  # oldest-finished-first
    queued: list[str] = []
    for _mtime, project, plan, paths in candidates[: settings.auto_publish_max_per_day]:
        argv = [
            "--title", plan.title,
            "--description", "",
            "--tags", "",
            "--privacy", "public",
        ]
        _enqueue(session, project, "youtube_publish", argv)
        queued.append(project.slug)
        log.info("auto-publish: queued %s for YouTube upload", project.slug)
    return queued


def _auto_publish_scheduler_loop() -> None:
    """Background thread for `RENDERFLOW_AUTO_PUBLISH=1` — wakes once a
    day at `RENDERFLOW_AUTO_PUBLISH_HOUR` and runs `_run_auto_publish_batch`.
    Re-reads `Settings.load()` on every wake (not just once at thread
    start), so toggling the feature off in `.env` takes effect on the
    *next* scheduled wake without needing a full api.py restart — though
    a change to the hour itself only takes effect the day after, since
    the sleep duration for the *current* wait was already computed
    before the edit.
    """
    while True:
        settings = Settings.load()
        delay = _seconds_until_next_run(settings.auto_publish_hour)
        time.sleep(delay)
        settings = Settings.load()
        if not settings.auto_publish_enabled:
            continue
        session = db.new_session()
        try:
            _run_auto_publish_batch(session, settings)
        except Exception:
            log.exception("auto-publish batch failed")
        finally:
            session.close()


@app.on_event("startup")
def startup() -> None:
    settings = Settings.load()
    if not settings.secret_key:
        raise RuntimeError(
            "RENDERFLOW_SECRET_KEY is not set — generate one with "
            "`python -c 'import secrets; print(secrets.token_hex(32))'` "
            "and add it to .env"
        )
    if settings.env == "production":
        # The dev conveniences are auth/paywall bypasses — a production
        # instance must be impossible to start with them configured.
        if settings.dev_login_email or settings.dev_login_password:
            raise RuntimeError(
                "RENDERFLOW_DEV_LOGIN_EMAIL/_PASSWORD must not be set in "
                "production — remove them from .env"
            )
        if settings.dev_checkout:
            raise RuntimeError(
                "RENDERFLOW_DEV_CHECKOUT must not be set in production — "
                "it activates subscriptions without payment"
            )
        if settings.celery_eager:
            raise RuntimeError(
                "RENDERFLOW_CELERY_EAGER must not be set in production — it "
                "runs the pipeline synchronously inside the request handler"
            )
        if "renderflow:renderflow@" in settings.database_url:
            raise RuntimeError(
                "the database is still using the default dev password — set "
                "RENDERFLOW_PG_PASSWORD and RENDERFLOW_DATABASE_URL in .env"
            )
    db.init_db()
    _projects_dir().mkdir(parents=True, exist_ok=True)
    if settings.celery_eager:
        # See _recover_orphaned_jobs_eager's own docstring — a real Celery
        # worker reconciles its own orphans via recover_orphaned_jobs
        # (worker_ready signal), which never fires in this in-process
        # dispatch mode, so api.py must do it on its own boot instead.
        session = db.new_session()
        try:
            _recover_orphaned_jobs_eager(session)
        finally:
            session.close()
    if settings.auto_publish_enabled:
        threading.Thread(target=_auto_publish_scheduler_loop, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    # Localhost-only bind: in production Caddy is the sole public listener
    # and proxies here; proxy_headers makes uvicorn trust its
    # X-Forwarded-For/Proto so request scheme and client IPs are right.
    uvicorn.run(app, host="127.0.0.1", port=8321, proxy_headers=True)
