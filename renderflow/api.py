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
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from renderflow import db
from renderflow.auth import current_user
from renderflow.auth import router as auth_router
from renderflow.billing import consume_credit, entitlement
from renderflow.billing import router as billing_router
from renderflow.config import Settings
from renderflow.db import Job, Project, User, get_db
from renderflow.pipeline.script import (
    effective_avatar_layout,
    scene_is_avatar_solo,
    scene_is_visual_only,
)
from renderflow.schema import AssetStatus, ProjectPerformance, Scene, ScenePlan
from renderflow.storage import (
    ProjectPaths,
    load_performance,
    load_plan,
    save_performance,
    save_plan,
    slugify,
)
from renderflow.tasks import cancel_job, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"

app = FastAPI(title="RenderFlow")
app.include_router(auth_router)
app.include_router(billing_router)


def _projects_dir() -> Path:
    return Settings.load().projects_dir


def _active_job(session: Session, project: Project) -> Job | None:
    return db.active_job(session, project.id)


def _enqueue(session: Session, project: Project, kind: str, argv: list[str]) -> Job:
    """Queue a pipeline run for the Celery worker.

    The Job row must be committed *before* .delay() — the worker can pick
    the message up faster than this request finishes, and an uncommitted
    job id would look like a stale delivery and be dropped.
    """
    job = Job(project_id=project.id, kind=kind, argv=argv)
    session.add(job)
    session.commit()
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
    # Also gate on "not run": ffmpeg creates final.mp4 on disk the instant it
    # starts encoding, with a fresh mtime — a still-active render would
    # otherwise look "ready" (and downloadable) while the file is mid-write.
    final_ready = (
        not run
        and all_done
        and final.exists()
        and final.stat().st_mtime >= paths.scenes_json.stat().st_mtime
    )

    if run:
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

    cost_by_category = {
        "Images": sum(
            (s.assets.image.cost or 0.0) + (s.assets.avatar_image.cost or 0.0)
            for s in plan.scenes
        ),
        "Voice": sum(s.assets.voice.cost or 0.0 for s in plan.scenes),
        "Avatar": sum(s.assets.avatar_clip.cost or 0.0 for s in plan.scenes),
    }

    cost = plan.total_asset_cost()
    perf = _load_performance_view(paths, final_ready, final)
    production_time_sec = (
        perf.completed_at - perf.created_at
        if perf.completed_at is not None and perf.created_at is not None
        else None
    )
    profit = perf.revenue_usd - cost if perf.revenue_usd is not None else None

    return {
        "slug": slug,
        "title": plan.title,
        "style": plan.style,
        "status": status,
        "progress": progress,
        "cost": cost,
        "costByCategory": cost_by_category,
        "estDurationSec": est_sec,
        "createdLabel": datetime.fromtimestamp(
            paths.scenes_json.stat().st_mtime
        ).strftime("%b %-d"),
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
    return {
        "slug": project.slug,
        "title": project.title,
        "style": "",
        "status": status,
        "progress": 0,
        "cost": 0.0,
        "costByCategory": {"Images": 0.0, "Voice": 0.0, "Avatar": 0.0},
        "estDurationSec": 0,
        "createdLabel": datetime.fromtimestamp(project.created_at).strftime("%b %-d"),
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
    projects.sort(key=lambda p: (not p["running"], p["title"].lower()))
    return {
        "projects": projects,
        "jobs": _jobs_view(projects),
        # Polled with the rest of state so the sidebar's credits/plan line
        # is always current (e.g. drops right after a create).
        "billing": entitlement(session, user),
    }


class NewProject(BaseModel):
    title: str
    script: str
    style: str = "documentary"


@app.post("/api/projects", status_code=201)
def create_project(
    body: NewProject,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict[str, str]:
    title = " ".join(body.title.split())
    script = body.script.strip()
    if not title:
        raise HTTPException(422, "title must not be empty")
    if not script:
        raise HTTPException(422, "script must not be empty")

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
    source = paths.script / "source.txt"
    source.write_text(script)
    save_performance(ProjectPerformance(created_at=time.time()), paths)
    # --skip-render: stop after assets so the project lands on "Paused" —
    # the user gets a chance to regenerate scenes or change avatar layouts
    # (now all split-screen by default, see scene_is_avatar_solo) before
    # committing to a multi-minute FFmpeg pass, instead of it rendering
    # immediately with whatever the first generation happened to produce.
    # The dashboard's existing "Resume run" button (shown for Paused
    # projects) does the render pass whenever they're ready.
    _enqueue(
        session,
        project,
        "create",
        ["--script-file", str(source), "--style", body.style, "--title", title, "--skip-render"],
    )
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
        if "renderflow:renderflow@" in settings.database_url:
            raise RuntimeError(
                "the database is still using the default dev password — set "
                "RENDERFLOW_PG_PASSWORD and RENDERFLOW_DATABASE_URL in .env"
            )
    db.init_db()
    _projects_dir().mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import uvicorn

    # Localhost-only bind: in production Caddy is the sole public listener
    # and proxies here; proxy_headers makes uvicorn trust its
    # X-Forwarded-For/Proto so request scheme and client IPs are right.
    uvicorn.run(app, host="127.0.0.1", port=8321, proxy_headers=True)
