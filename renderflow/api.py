"""Local dashboard API: read project state, launch pipeline runs, serve files.

The pipeline itself always runs as a `make_video.py` subprocess — never
inside a request handler. This is the dev, single-machine stand-in for the
Week-2 worker queue; endpoints are shaped so a Celery/Redis backend can
replace the subprocess layer without changing the frontend.

Run:  .venv/bin/python -m renderflow.api   (serves http://127.0.0.1:8321)
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from renderflow.config import Settings
from renderflow.pipeline.script import scene_is_avatar_solo
from renderflow.schema import AssetStatus, ProjectPerformance, Scene, ScenePlan
from renderflow.storage import (
    ProjectPaths,
    load_performance,
    load_plan,
    save_performance,
    save_plan,
    slugify,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"

app = FastAPI(title="RenderFlow")

# slug -> {"proc": Popen, "started": float}; dev-only, lost on restart
# (statuses then fall back to what scenes.json says on disk).
_runs: dict[str, dict[str, Any]] = {}


def _projects_dir() -> Path:
    return Settings.load().projects_dir


def _pid_is_pipeline(pid: int) -> bool:
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
    )
    return proc.returncode == 0 and "make_video.py" in proc.stdout


def _run_active(slug: str) -> dict[str, Any] | None:
    run = _runs.get(slug)
    if run and run["proc"].poll() is None:
        return run
    # Fall back to the pid file so runs survive an API restart: the
    # subprocess keeps working either way, and the UI must keep seeing it.
    pid_file = _projects_dir() / slug / "logs" / "run.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            return None
        if _pid_is_pipeline(pid):
            return {"proc": None, "started": pid_file.stat().st_mtime, "pid": pid}
    return None


def _spawn(slug: str, args: list[str]) -> None:
    paths = ProjectPaths.create(_projects_dir(), slug)
    log_file = (paths.logs / "run.log").open("a")
    log_file.write(f"\n=== {datetime.now().isoformat()} {' '.join(args)} ===\n")
    log_file.flush()
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "make_video.py"), *args, "--slug", slug],
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # New session (own process group) so _kill_run can signal the whole
        # tree at once — make_video.py shells out to real subprocesses of
        # its own (wav2lip's inference.py, ffmpeg); killing only the direct
        # child left those running to completion as orphans (found live
        # 2026-07 testing cancel: a wav2lip inference.py kept burning CPU
        # for a minute after "cancel" had already returned).
        start_new_session=True,
    )
    (paths.logs / "run.pid").write_text(str(proc.pid))
    _runs[slug] = {"proc": proc, "started": time.time()}


def _kill_run(slug: str) -> bool:
    """Stop an active run and its whole subprocess tree, if any.

    make_video.py is spawned in its own process group (_spawn,
    start_new_session=True) so a single os.killpg reaches the pipeline's own
    subprocess children too (wav2lip's inference.py, ffmpeg) — killing only
    the direct child left those running to completion as orphans (found
    live 2026-07 testing cancel: a wav2lip inference.py kept burning CPU for
    a minute after "cancel" had already returned). Falls back to signaling
    just the pid for a run spawned before this change, which predates the
    process group and so has none to target.

    Whatever asset was mid-generation is left in RUNNING state on disk —
    that's already handled without any extra bookkeeping here: `_start`
    (pipeline/assets.py) treats an orphaned RUNNING asset exactly like one
    left behind by a crash, routing it through FAILED -> RETRYING the next
    time a run starts on this project.
    """
    run = _run_active(slug)
    if not run:
        return False
    pid = run["proc"].pid if run.get("proc") is not None else run["pid"]

    def _signal(sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    _signal(signal.SIGTERM)
    for _ in range(20):
        if not _pid_is_pipeline(pid):
            break
        time.sleep(0.25)
    else:
        _signal(signal.SIGKILL)
    proc = run.get("proc")
    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    _runs.pop(slug, None)
    (_projects_dir() / slug / "logs" / "run.pid").unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# State assembly
# ---------------------------------------------------------------------------

def _scene_assets(scene: Scene) -> dict[str, str]:
    assets: dict[str, str] = {}
    # Solo-layout scenes never generate a background image (see
    # scene_is_avatar_solo) — showing an "Image: pending" chip that can
    # never complete looked like a stuck pipeline step.
    if not (scene.type == "talking_avatar" and scene_is_avatar_solo(scene)):
        assets["image"] = scene.assets.image.status.value
    assets["voice"] = scene.assets.voice.status.value
    if scene.type == "talking_avatar":
        assets["avatar"] = scene.assets.avatar_clip.status.value
    return assets


def _file_url(slug: str, path: str | Path | None) -> str | None:
    """Map an absolute asset path to its /files URL (projects dir only).

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
        rel = file_path.resolve().relative_to(_projects_dir().resolve() / slug)
    except ValueError:
        return None
    try:
        version = int(file_path.stat().st_mtime)
    except OSError:
        version = 0
    return f"/files/{slug}/{rel.as_posix()}?v={version}"


def _scene_thumb(slug: str, scene: Scene) -> str | None:
    if scene.assets.image.status is AssetStatus.COMPLETED:
        return _file_url(slug, scene.assets.image.path)
    # Solo-layout scenes have no background image — preview the avatar
    # portrait instead of leaving the card blank.
    if scene.assets.avatar_image.status is AssetStatus.COMPLETED:
        return _file_url(slug, scene.assets.avatar_image.path)
    return None


def _refs(scene: Scene):
    # Solo-layout scenes never get a background image (see
    # scene_is_avatar_solo) — counting it here would keep progress stuck
    # below 100% forever.
    if not (scene.type == "talking_avatar" and scene_is_avatar_solo(scene)):
        yield scene.assets.image
    yield scene.assets.voice
    if scene.type == "talking_avatar":
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


def _project_view(slug: str, plan: ScenePlan, paths: ProjectPaths) -> dict[str, Any]:
    refs = [ref for scene in plan.scenes for ref in _refs(scene)]
    total = len(refs)
    done = sum(1 for r in refs if r.status is AssetStatus.COMPLETED)
    any_failed = any(r.status is AssetStatus.FAILED for r in refs)
    all_done = total > 0 and done == total
    final = paths.output / "final.mp4"
    run = _run_active(slug)

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
            "thumb": _scene_thumb(slug, s),
            "avatarLayout": s.avatar_layout if s.type == "talking_avatar" else None,
            "effectiveLayout": (
                ("solo" if scene_is_avatar_solo(s) else "split")
                if s.type == "talking_avatar" else None
            ),
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
        "videoUrl": _file_url(slug, paths.output / "final.mp4") if final_ready else None,
        "thumbnailUrl": _file_url(slug, paths.output / "thumbnail.jpg")
        if (paths.output / "thumbnail.jpg").exists() else None,
        "running": bool(run),
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
        run = _run_active(project["slug"])
        elapsed = int(time.time() - run["started"]) if run else 0
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


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    projects: list[dict[str, Any]] = []
    projects_dir = _projects_dir()
    if projects_dir.exists():
        for entry in sorted(projects_dir.iterdir()):
            paths = ProjectPaths(root=entry)
            if not paths.scenes_json.exists():
                continue
            try:
                plan = load_plan(paths)
            except (ValueError, json.JSONDecodeError):
                continue  # mid-write or hand-edited; next poll picks it up
            projects.append(_project_view(entry.name, plan, paths))
    projects.sort(key=lambda p: (not p["running"], p["title"].lower()))
    return {"projects": projects, "jobs": _jobs_view(projects)}


class NewProject(BaseModel):
    title: str
    script: str
    style: str = "documentary"


@app.post("/api/projects", status_code=201)
def create_project(body: NewProject) -> dict[str, str]:
    title = " ".join(body.title.split())
    script = body.script.strip()
    if not title:
        raise HTTPException(422, "title must not be empty")
    if not script:
        raise HTTPException(422, "script must not be empty")
    slug = slugify(title)
    if _run_active(slug):
        raise HTTPException(409, f"project {slug!r} already has a run in progress")
    if (ProjectPaths(root=_projects_dir() / slug)).scenes_json.exists():
        raise HTTPException(409, f"a project titled {title!r} already exists")
    paths = ProjectPaths.create(_projects_dir(), slug)
    source = paths.script / "source.txt"
    source.write_text(script)
    save_performance(ProjectPerformance(created_at=time.time()), paths)
    _spawn(slug, ["--script-file", str(source), "--style", body.style, "--title", title])
    return {"slug": slug}


def _existing_project(slug: str) -> ProjectPaths:
    """Resolve a slug to its project dir, rejecting traversal and unknowns."""
    if slug != slugify(slug):
        raise HTTPException(404, f"no project {slug!r}")
    paths = ProjectPaths(root=_projects_dir() / slug)
    if not paths.scenes_json.exists():
        raise HTTPException(404, f"no project {slug!r}")
    return paths


@app.delete("/api/projects/{slug}")
def delete_project(slug: str) -> dict[str, str]:
    paths = _existing_project(slug)
    # Stop an active run before removing its files out from under it, rather
    # than blocking the delete — the client asked to be able to delete a
    # project while it's still generating, not just after.
    _kill_run(slug)
    shutil.rmtree(paths.root)
    return {"deleted": slug}


@app.post("/api/projects/{slug}/cancel")
def cancel_project(slug: str) -> dict[str, str]:
    _existing_project(slug)
    if not _kill_run(slug):
        raise HTTPException(409, "no run in progress")
    return {"slug": slug}


@app.post("/api/projects/{slug}/resume")
def resume_project(slug: str) -> dict[str, str]:
    paths = _existing_project(slug)
    if _run_active(slug):
        raise HTTPException(409, "run already in progress")
    _spawn(slug, ["--scenes-file", str(paths.scenes_json)])
    return {"slug": slug}


class PerformanceUpdate(BaseModel):
    views: int | None = None
    watchTimeMinutes: float | None = None
    revenueUsd: float | None = None
    notes: str = ""


@app.post("/api/projects/{slug}/performance")
def set_performance(slug: str, body: PerformanceUpdate) -> dict[str, str]:
    """Manual YouTube performance entry — there is no YouTube API integration,
    the dashboard's Revenue form always submits the full set of fields at
    once, so this is a full replace of the user-entered fields (not a partial
    merge) — that's what lets clearing a field back to blank actually stick.
    created_at/completed_at are untouched; the pipeline owns those."""
    paths = _existing_project(slug)
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
def regenerate_scene(slug: str, scene_id: int) -> dict[str, str]:
    paths = _existing_project(slug)
    if _run_active(slug):
        raise HTTPException(409, "run already in progress")
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
    _spawn(slug, ["--scenes-file", str(paths.scenes_json), "--skip-render"])
    return {"slug": slug}


class SceneLayoutUpdate(BaseModel):
    layout: str  # "auto" | "solo" | "split"


@app.post("/api/projects/{slug}/scenes/{scene_id}/layout")
def set_scene_layout(slug: str, scene_id: int, body: SceneLayoutUpdate) -> dict[str, str]:
    """Override solo-vs-split for one scene — e.g. the user doesn't want
    the generated visual for this beat and would rather the host just talk
    full-screen instead (or the reverse). Persisted on the scene itself
    (`avatar_layout`); see scene_is_avatar_solo for how "auto" falls back to
    the default cycle."""
    if body.layout not in ("auto", "solo", "split"):
        raise HTTPException(422, "layout must be 'auto', 'solo', or 'split'")
    paths = _existing_project(slug)
    if _run_active(slug):
        raise HTTPException(409, "run already in progress")
    plan = load_plan(paths)
    scene = next((s for s in plan.scenes if s.id == scene_id), None)
    if scene is None:
        raise HTTPException(404, f"no scene {scene_id} in {slug!r}")
    if scene.type != "talking_avatar":
        raise HTTPException(422, "layout override only applies to talking-avatar scenes")

    scene.avatar_layout = body.layout
    # Split needs a background visual — generate one if this scene never
    # had one (it was solo up to now). Solo needs nothing generated; any
    # existing image is just left alone, unused, in case they switch back.
    needs_image = not scene_is_avatar_solo(scene) and scene.assets.image.status != AssetStatus.COMPLETED
    (paths.output / "final.mp4").unlink(missing_ok=True)
    save_plan(plan, paths)
    if needs_image:
        _spawn(slug, ["--scenes-file", str(paths.scenes_json), "--skip-render"])
    return {"slug": slug}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/logo.png")
def logo() -> FileResponse:
    return FileResponse(WEB_DIR / "logo.png")


@app.on_event("startup")
def mount_files() -> None:
    projects_dir = _projects_dir()
    projects_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=projects_dir), name="files")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8321)
