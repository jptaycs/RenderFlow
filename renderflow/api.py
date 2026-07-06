"""Local dashboard API: read project state, launch pipeline runs, serve files.

The pipeline itself always runs as a `make_video.py` subprocess — never
inside a request handler. This is the dev, single-machine stand-in for the
Week-2 worker queue; endpoints are shaped so a Celery/Redis backend can
replace the subprocess layer without changing the frontend.

Run:  .venv/bin/python -m renderflow.api   (serves http://127.0.0.1:8321)
"""

from __future__ import annotations

import json
import shutil
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
from renderflow.schema import AssetStatus, Scene, ScenePlan
from renderflow.storage import ProjectPaths, load_plan, save_plan, slugify

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
    )
    (paths.logs / "run.pid").write_text(str(proc.pid))
    _runs[slug] = {"proc": proc, "started": time.time()}


# ---------------------------------------------------------------------------
# State assembly
# ---------------------------------------------------------------------------

_ASSET_KEYS = ("image", "voice", "avatar_clip")


def _scene_assets(scene: Scene) -> dict[str, str]:
    assets = {
        "image": scene.assets.image.status.value,
        "voice": scene.assets.voice.status.value,
    }
    if scene.type == "talking_avatar":
        assets["avatar"] = scene.assets.avatar_clip.status.value
    return assets


def _file_url(slug: str, path: str | None) -> str | None:
    """Map an absolute asset path to its /files URL (projects dir only)."""
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(_projects_dir().resolve() / slug)
    except ValueError:
        return None
    return f"/files/{slug}/{rel.as_posix()}"


def _refs(scene: Scene):
    yield scene.assets.image
    yield scene.assets.voice
    if scene.type == "talking_avatar":
        yield scene.assets.avatar_image
        yield scene.assets.avatar_clip


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
    final_ready = (
        all_done
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
            "thumb": _file_url(slug, s.assets.image.path)
            if s.assets.image.status is AssetStatus.COMPLETED else None,
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

    return {
        "slug": slug,
        "title": plan.title,
        "style": plan.style,
        "status": status,
        "progress": progress,
        "cost": plan.total_asset_cost(),
        "costByCategory": cost_by_category,
        "estDurationSec": est_sec,
        "createdLabel": datetime.fromtimestamp(
            paths.scenes_json.stat().st_mtime
        ).strftime("%b %-d"),
        "stages": stages,
        "scenes": scenes,
        "videoUrl": f"/files/{slug}/output/final.mp4" if final_ready else None,
        "thumbnailUrl": f"/files/{slug}/output/thumbnail.jpg"
        if (paths.output / "thumbnail.jpg").exists() else None,
        "running": bool(run),
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
    if _run_active(slug):
        raise HTTPException(409, "run in progress — wait for it to finish first")
    shutil.rmtree(paths.root)
    _runs.pop(slug, None)
    return {"deleted": slug}


@app.post("/api/projects/{slug}/resume")
def resume_project(slug: str) -> dict[str, str]:
    paths = _existing_project(slug)
    if _run_active(slug):
        raise HTTPException(409, "run already in progress")
    _spawn(slug, ["--scenes-file", str(paths.scenes_json)])
    return {"slug": slug}


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

    scene.assets = SceneAssets()
    (paths.output / "final.mp4").unlink(missing_ok=True)
    save_plan(plan, paths)
    _spawn(slug, ["--scenes-file", str(paths.scenes_json)])
    return {"slug": slug}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.on_event("startup")
def mount_files() -> None:
    projects_dir = _projects_dir()
    projects_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=projects_dir), name="files")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8321)
