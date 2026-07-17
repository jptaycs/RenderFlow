"""Celery worker: executes pipeline jobs from the DB queue.

Replaces api.py's dev-only subprocess spawner. The pipeline still runs as a
`make_video.py` subprocess (never inside the worker's Python process — it
needs its own process group so cancellation can kill the whole tree:
wav2lip's inference.py and ffmpeg used to survive as orphans when only the
direct child was signalled, found live 2026-07).

Run the worker (host, not Docker — the pipeline needs host ffmpeg and the
local model dirs):

    .venv/bin/celery -A renderflow.tasks worker --concurrency=1 --loglevel=info

Cancellation model (Phase 1, single host): the API marks the Job row
`cancelled` and kills the recorded pid's process group directly; this task
notices the status after `wait()` returns and leaves it alone. Queued jobs
are cancelled via plain `revoke()` plus the status check at task start.
When workers move to separate machines (Phase 2 deployment), running-job
cancellation must become a worker-side signal instead — the API won't share
a pid namespace with the worker anymore.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from celery import Celery
from celery.signals import worker_ready
from sqlalchemy.orm import Session

from renderflow.config import Settings
from renderflow.db import Job, Project, init_db, new_session
from renderflow.storage import ProjectPaths

REPO_ROOT = Path(__file__).resolve().parent.parent
ERROR_TAIL_CHARS = 2000

_settings = Settings.load()
celery_app = Celery(
    "renderflow", broker=_settings.redis_url, backend=_settings.redis_url
)
celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    # A lost worker must not silently re-run a half-finished pipeline job;
    # boot-time recovery (below) marks orphans failed for an explicit resume.
    task_acks_late=False,
)


def pid_is_pipeline(pid: int) -> bool:
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
    )
    return proc.returncode == 0 and "make_video.py" in proc.stdout


def kill_pipeline_pgid(pid: int) -> None:
    """SIGTERM the pipeline's whole process group, escalating to SIGKILL.

    Ported from api._kill_run: make_video.py is spawned with
    start_new_session=True, so killpg reaches its own children (wav2lip
    inference.py, ffmpeg) too.
    """

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
        if not pid_is_pipeline(pid):
            return
        time.sleep(0.25)
    _signal(signal.SIGKILL)


def cancel_job(session: Session, job: Job) -> None:
    """Cancel a queued or running job. Commits the status change itself —
    the cancelled status must be visible to the worker *before* the child
    dies, or the task's post-wait check would overwrite it with 'failed'."""
    job.status = "cancelled"
    job.finished_at = time.time()
    session.commit()
    if job.celery_task_id:
        celery_app.control.revoke(job.celery_task_id)
    if job.pid:
        kill_pipeline_pgid(job.pid)


def _log_tail(log_file: Path) -> str | None:
    try:
        return log_file.read_text(errors="replace")[-ERROR_TAIL_CHARS:]
    except OSError:
        return None


@celery_app.task(name="renderflow.run_pipeline")
def run_pipeline(job_id: int) -> None:
    session = new_session()
    try:
        job = session.get(Job, job_id)
        if job is None or job.status != "queued":
            return  # cancelled while queued, or a stale/duplicate delivery
        project = session.get(Project, job.project_id)

        # Safety net behind the API's 409 check: never two writers on one
        # project's scenes.json.
        clash = (
            session.query(Job)
            .filter(
                Job.project_id == job.project_id,
                Job.status == "running",
                Job.id != job.id,
            )
            .first()
        )
        if clash:
            job.status = "failed"
            job.error = f"another run (job {clash.id}) is already active for this project"
            job.finished_at = time.time()
            session.commit()
            return

        project_dir = Path(project.dir_path)
        paths = ProjectPaths.create(project_dir.parent, project_dir.name)
        argv = [
            sys.executable,
            str(REPO_ROOT / "make_video.py"),
            *job.argv,
            "--slug",
            project_dir.name,
            # CLI flag, never env: Settings.load() uses
            # load_dotenv(override=True), which clobbers inherited env vars.
            "--projects-dir",
            str(project_dir.parent),
        ]
        log_path = paths.logs / "run.log"
        with log_path.open("a") as log_file:
            log_file.write(
                f"\n=== {datetime.now().isoformat()} job {job.id} ({job.kind}) "
                f"{' '.join(job.argv)} ===\n"
            )
            log_file.flush()
            proc = subprocess.Popen(
                argv,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            job.status = "running"
            job.started_at = time.time()
            job.pid = proc.pid
            session.commit()
            returncode = proc.wait()

        session.refresh(job)  # the API may have marked it cancelled mid-run
        if job.status == "cancelled":
            return
        job.status = "succeeded" if returncode == 0 else "failed"
        if returncode != 0:
            job.error = (
                f"make_video.py exited with code {returncode}\n"
                f"{_log_tail(log_path) or '(no log output)'}"
            )
        job.finished_at = time.time()
        session.commit()
    except Exception as exc:  # a worker bug must never strand a job as 'running'
        session.rollback()
        job = session.get(Job, job_id)
        if job is not None and job.status in ("queued", "running"):
            job.status = "failed"
            job.error = f"worker error: {exc}"
            job.finished_at = time.time()
            session.commit()
        raise
    finally:
        session.close()


@worker_ready.connect
def recover_orphaned_jobs(**_kwargs) -> None:
    """On worker boot, fail any job left 'running' by a dead worker.

    If the pipeline subprocess itself is somehow still alive (worker was
    killed, child survived in its own session), kill it too — nothing is
    waiting on it anymore, and a future resume starting alongside it would
    mean two writers on the same scenes.json. Asset-level state is safe
    either way: pipeline/assets.py routes orphaned RUNNING assets through
    failed -> retrying on the next run.
    """
    init_db()
    session = new_session()
    try:
        for job in session.query(Job).filter(Job.status == "running").all():
            if job.pid and pid_is_pipeline(job.pid):
                kill_pipeline_pgid(job.pid)
            job.status = "failed"
            job.error = "worker restarted while this job was running — resume to retry"
            job.finished_at = time.time()
        session.commit()
    finally:
        session.close()
