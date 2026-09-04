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

import contextlib
import logging
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

log = logging.getLogger("renderflow.tasks")

# Windows SetThreadExecutionState flags — see _prevent_system_sleep below.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def _prevent_system_sleep():
    """Hold off Windows system sleep for the duration of a pipeline run.

    Added 2026-09 after a real render appeared to take ~9 hours: the log
    showed normal ~5s status-poll cadence, then a silent multi-hour gap,
    then polling resumed at the same cadence like nothing happened — the
    signature of the OS suspending the process (laptop slept), not
    anything actually slow. In eager mode (this dev machine) or a real
    Celery worker, the thread that calls proc.wait() blocks synchronously
    on network I/O for the whole run; if the machine sleeps mid-render, a
    render that would otherwise finish in minutes just freezes until
    someone wakes the machine back up, then continues from where it left
    off — which reads as "rendering takes forever."

    ES_SYSTEM_REQUIRED (not ES_DISPLAY_REQUIRED) only blocks *system*
    sleep — the display can still turn off/lock normally, this doesn't
    keep the screen on. Windows-only: SetThreadExecutionState doesn't
    exist elsewhere, and a deployed Linux server doesn't sleep on its own
    the way a laptop does, so this is a no-op there by construction.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS)

REPO_ROOT = Path(__file__).resolve().parent.parent
ERROR_TAIL_CHARS = 2000
# Every job kind runs make_video.py except youtube_publish — added 2026-09
# alongside renderflow/youtube.py + publish_youtube.py. Same subprocess/
# logging/cancellation machinery below serves both scripts unmodified.
SCRIPT_BY_JOB_KIND = {"youtube_publish": "publish_youtube.py"}
# Auto-retry a failed make_video.py run up to this many attempts before
# giving up — see the retry loop in run_pipeline for why this is safe.
MAX_RUN_ATTEMPTS = 10
RUN_RETRY_DELAY_SEC = 5


def _resumable_job_argv(job_argv: list[str], scenes_json: Path) -> list[str]:
    """The argv a *retry* attempt should actually use, so it resumes from
    whatever the previous attempt already generated instead of starting
    the whole video over.

    Bug fixed 2026-09, caught live on a real 112-scene project: the retry
    loop below reused the exact same `job.argv` on every attempt, which is
    only actually resumable for "resume"/"regenerate"/"thumbnail" jobs —
    every one of those is *already* built with `--scenes-file` (api.py),
    so re-running them just reloads the persisted, partially-completed
    plan and `AssetRef`'s state machine skips whatever's already COMPLETED
    (per the docstring above this constant). A "create" job is different:
    it's built with `--script-file`/`--topic` (never `--scenes-file`,
    since there's no scenes.json yet when it's first enqueued) — and
    `make_video.py` re-derives a *brand new* `ScenePlan` from the raw
    script/topic on *every* invocation, regardless of any scenes.json
    that a previous attempt already wrote to disk. So a mid-run failure
    on a create job (e.g. a transient 69labs/Claude error at scene 90 of
    112) auto-retried into silently discarding every already-generated,
    already-paid-for asset and regenerating the entire video from
    scratch — the exact opposite of what auto-retry was built for
    (see MAX_RUN_ATTEMPTS's own docstring: "re-invoking ... just picks up
    whatever didn't finish, never redoes completed work").

    Once a plan has actually been persisted (`scenes_json.exists()` —
    `make_video.py` calls `save_plan()` immediately after resolving
    title/format, before any asset generation starts), every later
    attempt should load *that* instead of re-parsing the source: same
    `--scenes-file` argv shape "resume" already uses successfully,
    dropping the script/topic-construction flags (title/format inside
    the persisted plan are already correct — attempt 1 applied them
    before its own `save_plan()` — so there's nothing left to reapply
    except `--skip-render`, which every create job always sets and a
    `--scenes-file` invocation understands identically).

    A no-op for every job kind that already passes `--scenes-file` (every
    non-create kind), and for a create job whose attempt 1 crashed before
    ever reaching `save_plan()` (nothing to resume from yet either way).
    """
    if "--scenes-file" in job_argv or not scenes_json.exists():
        return job_argv
    extra = ["--skip-render"] if "--skip-render" in job_argv else []
    return ["--scenes-file", str(scenes_json), *extra]

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
if _settings.celery_eager:
    # See Settings.celery_eager's docstring: dev/test-only, no Redis or
    # `celery worker` process needed. api.startup refuses this in production.
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)


def pid_is_pipeline(pid: int) -> bool:
    # Windows has no `ps` binary and no equivalent -o command= flag on
    # `tasklist` — bug found live 2026-09 while verifying the eager-mode
    # threaded-dispatch fix above: cancelling a genuinely-running project
    # 500'd instead of stopping it, leaving the subprocess orphaned and
    # still running. tasklist can't give us the full command line without
    # WMI/psutil (neither in this project's dependencies), so this checks
    # image name only — make_video.py/publish_youtube.py are always
    # spawned as `sys.executable <script>`, i.e. python.exe, so this is
    # the same "plausibly our pipeline, not some unrelated process reusing
    # the pid" approximation the POSIX branch already made, just coarser.
    if sys.platform == "win32":
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True,
        )
        return proc.returncode == 0 and "python.exe" in proc.stdout.lower()
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return False
    return "make_video.py" in proc.stdout or any(
        script in proc.stdout for script in SCRIPT_BY_JOB_KIND.values()
    )


def kill_pipeline_pgid(pid: int) -> None:
    """Kill the pipeline's whole process tree.

    POSIX: SIGTERM the process group (make_video.py is spawned with
    start_new_session=True, so killpg reaches its own children too —
    wav2lip inference.py, ffmpeg), escalating to SIGKILL if it doesn't
    exit within 5s. Windows has neither process groups nor SIGKILL —
    os.killpg/signal.SIGKILL don't exist there at all (AttributeError,
    caught live 2026-09: "Cancel run" / deleting a running project 500'd
    instead of stopping it). `taskkill /F /T` kills the whole descendant
    tree unconditionally in one call — no graceful-then-forceful two-step
    the way POSIX gets from SIGTERM to SIGKILL, since Windows has nothing
    analogous to SIGTERM for an ordinary process either.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, text=True
        )
        return

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
        script = SCRIPT_BY_JOB_KIND.get(job.kind, "make_video.py")

        def _effective_job_argv(attempt: int) -> list[str]:
            # Attempt 1 always uses the job's own argv verbatim; a later
            # attempt resumes from whatever got persisted instead of
            # re-deriving the plan from scratch — see
            # _resumable_job_argv's docstring for why that distinction
            # matters (only "create" jobs are actually affected).
            if attempt == 1:
                return job.argv
            return _resumable_job_argv(job.argv, paths.scenes_json)

        def _argv_for_attempt(attempt: int) -> list[str]:
            return [
                sys.executable,
                str(REPO_ROOT / script),
                *_effective_job_argv(attempt),
                "--slug",
                project_dir.name,
                # CLI flag, never env: Settings.load() uses
                # load_dotenv(override=True), which clobbers inherited env vars.
                "--projects-dir",
                str(project_dir.parent),
            ]

        log_path = paths.logs / "run.log"
        # Auto-retry only the resumable pipeline (make_video.py): a failed
        # asset leaves completed ones in place (schema.AssetRef's state
        # machine skips them on re-run), so re-invoking after a failure is
        # safe and just picks up whatever didn't finish. youtube_publish is
        # a one-shot, human-reviewed action with no such resumability, so
        # it keeps the original single-attempt behavior. Added 2026-09
        # after a run failed on 100% of its scenes from a transient local
        # TLS-interception error (see the truststore fix) that a plain
        # retry would have recovered from without anyone noticing the
        # "Failed" status and clicking Resume by hand.
        max_attempts = MAX_RUN_ATTEMPTS if script == "make_video.py" else 1
        returncode = 1
        for attempt in range(1, max_attempts + 1):
            session.refresh(job)  # the API may have cancelled it between attempts
            if job.status == "cancelled":
                return
            argv = _argv_for_attempt(attempt)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n=== {datetime.now().isoformat()} job {job.id} ({job.kind}) "
                    f"attempt {attempt}/{max_attempts} "
                    f"{' '.join(_effective_job_argv(attempt))} ===\n"
                )
                log_file.flush()
                proc = subprocess.Popen(
                    argv,
                    cwd=REPO_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    # make_video.py's console output uses plain Unicode (→, ✦,
                    # …) with no thought to the terminal encoding — fine on a
                    # UTF-8 default (macOS/Linux), but Windows' legacy console
                    # codepage (cp1252 etc.) can't encode those characters, and
                    # a redirected-to-file stdout still uses that same locale
                    # encoding unless overridden. Crashed a real run live
                    # (UnicodeEncodeError on '→') the first time this
                    # dashboard ran on a Windows box. PYTHONIOENCODING forces
                    # UTF-8 for the child regardless of host locale.
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                job.status = "running"
                if attempt == 1:
                    job.started_at = time.time()
                job.pid = proc.pid
                session.commit()
                # See _prevent_system_sleep's docstring — a Windows sleep
                # mid-wait() would otherwise freeze this render until
                # someone wakes the machine, then silently resume.
                with _prevent_system_sleep():
                    returncode = proc.wait()

            session.refresh(job)  # the API may have marked it cancelled mid-run
            if job.status == "cancelled":
                return
            if returncode == 0:
                break
            if attempt < max_attempts:
                log.warning(
                    "job %d (%s) attempt %d/%d failed (exit %d), retrying in %ds",
                    job.id, job.kind, attempt, max_attempts, returncode,
                    RUN_RETRY_DELAY_SEC,
                )
                time.sleep(RUN_RETRY_DELAY_SEC)

        job.status = "succeeded" if returncode == 0 else "failed"
        if returncode != 0:
            attempts_label = (
                f"{max_attempts} attempts" if max_attempts > 1 else "1 attempt"
            )
            job.error = (
                f"{script} exited with code {returncode} after {attempts_label}\n"
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
