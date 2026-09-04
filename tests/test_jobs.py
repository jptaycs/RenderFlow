"""Worker job lifecycle: run_pipeline transitions, cancellation, recovery.

The Celery task function is called directly (synchronously) — no broker.
subprocess.Popen is stubbed so no real make_video.py ever runs.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from renderflow import db as rdb
from renderflow import tasks


def _make_project(saas_env, session, slug="demo") -> rdb.Project:
    user = rdb.User(email="worker-test@example.com", password_hash="x")
    session.add(user)
    session.flush()
    project = rdb.Project(
        owner_id=user.id,
        slug=slug,
        title="Demo",
        dir_path=str(saas_env.projects_dir / f"u{user.id}" / slug),
        created_at=0.0,
    )
    session.add(project)
    session.flush()
    return project


def _make_job(session, project, status="queued", kind="resume", **kwargs) -> rdb.Job:
    job = rdb.Job(
        project_id=project.id, kind=kind, argv=["--scenes-file", "x"], status=status, **kwargs
    )
    session.add(job)
    session.commit()
    return job


class FakeProc:
    def __init__(self, returncode=0, on_wait=None):
        self.pid = 4242
        self._returncode = returncode
        self._on_wait = on_wait

    def wait(self):
        if self._on_wait:
            self._on_wait()
        return self._returncode


def _stub_popen(monkeypatch, returncode=0, on_wait=None, calls=None):
    def popen(argv, **kwargs):
        if calls is not None:
            calls.append(argv)
        return FakeProc(returncode, on_wait)

    monkeypatch.setattr(tasks.subprocess, "Popen", popen)


def test_run_pipeline_success_lifecycle(saas_env, monkeypatch):
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        job = _make_job(session, project)
        job_id, project_dir = job.id, project.dir_path

    calls: list[list[str]] = []
    _stub_popen(monkeypatch, returncode=0, calls=calls)
    tasks.run_pipeline(job_id)

    with rdb.new_session() as session:
        job = session.get(rdb.Job, job_id)
        assert job.status == "succeeded"
        assert job.pid == 4242
        assert job.started_at is not None and job.finished_at is not None
    # The worker appends --slug and --projects-dir derived from the row.
    argv = calls[0]
    assert argv[argv.index("--slug") + 1] == "demo"
    assert argv[argv.index("--projects-dir") + 1] == str(saas_env.projects_dir / "u1")


def test_run_pipeline_failure_records_log_tail(saas_env, monkeypatch):
    # A make_video.py-kind job (the default "resume" here) auto-retries up
    # to MAX_RUN_ATTEMPTS — sleep is stubbed out so a 10-attempt failure
    # doesn't actually wait RUN_RETRY_DELAY_SEC between each one.
    monkeypatch.setattr(tasks.time, "sleep", lambda seconds: None)
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        job = _make_job(session, project)
        job_id = job.id
        log = saas_env.projects_dir / "u1" / "demo" / "logs" / "run.log"

    def write_log():
        log.write_text("boom: provider exploded")

    calls: list[list[str]] = []
    _stub_popen(monkeypatch, returncode=1, on_wait=write_log, calls=calls)
    tasks.run_pipeline(job_id)

    with rdb.new_session() as session:
        job = session.get(rdb.Job, job_id)
        assert job.status == "failed"
        assert "exited with code 1" in job.error
        assert "provider exploded" in job.error
        assert f"after {tasks.MAX_RUN_ATTEMPTS} attempts" in job.error
    assert len(calls) == tasks.MAX_RUN_ATTEMPTS


def test_run_pipeline_retries_and_recovers(saas_env, monkeypatch):
    # The exact scenario that motivated auto-retry: a transient failure
    # (e.g. the TLS-interception issue that killed 100% of a run's scenes)
    # should be recovered from automatically, not require a manual Resume.
    monkeypatch.setattr(tasks.time, "sleep", lambda seconds: None)
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        job = _make_job(session, project)
        job_id = job.id

    returncodes = iter([1, 1, 0])  # fails twice, succeeds on the 3rd attempt
    calls: list[list[str]] = []

    def popen(argv, **kwargs):
        calls.append(argv)
        return FakeProc(returncode=next(returncodes))

    monkeypatch.setattr(tasks.subprocess, "Popen", popen)
    tasks.run_pipeline(job_id)

    with rdb.new_session() as session:
        job = session.get(rdb.Job, job_id)
        assert job.status == "succeeded"
        assert job.error is None
    assert len(calls) == 3  # stopped retrying the moment it succeeded


def test_run_pipeline_does_not_retry_youtube_publish(saas_env, monkeypatch):
    # A one-shot, human-reviewed action with no resumability — unlike
    # make_video.py, retrying it blindly on failure isn't safe.
    monkeypatch.setattr(tasks.time, "sleep", lambda seconds: None)
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        job = _make_job(session, project, kind="youtube_publish")
        job_id = job.id

    calls: list[list[str]] = []
    _stub_popen(monkeypatch, returncode=1, calls=calls)
    tasks.run_pipeline(job_id)

    with rdb.new_session() as session:
        job = session.get(rdb.Job, job_id)
        assert job.status == "failed"
        assert "after 1 attempt" in job.error
    assert len(calls) == 1


def test_run_pipeline_skips_job_cancelled_while_queued(saas_env, monkeypatch):
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        job = _make_job(session, project, status="cancelled")
        job_id = job.id

    def never_called(*a, **k):
        raise AssertionError("Popen must not run for a cancelled job")

    monkeypatch.setattr(tasks.subprocess, "Popen", never_called)
    tasks.run_pipeline(job_id)

    with rdb.new_session() as session:
        assert session.get(rdb.Job, job_id).status == "cancelled"


def test_run_pipeline_keeps_cancellation_set_mid_run(saas_env, monkeypatch):
    """The API cancels a running job by marking the row and killing the pid;
    the worker's post-wait check must not overwrite that with failed."""
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        job = _make_job(session, project)
        job_id = job.id

    def cancel_from_api():
        with rdb.new_session() as other:
            row = other.get(rdb.Job, job_id)
            row.status = "cancelled"
            other.commit()

    # Killed process exits non-zero; status must stay cancelled, not failed.
    _stub_popen(monkeypatch, returncode=-15, on_wait=cancel_from_api)
    tasks.run_pipeline(job_id)

    with rdb.new_session() as session:
        job = session.get(rdb.Job, job_id)
        assert job.status == "cancelled"
        assert job.error is None


def test_run_pipeline_fails_fast_when_another_run_is_active(saas_env, monkeypatch):
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        _make_job(session, project, status="running")
        job = _make_job(session, project)
        job_id = job.id

    def never_called(*a, **k):
        raise AssertionError("Popen must not run while another job is active")

    monkeypatch.setattr(tasks.subprocess, "Popen", never_called)
    tasks.run_pipeline(job_id)

    with rdb.new_session() as session:
        job = session.get(rdb.Job, job_id)
        assert job.status == "failed"
        assert "already active" in job.error


def test_recover_orphaned_jobs_fails_dead_running_jobs(saas_env, monkeypatch):
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        dead = _make_job(session, project, status="running", pid=99999)
        done = _make_job(session, project, status="succeeded")
        dead_id, done_id = dead.id, done.id

    monkeypatch.setattr(tasks, "pid_is_pipeline", lambda pid: False)
    killed: list[int] = []
    monkeypatch.setattr(tasks, "kill_pipeline_pgid", lambda pid: killed.append(pid))
    tasks.recover_orphaned_jobs()

    with rdb.new_session() as session:
        assert session.get(rdb.Job, dead_id).status == "failed"
        assert "worker restarted" in session.get(rdb.Job, dead_id).error
        assert session.get(rdb.Job, done_id).status == "succeeded"
    assert killed == []  # dead pid — nothing to kill


def test_recover_orphaned_jobs_kills_surviving_subprocess(saas_env, monkeypatch):
    with rdb.new_session() as session:
        project = _make_project(saas_env, session)
        job = _make_job(session, project, status="running", pid=1234)
        job_id = job.id

    monkeypatch.setattr(tasks, "pid_is_pipeline", lambda pid: True)
    killed: list[int] = []
    monkeypatch.setattr(tasks, "kill_pipeline_pgid", lambda pid: killed.append(pid))
    tasks.recover_orphaned_jobs()

    with rdb.new_session() as session:
        assert session.get(rdb.Job, job_id).status == "failed"
    assert killed == [1234]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific process kill path")
def test_kill_pipeline_pgid_actually_kills_a_real_process_on_windows():
    # Regression: os.killpg/signal.SIGKILL don't exist on Windows at all
    # (AttributeError) — caught live 2026-09 when "Cancel run" / deleting
    # a running project 500'd instead of stopping it, leaving the
    # subprocess orphaned and running indefinitely. taskkill /F /T is the
    # actual fix; this spawns a real sleeping process to prove it works,
    # not just that the code path avoids the AttributeError.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(0.5)
        assert tasks.pid_is_pipeline(proc.pid) is True

        tasks.kill_pipeline_pgid(proc.pid)
        time.sleep(0.5)

        assert tasks.pid_is_pipeline(proc.pid) is False
        assert proc.poll() is not None  # the process actually exited
    finally:
        proc.kill()  # safety net if the assertion above ever fails
