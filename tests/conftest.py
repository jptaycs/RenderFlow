"""Shared fixtures for the SaaS-layer tests (auth, jobs, API scoping).

DB tests run on in-memory SQLite (StaticPool = one shared connection) via
db.configure — the models deliberately use portable column types so no
Postgres/Docker is needed, per the repo rule that tests never hit live
services. Celery is never contacted either: run_pipeline is stubbed at the
api module boundary, and job tests call the task function directly.

Settings.load is monkeypatched wholesale rather than via env vars because
load_dotenv(override=True) makes the repo's .env clobber any monkeypatched
environment variables (the documented dotenv gotcha).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from renderflow import db as rdb
from renderflow.config import Settings


def make_settings(**overrides) -> Settings:
    """A fully-populated Settings for tests — pair with
    `monkeypatch.setattr(Settings, "load", classmethod(lambda cls: s))`.
    Patching Settings.load wholesale (instead of env vars) matters twice
    over: load_dotenv(override=True) clobbers monkeypatched env vars with
    .env values, and it does so lazily — mid-test, the first time any code
    path calls Settings.load()."""
    defaults = dict(
        llm_provider="claude",
        image_provider="pollinations",
        thumbnail_bg_provider="",
        thumbnail_reaction_provider="",
        tts_provider="kokoro",
        avatar_provider="ffmpeg-still",
        llm_model="test",
        tts_voice="test",
        tts_length_scale=1.4,
        tts_sentence_pause=0.45,
        avatar_image=None,
        projects_dir=Path("projects"),
        database_url="sqlite://",
        redis_url="redis://unused",
        secret_key="test-secret-key",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def saas_env(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    rdb.configure(engine)
    rdb.Base.metadata.create_all(engine)

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    settings = make_settings(projects_dir=projects_dir)
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    return SimpleNamespace(engine=engine, projects_dir=projects_dir, settings=settings)


class StubPipeline:
    """Stands in for tasks.run_pipeline at the API boundary — records what
    would have been queued instead of talking to a Celery broker."""

    def __init__(self) -> None:
        self.delayed: list[int] = []

    def delay(self, job_id: int):
        self.delayed.append(job_id)
        return SimpleNamespace(id=f"stub-task-{job_id}")


@pytest.fixture
def pipeline_stub(saas_env, monkeypatch) -> StubPipeline:
    from renderflow import api

    stub = StubPipeline()
    monkeypatch.setattr(api, "run_pipeline", stub)
    return stub


@pytest.fixture
def make_client(pipeline_stub):
    """Factory for API clients, each with its own cookie jar (its own user)."""
    from fastapi.testclient import TestClient

    from renderflow import api

    def _make() -> "TestClient":
        return TestClient(api.app)

    return _make


@pytest.fixture
def client(make_client):
    return make_client()


def register(client, email: str, password: str = "password123") -> dict:
    res = client.post("/api/auth/register", json={"email": email, "password": password})
    assert res.status_code == 201, res.text
    return res.json()
