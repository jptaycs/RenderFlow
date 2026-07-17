"""Dashboard API pure-function tests — no server/HTTP needed for these."""

from __future__ import annotations

from renderflow.api import _scene_assets
from renderflow.schema import AssetStatus, AvatarSpec, Scene

AVATAR = AvatarSpec(name="Host", description="a documentary host")


def _scene(scene_id: int, scene_type: str = "narration", avatar_layout: str = "auto") -> Scene:
    return Scene(
        id=scene_id,
        type=scene_type,
        duration_estimate_sec=5.0,
        narration="Some narration.",
        image_prompt="A photo.",
        avatar=AVATAR if scene_type == "talking_avatar" else None,
        avatar_layout=avatar_layout,
    )


def test_scene_assets_includes_image_for_narration_scenes():
    assert _scene_assets(_scene(2, "narration")) == {
        "image": AssetStatus.PENDING.value,
        "voice": AssetStatus.PENDING.value,
    }


def test_scene_assets_includes_image_for_split_avatar_scenes():
    # "auto" always means split-screen now — every talking-avatar scene
    # gets a background image unless manually overridden to "solo".
    assets = _scene_assets(_scene(2, "talking_avatar"))
    assert assets["image"] == AssetStatus.PENDING.value
    assert assets["avatar"] == AssetStatus.PENDING.value


def test_scene_assets_omits_image_for_solo_avatar_scenes():
    # Solo is opt-in — it never gets a background image, so the chip must
    # not appear at all (it used to show a permanently-"pending" chip).
    assets = _scene_assets(_scene(1, "talking_avatar", avatar_layout="solo"))
    assert "image" not in assets
    assert assets == {
        "voice": AssetStatus.PENDING.value,
        "avatar": AssetStatus.PENDING.value,
    }


def test_scene_assets_respects_manual_solo_override():
    # "auto" would be split by default — forcing "solo" must drop the image
    # chip.
    assets = _scene_assets(_scene(2, "talking_avatar", avatar_layout="solo"))
    assert "image" not in assets


def test_scene_assets_respects_manual_split_override():
    # Explicit "split" behaves the same as the "auto" default.
    assets = _scene_assets(_scene(1, "talking_avatar", avatar_layout="split"))
    assert "image" in assets


def test_scene_assets_omits_avatar_for_visual_only_scenes():
    # Visual-only shows the background image (like split) but never an
    # avatar chip — no lip-synced clip is ever generated for it.
    assets = _scene_assets(_scene(1, "talking_avatar", avatar_layout="visual"))
    assert assets == {
        "image": AssetStatus.PENDING.value,
        "voice": AssetStatus.PENDING.value,
    }


def test_file_url_includes_cache_busting_mtime(tmp_path):
    """Regression: scene/thumbnail/video filenames are deterministic
    (scene_002.png, thumbnail.jpg, final.mp4) — regenerating overwrites the
    same path, so without a cache-busting query param the URL is
    byte-identical to before and the browser keeps showing the stale
    cached image, making "regenerate scene" look like it did nothing."""
    import os
    import time

    from renderflow import api
    from renderflow.storage import ProjectPaths

    slug = "demo"
    paths = ProjectPaths(root=tmp_path / slug)
    img = tmp_path / slug / "images" / "scene_001.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"one")

    url1 = api._file_url(paths, slug, img)
    assert url1 == f"/files/{slug}/images/scene_001.png?v={int(img.stat().st_mtime)}"

    # Simulate a regenerate: same path, new content, new mtime.
    future = time.time() + 100
    os.utime(img, (future, future))
    url2 = api._file_url(paths, slug, img)
    assert url2 != url1


def test_file_url_returns_none_for_missing_path(tmp_path):
    from renderflow import api
    from renderflow.storage import ProjectPaths

    assert api._file_url(ProjectPaths(root=tmp_path / "demo"), "demo", None) is None


def test_vary_prompt_appends_a_variation_clause():
    from renderflow.api import _REGENERATE_VARIATIONS, _vary_prompt

    varied = _vary_prompt("A photo of a barn.")
    assert varied.startswith("A photo of a barn.")
    assert any(v in varied for v in _REGENERATE_VARIATIONS)


def test_vary_prompt_does_not_grow_unbounded_across_repeated_calls():
    from renderflow.api import _vary_prompt

    base = "A photo of a barn."
    once = _vary_prompt(base)
    twice = _vary_prompt(once)
    thrice = _vary_prompt(twice)
    # Every call strips the previous variation clause before adding a new
    # one, so the prompt never grows past one clause no matter how many
    # times "Regenerate" is clicked in a row.
    assert twice.count("Try this take:") == 1
    assert thrice.count("Try this take:") == 1
    assert thrice.startswith(base)


def test_load_performance_returns_default_when_missing(tmp_path):
    from renderflow.schema import ProjectPerformance
    from renderflow.storage import ProjectPaths, load_performance

    paths = ProjectPaths.create(tmp_path, "demo")
    perf = load_performance(paths)
    assert perf == ProjectPerformance()


def test_save_and_load_performance_roundtrip(tmp_path):
    from renderflow.schema import ProjectPerformance
    from renderflow.storage import ProjectPaths, load_performance, save_performance

    paths = ProjectPaths.create(tmp_path, "demo")
    save_performance(
        ProjectPerformance(views=1000, watch_time_minutes=42.5, revenue_usd=12.34, notes="ok"),
        paths,
    )
    reloaded = load_performance(paths)
    assert reloaded.views == 1000
    assert reloaded.revenue_usd == 12.34
    assert reloaded.notes == "ok"


def test_project_view_computes_profit_and_production_time(tmp_path):
    import os
    import time

    import pytest

    from renderflow import api
    from renderflow.db import Project
    from renderflow.schema import (
        AssetRef,
        AssetStatus,
        ProjectPerformance,
        Scene,
        SceneAssets,
        ScenePlan,
    )
    from renderflow.storage import ProjectPaths, save_performance, save_plan

    slug = "demo-project"
    paths = ProjectPaths.create(tmp_path, slug)

    scene = Scene(
        id=1,
        type="narration",
        duration_estimate_sec=5.0,
        narration="Hello.",
        image_prompt="A photo.",
        assets=SceneAssets(
            image=AssetRef(status=AssetStatus.COMPLETED, path=str(paths.images / "scene_001.png"), cost=0.0),
            voice=AssetRef(status=AssetStatus.COMPLETED, path=str(paths.voice / "scene_001.wav"), cost=0.0),
        ),
    )
    plan = ScenePlan(title="Demo", style="documentary", scenes=[scene])
    save_plan(plan, paths)

    final = paths.output / "final.mp4"
    final.write_bytes(b"data")
    now = time.time() + 5  # ensure final.mp4 is newer than scenes.json
    os.utime(final, (now, now))

    created_at = now - 120
    completed_at = now
    save_performance(
        ProjectPerformance(created_at=created_at, completed_at=completed_at, revenue_usd=25.0),
        paths,
    )

    project = Project(
        owner_id=1, slug=slug, title="Demo", dir_path=str(paths.root), created_at=created_at
    )
    view = api._project_view(project, plan, paths, job=None)
    assert view["status"] == "Complete"
    assert view["productionTimeSec"] == pytest.approx(completed_at - created_at)
    assert view["profit"] == pytest.approx(25.0 - view["cost"])


"""Endpoint tests: ownership scoping, the job queue, quotas, file serving.

These use the fixtures in conftest.py — in-memory SQLite and a stubbed
run_pipeline; no Postgres, Redis, or subprocess is ever touched."""

from tests.conftest import register


def _create_project(client, title="My First Video"):
    res = client.post(
        "/api/projects", json={"title": title, "script": "One sentence of narration."}
    )
    assert res.status_code == 201, res.text
    return res.json()["slug"]


def test_create_project_enqueues_a_job(client, pipeline_stub, saas_env):
    register(client, "admin@example.com")
    slug = _create_project(client)

    from renderflow import db as rdb

    with rdb.new_session() as session:
        jobs = session.query(rdb.Job).all()
        assert len(jobs) == 1
        assert jobs[0].kind == "create"
        assert jobs[0].status == "queued"
        assert "--skip-render" in jobs[0].argv
        assert pipeline_stub.delayed == [jobs[0].id]
        project = session.query(rdb.Project).one()
    # New projects live in the per-user namespace.
    assert f"u{1}/{slug}" in project.dir_path
    assert (saas_env.projects_dir / "u1" / slug / "script" / "source.txt").exists()


def test_project_with_active_job_rejects_further_runs(client):
    register(client, "admin@example.com")
    slug = _create_project(client)
    # The stubbed worker never runs, so the create job stays queued = active.
    assert client.post(f"/api/projects/{slug}/resume").status_code == 409


def test_new_project_appears_as_placeholder_before_scenes_exist(client):
    register(client, "admin@example.com")
    slug = _create_project(client)
    projects = client.get("/api/state").json()["projects"]
    assert [p["slug"] for p in projects] == [slug]
    assert projects[0]["status"] == "Generating"
    assert projects[0]["running"] is True
    assert projects[0]["scenes"] == []


def test_projects_are_isolated_between_users(make_client, saas_env):
    alice = make_client()
    register(alice, "alice@example.com")
    slug = _create_project(alice)
    asset = saas_env.projects_dir / "u1" / slug / "images" / "scene_001.png"
    asset.write_bytes(b"pixels")

    bob = make_client()
    register(bob, "bob@example.com")
    # Same 404 as a nonexistent slug — no existence leak.
    assert bob.get("/api/state").json()["projects"] == []
    assert bob.post(f"/api/projects/{slug}/resume").status_code == 404
    assert bob.post(f"/api/projects/{slug}/scenes/1/regenerate").status_code == 404
    assert bob.get(f"/files/{slug}/images/scene_001.png").status_code == 404
    # The owner can still fetch their own file.
    assert alice.get(f"/files/{slug}/images/scene_001.png").status_code == 200


def test_files_route_rejects_traversal(client, saas_env):
    register(client, "admin@example.com")
    slug = _create_project(client)
    (saas_env.projects_dir / "secret.txt").write_text("top secret")
    res = client.get(f"/files/{slug}/%2e%2e/%2e%2e/secret.txt")
    assert res.status_code == 404


def test_trial_credits_gate_project_creation(make_client):
    register(make_client(), "admin@example.com")  # admin: exempt
    free = make_client()
    register(free, "free@example.com")

    from renderflow.config import TRIAL_CREDITS

    for i in range(TRIAL_CREDITS):
        _create_project(free, title=f"Video {i}")
    res = free.post(
        "/api/projects", json={"title": "One Too Many", "script": "text"}
    )
    assert res.status_code == 402
    assert "trial" in res.json()["detail"]


def test_admin_is_exempt_from_paywall(client):
    register(client, "admin@example.com")

    from renderflow.config import TRIAL_CREDITS

    for i in range(TRIAL_CREDITS + 1):
        _create_project(client, title=f"Video {i}")  # no 402


def _finish_create_with_plan(saas_env, slug: str, owner_dir: str = "u1"):
    """Simulate the worker having produced a one-scene plan + final.mp4."""
    from renderflow import db as rdb
    from renderflow.schema import Scene, ScenePlan
    from renderflow.storage import ProjectPaths, save_plan

    with rdb.new_session() as session:
        job = session.query(rdb.Job).order_by(rdb.Job.id.desc()).first()
        job.status = "succeeded"
        session.commit()
    paths = ProjectPaths.create(saas_env.projects_dir / owner_dir, slug)
    plan = ScenePlan(
        title="My First Video",
        style="documentary",
        scenes=[
            Scene(
                id=1,
                type="narration",
                duration_estimate_sec=5.0,
                narration="Hello.",
                image_prompt="A photo.",
            )
        ],
    )
    save_plan(plan, paths)
    (paths.output / "final.mp4").write_bytes(b"video")
    return paths


def test_scene_broll_toggle(client, saas_env, pipeline_stub):
    register(client, "admin@example.com")
    slug = _create_project(client)
    paths = _finish_create_with_plan(saas_env, slug)

    from renderflow.storage import load_plan

    res = client.post(f"/api/projects/{slug}/scenes/1/broll", json={"mode": "off"})
    assert res.status_code == 200, res.text
    assert load_plan(paths).scenes[0].broll_mode == "off"
    # Changing the visual invalidates the stale final render.
    assert not (paths.output / "final.mp4").exists()
    # "off" never queues a generation run (nothing to fetch).
    assert len(pipeline_stub.delayed) == 1  # only the original create job

    assert (
        client.post(f"/api/projects/{slug}/scenes/1/broll", json={"mode": "sideways"})
        .status_code
        == 422
    )


def test_scene_broll_toggle_is_owner_scoped(make_client, saas_env, pipeline_stub):
    alice = make_client()
    register(alice, "alice@example.com")
    slug = _create_project(alice)
    _finish_create_with_plan(saas_env, slug)

    bob = make_client()
    register(bob, "bob@example.com")
    res = bob.post(f"/api/projects/{slug}/scenes/1/broll", json={"mode": "off"})
    assert res.status_code == 404


def test_cancel_without_active_job_is_409(client, monkeypatch):
    register(client, "admin@example.com")
    slug = _create_project(client)

    from renderflow import api
    from renderflow import db as rdb

    # Simulate the worker having finished the create job.
    with rdb.new_session() as session:
        job = session.query(rdb.Job).one()
        job.status = "succeeded"
        session.commit()
    assert client.post(f"/api/projects/{slug}/cancel").status_code == 409


def test_cancel_marks_active_job_cancelled(client, monkeypatch):
    register(client, "admin@example.com")
    slug = _create_project(client)

    from renderflow import api

    cancelled = []
    monkeypatch.setattr(
        api, "cancel_job", lambda session, job: cancelled.append(job.id)
    )
    assert client.post(f"/api/projects/{slug}/cancel").status_code == 200
    assert len(cancelled) == 1


def test_delete_project_removes_rows_and_files(client, saas_env, monkeypatch):
    register(client, "admin@example.com")
    slug = _create_project(client)

    from renderflow import api
    from renderflow import db as rdb

    # Avoid the Celery revoke call inside cancel_job (no broker in tests).
    def _fake_cancel(session, job):
        job.status = "cancelled"
        session.commit()

    monkeypatch.setattr(api, "cancel_job", _fake_cancel)
    assert client.delete(f"/api/projects/{slug}").status_code == 200
    assert not (saas_env.projects_dir / "u1" / slug).exists()
    with rdb.new_session() as session:
        assert session.query(rdb.Project).count() == 0
        assert session.query(rdb.Job).count() == 0
