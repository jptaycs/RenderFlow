"""make_video --thumbnail-only orchestration: regenerate the thumbnail
without ever making a valid final.mp4 look stale (or a stale one fresh)."""

from __future__ import annotations

import os
import time

import make_video
from renderflow.schema import AssetRef, AssetStatus, Scene, ScenePlan
from renderflow.storage import ProjectPaths, save_plan
from tests.stubs import StubImage


def _plan() -> ScenePlan:
    scene = Scene(
        id=1,
        type="narration",
        duration_estimate_sec=5.0,
        narration="Hello.",
        image_prompt="A photo.",
    )
    plan = ScenePlan(title="Demo", style="documentary", scenes=[scene])
    plan.thumbnail = AssetRef(status=AssetStatus.COMPLETED, path="old-thumb")
    return plan


def _fakes(monkeypatch, paths: ProjectPaths):
    seen = {}

    def fake_generate(plan, image, paths_, reaction_provider=None):
        # The real generate_thumbnail persists state as it goes — this
        # mtime bump on scenes.json is exactly what the freshness restore
        # exists to compensate for.
        save_plan(plan, paths_)
        (paths_.output / "thumbnail_src.png").write_bytes(b"new-src")

    def fake_render(plan, paths_, avatar_image=None):
        # The real render_thumbnail skips when thumbnail.jpg exists — the
        # helper must have deleted the old one before calling it.
        seen["old_thumb_gone"] = not (paths_.output / "thumbnail.jpg").exists()
        (paths_.output / "thumbnail.jpg").write_bytes(b"new-thumb")

    monkeypatch.setattr(make_video, "generate_thumbnail", fake_generate)
    monkeypatch.setattr(make_video, "render_thumbnail", fake_render)
    return seen


def test_fresh_final_stays_fresh(tmp_path, monkeypatch):
    paths = ProjectPaths.create(tmp_path, "demo")
    plan = _plan()
    save_plan(plan, paths)
    (paths.output / "thumbnail.jpg").write_bytes(b"old-thumb")
    final = paths.output / "final.mp4"
    final.write_bytes(b"video")
    now = time.time()
    os.utime(final, (now + 5, now + 5))  # fresh: newer than scenes.json

    seen = _fakes(monkeypatch, paths)
    assert make_video._regenerate_thumbnail(plan, paths, StubImage(), StubImage(), None) == 0

    assert (paths.output / "thumbnail.jpg").read_bytes() == b"new-thumb"
    assert seen["old_thumb_gone"]
    # The whole point: scenes.json was rewritten mid-run, but final.mp4
    # must still count as fresh so the download card doesn't vanish.
    assert final.stat().st_mtime >= paths.scenes_json.stat().st_mtime


def test_stale_final_stays_stale(tmp_path, monkeypatch):
    paths = ProjectPaths.create(tmp_path, "demo")
    plan = _plan()
    save_plan(plan, paths)
    final = paths.output / "final.mp4"
    final.write_bytes(b"video")
    now = time.time()
    os.utime(final, (now - 100, now - 100))  # stale: user edited scenes since
    os.utime(paths.scenes_json, (now, now))

    _fakes(monkeypatch, paths)
    assert make_video._regenerate_thumbnail(plan, paths, StubImage(), StubImage(), None) == 0

    # A genuinely outdated render must not be promoted to "fresh" by a
    # thumbnail-only run — the Resume/re-render signal has to survive.
    assert final.stat().st_mtime < paths.scenes_json.stat().st_mtime


def test_failed_generation_keeps_old_thumbnail_and_freshness(tmp_path, monkeypatch):
    paths = ProjectPaths.create(tmp_path, "demo")
    plan = _plan()
    save_plan(plan, paths)
    (paths.output / "thumbnail.jpg").write_bytes(b"old-thumb")
    final = paths.output / "final.mp4"
    final.write_bytes(b"video")
    now = time.time()
    os.utime(final, (now + 5, now + 5))

    def failing_generate(plan_, image, paths_, reaction_provider=None):
        save_plan(plan_, paths_)  # state was persisted before the failure
        raise RuntimeError("provider down")

    monkeypatch.setattr(make_video, "generate_thumbnail", failing_generate)

    import pytest

    with pytest.raises(RuntimeError):
        make_video._regenerate_thumbnail(plan, paths, StubImage(), StubImage(), None)

    # Old thumbnail survives a failed regenerate, and the still-valid
    # final.mp4 must not have been demoted to stale by the aborted run.
    assert (paths.output / "thumbnail.jpg").read_bytes() == b"old-thumb"
    assert final.stat().st_mtime >= paths.scenes_json.stat().st_mtime
