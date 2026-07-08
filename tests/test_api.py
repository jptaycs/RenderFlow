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
    # id 2 is split-layout (scene_is_avatar_solo cycles 1, 4, 7, ...)
    assets = _scene_assets(_scene(2, "talking_avatar"))
    assert assets["image"] == AssetStatus.PENDING.value
    assert assets["avatar"] == AssetStatus.PENDING.value


def test_scene_assets_omits_image_for_solo_avatar_scenes():
    # id 1 is solo-layout — it never gets a background image, so the chip
    # must not appear at all (it used to show a permanently-"pending" chip).
    assets = _scene_assets(_scene(1, "talking_avatar"))
    assert "image" not in assets
    assert assets == {
        "voice": AssetStatus.PENDING.value,
        "avatar": AssetStatus.PENDING.value,
    }


def test_scene_assets_respects_manual_solo_override():
    # id 2 would be split under the default cycle — forcing "solo" must
    # drop the image chip just like a naturally-solo scene would.
    assets = _scene_assets(_scene(2, "talking_avatar", avatar_layout="solo"))
    assert "image" not in assets


def test_scene_assets_respects_manual_split_override():
    # id 1 would be solo under the default cycle — forcing "split" must
    # bring the image chip back.
    assets = _scene_assets(_scene(1, "talking_avatar", avatar_layout="split"))
    assert "image" in assets


def test_file_url_includes_cache_busting_mtime(tmp_path, monkeypatch):
    """Regression: scene/thumbnail/video filenames are deterministic
    (scene_002.png, thumbnail.jpg, final.mp4) — regenerating overwrites the
    same path, so without a cache-busting query param the URL is
    byte-identical to before and the browser keeps showing the stale
    cached image, making "regenerate scene" look like it did nothing."""
    import os
    import time

    from renderflow import api

    monkeypatch.setattr(api, "_projects_dir", lambda: tmp_path)
    slug = "demo"
    img = tmp_path / slug / "images" / "scene_001.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"one")

    url1 = api._file_url(slug, img)
    assert url1 == f"/files/{slug}/images/scene_001.png?v={int(img.stat().st_mtime)}"

    # Simulate a regenerate: same path, new content, new mtime.
    future = time.time() + 100
    os.utime(img, (future, future))
    url2 = api._file_url(slug, img)
    assert url2 != url1


def test_file_url_returns_none_for_missing_path(tmp_path, monkeypatch):
    from renderflow import api

    monkeypatch.setattr(api, "_projects_dir", lambda: tmp_path)
    assert api._file_url("demo", None) is None


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


def test_project_view_computes_profit_and_production_time(tmp_path, monkeypatch):
    import os
    import time

    import pytest

    from renderflow import api
    from renderflow.schema import (
        AssetRef,
        AssetStatus,
        ProjectPerformance,
        Scene,
        SceneAssets,
        ScenePlan,
    )
    from renderflow.storage import ProjectPaths, save_performance, save_plan

    monkeypatch.setattr(api, "_projects_dir", lambda: tmp_path)
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

    view = api._project_view(slug, plan, paths)
    assert view["status"] == "Complete"
    assert view["productionTimeSec"] == pytest.approx(completed_at - created_at)
    assert view["profit"] == pytest.approx(25.0 - view["cost"])
