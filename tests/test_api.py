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
