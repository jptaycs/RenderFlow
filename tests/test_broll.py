"""Stock-video B-roll: generation eligibility, resume, optionality."""

from __future__ import annotations

from pathlib import Path

from renderflow.pipeline.assets import generate_broll
from renderflow.schema import AssetStatus, AvatarSpec, Scene, ScenePlan
from renderflow.storage import ProjectPaths
from tests.stubs import StubVideo

AVATAR = AvatarSpec(name="Host", description="a documentary host")


def _plan(*scenes: Scene) -> ScenePlan:
    return ScenePlan(title="T", style="documentary", scenes=list(scenes))


def _scene(scene_id: int, scene_type: str = "narration", **kwargs) -> Scene:
    return Scene(
        id=scene_id,
        type=scene_type,
        duration_estimate_sec=5.0,
        narration="Some narration.",
        image_prompt="A photo.",
        avatar=AVATAR if scene_type == "talking_avatar" else None,
        **kwargs,
    )


def test_broll_fetches_for_eligible_scenes(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path, "demo")
    narration = _scene(1)
    visual_only = _scene(2, "talking_avatar", avatar_layout="visual")
    split = _scene(3, "talking_avatar")  # split-screen — not eligible (v1)
    plan = _plan(narration, visual_only, split)

    provider = StubVideo()
    generate_broll(plan, provider, paths)

    assert narration.assets.broll.status is AssetStatus.COMPLETED
    assert Path(narration.assets.broll.path).read_bytes() == b"fake-broll-mp4"
    assert visual_only.assets.broll.status is AssetStatus.COMPLETED
    assert split.assets.broll.status is AssetStatus.PENDING
    assert narration.assets.broll.cost == 0.0


def test_broll_respects_per_scene_off(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path, "demo")
    scene = _scene(1, broll_mode="off")
    provider = StubVideo()
    generate_broll(_plan(scene), provider, paths)
    assert scene.assets.broll.status is AssetStatus.PENDING
    assert provider.calls == []


def test_broll_skips_completed_on_resume(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path, "demo")
    scene = _scene(1)
    provider = StubVideo()
    generate_broll(_plan(scene), provider, paths)
    assert len(provider.calls) == 1
    generate_broll(_plan(scene), provider, paths)  # resume: no second fetch
    assert len(provider.calls) == 1


def test_broll_failure_marks_failed_and_continues(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path, "demo")
    first = _scene(1)
    second = _scene(2)
    failing = StubVideo(fail=True)
    generate_broll(_plan(first, second), failing, paths)  # must not raise
    assert first.assets.broll.status is AssetStatus.FAILED
    assert second.assets.broll.status is AssetStatus.FAILED
    assert len(failing.calls) == 2


def test_registry_returns_none_when_disabled():
    from renderflow.providers import build_broll
    from tests.conftest import make_settings

    assert build_broll(make_settings(broll_provider="")) is None


def test_old_scenes_json_loads_with_broll_defaults():
    """Backward compat: plans persisted before the broll/music fields
    existed must validate with the new defaults, no migration."""
    legacy = {
        "title": "Old",
        "style": "documentary",
        "scenes": [
            {
                "id": 1,
                "type": "narration",
                "duration_estimate_sec": 5.0,
                "narration": "Hi.",
                "image_prompt": "A photo.",
            }
        ],
    }
    plan = ScenePlan.model_validate(legacy)
    assert plan.music_track is None
    assert plan.scenes[0].broll_mode == "auto"
    assert plan.scenes[0].assets.broll.status is AssetStatus.PENDING
