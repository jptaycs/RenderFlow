"""Stock-video B-roll: generation eligibility, resume, optionality."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from renderflow.providers.base import GeneratedAsset, LLMResult
from renderflow.pipeline.assets import generate_broll
from renderflow.schema import AssetStatus, AvatarSpec, Scene, ScenePlan
from renderflow.storage import ProjectPaths, load_plan
from tests.stubs import StubVideo

AVATAR = AvatarSpec(name="Host", description="a documentary host")


class FakeRewriteLLM:
    """Minimal LLMProvider fake for the video-prompt-rewrite tests — real
    StubLLM (tests/stubs.py) always returns a canned scene-plan JSON blob,
    which isn't a useful "rewritten prompt" string to assert against."""

    name = "fake-rewrite-llm"

    def __init__(self, text: str = "REWRITTEN: camera slowly pans.", cost: float = 0.002, fail: bool = False):
        self.text = text
        self.cost = cost
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str, **params: Any) -> LLMResult:
        self.calls.append((system, prompt))
        if self.fail:
            raise RuntimeError("fake LLM failure")
        return LLMResult(text=self.text, provider=self.name, cost=self.cost)


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


class _ConcurrencyTrackingVideo:
    """Records how many find_clip calls were ever in flight at once, and
    briefly sleeps inside the call so overlapping calls actually have a
    chance to overlap — proves generate_broll's ThreadPoolExecutor really
    runs scenes concurrently rather than only appearing to (e.g. a bug
    that accidentally held a lock around the whole call, not just save)."""

    name = "concurrency-tracking-video"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def find_clip(self, prompt: str, min_duration_sec: float, **params: Any):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return GeneratedAsset(
            data=b"fake-broll-mp4", provider=self.name, cost=0.001,
        )


def test_broll_runs_scenes_concurrently_and_writes_valid_state(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path, "demo")
    scenes = [_scene(i) for i in range(1, 7)]
    plan = _plan(*scenes)
    provider = _ConcurrencyTrackingVideo()

    generate_broll(plan, provider, paths, max_concurrency=3)

    assert provider.calls == 6
    # Real overlap, not accidental full serialization behind a lock.
    assert provider.max_active >= 2

    for scene in plan.scenes:
        assert scene.assets.broll.status is AssetStatus.COMPLETED
        assert Path(scene.assets.broll.path).exists()

    # Concurrent save_plan calls (guarded by generate_broll's own lock)
    # must never leave scenes.json truncated or otherwise corrupted.
    reloaded = load_plan(paths)
    assert len(reloaded.scenes) == 6
    assert all(s.assets.broll.status is AssetStatus.COMPLETED for s in reloaded.scenes)


def test_broll_concurrency_one_matches_old_sequential_behavior(tmp_path: Path):
    # max_concurrency=1 is the escape hatch back to the pre-2026-09
    # one-at-a-time behavior — must still fully work.
    paths = ProjectPaths.create(tmp_path, "demo")
    scenes = [_scene(i) for i in range(1, 4)]
    plan = _plan(*scenes)
    provider = StubVideo()

    generate_broll(plan, provider, paths, max_concurrency=1)

    assert len(provider.calls) == 3
    assert all(s.assets.broll.status is AssetStatus.COMPLETED for s in plan.scenes)


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


def test_broll_rewrites_prompt_for_video_when_llm_given(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path, "demo")
    scene = _scene(1)
    provider = StubVideo()
    llm = FakeRewriteLLM(text="REWRITTEN: slow dolly-in over the reef.")

    generate_broll(_plan(scene), provider, paths, llm=llm)

    assert provider.calls == ["REWRITTEN: slow dolly-in over the reef."]
    assert len(llm.calls) == 1
    # combined cost: the (free) stub video cost + the real rewrite cost.
    assert scene.assets.broll.cost == 0.002


def test_broll_falls_back_to_original_prompt_when_rewrite_fails(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path, "demo")
    scene = _scene(1)
    provider = StubVideo()
    llm = FakeRewriteLLM(fail=True)

    generate_broll(_plan(scene), provider, paths, llm=llm)  # must not raise

    assert provider.calls == [scene.image_prompt]
    assert scene.assets.broll.status is AssetStatus.COMPLETED


def test_broll_without_llm_uses_original_prompt_unchanged(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path, "demo")
    scene = _scene(1)
    provider = StubVideo()

    generate_broll(_plan(scene), provider, paths)  # llm=None, the default

    assert provider.calls == [scene.image_prompt]


def test_registry_returns_none_when_disabled():
    from renderflow.providers import build_broll
    from tests.conftest import make_settings

    assert build_broll(make_settings(broll_provider="")) is None


def test_registry_name_override_wins_over_configured_default(monkeypatch):
    # settings.shorts_broll_provider (Shorts can run a faster/cheaper
    # provider than landscape) is threaded through as this override.
    # PexelsVideo's constructor itself needs PEXELS_API_KEY present (no
    # network call at construction time, just a presence check) — not set
    # in this test environment, so provide a fake one.
    monkeypatch.setenv("PEXELS_API_KEY", "fake-test-key")
    from renderflow.providers import build_broll
    from tests.conftest import make_settings

    settings = make_settings(broll_provider="labs69")
    provider = build_broll(settings, "pexels-video")
    assert provider.name == "pexels-video"


def test_registry_override_of_empty_string_falls_back_to_configured_default(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "fake-test-key")
    from renderflow.providers import build_broll
    from tests.conftest import make_settings

    settings = make_settings(broll_provider="labs69")
    provider = build_broll(settings, None)
    assert provider.name == "labs69"


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
