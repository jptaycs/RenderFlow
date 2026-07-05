"""End-to-end skeleton test with stub providers (no network, no API keys)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from renderflow.pipeline.assets import (
    generate_avatar_clips,
    generate_images,
    generate_voice,
)
from renderflow.pipeline.script import generate_script
from renderflow.schema import AssetStatus
from renderflow.storage import ProjectPaths, load_plan, save_plan, slugify
from tests.stubs import StubAvatar, StubImage, StubLLM, StubTTS


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths.create(tmp_path, "test-video")


def test_slugify():
    assert slugify("The History of Amish Farming!") == "the-history-of-amish-farming"
    assert slugify("***") == "untitled"


def test_script_to_assets_end_to_end(paths: ProjectPaths):
    plan, result = generate_script(StubLLM(), "test topic", 1, "documentary")
    assert result.cost == 0.01
    assert len(plan.scenes) == 2
    save_plan(plan, paths)

    generate_images(plan, StubImage(), paths)
    generate_voice(plan, StubTTS(), "voice-id", paths)

    reloaded = load_plan(paths)
    for scene in reloaded.scenes:
        assert scene.assets.image.status is AssetStatus.COMPLETED
        assert scene.assets.voice.status is AssetStatus.COMPLETED
        assert Path(scene.assets.image.path).read_bytes() == b"fake-png"
        assert Path(scene.assets.voice.path).read_bytes() == b"fake-mp3"
        assert scene.assets.image.provider == "stub-image"
    assert reloaded.total_asset_cost() == pytest.approx((0.003 + 0.002) * 2)


def test_completed_assets_are_skipped(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    save_plan(plan, paths)
    generate_images(plan, StubImage(), paths)
    first_run = {s.id: s.assets.image.path for s in plan.scenes}
    # Second run must not regenerate (would raise InvalidTransition otherwise)
    generate_images(plan, StubImage(), paths)
    assert {s.id: s.assets.image.path for s in plan.scenes} == first_run


def test_talking_avatar_clip_generation(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "test topic", 1, "documentary")
    plan.scenes[0].type = "talking_avatar"
    save_plan(plan, paths)

    generate_images(plan, StubImage(), paths)
    generate_voice(plan, StubTTS(), "voice-id", paths)
    generate_avatar_clips(plan, StubAvatar(), paths)

    avatar_scene = load_plan(paths).scenes[0]
    narration_scene = load_plan(paths).scenes[1]
    assert avatar_scene.assets.avatar_clip.status is AssetStatus.COMPLETED
    assert Path(avatar_scene.assets.avatar_clip.path).read_bytes() == b"fake-mp4"
    assert avatar_scene.assets.avatar_clip.provider == "stub-avatar"
    assert narration_scene.assets.avatar_clip.status is AssetStatus.PENDING
    assert load_plan(paths).total_asset_cost() == pytest.approx(
        (0.003 + 0.002) * 2 + 0.004
    )


HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_integration(paths: ProjectPaths):
    from renderflow.pipeline.render import probe_duration, render_video

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")

    for scene in plan.scenes:
        img = paths.images / f"scene_{scene.id:03d}.png"
        audio = paths.voice / f"scene_{scene.id:03d}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=steelblue:s=640x360",
             "-frames:v", "1", str(img)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440",
             "-t", "1", str(audio)],
            check=True, capture_output=True,
        )
        for ref, path in ((scene.assets.image, img), (scene.assets.voice, audio)):
            ref.advance(AssetStatus.RUNNING)
            ref.path = str(path)
            ref.advance(AssetStatus.COMPLETED)

    final = render_video(plan, paths)
    assert final.exists()
    # Two 1-second scenes → roughly 2 seconds of video
    assert probe_duration(final) == pytest.approx(2.0, abs=0.5)


def test_split_script_preserves_flow(paths: ProjectPaths):
    from renderflow.pipeline.script import split_script

    plan, result = split_script(StubLLM(), "Some client script text.", "documentary")
    assert len(plan.scenes) == 2
    assert plan.style == "documentary"
    assert result.cost == 0.01
