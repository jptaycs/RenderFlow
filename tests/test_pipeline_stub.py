"""End-to-end skeleton test with stub providers (no network, no API keys)."""

import json
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


def test_resume_retries_failed_assets(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    save_plan(plan, paths)
    failed = plan.scenes[0].assets.image
    failed.advance(AssetStatus.RUNNING)
    failed.advance(AssetStatus.FAILED)
    # A re-run must route failed → retrying → running, not raise InvalidTransition
    generate_images(plan, StubImage(), paths)
    assert failed.status is AssetStatus.COMPLETED
    assert Path(failed.path).read_bytes() == b"fake-png"


def test_resume_recovers_assets_orphaned_in_running_state(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    save_plan(plan, paths)
    orphaned = plan.scenes[0].assets.image
    orphaned.advance(AssetStatus.RUNNING)  # run crashed/killed mid-asset
    generate_images(plan, StubImage(), paths)
    assert orphaned.status is AssetStatus.COMPLETED
    assert Path(orphaned.path).read_bytes() == b"fake-png"


def test_talking_avatar_clip_generation(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "test topic", 1, "documentary")
    plan.scenes[0].type = "talking_avatar"
    save_plan(plan, paths)

    generate_images(plan, StubImage(), paths)
    generate_voice(plan, StubTTS(), "voice-id", paths)
    generate_avatar_clips(plan, StubAvatar(), paths)

    avatar_scene = load_plan(paths).scenes[0]
    narration_scene = load_plan(paths).scenes[1]
    assert avatar_scene.assets.avatar_image.status is AssetStatus.COMPLETED
    assert Path(avatar_scene.assets.avatar_image.path).read_bytes() == b"fake-png"
    assert avatar_scene.assets.avatar_clip.status is AssetStatus.COMPLETED
    assert Path(avatar_scene.assets.avatar_clip.path).read_bytes() == b"fake-mp4"
    assert avatar_scene.assets.avatar_clip.provider == "stub-avatar"
    assert narration_scene.assets.avatar_clip.status is AssetStatus.PENDING
    assert load_plan(paths).total_asset_cost() == pytest.approx(
        (0.003 + 0.002) * 2 + 0.003 + 0.004
    )


def test_talking_avatar_uses_local_avatar_image(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "test topic", 1, "documentary")
    plan.scenes[0].type = "talking_avatar"
    avatar_image = paths.root / "host.jpg"
    avatar_image.write_bytes(b"local-host")
    save_plan(plan, paths)

    generate_images(plan, StubImage(), paths, avatar_image=avatar_image)
    generate_voice(plan, StubTTS(), "voice-id", paths, length_scale=1.25)

    avatar_scene = load_plan(paths).scenes[0]
    assert avatar_scene.assets.avatar_image.provider == "local-file"
    assert Path(avatar_scene.assets.avatar_image.path).read_bytes() == b"local-host"
    assert Path(avatar_scene.assets.voice.path).read_bytes() == b"fake-mp3"


HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_integration(paths: ProjectPaths, monkeypatch):
    from renderflow.pipeline.render import probe_duration, render_video

    # Keep the integration test hermetic: zoompan needs no depth model.
    monkeypatch.setenv("RENDERFLOW_MOTION", "zoompan")
    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    plan.scenes[1].type = "talking_avatar"

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
        if scene.type == "talking_avatar":
            avatar_clip = paths.avatar / f"scene_{scene.id:03d}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=darkred:s=640x360",
                    "-f", "lavfi", "-i", "sine=frequency=220",
                    "-t", "1", str(avatar_clip),
                ],
                check=True, capture_output=True,
            )
            scene.assets.avatar_clip.advance(AssetStatus.RUNNING)
            scene.assets.avatar_clip.path = str(avatar_clip)
            scene.assets.avatar_clip.advance(AssetStatus.COMPLETED)

    final = render_video(plan, paths)
    assert final.exists()
    # Two 1-second scenes → roughly 2 seconds of video
    assert probe_duration(final) == pytest.approx(2.0, abs=0.5)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(final),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {s["codec_type"] for s in json.loads(probe.stdout)["streams"]}
    assert stream_types == {"audio", "video"}


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_parallax_clip_renders_with_synthetic_depth(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pytest.importorskip("transformers")
    from renderflow.pipeline import parallax
    from renderflow.pipeline.render import probe_duration

    img = tmp_path / "img.png"
    arr = np.zeros((360, 640, 3), np.uint8)
    arr[:180], arr[180:] = 200, 60
    cv2.imwrite(str(img), arr)
    # Synthetic depth gradient — no model download in tests.
    monkeypatch.setattr(
        parallax, "_depth",
        lambda image: np.tile(
            np.linspace(0.0, 1.0, image.shape[0], dtype=np.float32)[:, None],
            (1, image.shape[1]),
        ),
    )

    out = tmp_path / "clip.mp4"
    assert parallax.render_parallax_clip(img, out, 0.5, 320, 180, "pan_right", 0.08)
    assert probe_duration(out) == pytest.approx(0.5, abs=0.2)


def test_generate_thumbnail_is_tracked_and_resumable(paths: ProjectPaths):
    from renderflow.pipeline.assets import generate_thumbnail

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    save_plan(plan, paths)
    generate_thumbnail(plan, StubImage(), paths)

    assert plan.thumbnail.status is AssetStatus.COMPLETED
    assert Path(plan.thumbnail.path).read_bytes() == b"fake-png"
    assert plan.thumbnail.cost == 0.003
    assert plan.total_asset_cost() == pytest.approx(0.003)  # images not yet run
    first_path = plan.thumbnail.path
    generate_thumbnail(plan, StubImage(), paths)  # resume skips
    assert plan.thumbnail.path == first_path


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_thumbnail(paths: ProjectPaths):
    from renderflow.pipeline.render import render_thumbnail

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    assert render_thumbnail(plan, paths) is None  # no images yet

    img = paths.images / "scene_001.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=steelblue:s=640x360",
         "-frames:v", "1", str(img)],
        check=True, capture_output=True,
    )
    scene = plan.scenes[0]
    scene.assets.image.advance(AssetStatus.RUNNING)
    scene.assets.image.path = str(img)
    scene.assets.image.advance(AssetStatus.COMPLETED)

    thumb = render_thumbnail(plan, paths)
    assert thumb == paths.output / "thumbnail.jpg" and thumb.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(thumb)],
        check=True, capture_output=True, text=True,
    )
    assert probe.stdout.strip() == "1280,720"
    # Resume: second call is a no-op that returns the existing file
    mtime = thumb.stat().st_mtime_ns
    assert render_thumbnail(plan, paths) == thumb
    assert thumb.stat().st_mtime_ns == mtime


def test_split_script_preserves_flow(paths: ProjectPaths):
    from renderflow.pipeline.script import split_script

    plan, result = split_script(StubLLM(), "Some client script text.", "documentary")
    assert len(plan.scenes) == 2
    assert plan.style == "documentary"
    assert result.cost == 0.01


def test_local_split_script_preserves_text_without_llm():
    from renderflow.pipeline.script import split_script_local

    script = (
        "First, this exact sentence should stay intact. "
        "Then this second sentence should follow it. "
        "Finally, this closing sentence should remain unchanged."
    )

    plan, result = split_script_local(script, "documentary")

    assert result.provider == "local-script-splitter"
    assert result.cost == 0.0
    assert plan.style == "documentary"
    assert " ".join(scene.narration for scene in plan.scenes) == script
    assert all(scene.image_prompt for scene in plan.scenes)
    assert plan.scenes[0].type == "talking_avatar"
    assert plan.scenes[0].avatar is not None


def test_local_split_script_paces_five_second_scenes():
    from renderflow.pipeline.script import split_script_local

    script = (
        "The Colosseum could seat fifty thousand spectators, empty in under "
        "ten minutes, and stage naval battles on a flooded arena floor built "
        "by thousands of enslaved workers. It still stands today."
    )

    plan, _ = split_script_local(script, "documentary")

    # ~2.5 words/sec: every scene stays around 5 seconds, long sentences
    # split at clause breaks, and the text survives verbatim.
    assert len(plan.scenes) >= 3
    assert all(len(s.narration.split()) <= 16 for s in plan.scenes)
    assert " ".join(s.narration for s in plan.scenes) == script
    assert all("photograph" in s.image_prompt for s in plan.scenes)
    # Composition rule: subject + context, never a lone face
    assert all("never a lone" in s.image_prompt for s in plan.scenes)
    assert all("lone face close-up" in s.negative_prompt for s in plan.scenes)


def test_local_split_script_ignores_block_comments():
    from renderflow.pipeline.script import split_script_local

    plan, result = split_script_local(
        "Keep this narration. /* Do not narrate this section. */ Resume here.",
        "documentary",
    )

    narration = " ".join(scene.narration for scene in plan.scenes)
    assert result.cost == 0.0
    assert "Keep this narration." in narration
    assert "Resume here." in narration
    assert "Do not narrate this section." not in narration
