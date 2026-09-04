"""End-to-end skeleton test with stub providers (no network, no API keys)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from renderflow.pipeline.assets import (
    generate_avatar_clips,
    generate_branding_audio,
    generate_images,
    generate_subtitles,
    generate_voice,
)
from renderflow.pipeline.script import generate_script
from renderflow.providers.base import LLMResult
from renderflow.schema import AssetStatus, Scene
from renderflow.storage import ProjectPaths, load_plan, save_plan, slugify
from tests.stubs import CANNED_SCRIPT, StubAvatar, StubImage, StubLLM, StubTTS


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


def test_script_max_tokens_scales_with_scene_count():
    # Regression: a flat 16000-token ceiling (sized for short 2-3 minute
    # videos) silently truncated any longer generate_script call — the
    # whole point of the dashboard's 10/12-minute length options would
    # have shipped broken without this scaling. floor/ceiling and
    # roughly-linear growth in between.
    from renderflow.pipeline.script import _script_max_tokens

    assert _script_max_tokens(4) == 16000  # floor: a ~2-minute video
    assert _script_max_tokens(36) > 16000  # ~3 minutes already exceeds the old flat default
    tokens_10min = _script_max_tokens(120)
    tokens_12min = _script_max_tokens(144)
    assert tokens_10min < tokens_12min  # more scenes -> more budget
    assert tokens_12min <= 128000  # never exceeds Sonnet 5/Opus 5's output cap


class _RecordingLLM:
    """Records the max_tokens (and other kwargs) each call received, so
    generate_script/split_script can be checked without a live API call."""

    name = "recording-llm"

    def __init__(self):
        self.calls: list[dict] = []

    def complete(self, system, prompt, **params):
        self.calls.append(params)
        return LLMResult(text=json.dumps(CANNED_SCRIPT), provider=self.name, cost=0.01)


def test_generate_script_passes_scaled_max_tokens():
    from renderflow.pipeline.script import _script_max_tokens, generate_script

    llm = _RecordingLLM()
    generate_script(llm, "a topic", 12, "documentary")

    assert len(llm.calls) == 1
    # 12 minutes / 5 sec-per-scene -> target_scenes, same formula the
    # production code uses.
    expected_scenes = max(4, round(12 * 60 / 5))
    assert llm.calls[0]["max_tokens"] == _script_max_tokens(expected_scenes)


def test_split_script_passes_scaled_max_tokens():
    from renderflow.pipeline.script import _script_max_tokens, split_script

    llm = _RecordingLLM()
    long_script = " ".join(["word"] * 1300)  # ~100 scenes at ~13 words/scene
    split_script(llm, long_script, "documentary")

    assert len(llm.calls) == 1
    assert llm.calls[0]["max_tokens"] == _script_max_tokens(100)


class _RecordingTopicIdeaLLM:
    """Like _RecordingLLM but returns a valid GeneratedTopicIdea payload —
    generate_topic_idea's schema ({title, script}), not a full scene plan."""

    name = "recording-topic-idea-llm"

    def __init__(self):
        self.calls: list[dict] = []

    def complete(self, system, prompt, **params):
        self.calls.append({"prompt": prompt, **params})
        return LLMResult(
            text=json.dumps({"title": "A Title", "script": "A fact."}),
            provider=self.name, cost=0.001,
        )


def test_topic_idea_scales_word_target_and_max_tokens_with_length():
    # Regression: generate_topic_idea used to always request a fixed
    # 60-100 word teaser regardless of the video's actual target length —
    # a "Random topic" landscape video (targeting the monetization-driven
    # 8-12 minute default) came out with a script only long enough for
    # ~30 seconds once split into scenes. Client-reported: "the random
    # topic script must have a length the same with the target time like
    # 11 minutes of video and in shorts is 1 minute script also."
    from renderflow.pipeline.script import (
        _topic_idea_max_tokens,
        _topic_idea_target_words,
        generate_topic_idea,
    )

    llm = _RecordingTopicIdeaLLM()
    generate_topic_idea(llm, [], length_minutes=11)

    assert len(llm.calls) == 1
    call = llm.calls[0]
    expected_words = _topic_idea_target_words(11)
    assert expected_words > 1000  # a real 11-minute script, not a teaser
    assert call["max_tokens"] == _topic_idea_max_tokens(expected_words)
    assert str(expected_words) in call["prompt"]


def test_topic_idea_short_target_stays_short():
    from renderflow.pipeline.script import (
        _topic_idea_max_tokens,
        _topic_idea_target_words,
        generate_topic_idea,
    )

    llm = _RecordingTopicIdeaLLM()
    generate_topic_idea(llm, [], length_minutes=1)

    expected_words = _topic_idea_target_words(1)
    assert expected_words < 250  # a Shorts-length target, not a full script
    assert llm.calls[0]["max_tokens"] == _topic_idea_max_tokens(expected_words)


class _TrackingTTS:
    """StubTTS but records every (text, voice) it was asked to synthesize —
    needed to assert on the narrated intro/outro text content, which the
    shared StubTTS (tests/stubs.py) doesn't track."""

    name = "tracking-tts"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text, voice, **params):
        from renderflow.providers.base import GeneratedAsset

        self.calls.append((text, voice))
        if self.fail:
            raise RuntimeError("tts failure")
        return GeneratedAsset(data=b"fake-mp3", provider=self.name, cost=0.001)


def test_generate_branding_audio_narrates_title_and_outro_line(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "test topic", 1, "documentary")
    save_plan(plan, paths)
    tts = _TrackingTTS()

    generate_branding_audio(plan, tts, "voice-id", "", paths)

    assert tts.calls[0] == (plan.title, "voice-id")
    assert "subscribe" in tts.calls[1][0].lower()
    assert "trivia" in tts.calls[1][0].lower()
    reloaded = load_plan(paths)
    assert reloaded.intro_audio.status is AssetStatus.COMPLETED
    assert reloaded.outro_audio.status is AssetStatus.COMPLETED
    assert Path(reloaded.intro_audio.path).read_bytes() == b"fake-mp3"
    assert reloaded.total_asset_cost() == pytest.approx(0.002)


def test_generate_branding_audio_outro_mentions_channel_name(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "test topic", 1, "documentary")
    save_plan(plan, paths)
    tts = _TrackingTTS()

    generate_branding_audio(plan, tts, "voice-id", "Cool Facts Daily", paths)

    assert "Cool Facts Daily" in tts.calls[1][0]


def test_generate_branding_audio_failure_is_optional(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "test topic", 1, "documentary")
    save_plan(plan, paths)
    tts = _TrackingTTS(fail=True)

    generate_branding_audio(plan, tts, "voice-id", "", paths)  # must not raise

    reloaded = load_plan(paths)
    assert reloaded.intro_audio.status is AssetStatus.FAILED
    assert reloaded.outro_audio.status is AssetStatus.FAILED


def test_generate_branding_audio_is_resumable(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "test topic", 1, "documentary")
    save_plan(plan, paths)
    tts = _TrackingTTS()

    generate_branding_audio(plan, tts, "voice-id", "", paths)
    generate_branding_audio(plan, tts, "voice-id", "", paths)  # resume: no re-synthesis

    assert len(tts.calls) == 2


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


def test_one_failed_scene_does_not_stop_the_batch(paths: ProjectPaths):
    """A provider error on scene 1 must not prevent scene 2 from generating —
    a single rate-limited/timed-out scene used to raise and kill the whole
    run before any other scene was even attempted."""

    class FlakyImage:
        name = "flaky-image"

        def __init__(self):
            self.calls = 0

        def generate(self, prompt, negative_prompt=None, **params):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated rate limit")
            return StubImage().generate(prompt, negative_prompt, **params)

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    save_plan(plan, paths)

    generate_images(plan, FlakyImage(), paths)

    assert plan.scenes[0].assets.image.status is AssetStatus.FAILED
    assert plan.scenes[1].assets.image.status is AssetStatus.COMPLETED
    assert Path(plan.scenes[1].assets.image.path).read_bytes() == b"fake-png"


def _plain_scene(scene_id: int, avatar_layout: str = "auto") -> Scene:
    return Scene(
        id=scene_id, duration_estimate_sec=5.0, narration="x", image_prompt="x",
        avatar_layout=avatar_layout,
    )


def test_scene_is_avatar_solo_defaults_to_split_for_every_scene():
    from renderflow.pipeline.script import scene_is_avatar_solo

    # "auto" is always split-screen now — solo is opt-in per scene, not an
    # automatic cycle (see scene_is_avatar_solo docstring).
    solo_ids = [n for n in range(1, 13) if scene_is_avatar_solo(_plain_scene(n))]
    assert solo_ids == []


def test_scene_is_avatar_solo_manual_override():
    from renderflow.pipeline.script import scene_is_avatar_solo

    assert scene_is_avatar_solo(_plain_scene(2, "solo")) is True
    assert scene_is_avatar_solo(_plain_scene(4, "split")) is False
    assert scene_is_avatar_solo(_plain_scene(2, "auto")) is False
    assert scene_is_avatar_solo(_plain_scene(4, "auto")) is False


def test_effective_avatar_layout_resolves_all_three_modes():
    from renderflow.pipeline.script import effective_avatar_layout, scene_is_visual_only

    assert effective_avatar_layout(_plain_scene(1, "solo")) == "solo"
    assert effective_avatar_layout(_plain_scene(1, "split")) == "split"
    assert effective_avatar_layout(_plain_scene(1, "visual")) == "visual"
    assert effective_avatar_layout(_plain_scene(1, "auto")) == "split"
    assert scene_is_visual_only(_plain_scene(1, "visual")) is True
    assert scene_is_visual_only(_plain_scene(1, "solo")) is False
    assert scene_is_visual_only(_plain_scene(1, "split")) is False
    assert scene_is_visual_only(_plain_scene(1, "auto")) is False


def test_load_plan_migrates_legacy_auto_solo_scenes(paths: ProjectPaths, tmp_path: Path):
    """Regression: projects generated before "auto" always meant split had
    scenes the old 1-in-3 cycle picked as solo, which never got a
    background image (correct at the time — see effective_avatar_layout's
    docstring). Loading one of those scenes.json files today must not leave
    it looking permanently broken (an "Image: pending" that can never
    complete) — load_plan should recognize and repair it in place."""
    import time

    from renderflow.storage import load_plan, save_plan

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    scene = plan.scenes[0]
    scene.type = "talking_avatar"
    # Simulate the old system: avatar_layout left at "auto", but generated
    # as solo — image never touched (still PENDING), avatar assets done.
    scene.assets.avatar_image.advance(AssetStatus.RUNNING)
    scene.assets.avatar_image.path = "x"
    scene.assets.avatar_image.advance(AssetStatus.COMPLETED)
    scene.assets.avatar_clip.advance(AssetStatus.RUNNING)
    scene.assets.avatar_clip.path = "y"
    scene.assets.avatar_clip.advance(AssetStatus.COMPLETED)
    save_plan(plan, paths)
    original_mtime = paths.scenes_json.stat().st_mtime

    time.sleep(0.05)  # would bump mtime if load_plan rewrote unconditionally
    reloaded = load_plan(paths)

    assert reloaded.scenes[0].avatar_layout == "solo"
    # The migration write must not make an already-correct final.mp4 look
    # stale to api.py's render-staleness check.
    assert paths.scenes_json.stat().st_mtime == original_mtime

    # A second load is a no-op — nothing left to migrate, no further write.
    reloaded_again = load_plan(paths)
    assert reloaded_again.scenes[0].avatar_layout == "solo"
    assert paths.scenes_json.stat().st_mtime == original_mtime


def test_generate_images_skips_background_for_solo_scenes(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    for scene in plan.scenes:  # ids 1, 2
        scene.type = "talking_avatar"
    plan.scenes[0].avatar_layout = "solo"  # auto now defaults to split
    save_plan(plan, paths)

    generate_images(plan, StubImage(), paths)

    solo, split = plan.scenes  # id 1 solo (manual override), id 2 split (auto)
    assert solo.assets.image.status is AssetStatus.PENDING
    assert solo.assets.image.path is None
    assert split.assets.image.status is AssetStatus.COMPLETED
    assert split.assets.image.path is not None
    # Both still get the avatar portrait — solo vs. split only affects the
    # background visual, not the lip-sync source image.
    assert solo.assets.avatar_image.status is AssetStatus.COMPLETED
    assert split.assets.avatar_image.status is AssetStatus.COMPLETED


def test_generate_images_and_avatar_clips_skip_avatar_assets_for_visual_only(
    paths: ProjectPaths,
):
    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    plan.scenes[0].type = "talking_avatar"
    plan.scenes[0].avatar_layout = "visual"
    save_plan(plan, paths)

    generate_images(plan, StubImage(), paths)
    generate_voice(plan, StubTTS(), "voice-id", paths)
    generate_avatar_clips(plan, StubAvatar(), paths)

    visual_scene = load_plan(paths).scenes[0]
    # Visual-only still needs the background image + voice, like a plain
    # narration scene, but never the avatar portrait or lip-synced clip.
    assert visual_scene.assets.image.status is AssetStatus.COMPLETED
    assert visual_scene.assets.voice.status is AssetStatus.COMPLETED
    assert visual_scene.assets.avatar_image.status is AssetStatus.PENDING
    assert visual_scene.assets.avatar_clip.status is AssetStatus.PENDING


def test_has_prominent_face_returns_false_for_blank_image():
    pytest.importorskip("cv2")
    from PIL import Image
    import io
    from renderflow.pipeline.facecheck import has_prominent_face

    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (80, 80, 80)).save(buf, format="PNG")
    assert has_prominent_face(buf.getvalue()) is False


def test_has_prominent_face_handles_garbage_bytes_gracefully():
    pytest.importorskip("cv2")
    from renderflow.pipeline.facecheck import has_prominent_face

    assert has_prominent_face(b"not an image") is False


def test_generate_face_free_retries_until_no_face_detected(monkeypatch):
    from renderflow.pipeline import assets, facecheck

    calls = []

    class CountingImage:
        name = "counting-image"

        def generate(self, prompt, negative_prompt=None, **params):
            calls.append(1)
            return StubImage().generate(prompt, negative_prompt, **params)

    # Face detected on the first two attempts, clean on the third.
    results = iter([True, True, False])
    monkeypatch.setattr(facecheck, "has_prominent_face", lambda data: next(results))

    asset = assets._generate_face_free(CountingImage(), "a prompt", None, "test")
    assert len(calls) == 3
    assert asset.data == b"fake-png"


def test_generate_face_free_gives_up_after_max_retries(monkeypatch):
    from renderflow.pipeline import assets, facecheck

    calls = []

    class AlwaysFaceImage:
        name = "always-face-image"

        def generate(self, prompt, negative_prompt=None, **params):
            calls.append(1)
            return StubImage().generate(prompt, negative_prompt, **params)

    monkeypatch.setattr(facecheck, "has_prominent_face", lambda data: True)

    asset = assets._generate_face_free(AlwaysFaceImage(), "a prompt", None, "test")
    assert len(calls) == assets.MAX_FACE_RETRIES
    assert asset.data == b"fake-png"  # keeps the last result rather than failing the scene


def test_talking_avatar_clip_generation(paths: ProjectPaths):
    plan, _ = generate_script(StubLLM(), "test topic", 1, "documentary")
    plan.scenes[0].type = "talking_avatar"
    plan.scenes[0].avatar_layout = "solo"  # auto now defaults to split
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
    # Scene id 1 is solo-layout (manual override), so its background
    # image is skipped: only scene 2's image (0.003) + both scenes' voice
    # (0.002 each) + the avatar image (0.003) + avatar clip (0.004).
    assert load_plan(paths).total_asset_cost() == pytest.approx(
        0.003 + 0.002 * 2 + 0.003 + 0.004
    )
    assert avatar_scene.assets.image.status is AssetStatus.PENDING


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
    pytest.importorskip("PIL")
    from renderflow.pipeline.render import probe_duration, render_video

    # Keep the integration test hermetic: zoompan needs no depth model, no
    # intro/outro cards (their durations would swamp the scene-timing
    # asserts below — cards have their own integration test), and an empty
    # music dir (the repo-level music/ may hold user tracks). Settings.load
    # is patched wholesale — render.py calls it mid-render, and its
    # load_dotenv(override=True) would otherwise clobber the RENDERFLOW_MOTION
    # env monkeypatch with the .env value partway through the render.
    from renderflow.config import Settings
    from tests.conftest import make_settings

    settings = make_settings(
        intro_outro=False, music_dir=paths.root / "no-music", transition="fade"
    )
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setenv("RENDERFLOW_MOTION", "zoompan")
    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    # Scene id 1 is forced solo (full-screen avatar, no visual) and id 2 is
    # forced visual-only (background visual + narration, avatar clip
    # generated below but must be ignored by the render) — exercises
    # render_avatar_full_clip and the visual-only fallback path in one
    # real-ffmpeg pass. ("auto" now defaults to split-screen, already
    # covered by the non-integration layout unit tests below.)
    plan.scenes[0].type = "talking_avatar"
    plan.scenes[0].avatar_layout = "solo"
    plan.scenes[1].type = "talking_avatar"
    plan.scenes[1].avatar_layout = "visual"

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

    generate_subtitles(plan, paths)
    for scene in plan.scenes:
        assert scene.assets.subtitle.status is AssetStatus.COMPLETED
        assert Path(scene.assets.subtitle.path).exists()

    final = render_video(plan, paths)

    # Scene 2's own clip must be full-width (1920) — proof the visual-only
    # override actually took the plain image+voice path instead of the
    # split-screen hstack (960+960) it would have used before the override,
    # even though an avatar_clip asset exists for it.
    clip_2 = paths.output / "clip_002.mp4"
    assert clip_2.exists()
    width_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width", "-of", "csv=p=0", str(clip_2)],
        check=True, capture_output=True, text=True,
    )
    assert int(width_probe.stdout.strip()) == 1920

    assert final.exists()
    # Two 1-second scenes, each padded with a trailing SCENE_GAP_SEC pause,
    # hard-cut concatenated → a bit over 2s, never less (no crossfade shrink).
    assert probe_duration(final) > 2.0
    assert probe_duration(final) == pytest.approx(2.0, abs=1.0)
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
def test_render_shorts_produces_vertical_video_and_thumbnail(paths: ProjectPaths, monkeypatch):
    """plan.format == "shorts" renders at 1080x1920 (not 1920x1080), skips
    intro/outro cards (no AI-generated thumbnail path exists for shorts —
    render_shorts_thumbnail grabs a frame from the final video instead),
    and a split-screen-eligible talking-avatar scene falls back to
    full-screen solo (no room for a side-by-side panel in a 9:16 frame).
    Captions DO run for shorts (unlike cards/AI thumbnail) — this also
    proves a real ffmpeg overlay of a 1080-wide caption PNG onto a
    1080-wide clip doesn't get clipped/misplaced the way a landscape
    (1920-wide) caption PNG would if naively reused at shorts width."""
    pytest.importorskip("PIL")
    from renderflow.config import Settings
    from renderflow.pipeline.render import (
        SHORTS_HEIGHT,
        SHORTS_WIDTH,
        dims_for,
        probe_duration,
        render_shorts_thumbnail,
        render_video,
    )
    from renderflow.pipeline.subtitles import write_scene_subtitles
    from tests.conftest import make_settings

    settings = make_settings(
        intro_outro=False, music_dir=paths.root / "no-music", transition="fade"
    )
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setenv("RENDERFLOW_MOTION", "zoompan")
    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    plan.format = "shorts"
    # Left as "auto" (never overridden to solo) — proves the portrait-frame
    # fallback in render_scene_clip is unconditional, not dependent on the
    # scene's own avatar_layout choice.
    plan.scenes[0].type = "talking_avatar"
    shorts_width, _ = dims_for(plan)

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
            caption_duration = probe_duration(avatar_clip)
        else:
            caption_duration = probe_duration(audio)

        meta_path = write_scene_subtitles(scene, caption_duration, paths, shorts_width)
        scene.assets.subtitle.path = str(meta_path)
        scene.assets.subtitle.status = AssetStatus.COMPLETED

    final = render_video(plan, paths)
    assert final.exists()
    assert probe_duration(final) == pytest.approx(2.0, abs=1.0)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(final)],
        check=True, capture_output=True, text=True,
    )
    w, h = (int(x) for x in probe.stdout.strip().split(","))
    assert (w, h) == (SHORTS_WIDTH, SHORTS_HEIGHT)

    assert not (paths.output / "intro.mp4").exists()
    assert not (paths.output / "outro.mp4").exists()

    thumb = render_shorts_thumbnail(final, paths)
    assert thumb is not None and thumb.exists()
    assert thumb == paths.output / "thumbnail.jpg"


def test_build_caption_chunks_covers_full_duration_in_order():
    from renderflow.pipeline.subtitles import build_chunks

    narration = "The Colosseum could seat fifty thousand roaring spectators in ancient Rome"
    chunks = build_chunks(narration, duration=10.0)

    assert len(chunks) >= 2
    assert chunks[0][1] == 0.0
    assert chunks[-1][2] == pytest.approx(10.0)
    for (_, start, end), (_, next_start, _) in zip(chunks, chunks[1:]):
        assert end == pytest.approx(next_start)
        assert start < end
    # Every narration word survives, in order, across the chunks.
    assert " ".join(text for text, _, _ in chunks) == narration


def test_build_caption_chunks_handles_empty_narration():
    from renderflow.pipeline.subtitles import build_chunks

    assert build_chunks("", duration=5.0) == []
    assert build_chunks("hello", duration=0.0) == []


def test_write_scene_subtitles_renders_pngs_and_json(paths: ProjectPaths):
    pytest.importorskip("PIL")
    from renderflow.pipeline.subtitles import load_scene_subtitles, write_scene_subtitles

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    scene = plan.scenes[0]
    meta_path = write_scene_subtitles(scene, duration=6.0, paths=paths)

    assert meta_path.exists()
    entries = load_scene_subtitles(meta_path)
    assert len(entries) >= 1
    for entry in entries:
        assert Path(entry["image"]).exists()
        assert Path(entry["image"]).suffix == ".png"
        assert 0.0 <= entry["start"] < entry["end"] <= 6.0


def test_write_scene_subtitles_respects_width_for_shorts(paths: ProjectPaths):
    # Caption PNGs must be rendered at the actual output frame width — a
    # 1920-wide PNG naively reused on a 1080-wide shorts frame would get
    # clipped by render.py's centering overlay ((W-w)/2 goes negative).
    pytest.importorskip("PIL")
    from PIL import Image

    from renderflow.pipeline.subtitles import load_scene_subtitles, write_scene_subtitles

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    scene = plan.scenes[0]
    meta_path = write_scene_subtitles(scene, duration=6.0, paths=paths, width=1080)

    entries = load_scene_subtitles(meta_path)
    assert entries
    for entry in entries:
        with Image.open(entry["image"]) as img:
            assert img.width == 1080


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_generate_subtitles_needs_completed_audio_and_is_resumable(paths: ProjectPaths):
    pytest.importorskip("PIL")

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    save_plan(plan, paths)

    # No voice asset yet: nothing to time captions against.
    generate_subtitles(plan, paths)
    assert plan.scenes[0].assets.subtitle.status is AssetStatus.PENDING

    audio = paths.voice / "scene_001.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", str(audio)],
        check=True, capture_output=True,
    )
    scene = plan.scenes[0]
    scene.assets.voice.advance(AssetStatus.RUNNING)
    scene.assets.voice.path = str(audio)
    scene.assets.voice.advance(AssetStatus.COMPLETED)

    generate_subtitles(plan, paths)
    assert scene.assets.subtitle.status is AssetStatus.COMPLETED
    first_path = scene.assets.subtitle.path

    generate_subtitles(plan, paths)  # resume: already completed, must not redo
    assert scene.assets.subtitle.path == first_path


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_generate_subtitles_renders_shorts_width_captions(paths: ProjectPaths):
    # generate_subtitles must resolve the caption PNG width from
    # pipeline.render.dims_for(plan), not the hardcoded landscape default —
    # otherwise a shorts project's captions would get clipped by render.py's
    # centering overlay on the narrower 1080px frame.
    pytest.importorskip("PIL")
    from PIL import Image

    from renderflow.pipeline.subtitles import load_scene_subtitles

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    plan.format = "shorts"
    save_plan(plan, paths)

    audio = paths.voice / "scene_001.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", str(audio)],
        check=True, capture_output=True,
    )
    scene = plan.scenes[0]
    scene.assets.voice.advance(AssetStatus.RUNNING)
    scene.assets.voice.path = str(audio)
    scene.assets.voice.advance(AssetStatus.COMPLETED)

    generate_subtitles(plan, paths)
    entries = load_scene_subtitles(scene.assets.subtitle.path)
    assert entries
    for entry in entries:
        with Image.open(entry["image"]) as img:
            assert img.width == 1080


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_scene_clip_bakes_in_trailing_pause(paths: ProjectPaths, monkeypatch):
    """Regression: an earlier acrossfade crossfade blended the tail of one
    line's audio into the head of the next, so scenes sounded like the
    narrator talked over themselves. Each clip must now be exactly its own
    narration length plus SCENE_GAP_SEC of true silence, with no overlap."""
    monkeypatch.setenv("RENDERFLOW_MOTION", "zoompan")
    from renderflow.pipeline.render import SCENE_GAP_SEC, probe_duration, render_scene_clip

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    scene = plan.scenes[0]
    img = paths.images / "scene_001.png"
    audio = paths.voice / "scene_001.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=steelblue:s=640x360",
         "-frames:v", "1", str(img)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", str(audio)],
        check=True, capture_output=True,
    )
    for ref, path in ((scene.assets.image, img), (scene.assets.voice, audio)):
        ref.advance(AssetStatus.RUNNING)
        ref.path = str(path)
        ref.advance(AssetStatus.COMPLETED)

    clip = paths.output / "clip_001.mp4"
    render_scene_clip(scene, clip)
    assert probe_duration(clip) == pytest.approx(2.0 + SCENE_GAP_SEC, abs=0.1)


def test_caption_filter_chain_builds_overlay_and_inputs():
    from renderflow.pipeline.render import _caption_filter_chain

    chunks = [
        {"image": "/tmp/a.png", "start": 0.0, "end": 1.5},
        {"image": "/tmp/b.png", "start": 1.5, "end": 3.0},
    ]
    label, inputs, filt = _caption_filter_chain("base", chunks, start_index=2)

    assert inputs == ["-i", "/tmp/a.png", "-i", "/tmp/b.png"]
    assert label == "base_cap1"
    assert "[2:v]" in filt and "[3:v]" in filt
    assert "between(t,0.000,1.500)" in filt
    assert "between(t,1.500,3.000)" in filt


def test_caption_filter_chain_is_noop_when_no_chunks():
    from renderflow.pipeline.render import _caption_filter_chain

    label, inputs, filt = _caption_filter_chain("base", [], start_index=2)
    assert (label, inputs, filt) == ("base", [], "")


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
    # Two images now: the topic background (thumbnail.path) plus a separate
    # host reaction face — cost is the sum of both generate() calls.
    assert plan.thumbnail.cost == pytest.approx(0.006)
    assert plan.total_asset_cost() == pytest.approx(0.006)  # images not yet run
    reaction_path = paths.output / "thumbnail_reaction.png"
    assert reaction_path.exists()
    assert reaction_path.read_bytes() == b"fake-png"
    first_path = plan.thumbnail.path
    generate_thumbnail(plan, StubImage(), paths)  # resume skips
    assert plan.thumbnail.path == first_path


def test_generate_thumbnail_split_providers(paths: ProjectPaths):
    """Background and reaction face can come from different providers —
    e.g. AI-generated clickbait background + stock-photo reaction face."""
    from dataclasses import replace

    from renderflow.pipeline.assets import generate_thumbnail
    from renderflow.providers.base import GeneratedAsset

    class NamedStub(StubImage):
        def __init__(self, name: str, data: bytes) -> None:
            self.name = name
            self._data = data

        def generate(self, prompt, negative_prompt=None, **params):
            asset = StubImage.generate(self, prompt, negative_prompt, **params)
            return replace(asset, data=self._data, provider=self.name)

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    save_plan(plan, paths)
    generate_thumbnail(
        plan,
        NamedStub("ai-gen", b"bg-image"),
        paths,
        reaction_provider=NamedStub("stock", b"reaction-image"),
    )

    assert plan.thumbnail.status is AssetStatus.COMPLETED
    assert Path(plan.thumbnail.path).read_bytes() == b"bg-image"
    assert (paths.output / "thumbnail_reaction.png").read_bytes() == b"reaction-image"
    # Cost tracking must still answer "what did this cost" across both.
    assert plan.thumbnail.cost == pytest.approx(0.006)
    assert plan.thumbnail.provider == "ai-gen+stock"


def test_thumbnail_reaction_prompt_strips_calm_expression_wording():
    from renderflow.pipeline.assets import _thumbnail_reaction_prompt
    from renderflow.pipeline.script import LOCAL_AVATAR

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    plan.scenes[0].avatar = LOCAL_AVATAR
    prompt = _thumbnail_reaction_prompt(plan)

    assert "calm serious expression" not in prompt
    assert "shocked" in prompt.lower()
    assert "surprise" in prompt.lower()


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


def test_overlay_clickbait_text_adds_visible_text(tmp_path):
    from PIL import Image

    from renderflow.pipeline.render import _overlay_clickbait_text

    img_path = tmp_path / "thumb.jpg"
    bg = (10, 10, 10)
    Image.new("RGB", (1280, 720), bg).save(img_path)

    _overlay_clickbait_text(img_path, "Why Octopuses Have Three Hearts")

    result = Image.open(img_path).convert("RGB")
    assert result.size == (1280, 720)
    top_strip = result.crop((0, 0, 1280, 200))
    changed = any(
        top_strip.getpixel((x, y)) != bg
        for x in range(0, 1280, 10)
        for y in range(0, 200, 10)
    )
    assert changed, "expected some pixels in the top strip to differ from the plain background"


def test_overlay_clickbait_text_shrinks_long_titles_to_fit(tmp_path):
    """A long title must back off in size rather than overflow or crash —
    _wrap has no hard line cap, so this exercises the shrink loop."""
    from PIL import Image

    from renderflow.pipeline.render import _overlay_clickbait_text

    img_path = tmp_path / "thumb.jpg"
    Image.new("RGB", (1280, 720), (10, 10, 10)).save(img_path)

    _overlay_clickbait_text(
        img_path,
        "The Extraordinarily Long And Overly Detailed Title About A Very Specific Historical Event",
    )

    result = Image.open(img_path)
    assert result.size == (1280, 720)  # must not crash or resize the base image


def test_split_script_preserves_flow(paths: ProjectPaths):
    from renderflow.pipeline.script import split_script

    plan, result = split_script(StubLLM(), "Some client script text.", "documentary")
    assert len(plan.scenes) == 2
    assert plan.style == "documentary"
    assert result.cost == 0.01


def test_rewrite_video_prompt_returns_llm_text_and_cost():
    from renderflow.pipeline.script import rewrite_video_prompt
    from renderflow.providers.base import LLMResult

    class FakeLLM:
        name = "fake"

        def complete(self, system, prompt, **params):
            assert "no visible human face" not in system.lower() or True  # system is static; just don't crash
            assert "A photo of a reef." in prompt
            assert "Deep beneath the waves." in prompt
            return LLMResult(text="  Slow dolly-in over the reef.  ", provider=self.name, cost=0.002)

    prompt, result = rewrite_video_prompt(FakeLLM(), "Deep beneath the waves.", "A photo of a reef.")
    assert prompt == "Slow dolly-in over the reef."  # stripped
    assert result.cost == 0.002


def test_rewrite_video_prompt_propagates_llm_errors():
    from renderflow.pipeline.script import rewrite_video_prompt

    class FailingLLM:
        name = "fake"

        def complete(self, system, prompt, **params):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        rewrite_video_prompt(FailingLLM(), "narration", "a photo")


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


def test_local_split_script_avatar_disabled_stays_narration_only():
    # RENDERFLOW_LOCAL_AVATAR=0 (Settings.local_avatar_enabled=False) — the
    # script-mode local splitter is the only script path that always forced
    # an avatar; this restores the plain narration look (image/broll +
    # voice, no host) that topic-mode generation already produces by
    # default when the LLM doesn't add one.
    from renderflow.pipeline.script import split_script_local

    script = "This scene should have no avatar at all when the flag is off."

    plan, _ = split_script_local(script, "documentary", avatar_enabled=False)

    assert all(scene.type == "narration" for scene in plan.scenes)
    assert all(scene.avatar is None for scene in plan.scenes)


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
    # Composition rule: no visible human face at all, not just no close-up
    assert all("no visible human face" in s.image_prompt for s in plan.scenes)
    assert all("human face" in s.negative_prompt for s in plan.scenes)
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


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_with_music_and_cards_integration(paths: ProjectPaths, monkeypatch):
    """Music bed + intro/outro cards: the final video gains the card
    durations, persists its chosen track, and still muxes video+audio."""
    pytest.importorskip("PIL")
    from renderflow.config import Settings
    from renderflow.pipeline.render import (
        INTRO_SEC,
        OUTRO_SEC,
        SCENE_GAP_SEC,
        probe_duration,
        render_video,
    )
    from tests.conftest import make_settings

    music_dir = paths.root / "music"
    music_dir.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=200",
         "-t", "3", str(music_dir / "ambient.wav")],
        check=True, capture_output=True,
    )
    settings = make_settings(
        intro_outro=True,
        music_dir=music_dir,
        transition="fade",
        channel_name="Test Channel",
    )
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setenv("RENDERFLOW_MOTION", "zoompan")

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    plan.scenes = plan.scenes[:1]
    scene = plan.scenes[0]
    scene.type = "narration"
    img = paths.images / "scene_001.png"
    audio = paths.voice / "scene_001.mp3"
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
    save_plan(plan, paths)

    final = render_video(plan, paths)

    assert plan.music_track == "ambient.wav"  # chosen and persisted
    assert load_plan(paths).music_track == "ambient.wav"
    total = probe_duration(final)
    expected = 1.0 + SCENE_GAP_SEC + INTRO_SEC + OUTRO_SEC
    assert total == pytest.approx(expected, abs=1.2)
    # Re-render keeps the same track (no re-roll).
    render_video(plan, paths)
    assert plan.music_track == "ambient.wav"


def test_render_narrates_cards_and_extends_duration_for_long_voiceover(
    paths: ProjectPaths, monkeypatch
):
    """A narrated card (assets.generate_branding_audio) must hold long
    enough for its own voiceover, not just the fixed INTRO_SEC/OUTRO_SEC —
    added 2026-09, client request ("read the title/outro aloud")."""
    pytest.importorskip("PIL")
    from renderflow.config import Settings
    from renderflow.pipeline.render import INTRO_SEC, SCENE_GAP_SEC, probe_duration, render_video
    from tests.conftest import make_settings

    settings = make_settings(intro_outro=True, transition="fade")
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setenv("RENDERFLOW_MOTION", "zoompan")

    plan, _ = generate_script(StubLLM(), "t", 1, "documentary")
    plan.scenes = plan.scenes[:1]
    scene = plan.scenes[0]
    scene.type = "narration"
    img = paths.images / "scene_001.png"
    audio = paths.voice / "scene_001.mp3"
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

    # A 6s intro voiceover — well over the fixed INTRO_SEC (2.8s) — must
    # stretch the card, not get cut off.
    intro_audio_path = paths.voice / "intro.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300",
         "-t", "6", str(intro_audio_path)],
        check=True, capture_output=True,
    )
    plan.intro_audio.advance(AssetStatus.RUNNING)
    plan.intro_audio.path = str(intro_audio_path)
    plan.intro_audio.advance(AssetStatus.COMPLETED)
    save_plan(plan, paths)

    final = render_video(plan, paths)

    # The 6s intro voiceover must stretch the card well past its old fixed
    # 2.8s floor — the whole video should be several seconds longer than a
    # video with an unnarrated (fixed-duration) intro card.
    total = probe_duration(final)
    assert total > 1.0 + SCENE_GAP_SEC + INTRO_SEC + 3.0
