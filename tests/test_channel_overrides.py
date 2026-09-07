"""Per-project channel_name/tts_voice overrides (added 2026-09) — a second
content "channel" (a distinct branding name/narrator voice for a batch of
videos, e.g. a food-facts channel alongside the main trivia channel)
without touching the global .env default every other video uses. Exercised
through the real CLI entrypoint, same pattern as test_shorts_subtitles.py.
"""

from __future__ import annotations

import json

import make_video
from tests.conftest import make_settings
from tests.stubs import StubImage, StubTTS


def _scenes_file(tmp_path):
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({
        "title": "Banana Facts",
        "style": "documentary",
        "format": "landscape",
        "scenes": [{
            "id": 1,
            "type": "narration",
            "duration_estimate_sec": 5.0,
            "narration": "Hi, I'm a banana.",
            "image_prompt": "A photo.",
            "negative_prompt": None,
            "avatar": None,
            "avatar_layout": "auto",
            "broll_mode": "off",
            "motion": {"effect": "zoom_in", "intensity": 0.08},
        }],
    }))
    return path


def _run(tmp_path, scenes_file, monkeypatch, extra_argv=None):
    settings = make_settings(
        intro_outro=False, channel_name="Cool Facts Daily", tts_voice="default-voice-id",
    )
    monkeypatch.setattr(make_video.Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(make_video, "build_image", lambda s: StubImage())
    monkeypatch.setattr(make_video, "build_tts", lambda s: StubTTS())
    monkeypatch.setattr(make_video, "build_broll", lambda s, name=None: None)
    monkeypatch.setattr(make_video, "generate_thumbnail", lambda *a, **k: None)
    monkeypatch.setattr(make_video, "render_thumbnail", lambda *a, **k: None)
    monkeypatch.setattr(make_video, "generate_subtitles", lambda plan, paths: None)

    calls: dict = {}

    def _fake_generate_voice(plan, tts, voice, paths, **kw):
        calls["voice"] = voice
        # _incomplete_scenes (make_video.py) checks scene.assets.voice.path
        # truthy regardless of --skip-render — this stub replaces real
        # voice generation entirely, so it must still satisfy that check
        # or the run bails out with "Stopped before rendering" before
        # ever reaching the code this test actually cares about.
        for scene in plan.scenes:
            scene.assets.voice.path = "fake-voice.mp3"

    monkeypatch.setattr(make_video, "generate_voice", _fake_generate_voice)

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_video.py",
            "--scenes-file", str(scenes_file),
            "--slug", "demo",
            "--projects-dir", str(tmp_path / "projects"),
            "--skip-render",
            *(extra_argv or []),
        ],
    )
    exit_code = make_video.main()
    assert exit_code == 0
    return calls


def test_tts_voice_override_used_instead_of_global_default(tmp_path, monkeypatch):
    scenes_file = _scenes_file(tmp_path)
    calls = _run(tmp_path, scenes_file, monkeypatch, extra_argv=["--tts-voice", "banana-voice-id"])
    assert calls["voice"] == "banana-voice-id"


def test_no_override_falls_back_to_the_global_default_voice(tmp_path, monkeypatch):
    scenes_file = _scenes_file(tmp_path)
    calls = _run(tmp_path, scenes_file, monkeypatch)
    assert calls["voice"] == "default-voice-id"


def test_channel_name_override_is_persisted_on_the_plan(tmp_path, monkeypatch):
    # Must survive on disk, not just the in-memory run: resume/regenerate
    # run in a later process and need the exact identity this project was
    # created with, not whatever the global .env says by then.
    from renderflow.storage import ProjectPaths, load_plan

    scenes_file = _scenes_file(tmp_path)
    _run(tmp_path, scenes_file, monkeypatch, extra_argv=["--channel-name", "Snack Facts Daily"])

    paths = ProjectPaths.create(tmp_path / "projects", "demo")
    plan = load_plan(paths)
    assert plan.channel_name == "Snack Facts Daily"


def test_no_channel_override_leaves_plan_field_none(tmp_path, monkeypatch):
    from renderflow.storage import ProjectPaths, load_plan

    scenes_file = _scenes_file(tmp_path)
    _run(tmp_path, scenes_file, monkeypatch)

    paths = ProjectPaths.create(tmp_path / "projects", "demo")
    plan = load_plan(paths)
    assert plan.channel_name is None
