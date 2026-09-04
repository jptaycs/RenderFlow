"""RENDERFLOW_SHORTS_SUBTITLES: per-format override for burned-in captions.

Shorts are watched sound-off while scrolling far more than landscape is, so
a user who prefers captions off for their (sound-on, cinematic-feeling)
landscape videos may still want them on for Shorts specifically. None
(unset) means "follow subtitles_enabled" — this file locks in that
resolution logic (make_video.py's `subtitles_on` ternary) through the real
CLI entrypoint, monkeypatching generate_subtitles to record whether it was
called rather than exercising the real Pillow/ffprobe pipeline against
stub (non-audio) bytes.
"""

from __future__ import annotations

import json

import make_video
from tests.conftest import make_settings
from tests.stubs import StubImage, StubTTS


def _scenes_file(tmp_path, fmt: str):
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({
        "title": "Demo",
        "style": "documentary",
        "format": fmt,
        "scenes": [{
            "id": 1,
            "type": "narration",
            "duration_estimate_sec": 5.0,
            "narration": "Hello there.",
            "image_prompt": "A photo.",
            "negative_prompt": None,
            "avatar": None,
            "avatar_layout": "auto",
            "broll_mode": "off",
            "motion": {"effect": "zoom_in", "intensity": 0.08},
        }],
    }))
    return path


def _run(tmp_path, scenes_file, monkeypatch, **settings_overrides):
    settings = make_settings(intro_outro=False, **settings_overrides)
    monkeypatch.setattr(make_video.Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(make_video, "build_image", lambda s: StubImage())
    monkeypatch.setattr(make_video, "build_tts", lambda s: StubTTS())
    monkeypatch.setattr(make_video, "build_broll", lambda s, name=None: None)
    # Landscape (unlike shorts) generates the AI clickbait thumbnail, which
    # would run real ffmpeg on StubImage's fake (non-decodable) PNG bytes —
    # not what this test is checking, so no-op both steps.
    monkeypatch.setattr(make_video, "generate_thumbnail", lambda *a, **k: None)
    monkeypatch.setattr(make_video, "render_thumbnail", lambda *a, **k: None)

    called = {"subtitles": False}
    monkeypatch.setattr(
        make_video, "generate_subtitles", lambda plan, paths: called.__setitem__("subtitles", True)
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_video.py",
            "--scenes-file", str(scenes_file),
            "--slug", "demo",
            "--projects-dir", str(tmp_path / "projects"),
            "--skip-render",
        ],
    )
    exit_code = make_video.main()
    assert exit_code == 0
    return called["subtitles"]


def test_shorts_override_turns_captions_on_when_landscape_default_is_off(tmp_path, monkeypatch):
    scenes_file = _scenes_file(tmp_path, "shorts")
    ran = _run(
        tmp_path, scenes_file, monkeypatch,
        subtitles_enabled=False, shorts_subtitles_enabled=True,
    )
    assert ran is True


def test_shorts_override_does_not_leak_into_landscape(tmp_path, monkeypatch):
    scenes_file = _scenes_file(tmp_path, "landscape")
    ran = _run(
        tmp_path, scenes_file, monkeypatch,
        subtitles_enabled=False, shorts_subtitles_enabled=True,
    )
    assert ran is False


def test_shorts_unset_override_inherits_landscape_default(tmp_path, monkeypatch):
    scenes_file = _scenes_file(tmp_path, "shorts")
    ran = _run(
        tmp_path, scenes_file, monkeypatch,
        subtitles_enabled=False, shorts_subtitles_enabled=None,
    )
    assert ran is False
