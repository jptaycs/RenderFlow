"""A B-roll provider that fails to *construct* (e.g. RENDERFLOW_SHORTS_
BROLL_PROVIDER=pexels-video set without PEXELS_API_KEY) must degrade the
run the same way a B-roll *generation* failure already does — never crash
the whole make_video.py run right after images were already paid for.
"""

from __future__ import annotations

import json

import make_video
from tests.conftest import make_settings
from tests.stubs import StubImage, StubTTS


def test_broll_provider_construction_failure_does_not_crash_the_run(tmp_path, monkeypatch, capsys):
    scenes_file = tmp_path / "scenes.json"
    scenes_file.write_text(json.dumps({
        "title": "Demo",
        "style": "documentary",
        # shorts format skips the AI clickbait thumbnail entirely (it only
        # runs for landscape) — avoids needing a real decodable PNG for
        # StubImage's fake bytes to survive an actual ffmpeg thumbnail
        # crop, which isn't what this test is checking.
        "format": "shorts",
        "scenes": [{
            "id": 1,
            "type": "narration",
            "duration_estimate_sec": 5.0,
            "narration": "Hello there.",
            "image_prompt": "A photo.",
            "negative_prompt": None,
            "avatar": None,
            "avatar_layout": "auto",
            "broll_mode": "auto",
            "motion": {"effect": "zoom_in", "intensity": 0.08},
        }],
    }))

    settings = make_settings(
        broll_provider="labs69", intro_outro=False, subtitles_enabled=False,
    )
    monkeypatch.setattr(make_video.Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(make_video, "build_image", lambda s: StubImage())
    monkeypatch.setattr(make_video, "build_tts", lambda s: StubTTS())

    def _raise_missing_key(settings, name=None):
        raise ValueError("pexels-video provider needs PEXELS_API_KEY in .env")

    monkeypatch.setattr(make_video, "build_broll", _raise_missing_key)

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
    assert "B-roll provider unavailable" in capsys.readouterr().out
