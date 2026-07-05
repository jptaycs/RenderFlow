"""Local talking-avatar placeholder.

This is not lip sync. It turns the generated avatar image and narration audio
into a short presenter clip so the pipeline can exercise the avatar contract
without paying for an external avatar/video API.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from renderflow.providers.base import GeneratedAsset


class FFMpegStillAvatar:
    name = "ffmpeg-still"

    def generate_clip(
        self,
        avatar_image: Path,
        voice_audio: Path,
        script_text: str,
        **params: Any,
    ) -> GeneratedAsset:
        disclosure = params.get("disclosure") or "AI-generated host"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "avatar_clip.mp4"
            filter_with_disclosure = (
                "[0:v]"
                "scale=1920:1080:force_original_aspect_ratio=increase,"
                "crop=1920:1080,"
                "drawbox=x=0:y=0:w=iw:h=72:color=black@0.45:t=fill,"
                f"drawtext=text='{_escape_drawtext(disclosure)}':"
                "x=40:y=24:fontsize=30:fontcolor=white,"
                "format=yuv420p[v]"
            )
            proc = _render_clip(avatar_image, voice_audio, out, filter_with_disclosure)
            if proc.returncode != 0 and "No such filter: 'drawtext'" in proc.stderr:
                fallback_filter = (
                    "[0:v]"
                    "scale=1920:1080:force_original_aspect_ratio=increase,"
                    "crop=1920:1080,format=yuv420p[v]"
                )
                proc = _render_clip(avatar_image, voice_audio, out, fallback_filter)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg avatar render failed: {proc.stderr[-2000:]}")
            return GeneratedAsset(
                data=out.read_bytes(),
                provider=self.name,
                params={
                    "avatar_image": str(avatar_image),
                    "voice_audio": str(voice_audio),
                    "script_chars": len(script_text),
                    "disclosure": disclosure,
                },
                cost=0.0,
                meta={"format": "mp4"},
            )


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _render_clip(
    avatar_image: Path, voice_audio: Path, out: Path, filter_complex: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(avatar_image),
            "-i",
            str(voice_audio),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
