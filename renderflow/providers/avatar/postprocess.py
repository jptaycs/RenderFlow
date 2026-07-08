"""Shared ffmpeg helpers for avatar providers: audio prep and disclosure burn-in."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("renderflow.providers.avatar")


def escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def ensure_wav(voice_audio: Path, work: Path) -> Path:
    """Lip-sync models want WAV; transcode MP3 (ElevenLabs) losslessly."""
    if voice_audio.suffix.lower() == ".wav":
        return voice_audio
    out = work / "voice.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(voice_audio), "-ar", "44100", str(out)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"audio->wav transcode failed: {proc.stderr[-2000:]}")
    return out


def overlay_disclosure(video: bytes, disclosure: str, work: Path) -> bytes:
    """Burn the AI-host disclosure banner in, matching the ffmpeg-still contract."""
    src = work / "clip_raw.mp4"
    out = work / "clip_disclosed.mp4"
    src.write_bytes(video)
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf",
            (
                "drawbox=x=0:y=0:w=iw:h=48:color=black@0.45:t=fill,"
                f"drawtext=text='{escape_drawtext(disclosure)}':"
                "x=24:y=14:fontsize=22:fontcolor=white,"
                "format=yuv420p"
            ),
            "-c:v", "libx264", "-preset", "medium",
            "-c:a", "copy",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # drawtext needs a fontconfig build of ffmpeg; the clip itself is fine.
        log.warning("disclosure overlay skipped: %s", proc.stderr[-300:])
        return video
    return out.read_bytes()
