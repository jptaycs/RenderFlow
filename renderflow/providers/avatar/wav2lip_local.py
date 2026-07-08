"""Free local lip sync via Wav2Lip (no API, no quota, runs on CPU).

One portrait image + narration audio -> clip whose lips follow the audio.
Wav2Lip does not synthesize head motion, so a slow camera push-in is added
to keep the shot alive. Setup (one-time, ~520 MB):

    .venv/bin/pip install '.[wav2lip]'
    .venv/bin/python scripts/setup_wav2lip.py
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from renderflow.providers.avatar.postprocess import ensure_wav, escape_drawtext
from renderflow.providers.base import GeneratedAsset

log = logging.getLogger("renderflow.providers.wav2lip")

FPS = 25  # Wav2Lip's output rate for still-image input


class Wav2LipLocal:
    name = "wav2lip-local"

    def __init__(
        self,
        wav2lip_dir: Path = Path(".wav2lip"),
        motion_intensity: float = 0.06,
    ) -> None:
        self.wav2lip_dir = wav2lip_dir
        self.checkpoint = wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"
        if not self.checkpoint.exists():
            raise ValueError(
                "wav2lip-local is not set up — run: "
                ".venv/bin/pip install '.[wav2lip]' && "
                ".venv/bin/python scripts/setup_wav2lip.py"
            )
        self.motion_intensity = motion_intensity

    def generate_clip(
        self,
        avatar_image: Path,
        voice_audio: Path,
        script_text: str,
        **params: Any,
    ) -> GeneratedAsset:
        disclosure = params.get("disclosure") or "AI-generated host"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            wav = ensure_wav(voice_audio, work)
            raw = work / "wav2lip_raw.mp4"
            log.info(
                "wav2lip inference (%d chars narration) — CPU, takes a bit",
                len(script_text),
            )
            proc = subprocess.run(
                [
                    sys.executable, "inference.py",
                    "--checkpoint_path", str(self.checkpoint.resolve()),
                    "--face", str(avatar_image.resolve()),
                    "--audio", str(wav.resolve()),
                    "--outfile", str(raw),
                ],
                cwd=self.wav2lip_dir,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not raw.exists():
                raise RuntimeError(
                    f"wav2lip inference failed: {proc.stderr[-2000:] or proc.stdout[-2000:]}"
                )
            out = work / "avatar_clip.mp4"
            self._finish(raw, out, disclosure)
            data = out.read_bytes()
        return GeneratedAsset(
            data=data,
            provider=self.name,
            params={
                "checkpoint": self.checkpoint.name,
                "motion_intensity": self.motion_intensity,
                "avatar_image": str(avatar_image),
                "voice_audio": str(voice_audio),
                "script_chars": len(script_text),
                "disclosure": disclosure,
            },
            cost=0.0,
            meta={"format": "mp4"},
        )

    def _finish(self, raw: Path, out: Path, disclosure: str) -> None:
        """Slow push-in (Wav2Lip leaves the head static) + disclosure + encode."""
        width, height, duration = _probe_video(raw)
        width, height = width // 2 * 2, height // 2 * 2
        frames = max(math.ceil(duration * FPS), 1)
        motion = (
            # Prescale 2x so sub-pixel zoom motion doesn't jitter.
            f"scale={width * 2}:{height * 2},"
            f"zoompan=z='1+{self.motion_intensity}*on/{frames}':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s={width}x{height}:fps={FPS}"
        )
        banner = (
            "drawbox=x=0:y=0:w=iw:h=48:color=black@0.45:t=fill,"
            f"drawtext=text='{escape_drawtext(disclosure)}':"
            "x=24:y=14:fontsize=22:fontcolor=white,"
        )
        proc = _encode(raw, out, f"{motion},{banner}format=yuv420p")
        if proc.returncode != 0 and "drawtext" in proc.stderr:
            proc = _encode(raw, out, f"{motion},format=yuv420p")
        if proc.returncode != 0:
            raise RuntimeError(f"wav2lip finishing pass failed: {proc.stderr[-2000:]}")


def _encode(src: Path, out: Path, vf: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            str(out),
        ],
        capture_output=True,
        text=True,
    )


def _probe_video(path: Path) -> tuple[int, int, float]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr}")
    lines = proc.stdout.strip().splitlines()
    width, height = (int(v) for v in lines[0].split(",")[:2])
    return width, height, float(lines[1])
