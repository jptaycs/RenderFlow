"""One-time download of Kokoro-82M model files into .kokoro/ (~340 MB).

Usage:
    .venv/bin/pip install '.[kokoro]'
    .venv/bin/python scripts/setup_kokoro.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
TARGET = Path(".kokoro")


def main() -> int:
    TARGET.mkdir(exist_ok=True)
    for name in FILES:
        dest = TARGET / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"[skip] {dest} already present")
            continue
        print(f"[get ] {BASE}/{name}")
        with httpx.stream(
            "GET", f"{BASE}/{name}", follow_redirects=True, timeout=None
        ) as response:
            response.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
        print(f"[ok  ] {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    print("Kokoro ready. Set RENDERFLOW_TTS_PROVIDER=kokoro in .env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
