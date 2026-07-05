"""Phase 1 provider verification: one minimal live call per provider.

Usage:
    .venv/bin/python scripts/check_providers.py [claude|image|tts ...]

With no arguments, checks all three. Prints result summary and cost per
provider; exits non-zero if any check fails. Total cost of a full run is
well under $0.02.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable

from renderflow.config import Settings
from renderflow.providers import build_image, build_llm, build_tts

OK = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check_claude(settings: Settings) -> str:
    llm = build_llm(settings)
    result = llm.complete(
        "You are a health check. Reply with exactly the word: pong",
        "ping",
        max_tokens=64,
    )
    assert "pong" in result.text.lower(), f"unexpected reply: {result.text!r}"
    return (
        f"model={result.meta['model']} reply={result.text.strip()!r} "
        f"tokens={result.meta['input_tokens']}/{result.meta['output_tokens']} "
        f"cost=${result.cost:.6f}"
    )


def check_image(settings: Settings) -> str:
    image = build_image(settings)
    asset = image.generate("A single red apple on a white table, studio lighting")
    assert len(asset.data) > 1000, f"suspiciously small image: {len(asset.data)} bytes"
    magic = asset.data[:8]
    kind = "png" if magic.startswith(b"\x89PNG") else ("jpeg" if magic.startswith(b"\xff\xd8") else "webp/other")
    return f"provider={asset.provider} {len(asset.data)} bytes ({kind}) cost=${asset.cost:.4f}"


def check_tts(settings: Settings) -> str:
    tts = build_tts(settings)
    asset = tts.synthesize("RenderFlow provider check.", settings.tts_voice)
    assert len(asset.data) > 1000, f"suspiciously small audio: {len(asset.data)} bytes"
    return (
        f"provider={asset.provider} voice={settings.tts_voice} "
        f"{len(asset.data)} bytes cost=${asset.cost:.5f}"
    )


CHECKS: dict[str, Callable[[Settings], str]] = {
    "claude": check_claude,
    "image": check_image,
    "tts": check_tts,
}


def main() -> int:
    settings = Settings.load()
    names = sys.argv[1:] or list(CHECKS)
    failures = 0
    total_cost_note = []
    for name in names:
        try:
            summary = CHECKS[name](settings)
            print(f"[{OK}] {name}: {summary}")
        except Exception as exc:
            failures += 1
            print(f"[{FAIL}] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
