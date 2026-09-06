"""Process-environment fixes applied at package import time
(renderflow/__init__.py) — no network, just checks the env mutation."""

from __future__ import annotations

import importlib
import os

import renderflow


def test_import_strips_sslkeylogfile(monkeypatch):
    # Regression test for a real live failure (2026-09): Avast's Web Shield
    # sets SSLKEYLOGFILE (a named pipe belonging to its own HTTPS-scanning
    # proxy, e.g. `\\.\aswMonFltProxy\<id>`) at the Windows user-env level.
    # ssl.create_default_context() honors that env var unconditionally, and
    # writing to Avast's device raised `PermissionError: [Errno 13]` deep
    # inside truststore's SSLContext proxy — every single httpx-based
    # provider call in the app failed identically (reproduced live: every
    # scene's image AND voice generation failed, all ten create-job
    # auto-retry attempts, all ten of a resume job's retries after that).
    # renderflow/__init__.py must strip it from *this process's* env
    # (never touching the real Windows env var) before any SSL context can
    # be created — reload the module with the var deliberately set to
    # prove that happens, since a normal test run wouldn't otherwise
    # exercise this line (the module is already imported by other tests).
    monkeypatch.setenv("SSLKEYLOGFILE", r"\\.\aswMonFltProxy\deadbeef")

    importlib.reload(renderflow)

    assert os.environ.get("SSLKEYLOGFILE") is None
