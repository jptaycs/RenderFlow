"""RenderFlow — AI video production orchestration."""

import os

__version__ = "0.1.0"

# Second Avast Web Shield fallout, hit live 2026-09 (same root cause as the
# truststore fix below, a different symptom): Avast sets SSLKEYLOGFILE
# (verified live: `\\.\aswMonFltProxy\<id>`, a named pipe belonging to its
# own HTTPS-scanning proxy driver) at the Windows user-env level, presumably
# so it can capture TLS session keys for its own traffic inspection.
# `ssl.create_default_context()` honors that env var unconditionally
# (`context.keylog_filename = keylogfile`) — every single provider call in
# this app failed identically with `PermissionError: [Errno 13] Permission
# denied: '\\.\aswMonFltProxy\...'` deep in truststore's SSLContext proxy,
# because the current (non-elevated) process can't write to Avast's device.
# Reproduced live: every image AND voice call failed on every scene of a
# real project, all ten create-job auto-retry attempts, and all ten of the
# resume job's retries after — a systemic, per-process env issue, not a
# per-call transient error retrying could ever fix. RenderFlow has no
# legitimate use for TLS key logging, so this just removes it from *this
# process's* environment (never the real Windows env var, and never
# anything Avast itself does) before any SSL context gets created — must
# run before the truststore import right below, and before any provider
# constructs an httpx/anthropic client.
os.environ.pop("SSLKEYLOGFILE", None)

# Verify outbound HTTPS against the OS trust store instead of the bundled
# `certifi` CA list, for every httpx-based provider call in this package
# (and the anthropic SDK, which also uses httpx). certifi ships only the
# public Mozilla-curated root list — it has no way to know about a root a
# local security product has injected into the OS store.
#
# Hit live 2026-09: Avast's HTTPS-scanning "Web Shield" MITMs outbound TLS
# and re-signs the connection with its own root CA (`Avast Web/Mail Shield
# Root`), which Avast itself installs into the Windows trust store — so
# curl and browsers (which consult the OS store) connect fine, but every
# httpx call failed with `CERTIFICATE_VERIFY_FAILED: unable to get local
# issuer certificate` — reproduced against 69labs.vip; any HTTPS provider
# a security product decides to intercept is equally exposed. `truststore`
# (stdlib `ssl` + the OS's native verifier — Windows CryptoAPI, macOS
# Security framework, Linux's system store) makes Python trust exactly
# what the OS already trusts, matching curl/browser behavior with no
# antivirus/proxy setting to change and no cert bundle to hand-patch.
# Must run before any provider constructs an httpx/anthropic client, so
# it's injected here at package import time rather than per-provider.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass  # truststore is a core dependency (pyproject.toml) — this is
    # only a safety net for an environment that skipped installing it.
