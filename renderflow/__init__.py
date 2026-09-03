"""RenderFlow — AI video production orchestration."""

__version__ = "0.1.0"

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
