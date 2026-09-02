"""Shared HTTP client for the 69labs API (https://69labs.vip/api/v1).

69labs is a job-queue REST API shared across images, video, and TTS:
POST .../generate returns a job id, GET .../status/{id} polls until a
terminal state, and GET .../download/{id} returns the file (direct bytes
or a 302 redirect to a presigned URL). All three product adapters
(image/labs69.py, tts/labs69_tts.py, video/labs69_video.py) share this
module instead of each re-implementing the submit/poll/download dance and
the idempotency/429/error-shape handling.

Deliberately NOT built on pipeline/retry.py's `@retryable` decorator: that
decorator re-enters the whole wrapped function on every attempt, which
would mint a fresh random job (and, worse, a fresh Idempotency-Key) per
retry. 69labs' own documented retry pattern is "generate one Idempotency-Key
per logical request, reuse it across retries of that same POST" — so
`submit()` retries in-place around a single httpx call with one key,
matching that contract exactly.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

import httpx

log = logging.getLogger("renderflow.providers.labs69")

API_BASE = "https://69labs.vip/api/v1"
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "CENSORED"}
# Status codes worth retrying in place (rate limit / transient server error).
# Everything else (400/401/402/403/404/409/410) is a real, non-retryable
# problem with the request or the account and should surface immediately.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class Labs69Error(RuntimeError):
    """A 69labs request failed, or a job ended in a non-COMPLETED terminal
    state (FAILED/CANCELLED/CENSORED)."""


def _api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("LABS69_API_KEY")
    if not key:
        raise ValueError(
            "69labs provider needs LABS69_API_KEY in .env (Settings > API "
            "Keys at 69labs.vip — the key starts with vk_)"
        )
    return key


def cost_from_status(status: dict[str, Any]) -> float | None:
    """USD cost of a completed job, read from the pay-per-use `billing`
    block. Credit-based accounts get `billing: null` — there is no way to
    derive a USD figure from a credit balance, so this returns None
    ("unknown"), not 0.0 ("free"). AssetRef.cost tolerates None (it's
    simply excluded from ScenePlan.total_asset_cost); a credit-account
    video's true cost will just be missing from the dashboard's cost
    tracking rather than reported as free."""
    billing = status.get("billing")
    if not billing:
        return None
    cents = billing.get("totalAmountCents")
    return cents / 100.0 if cents is not None else None


def require_completed(status: dict[str, Any], *, context: str) -> None:
    state = status.get("status")
    if state == "COMPLETED":
        return
    job_id = status.get("id", "?")
    if state == "CENSORED":
        raise Labs69Error(
            f"{context} (job {job_id}) was censored by 69labs — this needs "
            "a rewritten chunk resubmitted via POST /tts/retry-censored, "
            "which RenderFlow does not do automatically"
        )
    raise Labs69Error(f"{context} (job {job_id}) ended with status {state!r}")


class Labs69Client:
    """Thin REST client for one 69labs product family. `api_key` defaults
    to LABS69_API_KEY from the environment."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = _api_key(api_key)

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_retries: int = 4,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """POST a generate request, retrying 429/5xx in place with the same
        Idempotency-Key so a retried submit can never double-bill."""
        idempotency_key = str(uuid.uuid4())
        headers = self._headers(idempotency_key)
        delay = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                response = httpx.post(
                    f"{API_BASE}{path}", headers=headers, json=body, timeout=timeout
                )
            except httpx.TransportError as exc:
                if attempt == max_retries:
                    raise Labs69Error(f"69labs POST {path} failed: {exc}") from exc
                log.warning("69labs POST %s transport error, retrying: %s", path, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                retry_after = float(response.headers.get("Retry-After", delay))
                log.warning(
                    "69labs POST %s -> %d, retrying in %.1fs (attempt %d/%d)",
                    path, response.status_code, retry_after, attempt, max_retries,
                )
                time.sleep(retry_after)
                delay = min(delay * 2, 30.0)
                continue
            self._raise_for_status(response, path)
            return response.json()
        raise Labs69Error(f"69labs POST {path} exhausted retries")

    def poll_until_terminal(
        self, path: str, *, interval: float = 4.0, timeout: float = 600.0
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            response = httpx.get(
                f"{API_BASE}{path}", headers=self._headers(), timeout=30.0
            )
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", interval)))
                continue
            self._raise_for_status(response, path)
            data = response.json()
            if data.get("status") in TERMINAL_STATUSES:
                return data
            if time.monotonic() > deadline:
                raise Labs69Error(
                    f"69labs job at {path} timed out after {timeout:.0f}s "
                    f"(last status {data.get('status')!r})"
                )
            time.sleep(interval)

    def download(self, path: str, *, max_retries: int = 3) -> bytes:
        """GET and read the full body, retrying transient transport errors.

        Unlike `submit()`/`poll_until_terminal()`, this had no retry at all
        until a real mid-download drop on a ~4 MB video file
        (`httpx.RemoteProtocolError: peer closed connection without
        sending complete message body`) lost a whole B-roll clip live —
        the per-scene continue-on-failure design in generate_broll caught
        it fine, but a multi-megabyte transfer over a real network has a
        meaningfully higher chance of a mid-stream drop than the small
        JSON responses submit()/poll use, so it deserves its own retry.
        """
        delay = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                response = httpx.get(
                    f"{API_BASE}{path}",
                    headers=self._headers(),
                    timeout=300.0,
                    follow_redirects=True,
                )
            except httpx.TransportError as exc:
                if attempt == max_retries:
                    raise Labs69Error(f"69labs GET {path} failed: {exc}") from exc
                log.warning(
                    "69labs download %s transport error, retrying (attempt %d/%d): %s",
                    path, attempt, max_retries, exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            self._raise_for_status(response, path)
            return response.content
        raise Labs69Error(f"69labs GET {path} exhausted retries")

    def get_json(self, path: str) -> dict[str, Any]:
        response = httpx.get(
            f"{API_BASE}{path}", headers=self._headers(), timeout=30.0
        )
        self._raise_for_status(response, path)
        return response.json()

    @staticmethod
    def _raise_for_status(response: httpx.Response, path: str) -> None:
        # download() always passes follow_redirects=True, so a 302 to a
        # presigned URL is already resolved to its final response by the
        # time we get here — no redirect status ever reaches this check.
        if response.is_success:
            return
        try:
            body = response.json()
        except ValueError:
            body = {}
        message = body.get("error") or response.text[:300]
        code = body.get("code")
        raise Labs69Error(f"69labs {path} -> {response.status_code} ({code}): {message}")
