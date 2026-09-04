import json

import httpx
import pytest

from renderflow.providers import labs69_client
from renderflow.providers.base import ImageProvider, TTSProvider, VideoProvider
from renderflow.providers.labs69_client import (
    Labs69Client,
    Labs69Error,
    cost_from_status,
    require_completed,
)


# --- cost_from_status / require_completed (pure logic, no network) ---


def test_cost_from_status_reads_ppu_billing():
    status = {"billing": {"mode": "PAY_PER_USE", "totalAmountCents": 25}}
    assert cost_from_status(status) == pytest.approx(0.25)


def test_cost_from_status_none_for_credit_accounts():
    assert cost_from_status({"billing": None}) is None
    assert cost_from_status({}) is None


def test_require_completed_passes_on_completed():
    require_completed({"status": "COMPLETED", "id": "j1"}, context="job")


def test_require_completed_raises_on_censored():
    with pytest.raises(Labs69Error, match="censored"):
        require_completed({"status": "CENSORED", "id": "j1"}, context="tts job")


def test_require_completed_raises_on_failed():
    with pytest.raises(Labs69Error, match="FAILED"):
        require_completed({"status": "FAILED", "id": "j1"}, context="image job")


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("LABS69_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LABS69_API_KEY"):
        Labs69Client()


# --- Labs69Client submit/poll/download against mocked httpx ---


def _json_response(url, payload, status=200, headers=None):
    return httpx.Response(
        status,
        content=json.dumps(payload),
        headers=headers or {},
        request=httpx.Request("GET", url),
    )


def test_submit_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return _json_response(url, {"error": "slow down"}, status=429, headers={"Retry-After": "0"})
        return _json_response(url, {"id": "job1", "queuePosition": 0}, status=201)

    monkeypatch.setattr(labs69_client.httpx, "post", fake_post)
    client = Labs69Client()
    result = client.submit("/images/generate", {"prompt": "x"})
    assert result["id"] == "job1"
    assert len(calls) == 2


def test_submit_falls_back_to_default_delay_on_http_date_retry_after(monkeypatch):
    # Regression (full-app scan 2026-09): RFC 9110 allows Retry-After to be
    # an HTTP-date string instead of delta-seconds — bare float() raised an
    # uncaught ValueError on that form, aborting the whole retry loop
    # instead of just falling back to the default backoff delay.
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    monkeypatch.setattr(labs69_client.time, "sleep", lambda seconds: None)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return _json_response(
                url, {"error": "slow down"}, status=429,
                headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
            )
        return _json_response(url, {"id": "job1", "queuePosition": 0}, status=201)

    monkeypatch.setattr(labs69_client.httpx, "post", fake_post)
    result = Labs69Client().submit("/images/generate", {"prompt": "x"})
    assert result["id"] == "job1"
    assert len(calls) == 2


def test_submit_raises_on_non_retryable_error(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")

    def fake_post(url, **kwargs):
        return _json_response(url, {"error": "bad prompt", "code": "BAD_REQUEST"}, status=400)

    monkeypatch.setattr(labs69_client.httpx, "post", fake_post)
    with pytest.raises(Labs69Error, match="BAD_REQUEST"):
        Labs69Client().submit("/images/generate", {"prompt": "x"})


def test_poll_until_terminal_waits_then_returns(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        status = "PROCESSING" if len(calls) < 3 else "COMPLETED"
        return _json_response(url, {"status": status, "id": "job1"})

    monkeypatch.setattr(labs69_client.httpx, "get", fake_get)
    monkeypatch.setattr(labs69_client.time, "sleep", lambda _: None)
    result = Labs69Client().poll_until_terminal("/images/status/job1", interval=0.0)
    assert result["status"] == "COMPLETED"
    assert len(calls) == 3


def test_download_retries_transport_error_then_succeeds(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    monkeypatch.setattr(labs69_client.time, "sleep", lambda _: None)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("peer closed connection")
        return httpx.Response(200, content=b"video-bytes", request=httpx.Request("GET", url))

    monkeypatch.setattr(labs69_client.httpx, "get", fake_get)
    data = Labs69Client().download("/videos/download/job1")
    assert data == b"video-bytes"
    assert len(calls) == 2


def test_download_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    monkeypatch.setattr(labs69_client.time, "sleep", lambda _: None)

    def fake_get(url, **kwargs):
        raise httpx.RemoteProtocolError("peer closed connection")

    monkeypatch.setattr(labs69_client.httpx, "get", fake_get)
    with pytest.raises(Labs69Error, match="peer closed connection"):
        Labs69Client().download("/videos/download/job1", max_retries=2)


# --- Image / TTS / video adapters ---


def _mock_job_lifecycle(monkeypatch, status_payload, download_bytes):
    def fake_post(url, **kwargs):
        return _json_response(url, {"id": "job1", "queuePosition": 0}, status=201)

    def fake_get(url, **kwargs):
        if "/status/" in url:
            return _json_response(url, status_payload)
        if "/download/" in url:
            return httpx.Response(200, content=download_bytes, request=httpx.Request("GET", url))
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(labs69_client.httpx, "post", fake_post)
    monkeypatch.setattr(labs69_client.httpx, "get", fake_get)


def test_labs69_image_satisfies_protocol(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.image.labs69 import Labs69Image

    assert isinstance(Labs69Image(), ImageProvider)


def test_labs69_image_generate_happy_path(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.image.labs69 import Labs69Image

    _mock_job_lifecycle(
        monkeypatch,
        {
            "status": "COMPLETED",
            "id": "job1",
            "outputMetadata": {"format": "png"},
            "billing": {"totalAmountCents": 25},
        },
        b"\x89PNGfake",
    )
    asset = Labs69Image().generate("a mountain lake")
    assert asset.data == b"\x89PNGfake"
    assert asset.provider == "labs69"
    assert asset.cost == pytest.approx(0.25)
    assert asset.meta["format"] == "png"


def test_labs69_image_raises_on_failed_job(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.image.labs69 import Labs69Image

    _mock_job_lifecycle(monkeypatch, {"status": "FAILED", "id": "job1"}, b"")
    with pytest.raises(Labs69Error, match="FAILED"):
        Labs69Image().generate("a mountain lake")


def test_labs69_tts_satisfies_protocol(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.tts.labs69_tts import Labs69TTS

    assert isinstance(Labs69TTS(), TTSProvider)


def test_labs69_tts_synthesize_happy_path(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.tts.labs69_tts import Labs69TTS

    _mock_job_lifecycle(
        monkeypatch, {"status": "COMPLETED", "id": "job1", "billing": None}, b"ID3fakemp3"
    )
    asset = Labs69TTS().synthesize("Hello world", "21m00Tcm4TlvDq8ikWAM")
    assert asset.data == b"ID3fakemp3"
    assert asset.cost is None
    assert asset.meta["characters"] == len("Hello world")
    assert asset.params["voiceProvider"] == "elevenlabs"


def test_labs69_tts_censored_raises(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.tts.labs69_tts import Labs69TTS

    _mock_job_lifecycle(monkeypatch, {"status": "CENSORED", "id": "job1"}, b"")
    with pytest.raises(Labs69Error, match="censored"):
        Labs69TTS().synthesize("bad text", "voice1")


def test_labs69_video_satisfies_protocol(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.video.labs69_video import Labs69Video

    assert isinstance(Labs69Video(), VideoProvider)


def test_labs69_video_pick_duration_prefers_smallest_covering():
    from renderflow.providers.video.labs69_video import Labs69Video

    model_info = {"durations": ["4", "6", "8", "10"]}
    assert Labs69Video._pick_duration(model_info, 5.0) == "6"
    assert Labs69Video._pick_duration(model_info, 20.0) == "10"  # nothing long enough
    assert Labs69Video._pick_duration(None, 5.0) is None


def test_labs69_video_pick_aspect_ratio():
    from renderflow.providers.video.labs69_video import Labs69Video

    model_info = {"aspectRatios": [{"value": "1:1"}, {"value": "16:9"}]}
    assert Labs69Video._pick_aspect_ratio(model_info) == "16:9"
    assert Labs69Video._pick_aspect_ratio({"aspectRatios": [{"value": "1:1"}]}) is None
    assert Labs69Video._pick_aspect_ratio(None) is None


def test_labs69_video_lookup_model_is_race_free_under_concurrency(monkeypatch):
    # Regression (full-app scan 2026-09): the old check-then-set
    # (_model_info_fetched = True *before* the network call returned) let
    # a thread arriving mid-fetch see the flag already True and return
    # _model_info while it was still None — not a benign duplicate fetch,
    # a real permanent miss for that thread's find_clip call. Forces the
    # race window with a barrier so a second caller is guaranteed to be
    # blocked mid-fetch when it calls _lookup_model, then asserts it still
    # gets the real (non-None) catalog entry, not a premature None, and
    # that the network was only hit once.
    import threading

    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.video.labs69_video import Labs69Video

    fetch_started = threading.Event()
    release_fetch = threading.Event()
    fetch_calls = []

    def fake_get_json(self, path):
        fetch_calls.append(path)
        fetch_started.set()
        assert release_fetch.wait(timeout=5), "second caller never reached _lookup_model"
        return {
            "models": [{"id": "m1", "durations": ["6"], "aspectRatios": [{"value": "16:9"}]}],
            "defaultModelId": "m1",
        }

    monkeypatch.setattr(labs69_client.Labs69Client, "get_json", fake_get_json)
    provider = Labs69Video()

    results: list[dict | None] = [None, None]

    def call_first():
        results[0] = provider._lookup_model()

    def call_second():
        fetch_started.wait(timeout=5)  # ensure we arrive while the fetch is in flight
        release_fetch.set()
        results[1] = provider._lookup_model()

    t1 = threading.Thread(target=call_first)
    t2 = threading.Thread(target=call_second)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(fetch_calls) == 1  # only one real fetch, not one per thread
    assert results[0] is not None and results[0]["id"] == "m1"
    assert results[1] is not None and results[1]["id"] == "m1"  # not the stale None


def test_labs69_video_find_clip_happy_path(monkeypatch):
    monkeypatch.setenv("LABS69_API_KEY", "vk_test")
    from renderflow.providers.video.labs69_video import Labs69Video

    def fake_post(url, **kwargs):
        return _json_response(url, {"id": "job1"}, status=201)

    def fake_get(url, **kwargs):
        if "/models" in url:
            return _json_response(
                url,
                {
                    "models": [
                        {
                            "id": "grok-imagine-video",
                            "durations": ["6", "10"],
                            "aspectRatios": [{"value": "16:9"}],
                        }
                    ],
                    "defaultModelId": "grok-imagine-video",
                },
            )
        if "/status/" in url:
            return _json_response(
                url, {"status": "COMPLETED", "id": "job1", "outputMetadata": {"durationSeconds": 6}}
            )
        if "/download/" in url:
            return httpx.Response(200, content=b"fakemp4", request=httpx.Request("GET", url))
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(labs69_client.httpx, "post", fake_post)
    monkeypatch.setattr(labs69_client.httpx, "get", fake_get)
    asset = Labs69Video().find_clip("a river at dawn", min_duration_sec=5.0)
    assert asset.data == b"fakemp4"
    assert asset.params["duration"] == "6"
    assert asset.params["aspectRatio"] == "16:9"
