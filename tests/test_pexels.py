import json

import httpx
import pytest

from renderflow.providers.base import ImageProvider
from renderflow.providers.image import pexels
from renderflow.providers.image.pexels import PexelsImage, _search_query


def test_pexels_satisfies_image_protocol(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    assert isinstance(PexelsImage(), ImageProvider)


def test_pexels_requires_key(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PEXELS_API_KEY"):
        PexelsImage()


def test_search_query_from_scene_prompt():
    # Shape produced by pipeline/script.py::_local_image_prompt.
    prompt = (
        "Wide or medium documentary b-roll photograph about solar panels — "
        "absolutely no visible human face, never a portrait, never a "
        "person looking at the camera. If people appear, show them from "
        "behind, in silhouette, at a distance, or cropped to hands/objects "
        "only — never a face. The photo must clearly visually relate to "
        "solar panels: show its concrete setting, objects, tools, or the "
        "specific action described here: The salesman promised the roof "
        "would pay for itself. Keep solar panels recognizably present in "
        "the frame even if this line doesn't name it directly."
    )
    query = _search_query(prompt)
    assert query.startswith("solar panels")
    assert len(query.split()) <= 6
    assert "face" not in query


def test_search_query_strips_variation_clause():
    prompt = (
        "Wide or medium documentary b-roll photograph about ants — "
        "described here: The colony moved on. Keep ants present."
        " Try this take: from a different angle."
    )
    assert "angle" not in _search_query(prompt)


def test_search_query_from_thumbnail_prompt():
    prompt = (
        "Viral YouTube thumbnail background: a dramatic photograph of "
        "beer flood. One instantly recognizable beer flood scene."
    )
    assert _search_query(prompt) == "beer flood"


def test_search_query_fallback_on_unknown_shape():
    query = _search_query("An abandoned brewery at dawn, mist over the vats")
    assert "brewery" in query
    assert len(query.split()) <= 6


JPEG = b"\xff\xd8\xe0fake-jpeg-bytes"


def _fake_httpx_get(photos):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url == pexels.API_URL:
            body = json.dumps({"photos": photos})
            return httpx.Response(200, content=body, request=httpx.Request("GET", url))
        return httpx.Response(200, content=JPEG, request=httpx.Request("GET", url))

    return fake_get, calls


def _photo(photo_id, alt=""):
    return {
        "id": photo_id,
        "alt": alt,
        "url": f"https://pexels.com/photo/{photo_id}",
        "photographer": "Someone",
        "src": {"original": f"https://images.pexels.com/{photo_id}.jpg"},
    }


def test_generate_downloads_photo_and_costs_nothing(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.setattr(pexels, "MIN_INTERVAL", 0.0)
    fake_get, _ = _fake_httpx_get([_photo(1), _photo(2)])
    monkeypatch.setattr(pexels.httpx, "get", fake_get)
    asset = PexelsImage().generate("a photo about ants — described")
    assert asset.data == JPEG
    assert asset.cost == 0.0
    assert asset.provider == "pexels"
    assert asset.params["photo_id"] in (1, 2)


def test_repeat_calls_return_different_photos(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.setattr(pexels, "MIN_INTERVAL", 0.0)
    fake_get, _ = _fake_httpx_get([_photo(1), _photo(2), _photo(3)])
    monkeypatch.setattr(pexels.httpx, "get", fake_get)
    provider = PexelsImage()
    prompt = "a photo about ants"
    ids = {provider.generate(prompt).params["photo_id"] for _ in range(3)}
    assert ids == {1, 2, 3}


def test_facey_alt_photos_are_deprioritized(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.setattr(pexels, "MIN_INTERVAL", 0.0)
    monkeypatch.setattr(pexels, "TOP_POOL", 1)
    photos = [_photo(1, alt="portrait of a smiling man"), _photo(2, alt="an ant hill")]
    fake_get, _ = _fake_httpx_get(photos)
    monkeypatch.setattr(pexels.httpx, "get", fake_get)
    asset = PexelsImage().generate("a photo about ants")
    assert asset.params["photo_id"] == 2


def test_generate_raises_on_no_results(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.setattr(pexels, "MIN_INTERVAL", 0.0)
    fake_get, calls = _fake_httpx_get([])
    monkeypatch.setattr(pexels.httpx, "get", fake_get)
    with pytest.raises(ValueError, match="no photos"):
        PexelsImage().generate("a photo about mosquito bites itching")
    # Progressively shorter fallback queries were attempted.
    assert len(calls) > 1
