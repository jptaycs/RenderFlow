"""YouTube publishing: upload request shape, thumbnail wiring, storage
round-trip. Never hits the real API — googleapiclient's client object is
faked out at renderflow.youtube._client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from renderflow.schema import YouTubePublish
from renderflow.storage import ProjectPaths, load_youtube_publish, save_youtube_publish


class _FakeRequest:
    """Mimics googleapiclient's resumable-upload request: next_chunk()
    returns (status, None) while in progress, then (None, response)."""

    def __init__(self, response: dict[str, Any]):
        self._response = response
        self._calls = 0

    def next_chunk(self):
        self._calls += 1
        if self._calls == 1:
            from types import SimpleNamespace

            return SimpleNamespace(progress=lambda: 0.5), None
        return None, self._response


class _FakeVideosResource:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.insert_calls: list[dict[str, Any]] = []

    def insert(self, part, body, media_body):
        self.insert_calls.append({"part": part, "body": body, "media_body": media_body})
        return _FakeRequest(self.response)


class _FakeThumbnailsResource:
    def __init__(self):
        self.set_calls: list[dict[str, Any]] = []

    def set(self, videoId, media_body):
        self.set_calls.append({"videoId": videoId, "media_body": media_body})

        class _Executable:
            def execute(self_inner):
                return {}

        return _Executable()


class _FakeYouTubeClient:
    def __init__(self, response: dict[str, Any]):
        self.videos_resource = _FakeVideosResource(response)
        self.thumbnails_resource = _FakeThumbnailsResource()

    def videos(self):
        return self.videos_resource

    def thumbnails(self):
        return self.thumbnails_resource


def test_is_connected_reflects_token_file(tmp_path, monkeypatch):
    from renderflow import youtube as yt

    token = tmp_path / "token.json"
    monkeypatch.setattr(yt, "TOKEN_PATH", token)
    assert yt.is_connected() is False
    token.write_text("{}")
    assert yt.is_connected() is True


def test_credentials_raises_when_not_connected(tmp_path, monkeypatch):
    from renderflow import youtube as yt

    monkeypatch.setattr(yt, "TOKEN_PATH", tmp_path / "missing.json")
    with pytest.raises(yt.YouTubeNotConnected, match="setup_youtube"):
        yt._credentials()


def test_upload_video_builds_correct_request_body(tmp_path, monkeypatch):
    from renderflow import youtube as yt

    fake_client = _FakeYouTubeClient({"id": "vid123"})
    monkeypatch.setattr(yt, "_client", lambda: fake_client)

    video_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"fake video bytes")

    result = yt.upload_video(
        video_path,
        title="My Video",
        description="desc",
        tags=["a", "b"],
        privacy_status="unlisted",
        contains_synthetic_media=True,
        made_for_kids=False,
    )

    assert result == {"video_id": "vid123", "url": "https://youtu.be/vid123"}
    call = fake_client.videos_resource.insert_calls[0]
    assert call["body"]["snippet"]["title"] == "My Video"
    assert call["body"]["snippet"]["description"] == "desc"
    assert call["body"]["snippet"]["tags"] == ["a", "b"]
    assert call["body"]["status"]["privacyStatus"] == "unlisted"
    assert call["body"]["status"]["containsSyntheticMedia"] is True
    assert call["body"]["status"]["selfDeclaredMadeForKids"] is False
    # No thumbnail_path given — thumbnails.set must never be called.
    assert fake_client.thumbnails_resource.set_calls == []


def test_upload_video_sets_thumbnail_when_given(tmp_path, monkeypatch):
    from renderflow import youtube as yt

    fake_client = _FakeYouTubeClient({"id": "vid456"})
    monkeypatch.setattr(yt, "_client", lambda: fake_client)

    video_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"fake video bytes")
    thumb_path = tmp_path / "thumbnail.jpg"
    thumb_path.write_bytes(b"fake jpg bytes")

    yt.upload_video(video_path, title="T", thumbnail_path=thumb_path)

    assert fake_client.thumbnails_resource.set_calls[0]["videoId"] == "vid456"


def test_upload_video_skips_thumbnail_when_path_missing(tmp_path, monkeypatch):
    """A configured-but-nonexistent thumbnail path must not crash the
    upload — thumbnails are a nice-to-have, not required."""
    from renderflow import youtube as yt

    fake_client = _FakeYouTubeClient({"id": "vid789"})
    monkeypatch.setattr(yt, "_client", lambda: fake_client)

    video_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"fake video bytes")

    yt.upload_video(video_path, title="T", thumbnail_path=tmp_path / "nope.jpg")

    assert fake_client.thumbnails_resource.set_calls == []


def test_upload_video_defaults_contains_synthetic_media_true(tmp_path, monkeypatch):
    """Every RenderFlow video is AI-narrated over AI-generated visuals —
    the disclosure flag must default on, not require opting in."""
    from renderflow import youtube as yt

    fake_client = _FakeYouTubeClient({"id": "vid999"})
    monkeypatch.setattr(yt, "_client", lambda: fake_client)

    video_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"fake video bytes")

    yt.upload_video(video_path, title="T")

    body = fake_client.videos_resource.insert_calls[0]["body"]
    assert body["status"]["containsSyntheticMedia"] is True


def test_youtube_publish_storage_round_trip(tmp_path):
    paths = ProjectPaths.create(tmp_path, "demo")
    assert load_youtube_publish(paths) is None  # never published

    result = YouTubePublish(
        video_id="abc123",
        url="https://youtu.be/abc123",
        privacy_status="public",
        contains_synthetic_media=True,
        published_at=1234.5,
    )
    save_youtube_publish(result, paths)

    reloaded = load_youtube_publish(paths)
    assert reloaded == result
