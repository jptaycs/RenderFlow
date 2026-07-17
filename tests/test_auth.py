"""Auth: register/login/logout, cookie sessions, first-user adoption."""

from __future__ import annotations

import json

from tests.conftest import register


def test_register_signs_in_and_me_roundtrips(client):
    body = register(client, "creator@example.com")
    assert body["email"] == "creator@example.com"
    assert body["isAdmin"] is True  # first user ever

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "creator@example.com"


def test_register_normalizes_email_case(client):
    register(client, "MixedCase@Example.COM")
    res = client.post(
        "/api/auth/login",
        json={"email": "mixedcase@example.com", "password": "password123"},
    )
    assert res.status_code == 200


def test_second_user_is_not_admin(make_client):
    register(make_client(), "first@example.com")
    body = register(make_client(), "second@example.com")
    assert body["isAdmin"] is False


def test_duplicate_email_is_rejected(client):
    register(client, "dup@example.com")
    res = client.post(
        "/api/auth/register",
        json={"email": "dup@example.com", "password": "password456"},
    )
    assert res.status_code == 409


def test_invalid_email_and_short_password_rejected(client):
    assert (
        client.post(
            "/api/auth/register", json={"email": "not-an-email", "password": "password123"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/auth/register", json={"email": "ok@example.com", "password": "short"}
        ).status_code
        == 422
    )


def test_login_rejects_wrong_password_and_unknown_email(client):
    register(client, "user@example.com")
    client.cookies.clear()
    wrong = client.post(
        "/api/auth/login", json={"email": "user@example.com", "password": "wrong-pass"}
    )
    unknown = client.post(
        "/api/auth/login", json={"email": "nobody@example.com", "password": "password123"}
    )
    assert wrong.status_code == unknown.status_code == 401
    # Same message either way — no probing which emails have accounts.
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_logout_clears_the_session(client):
    register(client, "user@example.com")
    assert client.get("/api/auth/me").status_code == 200
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_endpoints_require_auth(client):
    assert client.get("/api/state").status_code == 401
    assert client.post("/api/projects", json={"title": "X", "script": "Y"}).status_code == 401
    assert client.post("/api/projects/x/resume").status_code == 401
    assert client.get("/files/x/output/final.mp4").status_code == 401


def test_dev_login_prefill_is_disabled_by_default(client):
    """No dev credentials configured (the deployed posture) — the endpoint
    exposes nothing, and no POST route exists at all (the button submits
    through the normal /login and /register endpoints)."""
    assert client.get("/api/auth/dev-login").json() == {"enabled": False}
    assert client.post("/api/auth/dev-login").status_code == 405


def _enable_dev_login(saas_env, monkeypatch, email="dev@renderflow.local", password="dev-pass-123"):
    import dataclasses

    from renderflow.config import Settings

    dev_settings = dataclasses.replace(
        saas_env.settings, dev_login_email=email, dev_login_password=password
    )
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: dev_settings))


def test_dev_login_prefill_returns_configured_credentials(saas_env, client, monkeypatch):
    _enable_dev_login(saas_env, monkeypatch)
    assert client.get("/api/auth/dev-login").json() == {
        "enabled": True,
        "email": "dev@renderflow.local",
        "password": "dev-pass-123",
    }


def test_dev_credentials_only_work_through_the_real_password_check(
    saas_env, make_client, monkeypatch
):
    """The button's flow is register-once then normal logins — a wrong
    password is rejected exactly like any other account's."""
    _enable_dev_login(saas_env, monkeypatch)
    dev = make_client()
    prefill = dev.get("/api/auth/dev-login").json()

    # Fresh DB: the button's fallback registers through the normal endpoint.
    res = dev.post(
        "/api/auth/register",
        json={"email": prefill["email"], "password": prefill["password"]},
    )
    assert res.status_code == 201

    # Thereafter it's an ordinary login; the real check still applies.
    fresh = make_client()
    ok = fresh.post(
        "/api/auth/login",
        json={"email": prefill["email"], "password": prefill["password"]},
    )
    bad = fresh.post(
        "/api/auth/login",
        json={"email": prefill["email"], "password": "not-the-dev-password"},
    )
    assert ok.status_code == 200
    assert bad.status_code == 401


def test_first_user_adopts_legacy_projects(saas_env, make_client):
    # A pre-auth project sitting at the old flat projects/<slug> location.
    legacy = saas_env.projects_dir / "old-video"
    (legacy / "script").mkdir(parents=True)
    (legacy / "script" / "scenes.json").write_text(
        json.dumps({"title": "Old Video", "style": "documentary", "scenes": []})
    )

    admin = make_client()
    register(admin, "admin@example.com")
    slugs = [p["slug"] for p in admin.get("/api/state").json()["projects"]]
    assert "old-video" in slugs

    # The second user must not see the adopted project.
    other = make_client()
    register(other, "other@example.com")
    assert other.get("/api/state").json()["projects"] == []
