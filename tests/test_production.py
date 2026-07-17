"""Production posture: secure cookies and the startup safety guards."""

from __future__ import annotations

import dataclasses

import pytest

from renderflow.config import Settings
from tests.conftest import register


def _production_settings(saas_env, **overrides):
    return dataclasses.replace(saas_env.settings, env="production", **overrides)


def _patch_settings(monkeypatch, settings):
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))


def test_session_cookie_is_secure_in_production(saas_env, make_client, monkeypatch):
    _patch_settings(monkeypatch, _production_settings(saas_env))
    client = make_client()
    res = client.post(
        "/api/auth/register",
        json={"email": "user@example.com", "password": "password123"},
    )
    assert res.status_code == 201
    assert "secure" in res.headers["set-cookie"].lower()


def test_session_cookie_is_not_secure_in_dev(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "user@example.com", "password": "password123"},
    )
    assert res.status_code == 201
    assert "secure" not in res.headers["set-cookie"].lower()


def test_startup_refuses_production_with_dev_login(saas_env, monkeypatch):
    from renderflow import api

    _patch_settings(
        monkeypatch,
        _production_settings(saas_env, dev_login_email="dev@renderflow.local"),
    )
    with pytest.raises(RuntimeError, match="DEV_LOGIN"):
        api.startup()


def test_startup_refuses_production_with_dev_checkout(saas_env, monkeypatch):
    from renderflow import api

    _patch_settings(monkeypatch, _production_settings(saas_env, dev_checkout=True))
    with pytest.raises(RuntimeError, match="DEV_CHECKOUT"):
        api.startup()


def test_startup_refuses_production_with_default_db_password(saas_env, monkeypatch):
    from renderflow import api

    _patch_settings(
        monkeypatch,
        _production_settings(
            saas_env,
            database_url="postgresql+psycopg://renderflow:renderflow@127.0.0.1:5433/renderflow",
        ),
    )
    with pytest.raises(RuntimeError, match="default dev password"):
        api.startup()


def test_startup_refuses_missing_secret_key(saas_env, monkeypatch):
    from renderflow import api

    _patch_settings(
        monkeypatch, dataclasses.replace(saas_env.settings, secret_key="")
    )
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        api.startup()


def test_startup_accepts_clean_production_config(saas_env, monkeypatch):
    from renderflow import api

    # Non-default DB creds, no dev flags — must boot. (init_db runs against
    # the test SQLite engine configured by the fixture.)
    _patch_settings(monkeypatch, _production_settings(saas_env))
    api.startup()
