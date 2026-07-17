"""Billing: trial credits, subscription entitlement, dev checkout, grants."""

from __future__ import annotations

import dataclasses
import time

from renderflow import db as rdb
from renderflow.config import PLANS, TRIAL_CREDITS, Settings
from tests.conftest import register


def _enable_dev_checkout(saas_env, monkeypatch):
    dev_settings = dataclasses.replace(saas_env.settings, dev_checkout=True)
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: dev_settings))


def _create_project(client, title):
    res = client.post("/api/projects", json={"title": title, "script": "text"})
    assert res.status_code == 201, res.text
    return res.json()["slug"]


def _billing(client) -> dict:
    return client.get("/api/state").json()["billing"]


def _user_row(email: str) -> rdb.User:
    with rdb.new_session() as session:
        return session.query(rdb.User).filter(rdb.User.email == email).one()


def _set_user(email: str, **fields) -> None:
    with rdb.new_session() as session:
        user = session.query(rdb.User).filter(rdb.User.email == email).one()
        for key, value in fields.items():
            setattr(user, key, value)
        session.commit()


def test_new_account_starts_on_trial(make_client):
    register(make_client(), "admin@example.com")
    user = make_client()
    register(user, "user@example.com")
    assert _billing(user) == {
        "kind": "trial",
        "plan": None,
        "remaining": TRIAL_CREDITS,
        "renewsAt": None,
    }


def test_trial_credits_decrement_and_block(make_client):
    register(make_client(), "admin@example.com")
    user = make_client()
    register(user, "user@example.com")

    _create_project(user, "First Video")
    assert _billing(user)["remaining"] == TRIAL_CREDITS - 1

    for i in range(TRIAL_CREDITS - 1):
        _create_project(user, f"Video {i}")
    assert _billing(user) == {
        "kind": "blocked",
        "plan": None,
        "remaining": 0,
        "renewsAt": None,
    }
    res = user.post("/api/projects", json={"title": "Blocked", "script": "text"})
    assert res.status_code == 402


def test_deleting_a_project_does_not_refund_a_trial_credit(make_client):
    register(make_client(), "admin@example.com")
    user = make_client()
    register(user, "user@example.com")
    slug = _create_project(user, "Throwaway")
    assert user.delete(f"/api/projects/{slug}").status_code == 200
    assert _billing(user)["remaining"] == TRIAL_CREDITS - 1


def test_admin_entitlement_is_unlimited(client):
    register(client, "admin@example.com")
    assert _billing(client)["kind"] == "admin"


def test_billing_catalog_endpoint(client):
    register(client, "admin@example.com")
    body = client.get("/api/billing").json()
    assert body["plans"] == PLANS
    assert body["devCheckout"] is False
    assert body["entitlement"]["kind"] == "admin"


def test_checkout_returns_503_without_a_payment_provider(make_client):
    register(make_client(), "admin@example.com")
    user = make_client()
    register(user, "user@example.com")
    res = user.post("/api/billing/checkout", json={"plan": "starter"})
    assert res.status_code == 503
    assert "not configured" in res.json()["detail"]


def test_checkout_rejects_unknown_plan(saas_env, make_client, monkeypatch):
    _enable_dev_checkout(saas_env, monkeypatch)
    user = make_client()
    register(user, "user@example.com")
    assert user.post("/api/billing/checkout", json={"plan": "gold"}).status_code == 422


def test_dev_checkout_activates_a_subscription(saas_env, make_client, monkeypatch):
    _enable_dev_checkout(saas_env, monkeypatch)
    register(make_client(), "admin@example.com")
    user = make_client()
    register(user, "user@example.com")
    _set_user("user@example.com", trial_credits=0)
    assert _billing(user)["kind"] == "blocked"

    res = user.post("/api/billing/checkout", json={"plan": "starter"})
    assert res.status_code == 200
    ent = res.json()["entitlement"]
    assert ent["kind"] == "subscription"
    assert ent["plan"] == "starter"
    assert ent["remaining"] == PLANS["starter"]["videos_per_month"]
    assert ent["renewsAt"] > time.time()

    # Creation works again, and trial credits stay untouched at 0 —
    # subscription usage is derived from the monthly project count.
    _create_project(user, "Back In Business")
    assert _billing(user)["remaining"] == PLANS["starter"]["videos_per_month"] - 1
    assert _user_row("user@example.com").trial_credits == 0


def test_subscription_monthly_allowance_blocks_past_quota(
    saas_env, make_client, monkeypatch
):
    _enable_dev_checkout(saas_env, monkeypatch)
    register(make_client(), "admin@example.com")
    user = make_client()
    register(user, "user@example.com")
    _set_user("user@example.com", trial_credits=0)
    user.post("/api/billing/checkout", json={"plan": "starter"})

    # Shrink the plan's allowance so the test doesn't create 10 projects.
    monkeypatch.setitem(PLANS["starter"], "videos_per_month", 2)
    _create_project(user, "One")
    _create_project(user, "Two")
    res = user.post("/api/projects", json={"title": "Three", "script": "text"})
    assert res.status_code == 402
    assert "monthly limit" in res.json()["detail"]


def test_expired_subscription_blocks(make_client, saas_env, monkeypatch):
    _enable_dev_checkout(saas_env, monkeypatch)
    register(make_client(), "admin@example.com")
    user = make_client()
    register(user, "user@example.com")
    _set_user(
        "user@example.com",
        trial_credits=0,
        tier="starter",
        subscription_expires_at=time.time() - 60,
    )
    assert _billing(user)["kind"] == "blocked"
    res = user.post("/api/projects", json={"title": "Lapsed", "script": "text"})
    assert res.status_code == 402


def test_grant_requires_admin(make_client):
    register(make_client(), "admin@example.com")
    user = make_client()
    register(user, "user@example.com")
    res = user.post(
        "/api/billing/grant",
        json={"email": "user@example.com", "plan": "starter", "months": 1},
    )
    assert res.status_code == 403


def test_admin_grant_activates_subscription(make_client):
    admin = make_client()
    register(admin, "admin@example.com")
    user = make_client()
    register(user, "user@example.com")

    res = admin.post(
        "/api/billing/grant",
        json={"email": "user@example.com", "plan": "creator", "months": 2},
    )
    assert res.status_code == 200
    assert res.json()["entitlement"]["kind"] == "subscription"
    assert _billing(user)["plan"] == "creator"

    missing = admin.post(
        "/api/billing/grant",
        json={"email": "ghost@example.com", "plan": "creator", "months": 1},
    )
    assert missing.status_code == 404
