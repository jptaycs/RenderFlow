"""Subscription entitlement: trial credits, plans, and the checkout seam.

The model (see CLAUDE.md): every new account gets TRIAL_CREDITS videos;
after that, creating videos requires an active subscription (User.tier is a
PLANS key and subscription_expires_at is in the future), which grants a
per-calendar-month allowance derived by counting projects — never a stored
counter. Admins are always unlimited. Only project *creation* is gated:
resume/regenerate/layout/thumbnail on an existing project stay free — the
credit bought that video, fixing it shouldn't cost more.

Real payments are not integrated yet. POST /checkout is the seam where
Stripe/Paddle plugs in: today it either simulates instantly
(RENDERFLOW_DEV_CHECKOUT=1, local development only) or returns 503; later
it will return a hosted-checkout URL, and the provider's webhook will set
the same two fields (tier, subscription_expires_at) the simulator sets.
"""

from __future__ import annotations

import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from renderflow.auth import current_user
from renderflow.config import PLANS, Settings
from renderflow.db import Project, User, get_db

SUBSCRIPTION_PERIOD_SEC = 30 * 24 * 3600

router = APIRouter(prefix="/api/billing")


def month_start_epoch() -> float:
    now = datetime.now()
    return datetime(now.year, now.month, 1).timestamp()


def subscription_active(user: User) -> bool:
    return (
        user.tier in PLANS
        and user.subscription_expires_at is not None
        and user.subscription_expires_at > time.time()
    )


def _videos_this_month(session: Session, user: User) -> int:
    return (
        session.query(Project)
        .filter(Project.owner_id == user.id, Project.created_at >= month_start_epoch())
        .count()
    )


def entitlement(session: Session, user: User) -> dict:
    """What this account may do right now — the one source of truth,
    consumed by create_project (enforcement) and the dashboard (display)."""
    if user.is_admin:
        return {"kind": "admin", "plan": None, "remaining": None, "renewsAt": None}
    if subscription_active(user):
        quota = PLANS[user.tier]["videos_per_month"]
        used = _videos_this_month(session, user)
        return {
            "kind": "subscription",
            "plan": user.tier,
            "remaining": max(quota - used, 0),
            "renewsAt": user.subscription_expires_at,
        }
    if user.trial_credits > 0:
        return {
            "kind": "trial",
            "plan": None,
            "remaining": user.trial_credits,
            "renewsAt": None,
        }
    return {"kind": "blocked", "plan": None, "remaining": 0, "renewsAt": None}


def consume_credit(session: Session, user: User) -> None:
    """Record one video creation. Trial credits decrement permanently;
    subscription usage is derived from the project count, so there is
    nothing to write for it; admins consume nothing."""
    if user.is_admin or subscription_active(user):
        return
    if user.trial_credits > 0:
        user.trial_credits -= 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
def billing_state(
    user: User = Depends(current_user), session: Session = Depends(get_db)
) -> dict:
    return {
        "entitlement": entitlement(session, user),
        "plans": PLANS,
        # Tells the pricing UI whether checkout actually works (local
        # simulator) or should show "payments coming soon".
        "devCheckout": Settings.load().dev_checkout,
    }


class CheckoutRequest(BaseModel):
    plan: str


@router.post("/checkout")
def checkout(
    body: CheckoutRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict:
    """The payment-provider seam (see module docstring)."""
    if body.plan not in PLANS:
        raise HTTPException(422, f"unknown plan {body.plan!r}")
    if not Settings.load().dev_checkout:
        raise HTTPException(
            503,
            "payments are not configured on this server yet — contact the operator",
        )
    user.tier = body.plan
    user.subscription_expires_at = time.time() + SUBSCRIPTION_PERIOD_SEC
    return {"entitlement": entitlement(session, user)}


class GrantRequest(BaseModel):
    email: str
    plan: str
    months: int = 1


@router.post("/grant")
def grant(
    body: GrantRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
) -> dict:
    """Admin-only manual subscription grant (support/comps — and the
    operator's only lever until a real payment provider is wired in)."""
    if not user.is_admin:
        raise HTTPException(403, "admin only")
    if body.plan not in PLANS:
        raise HTTPException(422, f"unknown plan {body.plan!r}")
    if not 1 <= body.months <= 24:
        raise HTTPException(422, "months must be 1-24")
    target = session.query(User).filter(User.email == body.email.strip().lower()).first()
    if target is None:
        raise HTTPException(404, f"no user {body.email!r}")
    target.tier = body.plan
    base = max(target.subscription_expires_at or 0, time.time())
    target.subscription_expires_at = base + body.months * SUBSCRIPTION_PERIOD_SEC
    return {"email": target.email, "entitlement": entitlement(session, target)}
