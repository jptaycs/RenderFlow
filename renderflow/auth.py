"""Accounts and sessions: bcrypt passwords, signed session cookie.

Self-contained email+password auth — no third-party auth service, works
fully offline. The session is an itsdangerous-signed cookie carrying the
user id (httponly, samesite=lax, 30-day expiry). No email
verification/password reset yet: those need an email-sending service and
are deferred until deployment.
"""

from __future__ import annotations

import re

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel
from sqlalchemy.orm import Session

from renderflow.config import Settings
from renderflow.db import User, adopt_legacy_projects, get_db

COOKIE_NAME = "renderflow_session"
SESSION_MAX_AGE_SEC = 30 * 24 * 3600

router = APIRouter(prefix="/api/auth")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def _serializer() -> URLSafeTimedSerializer:
    secret = Settings.load().secret_key
    if not secret:
        # api.py also refuses to boot without one; this is the backstop so a
        # misconfigured deployment can never mint unsigned-in-effect cookies.
        raise RuntimeError("RENDERFLOW_SECRET_KEY is not set")
    return URLSafeTimedSerializer(secret, salt="renderflow.session")


def set_session_cookie(response: Response, user_id: int) -> None:
    response.set_cookie(
        COOKIE_NAME,
        _serializer().dumps(user_id),
        max_age=SESSION_MAX_AGE_SEC,
        httponly=True,
        samesite="lax",
        # Production serves over TLS via Caddy — the cookie must never ride
        # plain http there. Dev is plain http on 127.0.0.1, so not secure.
        secure=Settings.load().env == "production",
    )


def current_user(request: Request, session: Session = Depends(get_db)) -> User:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(401, "not signed in")
    try:
        user_id = _serializer().loads(token, max_age=SESSION_MAX_AGE_SEC)
    except BadSignature:
        raise HTTPException(401, "invalid or expired session")
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(401, "invalid or expired session")
    return user


class Credentials(BaseModel):
    email: str
    password: str


def _user_view(user: User) -> dict:
    return {"email": user.email, "tier": user.tier, "isAdmin": user.is_admin}


@router.post("/register", status_code=201)
def register(
    body: Credentials, response: Response, session: Session = Depends(get_db)
) -> dict:
    email = body.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(422, "enter a valid email address")
    # bcrypt only hashes the first 72 bytes; reject rather than silently
    # truncate.
    if not 8 <= len(body.password.encode()) <= 72:
        raise HTTPException(422, "password must be 8-72 characters")
    if session.query(User).filter(User.email == email).first():
        raise HTTPException(409, "an account with this email already exists")

    first_user = session.query(User).first() is None
    user = User(
        email=email,
        password_hash=hash_password(body.password),
        is_admin=first_user,
    )
    session.add(user)
    session.flush()  # assign user.id
    if first_user:
        # The very first account (you) adopts every pre-auth project already
        # on disk, so nothing existing disappears from the dashboard.
        adopt_legacy_projects(session, user, Settings.load().projects_dir)
    set_session_cookie(response, user.id)
    return _user_view(user)


@router.post("/login")
def login(
    body: Credentials, response: Response, session: Session = Depends(get_db)
) -> dict:
    email = body.email.strip().lower()
    user = session.query(User).filter(User.email == email).first()
    # Same error for unknown email and wrong password — no account probing.
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "incorrect email or password")
    set_session_cookie(response, user.id)
    return _user_view(user)


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict:
    return _user_view(user)


@router.get("/dev-login")
def dev_login_prefill() -> dict:
    """Prefill values for the login page's "Developer login" button.

    Purely a convenience for local development: when
    RENDERFLOW_DEV_LOGIN_EMAIL/_PASSWORD are both set in .env, the button
    appears and submits these credentials through the NORMAL /login (and
    /register on a fresh DB) endpoints — the password is checked like any
    other; no bypass endpoint exists. With the env vars unset (any deployed
    instance), this returns enabled=false and exposes nothing.
    """
    settings = Settings.load()
    if not (settings.dev_login_email and settings.dev_login_password):
        return {"enabled": False}
    return {
        "enabled": True,
        "email": settings.dev_login_email,
        "password": settings.dev_login_password,
    }
