"""Environment-driven configuration. Never hardcode API keys."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Billing model (enforced in api.create_project via billing.entitlement):
# every new account gets TRIAL_CREDITS videos; after that an active
# subscription (User.tier in PLANS + unexpired subscription_expires_at) is
# required, with a per-calendar-month video allowance. Display values
# (price, label) live here too so the pricing UI has one source of truth.
TRIAL_CREDITS = 3
PLANS: dict[str, dict] = {
    "starter": {"label": "Starter", "price_usd": 19, "videos_per_month": 10},
    "creator": {"label": "Creator", "price_usd": 49, "videos_per_month": 30},
}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    image_provider: str
    # Thumbnail images may come from different providers than scene images
    # ("" = same as image_provider): the clickbait background wants AI
    # generation (stock search rarely has the dramatic saturated look),
    # while the reaction face can be a real stock photo.
    thumbnail_bg_provider: str
    thumbnail_reaction_provider: str
    tts_provider: str
    avatar_provider: str
    llm_model: str
    tts_voice: str
    tts_length_scale: float
    tts_sentence_pause: float
    avatar_image: Path | None
    projects_dir: Path
    # SaaS layer (api/auth/worker only — the pipeline itself never touches
    # the DB or Redis). secret_key signs session cookies; the API refuses to
    # start without one (make_video.py runs fine with it empty).
    database_url: str
    redis_url: str
    secret_key: str
    # "dev" (default) or "production". Production makes session cookies
    # Secure (TLS-only) and the API refuses to start with any dev
    # convenience flag set or the default DB password (see api.startup).
    env: str = "dev"
    # --- Output polish (Phase 4) ---
    # Stock-video B-roll provider ("" = disabled, "pexels-video"). Eligible
    # full-frame scenes use a real stock clip instead of still+motion.
    broll_provider: str = ""
    # Background music: directory of royalty-free tracks (empty/missing dir
    # = no music) and the pre-duck music volume (0..1).
    music_dir: Path = Path("music")
    music_volume: float = 0.20
    # Scene transitions: "fade" (video-only dip-through-black inside the
    # scene pause) or "none". NEVER an audio crossfade — see render.py.
    transition: str = "fade"
    # Intro/outro cards: on by default; channel name shown on both cards
    # when set.
    intro_outro: bool = True
    channel_name: str = ""
    # When both are set, the login page shows a one-click "Developer login"
    # button that prefills these credentials and submits them through the
    # normal password-checked login — there is no bypass endpoint. Local
    # development convenience only; leave empty on a deployed instance.
    dev_login_email: str = ""
    dev_login_password: str = ""
    # Enables the local checkout simulator (POST /api/billing/checkout
    # activates a plan instantly, no payment). This is the seam where
    # Stripe/Paddle plugs in later; leave unset on a deployed instance —
    # without it the endpoint returns 503 "payments not configured".
    dev_checkout: bool = False

    @classmethod
    def load(cls) -> "Settings":
        # override=True: .env is the source of truth, so edits apply to the
        # next run without restarting the API server (whose inherited env
        # would otherwise pin subprocesses to stale values).
        load_dotenv(override=True)
        return cls(
            llm_provider=os.getenv("RENDERFLOW_LLM_PROVIDER", "claude"),
            image_provider=os.getenv("RENDERFLOW_IMAGE_PROVIDER", "flux-replicate"),
            thumbnail_bg_provider=os.getenv("RENDERFLOW_THUMBNAIL_BG_PROVIDER", ""),
            thumbnail_reaction_provider=os.getenv(
                "RENDERFLOW_THUMBNAIL_REACTION_PROVIDER", ""
            ),
            tts_provider=os.getenv("RENDERFLOW_TTS_PROVIDER", "elevenlabs"),
            avatar_provider=os.getenv("RENDERFLOW_AVATAR_PROVIDER", "ffmpeg-still"),
            llm_model=os.getenv("RENDERFLOW_LLM_MODEL", "claude-opus-4-8"),
            tts_voice=os.getenv("RENDERFLOW_TTS_VOICE", "21m00Tcm4TlvDq8ikWAM"),
            tts_length_scale=float(os.getenv("RENDERFLOW_TTS_LENGTH_SCALE", "1.4")),
            tts_sentence_pause=float(os.getenv("RENDERFLOW_TTS_SENTENCE_PAUSE", "0.45")),
            avatar_image=(
                Path(value)
                if (value := os.getenv("RENDERFLOW_AVATAR_IMAGE"))
                else None
            ),
            projects_dir=Path(os.getenv("RENDERFLOW_PROJECTS_DIR", "projects")),
            database_url=os.getenv(
                "RENDERFLOW_DATABASE_URL",
                "postgresql+psycopg://renderflow:renderflow@127.0.0.1:5433/renderflow",
            ),
            redis_url=os.getenv("RENDERFLOW_REDIS_URL", "redis://127.0.0.1:6380/0"),
            secret_key=os.getenv("RENDERFLOW_SECRET_KEY", ""),
            dev_login_email=os.getenv("RENDERFLOW_DEV_LOGIN_EMAIL", ""),
            dev_login_password=os.getenv("RENDERFLOW_DEV_LOGIN_PASSWORD", ""),
            dev_checkout=os.getenv("RENDERFLOW_DEV_CHECKOUT", "").lower()
            in ("1", "true", "yes"),
            env=os.getenv("RENDERFLOW_ENV", "dev"),
            broll_provider=os.getenv("RENDERFLOW_BROLL_PROVIDER", ""),
            music_dir=Path(os.getenv("RENDERFLOW_MUSIC_DIR", "music")),
            music_volume=float(os.getenv("RENDERFLOW_MUSIC_VOLUME", "0.20")),
            transition=os.getenv("RENDERFLOW_TRANSITION", "fade"),
            intro_outro=os.getenv("RENDERFLOW_INTRO_OUTRO", "1").lower()
            in ("1", "true", "yes"),
            channel_name=os.getenv("RENDERFLOW_CHANNEL_NAME", ""),
        )
