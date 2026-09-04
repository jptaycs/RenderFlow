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
    # Whether pipeline/script.py::split_script_local puts every scene on
    # camera as a talking avatar (default, matches the documented "host
    # speaks in every scene" behavior) or leaves scenes as plain narration
    # (image/broll + voice, no host) — the local splitter is the only
    # script path that always forced an avatar; topic-mode generation lets
    # Claude decide per scene. Off restores the narration-only look.
    local_avatar_enabled: bool
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
    # Per-format override ("" = same as broll_provider), added 2026-09.
    # RENDERFLOW_BROLL_PROVIDER=labs69 (real AI generation, ~60-90s/clip —
    # see generate_broll) makes total render time scale with scene count
    # regardless of format: a 1-minute Short still runs 6-12 scenes, the
    # same per-scene cost as the first 6-12 scenes of a full-length video.
    # Set to "pexels-video" (free stock search, ~1-2s/scene) to make
    # Shorts specifically fast, independent of whatever the main
    # broll_provider is for landscape.
    shorts_broll_provider: str = ""
    # How many scenes' B-roll generate_broll fetches concurrently, added
    # 2026-09 — each find_clip call is a long, mostly-idle network wait
    # (labs69 real generation: ~60-90s/clip), so running them one at a
    # time meant total B-roll time scaled linearly with scene count for no
    # good reason. 3 is a modest default: real speedup without hammering
    # either provider's backend. 1 restores the old fully-sequential
    # behavior.
    broll_concurrency: int = 3
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
    # Burned-in captions: on by default. Off is a legitimate style choice
    # (e.g. real AI-generated video footage from labs69 already looks like
    # produced B-roll and can read as cluttered with captions on top) —
    # skips assets.generate_subtitles() entirely, so _subtitle_chunks()
    # (render.py) just sees an empty/PENDING ref and renders with no
    # overlay. Nothing else treats subtitles as required for completeness
    # (make_video._incomplete_scenes, api._scene_assets) — no migration or
    # UI change needed for existing projects either way.
    subtitles_enabled: bool = True
    # Per-format override (added 2026-09), None = follow subtitles_enabled
    # above. Shorts and landscape want opposite defaults for a real
    # reason, not just user taste: Shorts are watched sound-off while
    # scrolling far more than landscape is, so captions matter a lot more
    # there even for a user who prefers them off on their (sound-on,
    # cinematic-feeling) landscape videos.
    shorts_subtitles_enabled: bool | None = None
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
    # LOCAL DEV/TEST ONLY: makes tasks.py run Celery in "eager" mode
    # (task_always_eager) so `run_pipeline.delay(job.id)` executes
    # synchronously in the calling process instead of publishing to Redis
    # for a separate worker — lets the dashboard run with no Redis and no
    # `celery worker` process at all. The pipeline itself still runs as a
    # real `make_video.py` subprocess either way (see tasks.py), so this
    # only changes *dispatch*, not where the heavy work happens — but it
    # does mean the HTTP request blocks until that subprocess exits, which
    # is fine for a one-off manual test and wrong for anything else. Never
    # set this in production (refused at startup, see api.startup).
    celery_eager: bool = False

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
            local_avatar_enabled=os.getenv("RENDERFLOW_LOCAL_AVATAR", "1").lower()
            in ("1", "true", "yes"),
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
            celery_eager=os.getenv("RENDERFLOW_CELERY_EAGER", "").lower()
            in ("1", "true", "yes"),
            env=os.getenv("RENDERFLOW_ENV", "dev"),
            broll_provider=os.getenv("RENDERFLOW_BROLL_PROVIDER", ""),
            shorts_broll_provider=os.getenv("RENDERFLOW_SHORTS_BROLL_PROVIDER", ""),
            # Clamped to >=1 — ThreadPoolExecutor(max_workers=0) raises
            # ValueError uncaught, which would otherwise crash an entire
            # run (already-generated, already-paid-for images included)
            # instead of degrading gracefully like every other B-roll
            # failure mode. Found in a full-app scan 2026-09.
            broll_concurrency=max(1, int(os.getenv("RENDERFLOW_BROLL_CONCURRENCY", "3"))),
            music_dir=Path(os.getenv("RENDERFLOW_MUSIC_DIR", "music")),
            music_volume=float(os.getenv("RENDERFLOW_MUSIC_VOLUME", "0.20")),
            transition=os.getenv("RENDERFLOW_TRANSITION", "fade"),
            intro_outro=os.getenv("RENDERFLOW_INTRO_OUTRO", "1").lower()
            in ("1", "true", "yes"),
            channel_name=os.getenv("RENDERFLOW_CHANNEL_NAME", ""),
            subtitles_enabled=os.getenv("RENDERFLOW_SUBTITLES", "1").lower()
            in ("1", "true", "yes"),
            shorts_subtitles_enabled=(
                None
                if (raw := os.getenv("RENDERFLOW_SHORTS_SUBTITLES", "")) == ""
                else raw.lower() in ("1", "true", "yes")
            ),
        )
