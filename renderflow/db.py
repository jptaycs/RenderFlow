"""SaaS-layer database: users, project ownership, pipeline jobs.

Only the API server (renderflow/api.py) and the Celery worker
(renderflow/tasks.py) touch this — the pipeline itself (make_video.py and
everything under pipeline/) stays filesystem-only and user-agnostic.

Timestamps are float epoch seconds, matching the file-mtime convention used
throughout the rest of the codebase. Column types stick to portable ones
(JSON, not JSONB) so tests can run the same models on in-memory SQLite
without Docker.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sqlalchemy import JSON, Engine, ForeignKey, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from renderflow.config import TRIAL_CREDITS, Settings


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    # The subscription plan: "free" = no plan (trial-only account);
    # "starter"/"creator" = a PLANS key. Only meaningful together with
    # subscription_expires_at — see billing.subscription_active.
    tier: Mapped[str] = mapped_column(String(32), default="free")
    is_admin: Mapped[bool] = mapped_column(default=False)
    # Every new account gets TRIAL_CREDITS videos before a subscription is
    # required; consumed permanently on project creation while unsubscribed
    # (deleting a project does not refund — blocks create/delete/create).
    trial_credits: Mapped[int] = mapped_column(default=TRIAL_CREDITS)
    # Epoch seconds; the subscription is active while this is in the future.
    # Set by checkout (dev simulator today, payment-provider webhook later)
    # and by the admin grant endpoint.
    subscription_expires_at: Mapped[float | None] = mapped_column(default=None)
    created_at: Mapped[float] = mapped_column(default=time.time)

    projects: Mapped[list["Project"]] = relationship(back_populates="owner")


class Project(Base):
    __tablename__ = "projects"
    # Slugs are unique per owner, not globally — two users can both make
    # "The Shortest War in History" without leaking each other's existence.
    __table_args__ = (UniqueConstraint("owner_id", "slug"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    slug: Mapped[str] = mapped_column(String(255))
    title: Mapped[str] = mapped_column(String(512))
    # Absolute path to the project directory. New projects live under
    # projects/u{owner_id}/{slug}; adopted legacy projects keep their
    # original projects/{slug} location so paths inside scenes.json stay
    # valid (files are never moved).
    dir_path: Mapped[str] = mapped_column(String(1024), unique=True)
    created_at: Mapped[float] = mapped_column(default=time.time)

    owner: Mapped[User] = relationship(back_populates="projects")
    jobs: Mapped[list["Job"]] = relationship(back_populates="project")


ACTIVE_JOB_STATUSES = ("queued", "running")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # create|resume|regenerate|thumbnail|layout
    # Arguments for make_video.py, minus --slug/--projects-dir which the
    # worker derives from the project row itself.
    argv: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    # queued -> running -> succeeded | failed | cancelled
    celery_task_id: Mapped[str | None] = mapped_column(String(64), default=None)
    pid: Mapped[int | None] = mapped_column(default=None)
    created_at: Mapped[float] = mapped_column(default=time.time)
    started_at: Mapped[float | None] = mapped_column(default=None)
    finished_at: Mapped[float | None] = mapped_column(default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    project: Mapped[Project] = relationship(back_populates="jobs")


# ---------------------------------------------------------------------------
# Autopilot (renderflow/autopilot/): one YouTube channel, one video a day.
# ---------------------------------------------------------------------------


class AutopilotChannel(Base):
    """Settings for the daily trend-to-YouTube loop. One row per owner (the
    user runs a single channel covering every niche)."""

    __tablename__ = "autopilot_channels"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    name: Mapped[str] = mapped_column(String(255), default="Underrated Facts")
    enabled: Mapped[bool] = mapped_column(default=False)
    # Local hour (0-23, the worker PC's clock) the daily episode starts.
    run_hour: Mapped[int] = mapped_column(default=6)
    # Shorts per day (1-5). They run one after another: the next starts once
    # the previous is waiting for approval.
    episodes_per_day: Mapped[int] = mapped_column(default=1)
    length_minutes: Mapped[float] = mapped_column(default=10.0)
    # YouTube privacy on upload. Unaudited API projects are forced private
    # by YouTube regardless — see providers/publish/youtube.py.
    privacy: Mapped[str] = mapped_column(String(16), default="private")
    # [{"name", "kind": "fact"|"howto", "description"}] — see
    # autopilot.core.DEFAULT_NICHES.
    niches: Mapped[list] = mapped_column(JSON, default=list)
    # Last time the bot daemon checked in (dashboard shows online/offline).
    heartbeat_at: Mapped[float | None] = mapped_column(default=None)
    created_at: Mapped[float] = mapped_column(default=time.time)


# planning -> generating -> ready -> uploading -> uploaded
#                                 ↘ skipped        ↘ failed (retry = approve again)
# Any step can land on failed; "generating" is re-entered for a thumbnail redo.
# "replaced" = the user asked for a new topic; it doesn't use up a daily slot.
EPISODE_ACTIVE_STATUSES = ("planning", "generating", "uploading")


class Episode(Base):
    """One day's video: the trend it came from, its project, SEO, upload."""

    __tablename__ = "autopilot_episodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("autopilot_channels.id"), index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), default=None)
    day: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD, local
    status: Mapped[str] = mapped_column(String(16), default="planning", index=True)
    niche: Mapped[str] = mapped_column(String(255), default="")
    kind: Mapped[str] = mapped_column(String(16), default="")  # fact|howto
    trend: Mapped[str] = mapped_column(String(512), default="")
    topic: Mapped[str] = mapped_column(String(512), default="")
    angle: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str] = mapped_column(String(512), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    # Every scored candidate from the picker — shown in the dashboard so
    # the choice is reviewable, and used by "new topic" to try the next one.
    candidates: Mapped[list] = mapped_column(JSON, default=list)
    llm_cost: Mapped[float] = mapped_column(default=0.0)
    # 69labs credits, measured as balance deltas around each of the
    # episode's render jobs (exact: jobs run one at a time). credits_before
    # is the snapshot taken when a job starts; used/left are
    # {"images", "videos", "motion", "tools"} -> int. NULL on old rows.
    credits_before: Mapped[dict | None] = mapped_column(JSON, default=None)
    credits_used: Mapped[dict | None] = mapped_column(JSON, default=None)
    credits_left: Mapped[dict | None] = mapped_column(JSON, default=None)
    youtube_video_id: Mapped[str | None] = mapped_column(String(32), default=None)
    youtube_url: Mapped[str | None] = mapped_column(String(255), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[float] = mapped_column(default=time.time)
    updated_at: Mapped[float] = mapped_column(default=time.time)
    uploaded_at: Mapped[float | None] = mapped_column(default=None)


# ---------------------------------------------------------------------------
# Engine / session plumbing
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def configure(engine: Engine) -> None:
    """Bind the module to an engine (tests inject in-memory SQLite here)."""
    global _engine, _session_factory
    _engine = engine
    _session_factory = sessionmaker(bind=engine, expire_on_commit=False)


def get_engine() -> Engine:
    if _engine is None:
        url = Settings.load().database_url
        # A file-based (non-Postgres) DATABASE_URL is a dev/small-deployment
        # path — FastAPI's threadpool for sync routes, plus eager-mode
        # Celery, can hand a session's connection across threads, which
        # sqlite3 refuses by default. Harmless for Postgres, so only set it
        # for sqlite URLs. Tests bypass this entirely via configure().
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        configure(create_engine(url, connect_args=connect_args))
    return _engine


def new_session() -> Session:
    get_engine()
    assert _session_factory is not None
    return _session_factory()


def get_db():
    """FastAPI dependency: one session per request, commit on success."""
    session = new_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create tables that don't exist yet, then apply hand-rolled column
    migrations (Alembic deferred until post-launch — see CLAUDE.md).

    create_all never alters existing tables, so columns added after a table
    first shipped need explicit idempotent ALTERs. The users ones are
    Postgres only (the test suite always creates a fresh SQLite schema);
    _LATE_COLUMNS runs on both, since the Windows PC's live DB is SQLite."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        from sqlalchemy import text

        with engine.begin() as conn:
            # Phase 2 (billing): added to users after the table shipped.
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                f"trial_credits INTEGER NOT NULL DEFAULT {int(TRIAL_CREDITS)}"
            ))
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                "subscription_expires_at DOUBLE PRECISION"
            ))
    _add_missing_columns(engine)


# Columns added to autopilot tables after they shipped. Checked by
# inspection rather than IF NOT EXISTS so they also reach the Windows PC's
# SQLite database (SQLite has no ADD COLUMN IF NOT EXISTS).
_LATE_COLUMNS = (
    ("autopilot_channels", "episodes_per_day", "INTEGER NOT NULL DEFAULT 1"),
    ("autopilot_episodes", "credits_before", "JSON"),
    ("autopilot_episodes", "credits_used", "JSON"),
    ("autopilot_episodes", "credits_left", "JSON"),
)


def _add_missing_columns(engine: Engine) -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, ddl in _LATE_COLUMNS:
            existing = {c["name"] for c in inspector.get_columns(table)}
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


# ---------------------------------------------------------------------------
# Helpers shared by api.py and tasks.py
# ---------------------------------------------------------------------------


def active_job(session: Session, project_id: int) -> Job | None:
    return (
        session.query(Job)
        .filter(Job.project_id == project_id, Job.status.in_(ACTIVE_JOB_STATUSES))
        .order_by(Job.created_at)
        .first()
    )


def adopt_legacy_projects(session: Session, admin: User, projects_dir: Path) -> int:
    """Register pre-auth on-disk projects (projects/<slug>) to the admin.

    Files are never moved — dir_path records where each project already
    lives, so absolute asset paths stored inside scenes.json stay valid.
    Runs when the first user registers; safe to re-run (dir_path is unique
    and already-registered paths are skipped).
    """
    known = {row.dir_path for row in session.query(Project.dir_path).all()}
    adopted = 0
    if not projects_dir.exists():
        return 0
    for entry in sorted(projects_dir.iterdir()):
        scenes_json = entry / "script" / "scenes.json"
        if not scenes_json.exists() or str(entry.resolve()) in known:
            continue
        try:
            title = json.loads(scenes_json.read_text()).get("title") or entry.name
        except (json.JSONDecodeError, OSError):
            title = entry.name
        session.add(
            Project(
                owner_id=admin.id,
                slug=entry.name,
                title=title,
                dir_path=str(entry.resolve()),
                created_at=scenes_json.stat().st_mtime,
            )
        )
        adopted += 1
    return adopted
