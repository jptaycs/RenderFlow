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
        configure(create_engine(Settings.load().database_url))
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
    first shipped need explicit idempotent ALTERs. Postgres only: the test
    suite always creates a fresh SQLite schema, which create_all fully
    covers (and SQLite lacks ADD COLUMN IF NOT EXISTS anyway)."""
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
