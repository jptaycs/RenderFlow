"""Autopilot: trend parsing, topic choice, episode lifecycle, bot, API.

No live services: LLM/trends are stubs, Celery .delay() calls are
captured by monkeypatching, the DB is the in-memory SQLite from conftest.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from renderflow import db as rdb
from renderflow.autopilot import bot, core, jobs
from renderflow.providers.base import LLMResult, TrendItem
from renderflow.providers.messaging.telegram import normalize_update
from renderflow.providers.publish.youtube import build_metadata, clean_tags
from renderflow.providers.trends.google_trends import parse_feed
from renderflow.providers.trends.youtube_trending import parse_response

from tests.conftest import register

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:ht="https://trends.google.com/trending/rss" version="2.0"><channel>
<item><title>first frost</title><ht:approx_traffic>200K+</ht:approx_traffic>
<link>https://trends.google.com/x</link>
<ht:news_item><ht:news_item_title>Frost warning issued for the Midwest</ht:news_item_title></ht:news_item>
</item>
<item><title>pumpkin spice</title><ht:approx_traffic>50K+</ht:approx_traffic></item>
</channel></rss>"""

PICK = {
    "candidates": [
        {"niche": "Garden & backyard", "kind": "howto", "trend": "first frost",
         "topic": "Protecting tomatoes from the first frost", "angle": "old tricks",
         "working_title": "The Old Trick That Saves Tomatoes From Frost", "score": 90, "reason": "frost trending"},
        {"niche": "Forgotten history", "kind": "fact", "trend": "pumpkin spice",
         "topic": "The surprising origin of pumpkin pie spice", "angle": "colonial",
         "working_title": "Pumpkin Spice Is Older Than America", "score": 70, "reason": "seasonal"},
    ]
}

SCRIPT = " ".join(["Frost can arrive overnight and ruin a whole season of work."] * 15)
EPISODE = {
    "title": "The Old Trick That Saves Tomatoes From Frost",
    "script": SCRIPT,
    "description": "Frost is coming — here is the trick.\n\nIn this video:\n• Old sheets",
    "tags": ["frost", "tomatoes", "gardening tips"],
    "hashtags": ["#gardening", "frost", "#tomatoes"],
}


class ScriptedLLM:
    name = "scripted"

    def __init__(self, *responses):
        self.responses = [json.dumps(r) for r in responses]
        self.calls = []

    def complete(self, system, prompt, **params):
        self.calls.append(prompt)
        return LLMResult(text=self.responses.pop(0), provider=self.name, cost=0.01)


class StaticTrends:
    name = "static"

    def fetch(self, **params):
        return [TrendItem(title="first frost", source="static", traffic="200K+")]


@pytest.fixture
def captured(monkeypatch):
    """Record every Celery .delay() the autopilot makes instead of queueing."""
    calls = SimpleNamespace(pipeline=[], start=[], package=[], upload=[])

    def fake(bucket):
        def delay(*args):
            getattr(calls, bucket).append(args)
            return SimpleNamespace(id=f"stub-{bucket}-{len(getattr(calls, bucket))}")
        def apply_async(args, countdown=0):
            return delay(*args)
        return SimpleNamespace(delay=delay, apply_async=apply_async)

    monkeypatch.setattr(jobs, "run_pipeline", fake("pipeline"))
    monkeypatch.setattr(jobs, "start_episode", fake("start"))
    monkeypatch.setattr(jobs, "package_episode", fake("package"))
    monkeypatch.setattr(jobs, "upload_episode", fake("upload"))
    monkeypatch.setattr(jobs, "messenger_or_none", lambda: None)
    return calls


def _admin_channel(session, enabled=True, run_hour=6):
    user = rdb.User(email="owner@example.com", password_hash="x", is_admin=True)
    session.add(user)
    session.flush()
    channel = rdb.AutopilotChannel(
        owner_id=user.id, niches=list(core.DEFAULT_NICHES), enabled=enabled, run_hour=run_hour
    )
    session.add(channel)
    session.commit()
    return user, channel


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def test_google_trends_feed_parses_titles_traffic_and_news():
    items = parse_feed(FEED)
    assert [i.title for i in items] == ["first frost", "pumpkin spice"]
    assert items[0].traffic == "200K+"
    assert items[0].context == ["Frost warning issued for the Midwest"]


def test_youtube_trending_parses_views():
    items = parse_response({"items": [{"id": "abc", "snippet": {"title": "Big video", "channelTitle": "Chan"},
                                       "statistics": {"viewCount": "2500000"}}]})
    assert items[0].traffic == "2.5M views"
    assert items[0].url.endswith("abc")


def test_youtube_metadata_always_discloses_synthetic_media_and_sanitizes():
    meta = build_metadata("A <b>title</b>", "desc <x>", ["a", "A", "two words"], privacy="public")
    assert meta["status"]["containsSyntheticMedia"] is True
    assert meta["status"]["selfDeclaredMadeForKids"] is False
    assert meta["status"]["privacyStatus"] == "public"
    assert "<" not in meta["snippet"]["title"] and ">" not in meta["snippet"]["description"]
    assert meta["snippet"]["tags"] == ["a", "two words"]  # case-insensitive dedupe


def test_youtube_tags_respect_total_length_limit():
    tags = clean_tags([f"tag number {i}" for i in range(100)])
    assert sum(len(t) + 3 for t in tags) <= 500


def test_telegram_update_normalization():
    msg = normalize_update({"update_id": 1, "message": {"chat": {"id": 42}, "text": "/status"}})
    assert msg == {"kind": "message", "chat_id": "42", "text": "/status", "callback_id": None}
    cb = normalize_update({"update_id": 2, "callback_query": {"id": "q", "data": "approve:3",
                                                              "message": {"chat": {"id": 42}}}})
    assert cb["kind"] == "callback" and cb["text"] == "approve:3" and cb["chat_id"] == "42"


# ---------------------------------------------------------------------------
# Picking + scheduling
# ---------------------------------------------------------------------------


def test_choose_candidate_never_repeats_yesterdays_niche():
    chosen = core.choose_candidate(PICK["candidates"], core.DEFAULT_NICHES, "Garden & backyard")
    assert chosen["niche"] == "Forgotten history"


def test_choose_candidate_rejects_unknown_niches_and_excluded_topics():
    cands = [{**PICK["candidates"][0], "niche": "Crypto tips"}, PICK["candidates"][1]]
    assert core.choose_candidate(cands, core.DEFAULT_NICHES, None)["niche"] == "Forgotten history"
    excluded = {PICK["candidates"][1]["topic"].lower()}
    assert core.choose_candidate(cands, core.DEFAULT_NICHES, None, excluded) is None


def test_choose_candidate_takes_kind_from_the_niche_list():
    cand = {**PICK["candidates"][0], "kind": "fact"}  # LLM got the kind wrong
    assert core.choose_candidate([cand], core.DEFAULT_NICHES, None)["kind"] == "howto"


def test_is_due_respects_enabled_hour_and_existing_episode(saas_env):
    session = rdb.new_session()
    _, channel = _admin_channel(session, run_hour=6)
    morning = datetime.now().replace(hour=5)
    later = datetime.now().replace(hour=15)
    assert not core.is_due(session, channel, morning)
    assert core.is_due(session, channel, later)
    core.new_episode(session, channel)
    assert not core.is_due(session, channel, later)
    channel.enabled = False
    assert not core.is_due(session, channel, later.replace(day=1) if later.day != 1 else later)


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------


def test_plan_episode_creates_a_full_render_project(saas_env, captured):
    session = rdb.new_session()
    user, channel = _admin_channel(session)
    episode = core.new_episode(session, channel)
    llm = ScriptedLLM(PICK, EPISODE)

    core.plan_episode(session, episode, llm, [StaticTrends()], jobs.enqueue_pipeline)

    assert episode.status == "generating"
    assert episode.niche == "Garden & backyard" and episode.kind == "howto"
    assert "AI-generated host" in episode.description
    assert "#frost" in episode.description  # hashtag normalized
    assert episode.llm_cost == pytest.approx(0.02)
    project = session.get(rdb.Project, episode.project_id)
    source = Path(project.dir_path) / "script" / "source.txt"
    assert source.read_text() == SCRIPT
    job = session.query(rdb.Job).filter(rdb.Job.project_id == project.id).one()
    assert job.kind == "create"
    assert "--skip-render" not in job.argv  # nobody reviews scenes: render straight through
    assert "--script-file" in job.argv and "documentary" in job.argv
    # Lands under its channel in the dashboard switcher, as a landscape video.
    assert job.argv[job.argv.index("--channel-name") + 1] == channel.name
    assert job.argv[job.argv.index("--format") + 1] == "landscape"
    assert captured.pipeline == [(job.id,)]
    assert "first frost" in llm.calls[0]  # trends reach the picker


def test_plan_episode_blocks_the_previous_episodes_niche(saas_env, captured):
    session = rdb.new_session()
    _, channel = _admin_channel(session)
    session.add(rdb.Episode(channel_id=channel.id, day="2000-01-01", status="uploaded",
                            niche="Garden & backyard", topic="old topic"))
    session.commit()
    episode = core.new_episode(session, channel)
    llm = ScriptedLLM(PICK, EPISODE)
    core.plan_episode(session, episode, llm, [], jobs.enqueue_pipeline)
    assert episode.niche == "Forgotten history"
    assert 'Do NOT use the niche "Garden & backyard"' in llm.calls[0]


def test_plan_episode_respects_the_paywall(saas_env, captured):
    session = rdb.new_session()
    user, channel = _admin_channel(session)
    user.is_admin = False
    user.trial_credits = 0
    session.commit()
    episode = core.new_episode(session, channel)
    with pytest.raises(RuntimeError, match="credits"):
        core.plan_episode(session, episode, ScriptedLLM(PICK, EPISODE), [], jobs.enqueue_pipeline)


def _generating_episode(session, saas_env, final=True):
    user, channel = _admin_channel(session)
    paths_root = saas_env.projects_dir / f"u{user.id}" / "ep"
    (paths_root / "output").mkdir(parents=True)
    (paths_root / "script").mkdir(parents=True)
    (paths_root / "script" / "scenes.json").write_text("{}")
    if final:
        (paths_root / "output" / "final.mp4").write_bytes(b"x")
    project = rdb.Project(owner_id=user.id, slug="ep", title="Ep", dir_path=str(paths_root), created_at=0)
    session.add(project)
    session.flush()
    episode = rdb.Episode(channel_id=channel.id, day=core.today(), status="generating",
                          project_id=project.id, topic="t", niche="Garden & backyard")
    session.add(episode)
    session.commit()
    return episode, project


def _job(session, project, status, kind="create"):
    job = rdb.Job(project_id=project.id, kind=kind, argv=[], status=status)
    session.add(job)
    session.commit()
    return job


def test_finished_render_packages_the_episode(saas_env, captured):
    session = rdb.new_session()
    episode, project = _generating_episode(session, saas_env)
    jobs.on_pipeline_job_finished(_job(session, project, "succeeded").id)
    assert captured.package == [(episode.id,)]


def test_failed_render_auto_resumes_then_gives_up(saas_env, captured):
    session = rdb.new_session()
    episode, project = _generating_episode(session, saas_env, final=False)
    for _ in range(core.STUCK_RESUME_LIMIT):
        jobs.on_pipeline_job_finished(_job(session, project, "failed").id)
    resumes = session.query(rdb.Job).filter(rdb.Job.kind == "resume").all()
    assert len(resumes) == core.STUCK_RESUME_LIMIT
    for r in resumes:
        r.status = "failed"
    session.commit()
    jobs.on_pipeline_job_finished(resumes[-1].id)
    session.refresh(episode)
    assert episode.status == "failed"


def test_normal_projects_are_ignored_by_the_hook(saas_env, captured):
    session = rdb.new_session()
    user = rdb.User(email="x@example.com", password_hash="x")
    session.add(user)
    session.flush()
    project = rdb.Project(owner_id=user.id, slug="p", title="P", dir_path="/tmp/nope-p", created_at=0)
    session.add(project)
    session.commit()
    jobs.on_pipeline_job_finished(_job(session, project, "failed").id)
    assert captured.package == [] and captured.pipeline == []


def test_actions_approve_thumb_skip(saas_env, captured):
    session = rdb.new_session()
    episode, project = _generating_episode(session, saas_env)
    with pytest.raises(jobs.ActionError):
        jobs.perform_action(session, episode, "approve")  # not ready yet
    episode.status = "ready"
    session.commit()

    jobs.perform_action(session, episode, "thumb")
    assert episode.status == "generating"
    thumb_job = session.query(rdb.Job).filter(rdb.Job.kind == "thumbnail").one()
    assert "--thumbnail-only" in thumb_job.argv

    episode.status = "ready"
    session.commit()
    jobs.perform_action(session, episode, "approve")
    assert episode.status == "uploading" and captured.upload == [(episode.id,)]
    with pytest.raises(jobs.ActionError):
        jobs.perform_action(session, episode, "skip")


def test_new_topic_skips_and_excludes_the_old_topic(saas_env, captured):
    session = rdb.new_session()
    episode, _ = _generating_episode(session, saas_env)
    episode.status = "ready"
    session.commit()
    jobs.perform_action(session, episode, "newtopic")
    assert episode.status == "skipped"
    new_id, exclude = captured.start[0]
    assert exclude == ["t"]
    assert session.get(rdb.Episode, new_id).status == "planning"


def test_cannot_start_two_episodes_at_once(saas_env, captured):
    session = rdb.new_session()
    _, channel = _admin_channel(session)
    jobs.start_new_episode(session, channel)
    with pytest.raises(jobs.ActionError):
        jobs.start_new_episode(session, channel)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


class FakeMessenger:
    def __init__(self):
        self.sent = []

    def send_text(self, text, buttons=None, chat_id=None):
        self.sent.append((text, chat_id))

    def answer_callback(self, callback_id, text=""):
        self.sent.append(("<answered>", callback_id))


def test_bot_pairing_reply_when_chat_id_unset(saas_env):
    session = rdb.new_session()
    m = FakeMessenger()
    bot.handle_event(session, m, {"kind": "message", "chat_id": "99", "text": "/start", "callback_id": None}, "")
    assert "99" in m.sent[0][0] and m.sent[0][1] == "99"


def test_bot_ignores_strangers(saas_env, captured):
    session = rdb.new_session()
    _admin_channel(session)
    m = FakeMessenger()
    bot.handle_event(session, m, {"kind": "message", "chat_id": "666", "text": "/today", "callback_id": None}, "42")
    assert m.sent == [] and captured.start == []


def test_bot_buttons_and_commands(saas_env, captured):
    session = rdb.new_session()
    episode, _ = _generating_episode(session, saas_env)
    channel = session.get(rdb.AutopilotChannel, episode.channel_id)
    assert "rendering" in bot.handle_command(session, channel, "/status")
    episode.status = "ready"
    session.commit()
    m = FakeMessenger()
    bot.handle_event(session, m, {"kind": "callback", "chat_id": "42", "text": f"approve:{episode.id}",
                                  "callback_id": "cb1"}, "42")
    assert captured.upload == [(episode.id,)]
    assert ("<answered>", "cb1") in m.sent
    assert "⚠️" in bot.handle_command(session, channel, "/approve")  # now uploading
    assert "paused" in bot.handle_command(session, channel, "/pause").lower()
    assert channel.enabled is False


def test_tick_schedule_starts_one_episode_when_due(saas_env, captured, monkeypatch):
    session = rdb.new_session()
    _admin_channel(session, run_hour=0)
    bot.tick_schedule(session)
    bot.tick_schedule(session)
    assert len(captured.start) == 1


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_api_state_creates_channel_with_default_niches(client, captured):
    register(client, "me@example.com")
    data = client.get("/api/autopilot").json()
    assert data["channel"]["name"] == "Underrated Facts"
    assert len(data["channel"]["niches"]) == len(core.DEFAULT_NICHES)
    assert data["episodes"] == [] and data["botOnline"] is False


def test_api_update_validates_and_saves(client, captured):
    register(client, "me@example.com")
    body = {"name": "Underrated Facts", "enabled": True, "runHour": 7, "lengthMinutes": 12,
            "privacy": "private", "niches": [{"name": "A", "kind": "fact"}, {"name": "B", "kind": "howto"}]}
    assert client.put("/api/autopilot/channel", json=body).status_code == 200
    assert client.get("/api/autopilot").json()["channel"]["runHour"] == 7
    bad = {**body, "niches": [{"name": "A", "kind": "fact"}]}
    assert client.put("/api/autopilot/channel", json=bad).status_code == 422
    assert client.put("/api/autopilot/channel", json={**body, "runHour": 30}).status_code == 422


def test_api_run_now_and_conflict(client, captured):
    register(client, "me@example.com")
    res = client.post("/api/autopilot/run-now")
    assert res.status_code == 202
    assert captured.start[0][0] == res.json()["episodeId"]
    assert client.post("/api/autopilot/run-now").status_code == 409


def test_api_actions_are_owner_scoped(make_client, captured):
    a, b = make_client(), make_client()
    register(a, "a@example.com")
    register(b, "b@example.com")
    ep = a.post("/api/autopilot/run-now").json()["episodeId"]
    assert b.post(f"/api/autopilot/episodes/{ep}/skip").status_code == 404
    assert a.post(f"/api/autopilot/episodes/{ep}/skip").status_code == 200
    assert a.post(f"/api/autopilot/episodes/{ep}/bogus").status_code == 404


# ---------------------------------------------------------------------------
# Integration with the Windows production setup (eager mode, shared YouTube
# logins, the 6pm auto-publish batch)
# ---------------------------------------------------------------------------


def _uploading_episode(session, saas_env):
    episode, project = _generating_episode(session, saas_env)
    episode.status = "uploading"
    episode.title, episode.description, episode.tags = "T", "D", ["a"]
    session.commit()
    return episode, project


def test_upload_uses_the_channels_token_and_records_the_publish(saas_env, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "messenger_or_none", lambda: None)  # not `captured`: it stubs upload_episode itself
    from renderflow.storage import ProjectPaths, load_youtube_publish

    session = rdb.new_session()
    episode, project = _uploading_episode(session, saas_env)
    token = tmp_path / ".youtube_token.json"
    token.write_text("{}")
    resolved = []
    monkeypatch.setattr(
        jobs.youtube_module, "resolve_token_path",
        lambda name, separate: resolved.append(name) or token,
    )
    used_tokens = []

    class Publisher:
        def upload(self, video, title, description, tags, **params):
            return SimpleNamespace(video_id="vid1", url="https://youtu.be/vid1", meta={"privacy": "private"})

    monkeypatch.setattr(
        jobs, "build_publisher", lambda settings, token_file: used_tokens.append(token_file) or Publisher()
    )
    jobs.upload_episode(episode.id)

    session.refresh(episode)
    assert episode.status == "uploaded"
    assert resolved == ["Underrated Facts"] and used_tokens == [token]
    record = load_youtube_publish(ProjectPaths(root=Path(project.dir_path)))
    assert record.video_id == "vid1" and record.contains_synthetic_media is True


def test_upload_fails_instead_of_using_the_wrong_account(saas_env, monkeypatch):
    monkeypatch.setattr(jobs, "messenger_or_none", lambda: None)  # not `captured`: it stubs upload_episode itself
    # A channel listed in RENDERFLOW_SEPARATE_YOUTUBE_CHANNELS with no token
    # of its own resolves to None — it must never fall back to the main one.
    session = rdb.new_session()
    episode, _ = _uploading_episode(session, saas_env)
    monkeypatch.setattr(jobs.youtube_module, "resolve_token_path", lambda name, separate: None)
    monkeypatch.setattr(jobs, "build_publisher", lambda *a: pytest.fail("must not upload"))
    jobs.upload_episode(episode.id)
    session.refresh(episode)
    assert episode.status == "failed" and "setup_youtube.py" in episode.error


def test_restart_recovery_fails_planning_and_uploading_episodes(saas_env, captured):
    session = rdb.new_session()
    _, channel = _admin_channel(session)
    planning = rdb.Episode(channel_id=channel.id, day="2000-01-01", status="planning")
    uploading = rdb.Episode(channel_id=channel.id, day="2000-01-02", status="uploading")
    ready = rdb.Episode(channel_id=channel.id, day="2000-01-03", status="ready")
    session.add_all([planning, uploading, ready])
    session.commit()
    jobs.recover_interrupted_episodes(session)
    assert planning.status == "failed" and "/today" in planning.error
    assert uploading.status == "failed" and "Studio" in uploading.error
    assert ready.status == "ready"


def test_eager_submit_runs_on_the_shared_worker_thread_not_inline(monkeypatch):
    # In eager mode .delay() would run a task inline — inside the bot loop or
    # an HTTP request — and outside the one-at-a-time render queue.
    import threading

    from renderflow import tasks
    from tests.conftest import make_settings

    eager = make_settings(projects_dir=Path("."), celery_eager=True)
    monkeypatch.setattr(tasks.Settings, "load", classmethod(lambda cls: eager))
    ran: list[tuple[int, str]] = []
    done = threading.Event()

    class Task:
        def run(self, value):
            ran.append((threading.get_ident(), value))
            if len(ran) == 2:
                done.set()

        def apply_async(self, *a, **k):
            pytest.fail("eager mode must not touch the broker")

    assert tasks.submit(Task(), "now") is None
    tasks.submit(Task(), "later", countdown=0.05)
    assert done.wait(5)
    assert [v for _, v in ran] == ["now", "later"]
    assert all(ident != threading.get_ident() for ident, _ in ran)
    assert len({ident for ident, _ in ran}) == 1  # one serialized worker
