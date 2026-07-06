# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RenderFlow is an AI video production **orchestration platform**: topic or script in → narrated, rendered MP4 out. It coordinates external AI providers (LLM, image, TTS, avatar) behind swappable interfaces — it is not a model itself. Target: long-form YouTube documentaries (20–30 min), currently at the Week-1 walking-skeleton stage producing short videos end-to-end.

`AGENTS.md` points here; this file is the single source of truth — keep it honest as things change.

**Documentation rule: every change updates this file, in the same session.** If a session changes behavior, adds/renames a file or env var, installs a dependency, switches a provider or voice, learns a provider quirk the hard way, or verifies something against a live service — record it here (Current state, Commands, Configuration, Provider notes, Layout, whichever fits) before finishing. Anything not written down here is knowledge the next session doesn't have.

## Current state

Working today, verified against live services:

- Synchronous CLI pipeline: scene plan → images → voice → FFmpeg 1080p MP4
- Providers: Claude (LLM, untested live — no API key yet), Pollinations + Flux-Replicate (image), Kokoro (**default TTS** — local, natural prosody) + Piper + ElevenLabs (TTS)
- Resumable asset generation (completed assets skip on re-run), per-asset cost tracking
- Talking-avatar scene contract with four providers: `wav2lip-local` (**default in practice** — free unlimited local lip sync + camera push-in, verified live, ~real-time on CPU), `ffmpeg-still` placeholder (static image + voice; no lip sync), `sadtalker-replicate` (paid lip sync + real head motion via Replicate; written, unverified — needs a funded `REPLICATE_API_TOKEN`), and `memo-hf` (free HF Space, demo-only: 4s audio cap)
- Piper narration pacing: `RENDERFLOW_TTS_LENGTH_SCALE` + sentence-pause insertion (see Provider notes)
- Fast visual pacing: scripts split into ~5-second scenes (clause-level splits for long sentences); image prompts are written as realistic photo captions with anti-AI-look negative prompts. Composition rule: never a lone human face — every image combines the sentence's subject with its topic context (setting/objects/actions; wide or medium shots), enforced in both LLM prompts and the local prompt builder + negative prompts
- Local web dashboard (`renderflow/api.py` + `web/index.html`): FastAPI serves project state, live scene thumbnails, and launches `make_video.py` runs as subprocesses (create / resume / regenerate-scene). Finished renders are download-only by design (no inline player — client preference); the download card shows the project thumbnail. Verified live end-to-end. Dev stand-in for the Week-2 worker queue — pipeline never runs inside a request handler.
- Project thumbnail: a **generated clickbait image + host badge**, not a scene crop. `assets.generate_thumbnail()` extracts the topic from the title (`_topic_from_title` strips clickbait filler words — do NOT put the full title in the image prompt, the model renders it as garbled text; learned 2026-07), generates a topic-literal dramatic image (subject right, no people) via the ImageProvider, tracked as `ScenePlan.thumbnail` (plan-level `AssetRef`, in `total_asset_cost()`, resume-aware). `render.render_thumbnail()` crops to 1280×720 `output/thumbnail.jpg` and composites `RENDERFLOW_AVATAR_IMAGE` as a white-ringed circular host badge bottom-left (PIL; skipped gracefully if PIL/portrait missing; falls back to the first scene image if generation never ran). The New Video modal has a "Make it clickbait" button that cycles the typed topic through title templates client-side.
- Dashboard render-state rule (`api.py`): a `final.mp4` counts as done only if **all assets are completed and the file is newer than scenes.json** (which is rewritten on every asset update) — a leftover video from an earlier run never shows as Complete/downloadable while assets are regenerating.
- Run tracking survives API restarts: `_spawn` writes `logs/run.pid`; `_run_active` falls back to it and verifies the pid is a live `make_video.py` process. Assets orphaned in `running` state by a crashed/killed run are recovered on resume (`_start` routes running → failed → retrying → running). Learned the hard way: an in-memory-only `_runs` dict made live runs show "Paused" after a restart, and Resume then spawned a colliding duplicate that died with `InvalidTransition: running -> running`.

Not built yet (roadmap order): stock-video B-roll (Pexels), subtitles (faster-whisper → ASS), narrator character consistency, more lip-sync avatar providers, background music/crossfades, Celery/Redis workers, Postgres, FastAPI, Next.js dashboard.

## Commands

```bash
# Python 3.12 venv (system python is 3.9 — do not use it)
.venv/bin/python -m pytest                 # run tests
.venv/bin/python -m pytest -k avatar       # run tests matching a keyword
.venv/bin/python make_video.py --topic "..." --length 3          # LLM writes script (needs ANTHROPIC_API_KEY)
.venv/bin/python make_video.py --script-file script.txt --slug x --title "..." # split client script locally (no key; --llm-split needs ANTHROPIC_API_KEY)
.venv/bin/python make_video.py --scenes-file plan.json --slug x  # from existing scene JSON (no key needed)
.venv/bin/python scripts/check_providers.py [claude|image|tts]   # live provider verification (< $0.02)
.venv/bin/python scripts/check_providers.py avatar               # live lip-sync clip (opt-in; saves projects/avatar_check.mp4)
.venv/bin/pip install '.[web]' && .venv/bin/python -m renderflow.api  # dashboard at http://127.0.0.1:8321
.venv/bin/pip install '.[parallax]'   # deps for depth-parallax scene motion (default RENDERFLOW_MOTION)
.venv/bin/pip install '.[wav2lip]' && .venv/bin/python scripts/setup_wav2lip.py  # one-time wav2lip-local setup (~520 MB)
.venv/bin/pip install '.[kokoro]' && .venv/bin/python scripts/setup_kokoro.py    # one-time kokoro TTS setup (~340 MB)
python -m piper.download_voices en_US-lessac-medium --data-dir .voices           # fetch a Piper voice
```

## Architecture — how a video gets made

```
make_video.py (CLI)
  [1] script:  topic --LLM--> GeneratedScript.to_plan() -> ScenePlan   (pipeline/script.py)
               or --scenes-file loads scenes.json directly (no LLM)
  [2] images:  per scene, ImageProvider.generate(image_prompt)          (pipeline/assets.py)
               talking_avatar scenes also get an avatar portrait
               (RENDERFLOW_AVATAR_IMAGE local file, or generated)
  [3] voice:   per scene, TTSProvider.synthesize(narration)             (pipeline/assets.py)
  [4] avatar:  talking_avatar scenes: AvatarProvider.generate_clip(
               portrait, voice, narration) -> lip-synced MP4            (pipeline/assets.py)
  [5] render:  per-scene clip (zoompan still, or avatar full/split),
               concat + loudnorm + faststart -> output/final.mp4        (pipeline/render.py)
```

Key mechanics that span multiple files:

- **`renderflow/schema.py` is THE central contract.** `ScenePlan` → `Scene` → `SceneAssets` (`image`, `avatar_image`, `voice`, `avatar_clip` as `AssetRef`s). Everything reads/writes these models; the plan is persisted to `projects/<slug>/script/scenes.json` after every asset state change. Treat schema changes like breaking API changes; update docs example and tests together. There is a parallel `Generated*` schema (what the LLM emits, no numeric constraints — unsupported by structured outputs) clamped into the real contract by `GeneratedScript.to_plan()`.
- **Asset state machine is exactly** `pending → running → completed | failed → retrying → running | cancelled`, enforced by `AssetRef.advance` (raises `InvalidTransition`).
- **Resume:** an asset with `status=completed` and a `path` is skipped on re-run. Consequence: config changes (voice, pacing, provider) do **not** retrofit an existing project — use a fresh `--slug` or delete the relevant `projects/<slug>/{voice,avatar}/` files.
- **Provider layer:** `providers/base.py` defines `Protocol`s (`LLMProvider`, `ImageProvider`, `TTSProvider`, `AvatarProvider`) returning `GeneratedAsset`/`LLMResult`; `providers/__init__.py` is the registry mapping env-selected names to adapters. `retry.py` supplies the backoff decorator adapters wrap their live calls with.
- **Rendering rules** (`pipeline/render.py`): scene duration always comes from ffprobe of the generated voice audio (or avatar clip) — never `duration_estimate_sec`. Scene visuals get **depth-parallax motion** by default (`pipeline/parallax.py`: Depth-Anything-V2-Small depth map + OpenCV remap, ~50 MB one-time HF download, ~2-5 s/scene on CPU; verified live 2026-07); `RENDERFLOW_MOTION=zoompan` opts out, and missing deps / depth failures fall back to zoompan automatically (flat Ken Burns, prescaled 2560×1440 to avoid sub-pixel jitter). Every talking-avatar scene renders split-screen (768px lip-synced avatar left, zoompan visual right); the local splitter marks **every** scene `talking_avatar`, so the host is on camera for the whole video (full-screen avatar path removed 2026-07). Audio is loudness-normalized per clip; final concat re-encodes with `+faststart`.
- **File formats ripple:** Piper emits WAV, ElevenLabs MP3 — `pipeline/assets.py` derives the file extension from `asset.meta["format"]`; lip-sync models want WAV, so `providers/avatar/postprocess.py` transcodes. That module also burns the "AI-generated host" disclosure banner into every avatar clip — **but the burn-in silently degrades to no banner when ffmpeg lacks `drawtext`, and the dev machine's ffmpeg build has no drawtext filter (discovered 2026-07), so local clips currently ship without the disclosure.** Fix path: reinstall ffmpeg with libfreetype, or overlay a pre-rendered PNG instead of drawtext.

## Layout

```
make_video.py               CLI entry point (the web API shells out to this too)
web/index.html              dashboard UI (vanilla JS, embedded fonts; polls /api/state
                            every 2.5s but re-renders only when state changed, preserving scroll)
renderflow/
  api.py                    FastAPI: /api/state, create/resume/regenerate/delete, /files static
  schema.py                 Scene JSON schema (Pydantic) + asset state machine — THE central contract
  config.py                 Settings from .env
  storage.py                projects/<slug>/{script,images,voice,avatar,subtitles,output,logs}/, plan save/load
  retry.py                  backoff decorator for external calls
  pipeline/
    script.py               topic→scenes and client-script→scenes prompts (LLM)
    assets.py               per-scene image/voice/avatar generation, state + cost persistence
    render.py               FFmpeg scene clips (split-screen avatar), concat, mux
    parallax.py             depth-parallax motion (Depth-Anything-V2 + cv2 remap)
  providers/
    base.py                 Protocol interfaces: LLMProvider, ImageProvider, TTSProvider, AvatarProvider + GeneratedAsset
    __init__.py             registry — env vars select active provider
    llm/claude.py           image/{flux_replicate,pollinations}.py   tts/{elevenlabs_tts,piper_tts,kokoro_tts}.py
    avatar/                 ffmpeg_still, wav2lip_local, sadtalker_replicate, memo_hf
                            + postprocess.py (shared wav transcode / disclosure burn-in)
scripts/
  check_providers.py        one minimal live call per provider
  setup_wav2lip.py          one-time .wav2lip/ download for wav2lip-local
  setup_kokoro.py           one-time .kokoro/ download for kokoro TTS
tests/                      stub providers, schema/state-machine/pipeline tests, FFmpeg integration test
```

Gitignored working dirs: `.venv/` (python 3.12), `.voices/` (Piper models), `.wav2lip/` (Wav2Lip code + weights), `projects/` (generated videos), `.env` (keys).

## Hard rules

- **Never call an external API from business logic.** Everything goes through the `Protocol` interfaces in `providers/base.py` and the registry in `providers/__init__.py`. Fix response-shape problems in the provider adapter, not the pipeline.
- **Cost tracking from day one.** Every provider call records its cost on the asset/result; the system must always be able to answer "this video cost $X".
- **Never commit `.env`.** Keys live only there. `.env.example` documents the variables.
- Backend stays Python-only. Type hints everywhere; Pydantic for schema and API contracts.
- Keep changes minimal and surgical — Week 2 (workers, Postgres, FastAPI) will build on this structure; don't restructure preemptively.

## Configuration (.env)

Free vs paid stacks are pure `.env` swaps — no code changes:

| Variable | Free stack | Paid stack |
|---|---|---|
| `RENDERFLOW_IMAGE_PROVIDER` | `pollinations` (keyless) | `flux-replicate` (`REPLICATE_API_TOKEN`) |
| `RENDERFLOW_TTS_PROVIDER` | `piper` (local) | `elevenlabs` (`ELEVENLABS_API_KEY`) |
| `RENDERFLOW_TTS_VOICE` | Piper voice name | ElevenLabs voice id |
| `RENDERFLOW_AVATAR_PROVIDER` | `wav2lip-local` (or `ffmpeg-still`) | `sadtalker-replicate` |

Tuning: `RENDERFLOW_TTS_LENGTH_SCALE` (Piper pace, higher = slower, 1.4 ≈ documentary), `RENDERFLOW_TTS_SENTENCE_PAUSE` (seconds between sentences), `RENDERFLOW_AVATAR_IMAGE` (local presenter portrait), `RENDERFLOW_LLM_MODEL`, `RENDERFLOW_PROJECTS_DIR`. See `.env.example` for the full annotated list.

## Provider notes (learned the hard way — do not rediscover)

- **ElevenLabs free tier blocks "library" voices over the API** (402 `paid_plan_required`). Only the account's *premade* voices work — e.g. George `JBFqnCBsd6RMkjVDRZzb` (documentary storyteller). Free tier = 10k chars/month, **no commercial rights**.
- **Pollinations** is keyless/free but anonymously caps at 1024×576 regardless of requested size (re-verified 2026-07); the renderer upscales, so output is 1080p but soft. A free registered token from auth.pollinations.ai (set `POLLINATIONS_TOKEN`) is sent as a Bearer header and lifts limits/watermark — whether it unlocks true 1080p is untested (no token yet). No `negative_prompt` input — the adapter folds it into the prompt. Can be slow (30–60 s/image). **Rate limits are real and sticky**: anonymous = 1 request/15 s, registered = 1/5 s; bursting past them earns persistent 429s that outlive short retries (hit 2026-07 after a 59-scene run). The adapter self-throttles to the allowed interval and retries with 20–60 s backoff — budget ~15-20 s/image anonymous.
- **Kokoro** (`kokoro-onnx`, Apache-2.0 — commercial OK) is the default narrator: far more natural prosody than Piper, runs local CPU faster than realtime, 24 kHz WAV. Setup: `pip install '.[kokoro]' && python scripts/setup_kokoro.py` (~340 MB into `.kokoro/`). Current voice `bm_george` (older British male); alternates `am_onyx` (deep US), `am_fenrir`. `speed` = 1/`RENDERFLOW_TTS_LENGTH_SCALE` (mapped in make_video); sentence pauses inserted like Piper. No literal breath sounds — that tier is paid (ElevenLabs).
- **Piper** voices live in `.voices/` (~60 MB each). MIT license — commercial use OK. Downloaded: `en_US-john-medium` (deep older male), `en_US-lessac-medium`, `en_US-ryan-low`. Fast but robotic — replaced by Kokoro as default 2026-07. Client rule: visuals must be AI-generated, created for the story — **no stock photos/footage** (a Pexels provider was built and removed 2026-07).
- **dotenv gotcha:** `Settings.load()` uses `load_dotenv(override=True)` — `.env` is the source of truth even for pipeline subprocesses spawned by a long-running API server (without override, the server's stale inherited env pinned subprocess config; hit 2026-07 when a TTS switch didn't apply).
- **Piper pacing:** raw Piper sounds rushed. `RENDERFLOW_TTS_LENGTH_SCALE` slows delivery; the adapter also splits text on `.!?…` and inserts `RENDERFLOW_TTS_SENTENCE_PAUSE` seconds of silence between sentences (an ellipsis in narration earns a dramatic pause). Remember the resume gotcha: existing projects keep their old audio.
- **Wav2Lip avatar** (`wav2lip-local`): `scripts/setup_wav2lip.py` pulls code **and** weights from the `camenduru/Wav2Lip` HF mirror into `.wav2lip/` (~520 MB — the official OneDrive checkpoint links are dead) and patches `audio.py` for librosa ≥ 0.10 (kwargs-only `filters.mel`). Runs ~real-time on Apple-Silicon CPU (13 s narration → 15 s). Lips only — the model doesn't move the head, so the adapter adds a slow zoompan push-in; for real head motion use `sadtalker-replicate`. Verified live 2026-07.
- **SadTalker avatar** (`cjwbw/sadtalker` on Replicate, version pinned in the adapter): billed per GPU second, so cost scales with narration length (~$0.08–0.15 for a short clip; cost computed from `predict_time`). Defaults `preprocess=full`, `still_mode=False` (head motion), `use_enhancer=True`; if the background warps around the moving head, set `still_mode=True`. The account's Replicate token exists but is **unfunded** — a credit purchase (min $10) is required, a saved card is not enough (402 `Insufficient credit`).
- **MEMO avatar** (`memo-hf`, public `fffiloni/MEMO` Gradio Space via `gradio_client`): demo-grade only — the Space **trims input audio to 4 seconds** and requests a 240s ZeroGPU slot that anonymous users are refused outright; a free `HF_TOKEN` is required just to be admitted. Verified 2026-07: every public talking-head Space has similar guards (EchoMimic trims to 5s / requests 200–300s). **There is no free hosted lip-sync API fit for full-length scene narrations.** Also: `gradio_client>=2` renamed `hf_token=` to `token=`, and needs `httpx_kwargs={"timeout": httpx.Timeout(30, read=None)}` or long generations die with `ReadTimeout`.
- **Claude LLM step:** uses structured outputs (`output_config.format` with `GeneratedScript.model_json_schema()`); the generation schema deliberately has no numeric constraints (unsupported there) — bounds are clamped in `GeneratedScript.to_plan()`. No `ANTHROPIC_API_KEY` funded yet; the key-free workflow is: a Claude Code session writes `scenes.json` by hand → `--scenes-file`.

## Testing conventions

- Stub providers in `tests/stubs.py` satisfy the Protocols — use them; never hit live APIs in tests.
- The FFmpeg integration test generates media with lavfi and is skipped when ffmpeg is absent.
- Live provider verification is `scripts/check_providers.py`, run manually, never in CI. The `avatar` check is excluded from the default run (slow; costs money on paid providers).

## What NOT to do

- Don't generate AI video clips (only stills + motion) — cost control, Phase 4+ decision. For "alive" B-roll the planned route is free stock footage (Pexels API), not video generation.
- Don't build auth, billing, publishing, or dashboards before the pipeline they'd serve exists.
- Don't render synchronously inside API requests once the FastAPI layer exists (Week 2+) — queue it.
- Don't change the scene schema casually, and don't bypass the provider registry.
