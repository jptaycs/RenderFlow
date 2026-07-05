# RenderFlow — Agent Instructions

AI video production **orchestration platform**: topic or script in → narrated, rendered MP4 out. It coordinates external AI providers (LLM, image, TTS) behind swappable interfaces — it is not a model itself. Target: long-form YouTube documentaries (20–30 min), currently at the Week-1 walking-skeleton stage producing short videos end-to-end.

## Current state (keep this section honest as things change)

Working today, verified against live services:

- Synchronous CLI pipeline: scene plan → images → voice → FFmpeg 1080p MP4
- Providers: Claude (LLM, untested live — no API key yet), Pollinations + Flux-Replicate (image), Piper + ElevenLabs (TTS)
- Resumable asset generation (completed assets skip on re-run), per-asset cost tracking
- Talking-avatar scene contract + local `ffmpeg-still` placeholder provider (image + voice → presenter MP4 with disclosure; not photoreal lip sync)

Not built yet (roadmap order): subtitles (faster-whisper → ASS), narrator character consistency, paid lip-sync avatar providers/split-screen scenes, background music/crossfades, Celery/Redis workers, Postgres, FastAPI, Next.js dashboard.

## Commands

```bash
# Python 3.12 venv (system python is 3.9 — do not use it)
.venv/bin/python -m pytest                 # run tests
.venv/bin/python make_video.py --topic "..." --length 3          # LLM writes script (needs ANTHROPIC_API_KEY)
.venv/bin/python make_video.py --script-file script.txt --slug x # split client script (needs ANTHROPIC_API_KEY)
.venv/bin/python make_video.py --scenes-file plan.json --slug x  # from existing scene JSON (no key needed)
.venv/bin/python scripts/check_providers.py [claude|image|tts]   # live provider verification (< $0.02)
```

## Layout

```
make_video.py               CLI entry point (the only executable surface today)
renderflow/
  schema.py                 Scene JSON schema (Pydantic) + asset state machine — THE central contract
  config.py                 Settings from .env
  storage.py                projects/<slug>/ layout, plan save/load
  retry.py                  backoff decorator for external calls
  pipeline/
    script.py               topic→scenes and client-script→scenes prompts (LLM)
    assets.py               per-scene image/voice generation, state + cost persistence
    render.py               FFmpeg zoompan, concat, mux
  providers/
    base.py                 Protocol interfaces: LLMProvider, ImageProvider, TTSProvider + GeneratedAsset
    __init__.py             registry — env vars select active provider
    llm/claude.py           image/{flux_replicate,pollinations}.py   tts/{elevenlabs_tts,piper_tts}.py
scripts/check_providers.py  one minimal live call per provider
tests/                      stub providers, schema/state-machine/pipeline tests, FFmpeg integration test
```

## Hard rules

- **Never call an external API from business logic.** Everything goes through the `Protocol` interfaces in `providers/base.py` and the registry in `providers/__init__.py`. Fix response-shape problems in the provider adapter, not the pipeline.
- **The scene JSON schema (`renderflow/schema.py`) is the central contract.** Treat changes like breaking API changes; update the docs example and tests together.
- **Asset state machine is exactly:** `pending → running → completed | failed → retrying → running | cancelled`. Transitions are enforced (`AssetRef.advance` raises on invalid moves).
- **Cost tracking from day one.** Every provider call records its cost on the asset/result; the system must always be able to answer "this video cost $X".
- **Never commit `.env`.** Keys live only there. `.env.example` documents the variables.
- Actual scene duration comes from the generated voice audio length (ffprobe), never from `duration_estimate_sec`.
- Backend stays Python-only. Type hints everywhere; Pydantic for schema and API contracts.
- Keep changes minimal and surgical — Week 2 (workers, Postgres, FastAPI) will build on this structure; don't restructure preemptively.

## Provider notes (learned the hard way — do not rediscover)

- **ElevenLabs free tier blocks "library" voices over the API** (402 `paid_plan_required`). Only the account's *premade* voices work — e.g. George `JBFqnCBsd6RMkjVDRZzb` (documentary storyteller). Free tier = 10k chars/month, **no commercial rights**. Voice output is MP3; Piper outputs WAV — `pipeline/assets.py` derives the extension from `asset.meta["format"]`.
- **Pollinations** is keyless/free but anonymously caps at 1024×576 regardless of requested size; the renderer upscales (prescale 2560×1440 before zoompan), so output is 1080p but soft. It has no `negative_prompt` input — the adapter folds it into the prompt. Can be slow (30–60 s/image); retries are configured.
- **Piper** voices live in `.voices/` (gitignored, ~60 MB each). Download: `python -m piper.download_voices en_US-lessac-medium --data-dir .voices`. MIT license — commercial use OK.
- **Free vs paid stacks** are pure `.env` swaps: `RENDERFLOW_IMAGE_PROVIDER=pollinations|flux-replicate`, `RENDERFLOW_TTS_PROVIDER=piper|elevenlabs`, `RENDERFLOW_TTS_VOICE=<piper voice name | elevenlabs voice id>`.
- **Claude LLM step:** uses structured outputs (`output_config.format` with `GeneratedScript.model_json_schema()`); the generation schema deliberately has no numeric constraints (unsupported there) — bounds are clamped in `GeneratedScript.to_plan()`. No `ANTHROPIC_API_KEY` funded yet; the key-free workflow is: a Claude Code session writes `scenes.json` by hand → `--scenes-file`.

## Testing conventions

- Stub providers in `tests/stubs.py` satisfy the Protocols — use them; never hit live APIs in tests.
- The FFmpeg integration test generates media with lavfi and is skipped when ffmpeg is absent.
- Live provider verification is `scripts/check_providers.py`, run manually, never in CI.

## What NOT to do

- Don't generate AI video clips (only stills + motion) — cost control, Phase 4+ decision.
- Don't build auth, billing, publishing, or dashboards before the pipeline they'd serve exists.
- Don't render synchronously inside API requests once the FastAPI layer exists (Week 2+) — queue it.
- Don't change the scene schema casually, and don't bypass the provider registry.
