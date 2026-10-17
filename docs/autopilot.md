# Autopilot: one YouTube Short a day, approved on Telegram

The **Autopilot** tab runs the Underrated Facts channel on its own:

1. Every day at the start hour (default 06:00, on the worker PC's clock), it
   scans **Google Trends** (plus YouTube's trending chart if
   `YOUTUBE_API_KEY` is set).
2. Claude picks the niche that's **trending hardest today**, never the same
   niche two days in a row, and finds an evergreen angle aimed at viewers 55+.
   It always skips sports, politics, tragedies, gossip, and health, legal or
   investing advice.
3. Claude writes a one-minute **Short** script (about 130 words, one fact or
   one quick trick) plus the SEO title, description and tags (`#Shorts` is
   always added). The Short is then made **exactly like a dashboard Short**
   (`make_video.py --script-file --format shorts`: same voice, visuals,
   captions, music and closing clip) and renders straight through.
4. Telegram gets the thumbnail, the SEO text (ready to copy) and the whole
   Short at preview quality, with buttons: **Approve & upload · New topic ·
   Skip today** (`/thumb` still makes a new thumbnail, but YouTube mostly
   ignores custom thumbnails on Shorts).
5. **Approve** uploads it to YouTube with "altered or synthetic content"
   switched on and an AI-host disclosure line in the description.

If a render stops partway (a provider rate limit, say), it resumes on its
own up to twice (each after a 30-minute wait, to outlast hourly rate
limits) before reporting failure.

## Commands (Telegram)

`/status` · `/today` (start now) · `/approve` · `/thumb` · `/newtopic` ·
`/skip` · `/pause` · `/resume` · `/trends`. Each takes an optional episode
number (`/approve 12`) and defaults to the latest episode. Only the chat in
`TELEGRAM_CHAT_ID` can control the bot. Anyone else gets ignored.

## Running it on the Windows PC as it is set up today (no WSL)

The PC already runs the dashboard natively in eager mode
(`RENDERFLOW_CELERY_EAGER=1`, SQLite, no Redis, no Celery worker; the API
starts at logon via `deploy/windows/`). In that mode the Autopilot needs
**no extra process**: the API runs the scheduler and the Telegram bot in a
background thread, and Autopilot renders wait their turn in the same
one-at-a-time queue as dashboard renders. Don't also start
`python -m renderflow.autopilot.bot` (it refuses to start in eager mode).

1. `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`: already set as Windows
   user environment variables on this PC. To set them up from scratch, see
   step 4 below.
2. YouTube: Autopilot uploads with the dashboard's existing login
   (`.youtube_token.json`, from `scripts/setup_youtube.py`). The Autopilot
   channel is "Underrated Facts", the main channel, so nothing new is
   needed. If you rename it to a channel listed in
   `RENDERFLOW_SEPARATE_YOUTUBE_CHANNELS`, connect that channel with
   `scripts/setup_youtube.py --channel "<name>"` first.
3. `ANTHROPIC_API_KEY` needs credit: picking the topic and writing the
   script are Claude calls.
4. Restart the API. Then, in the dashboard, open **Autopilot**, check the
   status tiles, and press **Make today's video now**.

The daily 6pm auto-publish never uploads Autopilot videos. They go out
only when you tap Approve.

## Setting it up on a Windows PC with WSL2 (real Celery worker)

The alternative: RenderFlow with Postgres, Redis and a Celery worker,
inside Ubuntu on WSL2.

1. **PowerShell (admin):** `wsl --install -d Ubuntu-24.04`, reboot, create
   your Linux user. Then in Ubuntu, enable systemd so the services start
   with WSL:
   ```bash
   printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
   ```
   Then run `wsl --shutdown` in PowerShell and reopen Ubuntu.
2. **Windows power settings:** set Sleep to *Never* while plugged in. A
   sleeping PC misses the schedule (it catches up when it wakes, the same
   day). To have WSL start at login, add a Task Scheduler task "At log on"
   that runs `wsl.exe -d Ubuntu-24.04 --exec sleep infinity`.
3. **Install RenderFlow (in Ubuntu):**
   ```bash
   sudo apt update && sudo apt install -y python3.12-venv ffmpeg git fonts-dejavu docker.io docker-compose-v2
   sudo usermod -aG docker $USER   # then close and reopen Ubuntu
   git clone https://github.com/jptaycs/RenderFlow.git ~/RenderFlow && cd ~/RenderFlow
   python3.12 -m venv .venv
   .venv/bin/pip install -e '.[web,kokoro,wav2lip,parallax]'
   .venv/bin/python scripts/setup_kokoro.py && .venv/bin/python scripts/setup_wav2lip.py
   ```
   Restore `.env` from Bitwarden (`scripts/env-pull.sh`) or copy it over by
   hand. It needs `ANTHROPIC_API_KEY`, `RENDERFLOW_SECRET_KEY`, the
   provider keys, and the new Autopilot keys below.
   **NVIDIA GPU?** Install the NVIDIA Windows driver (WSL2 picks up CUDA
   automatically), then `.venv/bin/pip install torch --index-url
   https://download.pytorch.org/whl/cu124`. Lip sync and depth motion then
   run on the GPU.
4. **Telegram bot:** in Telegram, message **@BotFather** → `/newbot` → copy
   the token into `.env` as `TELEGRAM_BOT_TOKEN`. Start the bot once (step
   7), send it any message, and it replies with your chat id. Put that in
   `TELEGRAM_CHAT_ID` and restart the bot.
5. **YouTube upload:** connect the channel the same way as the dashboard's
   Publish button (see `scripts/setup_youtube.py` and the YouTube publishing
   notes in CLAUDE.md):
   ```bash
   .venv/bin/python scripts/setup_youtube.py
   ```
   Choose the **Underrated Facts** channel and allow access. Optionally,
   create an **API key** in the same Google Cloud project and set it as
   `YOUTUBE_API_KEY` to add YouTube trending as a trend source.
6. **Start the datastores:** `docker compose up -d`.
7. **Run the four processes.** In separate terminals, or install the systemd
   units from `deploy/` and edit their `User=`/`WorkingDirectory=` to match
   your Linux user and `~/RenderFlow`:
   ```bash
   .venv/bin/celery -A renderflow.tasks worker --concurrency=1 --loglevel=info
   .venv/bin/python -m renderflow.api          # dashboard: http://localhost:8321 in Windows
   .venv/bin/python -m renderflow.autopilot.bot
   ```
8. Open the dashboard → **Autopilot**. Check that all five status tiles are
   green, look over the niches, tick *Make a video automatically every
   day*, and press **Make today's video now** for the first test.

## Things YouTube does that you'll notice

- **Uploads stay private** until Google audits your API project (apply in
  the Google Cloud console under YouTube API Services → Audit). Until then,
  tap the Studio link the bot sends and switch the video to Public.
- **Custom thumbnails** need a phone-verified channel
  (youtube.com/verify). Without it, the upload succeeds and the bot tells
  you to set the thumbnail in Studio.
- While the OAuth consent screen is in **Testing**, Google expires the
  connection after 7 days. Set it to **In production** to keep it.

## Length and cost

Every episode is a Short of about a minute (decided 2026-10-17; the old
Length setting is gone). Your dashboard Shorts have cost about $0.09 of
Claude each (median of 60), mostly the per-scene video-prompt rewrites; an
Autopilot Short adds two small calls (topic pick and script) on top. If a
render stops on a provider limit, the auto-resume (30 minutes apart) picks
it up.
