# Deploying RenderFlow to a production server

One Ubuntu 24.04 box runs everything: Postgres + Redis in Docker, the API
and Celery worker from a host venv under systemd, and Caddy terminating TLS.
This mirrors the dev setup exactly — same commands, same layout.

## What you need (the only manual steps)

1. **A server**: Ubuntu 24.04 LTS, at least 4 vCPU / 8 GB RAM / 80 GB disk
   (Hetzner CX32/CPX31-class, ~$15–30/mo, or a dedicated box for more render
   throughput). The pipeline is CPU-only — no GPU needed.
2. **A domain** with an A record pointing at the server's IP (e.g.
   `app.yourdomain.com`). Caddy gets its TLS certificate automatically once
   DNS resolves.

## First-time setup

SSH in as root and run:

```bash
git clone https://github.com/jptaycs/RenderFlow.git /opt/renderflow
cd /opt/renderflow
bash deploy/setup_server.sh app.yourdomain.com
```

The script is idempotent (safe to re-run) and does, in order: apt packages
(python3, ffmpeg, DejaVu fonts), Docker + Caddy from their official repos,
a `renderflow` system user, the venv with all pipeline extras, the one-time
model downloads (kokoro ~340 MB, wav2lip ~520 MB), a generated production
`.env` (fresh secret key + DB password — never overwrites an existing one),
the datastore containers, the systemd services, the daily backup timer, and
the Caddyfile for your domain.

Then:

1. `nano /opt/renderflow/.env` — fill in provider keys (`PEXELS_API_KEY`,
   `POLLINATIONS_TOKEN`, etc.) and check the provider selection. The
   template ships the free stack (kokoro TTS, pollinations images,
   wav2lip-local avatar).
2. `systemctl restart renderflow-api renderflow-worker`
3. Open `https://app.yourdomain.com` and **register your own account
   first** — the first account becomes the admin (unlimited videos, can
   grant subscriptions) and adopts any projects already on disk.
4. Smoke-test: create a short video end-to-end, then check two things that
   differ from macOS dev: captions render in a real bold font (DejaVu), and
   the "AI-generated host" disclosure banner appears on avatar clips
   (Linux ffmpeg has drawtext; the dev Mac's build doesn't).

Production safety is enforced at startup: the API refuses to boot if
`RENDERFLOW_ENV=production` is combined with any dev flag
(`RENDERFLOW_DEV_LOGIN_*`, `RENDERFLOW_DEV_CHECKOUT`) or the default
database password. Session cookies are Secure (TLS-only) in production.

## Updating to a new version

```bash
cd /opt/renderflow && bash deploy/deploy.sh
```

Pulls the latest code, installs dependencies, restarts both services
(database migrations run automatically on API startup), and health-checks
the result.

## Backups

`renderflow-backup.timer` runs `deploy/backup.sh` daily at 04:00: a
`pg_dump` of the database plus a tar of `projects/` (rendered videos,
scene plans, performance data) into `/var/backups/renderflow/`, keeping 14
days. **These backups live on the same disk** — for real durability, sync
them off-site, e.g. with rclone to Backblaze B2:

```bash
rclone sync /var/backups/renderflow b2:your-bucket/renderflow-backups
```

### Restore

```bash
systemctl stop renderflow-api renderflow-worker
gunzip -c /var/backups/renderflow/db-YYYY-MM-DD.sql.gz | \
  docker exec -i renderflow-postgres-1 psql -U renderflow renderflow
tar -xzf /var/backups/renderflow/projects-YYYY-MM-DD.tar.gz -C /opt/renderflow
systemctl start renderflow-api renderflow-worker
```

## Operations cheat-sheet

```bash
systemctl status renderflow-api renderflow-worker   # service health
journalctl -u renderflow-api -f                     # live API logs
journalctl -u renderflow-worker -f                  # live worker logs
docker compose -f /opt/renderflow/docker-compose.yml ps   # datastores
systemctl reload caddy                              # after Caddyfile edits
```
