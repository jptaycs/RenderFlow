#!/usr/bin/env bash
# One-time (idempotent — safe to re-run) production setup for RenderFlow on
# Ubuntu 24.04. Run as root from the repo checkout:
#
#   git clone <repo> /opt/renderflow
#   cd /opt/renderflow && bash deploy/setup_server.sh app.yourdomain.com
#
# See deploy/README.md for the full runbook.
set -euo pipefail

DOMAIN="${1:?usage: setup_server.sh <domain>  (e.g. app.yourdomain.com)}"
APP_DIR=/opt/renderflow
APP_USER=renderflow

if [[ $EUID -ne 0 ]]; then
    echo "run as root (sudo bash deploy/setup_server.sh $DOMAIN)" >&2
    exit 1
fi
if [[ ! -f "$APP_DIR/pyproject.toml" ]]; then
    echo "repo not found at $APP_DIR — clone it there first (see deploy/README.md)" >&2
    exit 1
fi

echo "==> apt packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -yq git curl ca-certificates gnupg ffmpeg \
    python3-venv python3-pip fonts-dejavu-core fonts-liberation

echo "==> docker"
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sh
fi

echo "==> caddy"
if ! command -v caddy >/dev/null; then
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -q
    apt-get install -yq caddy
fi

echo "==> service user"
if ! id "$APP_USER" >/dev/null 2>&1; then
    # Real home dir: huggingface/torch caches (parallax depth model) live in
    # ~/.cache, and the model setup scripts write into the repo dir.
    useradd --system --create-home --home-dir /var/lib/renderflow \
        --shell /usr/sbin/nologin "$APP_USER"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "==> python venv + dependencies (this can take a while: torch)"
sudo -u "$APP_USER" bash -c "
    set -euo pipefail
    cd '$APP_DIR'
    [ -d .venv ] || python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e '.[web,kokoro,parallax,wav2lip]'
"

echo "==> model downloads (kokoro ~340 MB, wav2lip ~520 MB; skipped if present)"
sudo -u "$APP_USER" bash -c "
    set -euo pipefail
    cd '$APP_DIR'
    [ -d .kokoro ] || .venv/bin/python scripts/setup_kokoro.py
    [ -d .wav2lip ] || .venv/bin/python scripts/setup_wav2lip.py
"

echo "==> production .env"
if [[ ! -f "$APP_DIR/.env" ]]; then
    SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    PG_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
    sed -e "s/__SECRET_KEY__/$SECRET_KEY/" \
        -e "s/__PG_PASSWORD__/$PG_PASSWORD/g" \
        "$APP_DIR/deploy/env.production.example" > "$APP_DIR/.env"
    chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    echo "    wrote $APP_DIR/.env (generated secret key + DB password)"
else
    echo "    $APP_DIR/.env already exists — leaving it untouched"
fi

echo "==> datastores (postgres + redis)"
(cd "$APP_DIR" && docker compose up -d)

echo "==> systemd services"
cp "$APP_DIR"/deploy/renderflow-api.service /etc/systemd/system/
cp "$APP_DIR"/deploy/renderflow-worker.service /etc/systemd/system/
cp "$APP_DIR"/deploy/renderflow-backup.service /etc/systemd/system/
cp "$APP_DIR"/deploy/renderflow-backup.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now renderflow-api renderflow-worker renderflow-backup.timer

echo "==> caddy vhost for $DOMAIN"
sed "s/__DOMAIN__/$DOMAIN/" "$APP_DIR/deploy/Caddyfile" > /etc/caddy/Caddyfile
systemctl reload caddy || systemctl restart caddy

echo
echo "Done. Next steps (deploy/README.md):"
echo "  1. Fill provider keys:  nano $APP_DIR/.env"
echo "  2. systemctl restart renderflow-api renderflow-worker"
echo "  3. Open https://$DOMAIN and register YOUR account first (it becomes admin)."
echo "  4. Smoke-test one video: captions font + avatar disclosure banner."
