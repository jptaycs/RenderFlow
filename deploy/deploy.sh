#!/usr/bin/env bash
# Update the running production instance to the latest code:
#
#   cd /opt/renderflow && bash deploy/deploy.sh
#
# DB migrations run automatically on API startup (db.init_db).
set -euo pipefail

APP_DIR=/opt/renderflow
APP_USER=renderflow

cd "$APP_DIR"

echo "==> pulling latest code"
sudo -u "$APP_USER" git pull --ff-only

echo "==> installing dependencies"
sudo -u "$APP_USER" .venv/bin/pip install -q -e '.[web]'

echo "==> restarting services"
systemctl restart renderflow-api renderflow-worker

echo "==> health check"
sleep 3
body=$(curl -sf http://127.0.0.1:8321/api/auth/dev-login)
if [[ "$body" != '{"enabled":false}' ]]; then
    echo "unexpected health-check response: $body" >&2
    echo "(dev login must be disabled in production; check .env and logs:" >&2
    echo " journalctl -u renderflow-api -n 50)" >&2
    exit 1
fi
echo "OK — API is up and in production posture."
