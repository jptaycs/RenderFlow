#!/usr/bin/env bash
# Daily backup: pg_dump of the database + tar of projects/ (rendered videos,
# scene plans, performance data) into /var/backups/renderflow, keeping
# RETENTION_DAYS. Runs via renderflow-backup.timer (04:00). These backups
# are on the SAME disk — sync them off-site for real durability (README).
set -euo pipefail

APP_DIR=/opt/renderflow
BACKUP_DIR=/var/backups/renderflow
RETENTION_DAYS=14
STAMP=$(date +%F)

mkdir -p "$BACKUP_DIR"

docker exec renderflow-postgres-1 pg_dump -U renderflow renderflow \
    | gzip > "$BACKUP_DIR/db-$STAMP.sql.gz"

tar -czf "$BACKUP_DIR/projects-$STAMP.tar.gz" \
    -C "$APP_DIR" projects

find "$BACKUP_DIR" -name '*.gz' -mtime +"$RETENTION_DAYS" -delete

echo "backup complete: $BACKUP_DIR/{db,projects}-$STAMP"
