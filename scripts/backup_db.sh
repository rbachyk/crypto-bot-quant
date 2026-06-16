#!/usr/bin/env bash
# Database backup (AGENTS.md Appendix B.15). Writes a timestamped custom-format
# pg_dump into the backup directory. Backups are mandatory before live (B.15),
# and a backup without a tested restore does NOT pass the BACKUP gate.
set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env if present (do not fail if absent).
if [[ -f .env ]]; then set -a; source .env; set +a; fi

DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://postgres:postgres@localhost:5432/trading_bot}"
PG_URL="${DATABASE_URL/+psycopg/}"
BACKUP_DIR="${BACKUP_PATH:-var/backups}"
mkdir -p "$BACKUP_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${BACKUP_DIR}/db_${STAMP}.dump"

echo "Backing up database -> ${OUT}"
pg_dump --format=custom --no-owner --no-privileges --dbname="$PG_URL" --file="$OUT"
echo "Backup complete: ${OUT} ($(du -h "$OUT" | cut -f1))"

# Keep a pointer to the latest backup for the restore test.
echo "$OUT" > "${BACKUP_DIR}/latest.txt"
