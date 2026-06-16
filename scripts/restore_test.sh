#!/usr/bin/env bash
# Restore test (AGENTS.md Appendix B.15). Restores the latest backup into a
# throwaway database and verifies the schema came back, then drops it. A backup
# is only trustworthy if its restore is tested; the BACKUP gate links this
# report. Runnable from the dashboard as the `run_restore_test_check` job.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then set -a; source .env; set +a; fi

DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://postgres:postgres@localhost:5432/trading_bot}"
PG_URL="${DATABASE_URL/+psycopg/}"
BACKUP_DIR="${BACKUP_PATH:-var/backups}"
REPORT_DIR="reports/backup"
mkdir -p "$REPORT_DIR"

# Ensure a backup exists.
if [[ ! -f "${BACKUP_DIR}/latest.txt" ]]; then
  echo "No backup found; creating one first."
  bash scripts/backup_db.sh
fi
DUMP="$(cat "${BACKUP_DIR}/latest.txt")"
echo "Restoring from: ${DUMP}"

# Derive base URL and a temp database name.
BASE="${PG_URL%/*}"          # strip /dbname
DBNAME="${PG_URL##*/}"
TMPDB="${DBNAME}_restore_test_$$"
ADMIN_URL="${BASE}/${DBNAME}"   # connect to existing db to issue CREATE DATABASE
TMP_URL="${BASE}/${TMPDB}"

cleanup() {
  psql "$ADMIN_URL" -v ON_ERROR_STOP=0 -c "DROP DATABASE IF EXISTS \"${TMPDB}\";" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Creating temp database ${TMPDB}"
psql "$ADMIN_URL" -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"${TMPDB}\";"

echo "Restoring dump into ${TMPDB}"
pg_restore --no-owner --no-privileges --dbname="$TMP_URL" "$DUMP" || true

TABLE_COUNT="$(psql "$TMP_URL" -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")"
echo "Restored table count: ${TABLE_COUNT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${REPORT_DIR}/restore_test_${STAMP}.json"
if [[ "${TABLE_COUNT}" -gt 0 ]]; then
  STATUS="PASS"
else
  STATUS="FAIL"
fi
cat > "$REPORT" <<EOF
{
  "status": "${STATUS}",
  "dump": "${DUMP}",
  "temp_database": "${TMPDB}",
  "restored_table_count": ${TABLE_COUNT},
  "timestamp": "${STAMP}"
}
EOF
echo "Restore-test report: ${REPORT} (status=${STATUS})"

[[ "${STATUS}" == "PASS" ]]
