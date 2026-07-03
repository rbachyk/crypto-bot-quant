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
# pg_restore exits non-zero when ANY error was ignored during the restore. The
# only benign errors for a custom-format dump taken with --no-owner
# --no-privileges are:
#   * the pre-installed plpgsql extension colliding with the dump's
#     CREATE/COMMENT ON EXTENSION entries (present in every fresh database);
#   * "unrecognized configuration parameter" on a dump-header SET (a newer
#     pg_dump client emitting session parameters — e.g. transaction_timeout
#     from PG17 — that an older server does not know; session config only,
#     no effect on restored schema/data).
# Anything else (missing tables, data errors, truncated dump) is a broken
# restore and MUST fail — no `|| true` here (audit M23).
set +e
RESTORE_OUT="$(pg_restore --no-owner --no-privileges --dbname="$TMP_URL" "$DUMP" 2>&1)"
RESTORE_RC=$?
set -e
[[ -n "$RESTORE_OUT" ]] && echo "$RESTORE_OUT"

RESTORE_OK=1
if [[ $RESTORE_RC -ne 0 ]]; then
  REAL_ERRORS="$(printf '%s\n' "$RESTORE_OUT" \
    | grep -Ei 'error|fatal|\[archiver\]' \
    | grep -Ev 'extension "plpgsql" already exists|COMMENT ON EXTENSION|must be owner of extension|errors ignored on restore|unrecognized configuration parameter' \
    || true)"
  if [[ -n "$REAL_ERRORS" ]]; then
    RESTORE_OK=0
    echo "pg_restore reported non-benign errors (exit=${RESTORE_RC}):"
    echo "$REAL_ERRORS"
  else
    echo "pg_restore exit=${RESTORE_RC} but only documented-benign errors were ignored."
  fi
fi

TABLE_COUNT="$(psql "$TMP_URL" -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")"
echo "Restored table count: ${TABLE_COUNT}"

# Row-count sanity: an exact total across all restored public tables. A backup
# of any initialised database contains rows (at minimum alembic_version), so a
# "successful" restore that brought back zero rows is a broken backup.
ROW_COUNT="$(psql "$TMP_URL" -tAc "
  SELECT coalesce(sum((xpath('/row/cnt/text()', x))[1]::text::bigint), 0)
  FROM (
    SELECT query_to_xml(format('SELECT count(*) AS cnt FROM %I.%I', table_schema, table_name),
                        false, true, '') AS x
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
  ) t;")"
echo "Restored total row count: ${ROW_COUNT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${REPORT_DIR}/restore_test_${STAMP}.json"
# PASS requires the restore itself to have succeeded AND the restored contents
# to be sane (schema came back and it is not empty).
if [[ "$RESTORE_OK" -eq 1 && "${TABLE_COUNT}" -gt 0 && "${ROW_COUNT}" -gt 0 ]]; then
  STATUS="PASS"
else
  STATUS="FAIL"
fi
cat > "$REPORT" <<EOF
{
  "status": "${STATUS}",
  "dump": "${DUMP}",
  "temp_database": "${TMPDB}",
  "restore_exit_code": ${RESTORE_RC},
  "restore_ok": $([[ "$RESTORE_OK" -eq 1 ]] && echo true || echo false),
  "restored_table_count": ${TABLE_COUNT},
  "restored_row_count": ${ROW_COUNT},
  "timestamp": "${STAMP}"
}
EOF
echo "Restore-test report: ${REPORT} (status=${STATUS})"

[[ "${STATUS}" == "PASS" ]]
