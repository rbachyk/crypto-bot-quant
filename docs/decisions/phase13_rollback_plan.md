# Phase 13 — Controlled Live Readiness: Rollback Plan & Deployment Decisions

## Decision context

Phase 13 delivers the final pre-live gates: SEC, DEPLOY, BACKUP, MON, CONFIG-FREEZE,
LEARN-PROMO-L, and LIVE. These gates confirm the system is operationally ready.

Live trading is never the default; it requires every gate to pass AND a manual "Go Live"
click on the dashboard. This document is the rollback plan required by the DEPLOY gate
(Appendix A, AGENTS.md).

---

## Rollback Plan

### Version rollback

Every git tag corresponds to a frozen config version. To roll back:

1. `git checkout <previous-tag>` on the production node.
2. `docker compose down` — stop all services.
3. `docker compose up -d` — restart with previous image tags (images are tagged by
   `CONFIG_VERSION`).
4. Run `make migrate` (Alembic `--sql` mode to verify it is a safe downgrade).
5. Run `make health` to confirm all services healthy.
6. Run `make -s run-gate GATE=INFRA FORMAT=json` to verify infrastructure gate.
7. Notify operator — resume paper mode only; live re-activation requires a full
   LIVE gate pass again.

### Learner rollback (separate from config rollback)

The `RollbackGuard` handles automatic learner rollback:

1. Trigger detected → `controller.freeze()` called → mode = FROZEN.
2. `frozen_fallback_policy` loaded from `var/adaptation/frozen_fallback.pkl`.
3. Trading continues on the last-approved deterministic policy.
4. Recovery: manual review → re-run LEARN-PROMO-L gate → manual approval.

The learner kill switch (`make kill`) is independent of the trading kill switch and
disables adaptation without stopping the trading engine.

### Database rollback

1. Run `make backup-db` before any schema migration.
2. To restore: `make restore-test` (dry-run) or `bash scripts/restore_test.sh`.
3. Full restore: `pg_restore --clean --dbname=$PG_URL $BACKUP_DIR/latest.dump`.

### Emergency close

If live positions must be closed immediately:
1. `make kill` — halts new order generation.
2. Dashboard → Emergency Close Mode (requires explicit confirmation + audit log).
3. Market orders close all open positions.
4. All actions logged to `audit_log`.

---

## Deployment decisions

### Decision: MVP single-node topology

The initial live deployment uses a single-node Docker Compose topology
(Appendix B.12). The live engine is isolated in its own container (`trading-engine-live`)
behind the `live` compose profile.

Why: simplicity; lower operational complexity for initial live validation; the risk
envelope and kill switch provide safety without requiring a split topology.

Assumption: if live performance is stable for 30+ days, migrate to the split topology
(research node + production node) per Appendix B.12.

### Decision: No real API keys in version control

All API keys are passed via `.env` (not committed). The `.env.example` contains only
safe placeholders. The DEPLOY gate and SEC gate enforce this.

### Decision: Config-freeze via git tags

The frozen `CONFIG_VERSION` is git-tagged as `config/<CONFIG_VERSION>` before any
live activation. The CONFIG-FREEZE gate verifies all version strings are non-empty.

### Decision: LIVE gate passes but Go-Live is still manual

The LIVE gate PASS means "the system is technically ready." The actual
"Go Live" action is a second manual step on the dashboard (Section 27 AGENTS.md).
This two-step design ensures no accidental live activation.

---

## Gate dependency map for Phase 13

```
SEC ──────────────┐
DEPLOY ───────────┤
BACKUP ───────────┤── CONFIG-FREEZE ──┐
MON ──────────────┘                   │
                                      └── LIVE
LEARN-PROMO-S ── LEARN-PROMO-L ───────┘  (+ all upstream Phases 1-12)
```

---

*Decision author: claude-sonnet-4-6, Phase 13, 2026-06-18*
*Assumption: Live activation will be preceded by an actual 72h paper/testnet soak run
as required by LIVE-1 criterion.*
