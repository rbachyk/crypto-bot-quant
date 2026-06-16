# Quant + ML + Adaptive Crypto Trading Bot

Production-grade, quant-first, risk-first crypto perpetual-futures trading
platform. **The single source of truth is [`AGENTS.md`](./AGENTS.md)** — read it
fully. The Priority Stack (Section 1) resolves every conflict; **capital
protection wins**. Live trading is never enabled by default and only becomes
possible after every required gate passes (Section 27).

This repository is built in phases (AGENTS.md Section 32). **Phase 1 —
Infrastructure Foundation** is implemented here: config + environment
validation, PostgreSQL/Alembic, a Redis-backed job system, health checks, a
dashboard shell with auth, an independent kill switch, exchange/universe
skeletons, a data lake, and the Gate Runner with the `INFRA`, `DB`, `QUEUE`,
`STORAGE`, `MON` (skeleton) and `BACKUP` (skeleton) gates.

## Local setup

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), and a reachable
PostgreSQL + Redis (either local services or `make docker-up`).

```bash
uv sync                      # or: make setup
cp .env.example .env         # edit DATABASE_URL / REDIS_URL / DASHBOARD_PASSWORD
make migrate                 # apply Alembic migrations (idempotent)
make health                  # probe db / redis / storage / kill switch
make test lint typecheck     # quality gates
```

Run the control center and a worker:

```bash
uv run uvicorn src.api.app:app --reload     # dashboard + API on :8000
make run-worker-data                        # a job worker (dedicated process)
```

Run a gate:

```bash
make run-gate GATE=INFRA          # or: python -m src.gates.runner --gate INFRA --json
make run-all-gates                # all gates in dependency order
```

## Docker (single-node MVP topology — Appendix B.12)

```bash
make docker-up      # postgres+timescale, redis, backend, dashboard, workers, caddy
make health
make docker-down
```

`trading-engine-live` is defined but **disabled by default** (it sits behind the
`live` compose profile and additionally refuses to boot unless
`APP_ENV=production` and `ENABLE_LIVE_TRADING=true`).

## Deployment setup (summary)

- Recommended split topology (Appendix B.12): a lightweight production node
  (backend, dashboard, paper/live engines, light monitoring) and a research node
  (data lake, backtest/ML/RL workers). Heavy research must never run inside the
  live execution process.
- Put the dashboard behind auth **and** a VPN (Tailscale/WireGuard); expose only
  443 + SSH; harden SSH (Appendix C, `SEC` gate).
- Configure scheduled DB + artifact backups and a **tested** restore
  (`make backup-db`, `make restore-test`) before any live activation.
- Freeze and git-tag all versions (`CONFIG_FREEZE` gate) before going live.

## Live safety checklist (never bypass)

Live trading is gated behind **every** item below (AGENTS.md Sections 2, 27;
Appendix A). Phase 1 satisfies none of the trading gates yet — this list is the
contract for later phases.

1. `TRADING_MODE=LIVE` only with `APP_ENV=production` **and**
   `ENABLE_LIVE_TRADING=true` (enforced in `src/config/settings.py`).
2. Exchange metadata `[VERIFIED]`; no `[UNVERIFIED]` for active symbols (`META`).
3. Exchange-resident stop-loss attached atomically at entry; native trailing
   (`EXEC`).
4. Risk envelope enforced: per-trade `risk_cap`, leverage cap, portfolio heat
   cap, net-beta cap, daily-loss & max-drawdown breakers (`RISK`).
5. Order ownership: every order carries `ORDER_CLIENT_ID_PREFIX`; foreign orders
   → halt + alert (`ORDER-OWN`).
6. Kill switch verified independent of the dashboard (`KILL`; `make kill`).
7. Startup reconciliation of all positions/orders; mismatch → halt + alert.
8. Backtest/walk-forward/fee/slippage gates pass; paper A+B pass.
9. Backups configured with a **tested** restore (`BACKUP`).
10. Monitoring + alerts deliver end-to-end (`MON`).
11. Config frozen + git-tagged (`CONFIG-FREEZE`); operator sign-off (`LIVE`).

**Profit alone is not success.** A failed gate is never a dead end — the
dashboard shows ordered remediation steps from `configs/gates.yaml`.

## Layout

```
src/config/      env-validated, versioned settings
src/db/          SQLAlchemy models + engine (jobs, gates, audit, …)
src/jobs/        Redis+Postgres job queue, worker, handlers
src/gates/       gate catalog, checks, runner (CLI: python -m src.gates.runner)
src/storage/     data lake + artifact store (versioned snapshots/manifests)
src/exchange/    exchange adapter skeleton (the only path to the venue)
src/universe/    universe builder skeleton
src/monitoring/  health checks + alert skeleton
src/api/         FastAPI backend + dashboard shell + auth
src/cli/         qbot CLI (kill switch, health, gate, worker)
migrations/      Alembic
configs/         gates.yaml (gate single source of truth)
scripts/         backup_db.sh, restore_test.sh
docs/decisions/  recorded assumptions
```
