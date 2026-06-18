# Quant + ML + Adaptive Crypto Trading Bot

Production-grade, quant-first, risk-first crypto perpetual-futures trading
platform. **The single source of truth is [`AGENTS.md`](./AGENTS.md)** — read it
fully. The Priority Stack (Section 1) resolves every conflict; **capital
protection wins**. Live trading is never enabled by default and only becomes
possible after every required gate passes (Section 27).

This repository is built in phases (AGENTS.md Section 32). **All 13 phases are
implemented**: infrastructure (config/env validation, PostgreSQL/Alembic, a
Redis-backed job system, health checks, dashboard + auth, independent kill
switch, data lake), the data platform + universe + feature pipeline, an
event-based backtest engine with walk-forward + fee/slippage stress, deterministic
quant strategies (families A/B/G) with a promotion/shelving research harness,
ranking + the immutable risk envelope + the execution core (atomic bracket orders,
ownership, reconciliation), a paper-trading engine, shadow ML + RL/online-learning
(shadow-only), and the full Gate Runner covering every gate from `INFRA` through
`LIVE`.

> **Scope note — real market data in, simulated execution out.**
> Real **public market data** is wired via `ccxt` (default **Bybit** USDT linear perps):
> `qbot download --exchange bybit ...` ingests real OHLCV / mark / index / funding /
> open-interest (+ estimated spread) into a versioned `DATA_VERSION` snapshot, so
> backtests/paper/shadow iterate on real data without losing prior versions. The default
> source stays the offline deterministic **skeleton** (tests + gates run with no network);
> a real exchange id opts in. **Order placement / account access is still skeleton** — the
> paper engine fills via a `SimulatedVenue`. Reaching real live trading still requires a
> real *trading* venue adapter, `[VERIFIED]` exchange metadata, and operator sign-off, and
> is hard-gated off by default (see the live safety checklist below).

## Local setup

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), and a reachable
PostgreSQL + Redis. If you don't run them locally, start the dockerised ones with
`make docker-up` **first** and wait for them to report healthy, then run the
host-side steps below (they connect to `127.0.0.1:5432` / `6379`).

```bash
uv sync                      # or: make setup
cp .env.example .env         # edit DASHBOARD_PASSWORD (DB/Redis URLs default to localhost)
make docker-up               # OPTIONAL: postgres + redis (+ stack) if not running locally
make migrate                 # apply Alembic migrations (idempotent; targets head)
make health                  # probe db / redis / storage / kill switch  → status: healthy
make test lint typecheck     # full quality suite (must be clean)
```

> macOS note: if a local (Homebrew) PostgreSQL already owns port 5432 it will
> shadow the docker one and `make migrate` will hit the wrong database — stop it
> (`brew services stop postgresql@<v>`) or remap the published port.

Once healthy you can exercise the system end-to-end without an exchange:

```bash
make run-all-gates                          # every gate in dependency order
uv run python -m src.gates.runner --gate BT --json    # one gate as JSON
```

Or download **real** market data (Bybit) into a versioned snapshot and backtest on it:

```bash
# Real Bybit public data → immutable DATA_VERSION snapshot (OI sampled at 1h).
uv run python -m src.cli.main download --config configs/data.bybit.yaml --days 5
# Then run the event-based engine on that snapshot (src.backtest.service.run_lake_backtest).
```

Each download is an immutable `DATA_VERSION` snapshot, so research iterations never lose
prior data. `build_lake_inputs()` feeds the lake through the **one** feature pipeline (the
Parity Rule) into the same engine the BT/WF/FEE/SLIP gates use. Note Bybit only serves recent
open-interest (≈8 days at 1h), so keep download windows recent or sample OI coarsely.

Run real-data iterations and compare them on a leaderboard (each run is tagged with its
`DATA_VERSION` so nothing is lost and iterations stay comparable):

```bash
uv run python -m src.cli.main backtest-lake --config configs/data.bybit.yaml --timeframe 1h
uv run python -m src.cli.main backtest-lake --strategy basis_reversion --timeframe 1h  # real candidate
uv run python -m src.cli.main leaderboard            # ranked by the profitability bar
```

`--strategy <candidate_id>` runs a real research strategy (families A/B/G) instead of the
reference self-test; omit it for the reference momentum baseline. (Note: candidate thresholds
are tuned for the synthetic fixtures — e.g. `basis_reversion`'s 15bps premium trigger exceeds
the real Bybit BTC basis of ≈5–8bps — so expect to retune params on real data via the leaderboard.)

The leaderboard ranks the best run per (strategy, snapshot, timeframe) by the profitability
bar (expectancy ≥ 0.03R, PF ≥ 1.10, max-DD ≤ 0.25, enough trades) — see the **Leaderboard**
dashboard page (`/dashboard/leaderboard`). It is a comparison aid; the BT/WF/FEE/SLIP gates
remain the binding profitability judgement before anything advances toward live.

Run the control center and a worker:

```bash
uv run uvicorn src.api.app:app --reload     # dashboard + API on :8000
make run-worker-data                        # a job worker (dedicated process)
```

The dashboard (one FastAPI app, served via Caddy at `https://localhost` under docker) has:
**Gates**, **Road to Live** (a live-readiness score over the `blocks_live` gates, in dependency
order, with a *Request live-activation approval* button at 100%), **Backtests** (*Run backtest* →
background `backtest` worker → stored in `backtest_runs`), **Leaderboard** (real-data iterations
ranked by the profitability bar, grouped by `DATA_VERSION` snapshot), **Paper** (*Run paper session* → sources
candidates only from **promoted** strategies → trades stored in `paper_trades`), **ML Shadow**
(*Run ML shadow pass* → `shadow_logs`, with the applied-count = 0 enforcement shown), **Jobs**
(with Cancel/Retry), **Statistics**, **Remediation**, **Approvals** (Approve/Reject), **Audit
Logs**, **Reports**, **Health**, plus a **Kill Switch** panel on the overview. Background work is
routed to dedicated per-class workers (`data`/`backtest`/`ml`/`rl`/`gates`/`default`), and the
`scheduler` service enqueues recurring shadow-only jobs (research re-validation, paper sessions,
ML shadow passes) gated by the `ENABLE_*` toggles.

End-to-end research flow: **research promotes candidates → `strategy_promotions` registry →
paper sources promoted strategies → `paper_trades` → dashboard**. Alerts deliver to the log/
dashboard sink plus optional Telegram/email transports (`ALERT_TELEGRAM_*` / `ALERT_EMAIL_*`).

Lifecycle is **backtest/research → paper → live** (there is no separate "demo" stage). The
backtest gates (BT/WF/FEE/SLIP) enforce the profitability bar (walk-forward expectancy ≥ 0.03R,
PF ≥ 1.10, max-DD ≤ 0.25, survives ×2 fees / +50% slippage); paper gates (PAPER-A/B) verify
pipeline correctness and trade volume, not paper profitability. Live stays disabled until the
full gate chain (incl. `LIVE`) passes **and** `TRADING_MODE=LIVE` + `APP_ENV=production` +
`ENABLE_LIVE_TRADING=true` + an operator sign-off — there is no "go live" button.

Training the learned layers (ML meta-labeler, RL/online learner) is **shadow-only** and
documented in [`docs/ml_rl_training.md`](docs/ml_rl_training.md) — when, how, and what data.

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
Appendix A). All of these gates are now implemented and enforced by the Gate
Runner — but live trading additionally requires wiring a real venue adapter (the
shipped one is an offline skeleton) and a manual operator sign-off, and stays
disabled unless `TRADING_MODE=LIVE` **and** `APP_ENV=production` **and**
`ENABLE_LIVE_TRADING=true`.

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
src/data/        data lake ingestion, coverage, quality, schema
src/universe/    tradable-universe builder
src/features/    the single causal feature pipeline (the Parity Rule)
src/backtest/    event-based engine, metrics, walk-forward, fee/slippage stress
src/strategies/  deterministic candidates (A/B/G) + research promotion harness
src/ranking/     setup-quality scoring + candidate ranking
src/risk/        immutable risk envelope, breakers, deterministic sizing
src/execution/   bracket orders, ownership, reconciliation, simulated venue
src/paper/       paper-trading engine (full pipeline on SimulatedVenue)
src/ml/          shadow ML (models, predictor, scorer, registry) — shadow-only
src/adaptation/  RL / online-learning policy (shadow-only)
src/gates/       gate catalog, checks (phase6–13), runner (python -m src.gates.runner)
src/exchange/    exchange adapter skeleton (the only path to the venue)
src/monitoring/  health checks + alerting
src/api/         FastAPI backend + dashboard + auth
src/cli/         qbot CLI (kill switch, health, gate, worker, enqueue)
src/killswitch.py  dual-backend (file + redis) fail-safe kill switch
migrations/      Alembic
configs/         gates.yaml + risk/strategy/feature/execution configs
scripts/         backup_db.sh, restore_test.sh
docs/decisions/  recorded assumptions
```
