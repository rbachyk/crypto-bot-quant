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

> macOS notes:
> - If a local (Homebrew) PostgreSQL already owns port 5432 it will shadow the docker one and
>   `make migrate` will hit the wrong database — stop it (`brew services stop postgresql@<v>`) or
>   remap the published port.
> - The dockerised `caddy` publishes **443** for the dashboard (`https://localhost`); if another
>   local web server holds 443, change the published port or skip Caddy and hit the backend
>   directly (`uv run uvicorn src.api.app:app --reload` → `http://localhost:8000`). The host-side
>   gate / health / CLI flow used for live-readiness does not need Caddy.

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

Forward-test on real data through the paper + shadow-ML pipelines (both shadow-only):

```bash
uv run python -m src.cli.main paper-lake --config configs/data.bybit.yaml --timeframe 1h
uv run python -m src.cli.main ml-shadow-lake --config configs/data.bybit.yaml --timeframe 1h
```

`paper-lake` derives a candidate stream from the real snapshot and runs it through the full
paper pipeline (ranking → risk → execution → SimulatedVenue), persisting real `paper_trades`
(shown on the **Paper** page). `ml-shadow-lake` scores those real candidates with the shadow
ML meta-labeler, logging every prediction with `applied=False` (never influences trading).

The **live trading loop** drives that same pipeline one decision time at a time against a
chosen venue:

```bash
uv run python -m src.cli.main live --mode paper    --config configs/data.bybit.yaml --timeframe 1h
uv run python -m src.cli.main live --mode testnet  --config configs/data.bybit.yaml --timeframe 1h  # real sandbox orders, no funds
uv run python -m src.cli.main live --mode testnet  --transport ws --timeframe 1h   # + real-time websocket data-integrity halt
# --mode live is real money and is REFUSED unless every safety condition holds (below)
```

`--transport ws|rest` attaches a live data feed (websocket via ccxt.pro, or REST polling) to a
`LiveDataManager` (Section 8): stale-stream / disconnect / ws-vs-REST integrity failures halt the
loop like the kill switch. Add **`--realtime`** to drive the candidate stream from the live feed
itself (rolling window → the one feature pipeline → the strategy on each newly-closed bar) instead
of snapshot replay — continuous live operation (Section 35), still venue-gated.

- `--mode` selects the **venue**: `paper` = offline `SimulatedVenue`; `testnet`/`live` = the real
  ccxt venue (`CcxtLiveVenue`). The **endpoint** is set separately by **`EXCHANGE_ENV`** — three
  *different* Bybit environments with different keys: **`testnet`** (`testnet.bybit.com`),
  **`demo`** (`api-demo.bybit.com` — mainnet market data + a virtual-funds demo account; use this
  if your keys came from Bybit *demo trading*), or **`live`** (real money). So to trade on your
  demo account: set `EXCHANGE_ENV=demo` + `EXCHANGE_API_KEY/SECRET`, then `--mode testnet`.
  Either way the entry carries its exchange-resident stop-loss/take-profit **atomically**
  (Section 2.2) and the ownership prefix as `clientOrderId` (Section 7); only `EXCHANGE_ENV=live`
  is real money (and is additionally gated).
- `live` (real-money mainnet) order placement passes through the **`LiveActivationGuard`**,
  which refuses unless ALL hold: `TRADING_MODE=LIVE` + `APP_ENV=production` +
  `ENABLE_LIVE_TRADING=true`; **every `blocks_live` gate PASSes** (Road to Live = 100%); an
  **APPROVED `live_activation` sign-off** exists; and the order is within the bounded-live
  caps in `configs/live.yaml` (max orders/session, max open positions, max notional %). A
  refused live order is a graceful non-placement, never a trade. There is still no "go live"
  button — live stays off by default.

Run the control center and a worker:

```bash
uv run uvicorn src.api.app:app --reload     # dashboard + API on :8000
make run-worker-data                        # a job worker (dedicated process)
```

The dashboard (one FastAPI app, served via Caddy at `https://localhost` under docker) has a
**fixed left sidebar** with a sticky topbar (page title + an environment chip that turns red only
when real-money live is armed). All **23 pages required by Section 25 / Appendix B.8** are present,
grouped logically into four sections (`tests/test_dashboard_pages.py` asserts every one is
reachable and linked):

- **Performance** — **Overview** (a TradeZella-style board: KPI cards for net P&L, win rate,
  expectancy R, profit factor, max drawdown, trades, avg win/loss, fees; an equity curve; and
  per-strategy/per-symbol breakdowns over a selectable period — computed from the real
  `paper_trades`), **Statistics** + per-symbol views, **Strategy / Regime / Session** analytics,
  **Execution Quality**, **Risk**, **Analytics**. A compact live-readiness widget persists here.
- **Research & Testing** — **Backtests** (every run starts from the same fixed initial equity —
  shown on the page — so runs are directly comparable), **Leaderboard** (real-data iterations
  ranked by the profitability bar, grouped by `DATA_VERSION`), **Paper Trading**, **Reports**.
- **Live & Learning** — **Live Trading** (dashboard Start/Stop/Reset controls, described above),
  **ML Shadow**
  (`shadow_logs`, applied-count = 0 enforcement), **Online Learning**, **RL**.
- **Operations** — **Control Center** (gate status, jobs, universe, **Kill Switch**), **Data
  Coverage**, **Universe**, **Jobs** (Cancel/Retry), **Gates**, **Road to Live** (live-readiness
  score over the `blocks_live` gates with a *Request live-activation approval* button at 100%),
  **Remediation**, **Approvals**, **Audit Logs**, **System Health** (per-component probes),
  **Settings**.

The UI is server-rendered (no SPA) with a custom dark design system — styled cards/KPIs, pill
status badges, a segmented period selector, and custom-styled form controls (no default browser
chrome). Every statistics page carries a time-period selector (Section 25).

Background work is routed to dedicated per-class workers
(`data`/`backtest`/`ml`/`rl`/`live`/`gates`/`default`), and the `scheduler` service enqueues
recurring shadow-only jobs (research re-validation, paper sessions, ML shadow passes) gated by
the `ENABLE_*` toggles.

**Dashboard-only demo/live operation (no terminal).** Everything the `qbot live` command does is
also driven from the **Live Trading** page: a **Start** button enqueues a `run_live_session` job
on the dedicated `live` worker, the page streams its per-tick progress, and a **Stop** button
halts it cleanly (whatever executed is still saved) so you can restart any time. Each run is
tagged by `EXCHANGE_ENV` — a Bybit **demo** run lands under `demo:` session ids — so its
statistics stay **separated** from paper/testnet/live. When `EXCHANGE_ENV=demo`, the page also
shows a **Reset demo statistics** button that zeroes only the `demo:` runs/trades/decision-logs/
explainability (leaving every other environment intact) — press it for a clean slate before a
fresh demo-testing run. So a full demo cycle is: set `EXCHANGE_ENV=demo` + your demo
`EXCHANGE_API_KEY/SECRET` in `.env`, `make docker-up`, open **Live Trading** → *Reset demo
statistics* → *Start demo session* → watch progress / *Stop*.

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

## Going through the gates (Road to Live)

The intended loop is **run gate → read remediation → fix → re-run → green → advance** until every
`blocks_live` gate passes (Section 27). Locally on a Mac:

```bash
make docker-up && make migrate && make health     # deps healthy
make run-all-gates                                 # full chain in dependency order (31 gates)
make run-gate GATE=BT                              # re-run a single gate after a fix
uv run uvicorn src.api.app:app --reload            # then open https://localhost → Road to Live
```

The **Road to Live** dashboard page shows the live-readiness score (% of `blocks_live` gates passed),
the blocking criterion + next action for each gate, and a re-run button; at 100% it exposes a
*Request live-activation approval* button (the operator sign-off the `LiveActivationGuard` requires).

> **Important — a green gate run is not, by itself, "ready for real money."** All 31 gates PASS in a
> clean local checkout, but several `LIVE`-gate criteria are operator-attested or run against
> synthetic/seeded data, and several AGENTS.md requirements are **not** asserted by any gate. Read
> **[`docs/spec_compliance.md`](docs/spec_compliance.md)** for the honest status: what is complete,
> the known gaps/divergences (incl. two safety items — regime no-trade protection and the learner
> revert path), and the concrete list of what real live additionally needs (a profitable edge,
> `[VERIFIED]` metadata, a real-time data feed, real testnet creds, and the operator sign-off).

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
