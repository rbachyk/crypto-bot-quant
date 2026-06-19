# Spec compliance vs AGENTS.md — status & known gaps

**Source of truth is [`AGENTS.md`](../AGENTS.md).** This file records, honestly, where the
implementation is complete and where it diverges or is incomplete, so that a green gate run is
not mistaken for full spec compliance or true live-readiness. Audited 2026-06-18.

## TL;DR

- **You can run everything locally on a Mac and pass the full gate chain** (`make run-all-gates`
  → 31/31 PASS incl. `LIVE`). The gate system and local setup are complete and consistent.
- **Spec compliance is now essentially complete.** The original audit's gaps (items 1–9) plus the
  websocket feed, real-time live-loop mode, entity-scoped stats filters and the report envelope
  have all been implemented and tested. What remains is **by design** (the learner stays
  shadow-only until promoted) or **refinements** (per-tick streaming of point-in-time series),
  listed under "Remaining" below.
- **Gate-green is still NOT the same as "ready for real money."** Several `LIVE`-gate criteria are
  operator-attested or run against synthetic/seeded data and PASS deterministically in this
  environment. Real live trading additionally requires the items in "Before real live" at the end
  (a profitable edge, `[VERIFIED]` metadata, real testnet creds, and the operator sign-off).

## Complete / strong (matches spec)

- **Gate system** — all 25 Appendix-A gates + 6 Appendix-D infra/RL gates implemented, runnable,
  with remediation wired; dependency graph + Road to Live + re-run flow. (`src/gates/`, `configs/gates.yaml`)
- **Order ownership** (Section 7) — prefixed client ids, foreign-order detection. (`src/execution/ownership.py`)
- **Immutable risk envelope core** (Section 17) — hard ceilings, exact capital-agnostic sizing,
  min-notional no-trade, heat/beta caps, daily-loss/drawdown/kill breakers. (`src/risk/`)
- **Execution core** (Section 18) — atomic exchange-resident bracket, native trailing, ownership,
  reconciliation, cancel/replace; live venue (testnet) carries SL/TP atomically. (`src/execution/`)
- **Event-based backtest** (Section 19) — genuinely event-driven, shares the one `compute_features`
  path (Parity Rule), walk-forward + locked hold-out + fee/slippage stress. (`src/backtest/`)
- **Setup-quality gate** (Section 15) — exact weights + multi-symbol attribution. (`src/ranking/`)
- **Strategy families A/B/G** (Section 12) with full hypothesis metadata; promote/shelve lifecycle. (`src/strategies/`)
- **ML / RL layers are correctly SHADOW-ONLY / gated** (Sections 20–21) — never bind to live decisions.
- **Live-safety guard chain** (Section 27) — `settings.live_trading_allowed` predicate +
  `LiveActivationGuard` (all `blocks_live` gates PASS + operator sign-off + bounded caps), wired
  through the venue so a denied live order is a graceful non-placement. (`src/config/settings.py`, `src/live/guard.py`)
- **Real-data pipeline (M1–M8)** — Bybit download → versioned snapshots → backtest → leaderboard →
  paper → ML-shadow → live loop (paper/testnet). Performance dashboard from real `paper_trades`.

## Known gaps & divergences (NOT asserted by the gates)

### Resolved (audit items 1–7, 2026-06-19)

1. **Regime detection (Section 11) — RESOLVED.** `src/regime/` is a deterministic engine
   emitting the 8 `R*` codes with `configs/regime.yaml` `priority` (safest wins) + anti-whipsaw
   tracker; `NO_TRADE_REGIMES` is the canonical set the pipeline now emits, so the setup-quality
   no-trade guard actually fires. (`src/ranking/setup_quality.py`, `src/paper/lake.py`, `src/backtest/engine.py`)
2. **Learner `rollback.revert()` (Section 21.7) — RESOLVED.** `revert()` now freezes onto the
   frozen fallback, cancels learner orders, alerts, and writes `learner_log`; the Phase-13 gate
   actually exercises it. (`src/adaptation/rollback.py`, `src/gates/phase13.py`)
3. **Live Data Manager (Section 8) — RESOLVED.** `src/live/data_manager.py` does staleness +
   disconnect detection, REST backfill-after-reconnect, ws-vs-REST compare, and symbol/exchange
   halts; the live loop halts on exchange-wide integrity failure. A real-time **websocket** feed
   (`src/live/websocket_feed.py`, ccxt.pro on a daemon asyncio thread) now feeds it —
   `qbot live --transport ws|rest` attaches a `LiveDataManager` so a live data-integrity failure
   halts the loop. **Real-time mode** (`src/live/realtime.py`) now drives candidate generation
   from the live stream too (rolling window → feature pipeline → strategy on each new bar) via
   `qbot live --realtime --transport ws`.
4. **Explainability (Section 24) — RESOLVED.** `decision_logs` + `trade_explainability` are real
   DB tables (migration 0009); `TradeExplainability.ensure_complete()` blocks any trade it can't
   explain; the paper engine builds one per executed trade. (`src/explainability.py`)
5. **Dashboard pages (Section 25) — RESOLVED.** All previously-missing pages exist (Data Coverage,
   Universe, Live Trading, Execution Quality, Risk, Online Learning, RL, Settings) plus dedicated
   Strategy/Regime/Session analytics. (Residual: the 7 **entity-scoped** time filters — by
   run/session/config/universe/strategy/model version — remain partial; period + symbol + strategy
   scope exist.)
6. **Anti-overfitting (Section 16) — RESOLVED.** `src/backtest/overfitting.py`: deflated Sharpe,
   probabilistic Sharpe, effective sample size, purged+embargoed CV, sample adequacy; surfaced in
   the walk-forward report.
7. **Risk checklist (Section 17) — RESOLVED.** weekly-loss / funding / **per-symbol** breakers +
   liquidation-distance + margin-availability pre-trade checks (`src/risk/breakers.py`).
8. **`LiveActivationRequest` (Section 27) — RESOLVED.** Typed `src/live/activation.py` record built
   only when gates are 100%, attached as the live_activation approval's evidence.
9. **Report envelope (Section 34) — PARTIAL.** `src/reporting.py` provides the standard envelope
   (versions/period/methodology/results/limitations/recommendations) + validator, wired into the
   backtest report writer. (Residual: not yet applied to every report writer, and a few named
   reports — live, RL, online-learning, live-readiness, daily-review — still lack dedicated generators.)

### Resolved (continued)

- **Entity-scoped stats filters (Section 25) — RESOLVED.** `compute_trading_stats` + the
  Performance/Analytics pages now scope by **strategy** and **session** (paper or live/testnet)
  in addition to symbol + time period; `/api/stats/scopes` + a period/strategy/session selector.
  Version-scoped analytics (by config/universe/model version) live on the Leaderboard's
  `dataset_version`/`strategy` filters.
- **Report envelope + missing reports (Section 34) — RESOLVED.** `src/reports.py` generates the
  previously-missing named reports (live, online_learning, rl_simulation, rl_shadow,
  live_readiness, daily_review) wrapped in the standard envelope (`qbot reports`); the envelope
  is also wired into the paper and data-quality report writers.

### Remaining (by design / refinements only)

- The learner is **shadow-only / not wired to the live trading path** — intentional per Section 21
  until it is promoted through the LEARN-PROMO gates + manual review.
- The real-time feed streams **OHLCV** and REST-seeds the point-in-time series (mark/index/
  funding/OI); per-tick streaming of those is a refinement, not a correctness gap.
- Per-trade version columns (config/universe/model version) aren't stored on `paper_trades`, so
  version-scoped *trade* filtering is served via the run-level Leaderboard rather than the
  per-trade stats — as complete as the data model supports.

## Doc inconsistencies (fixed / noted)

- AGENTS.md Appendix-A summary table lists 25 gates but Appendix D mandates 31 (adds INFRA/DB/QUEUE/
  STORAGE/RL-SIM/RL-SHADOW). The implementation is the correct superset.
- `LIVE-0` and `LIVE-8` (Paper-B) are enforced via the gate dependency graph rather than as named
  `live_0`/`live_8` criterion objects.

## Before real live (beyond gate-green)

1. Find a **profitable edge** — strategies trade on real BTC but are not yet profitable with the
   synthetic-tuned params; use the leaderboard to find a configuration that clears the bar.
2. Fix the two **[SAFETY]** items above (regime no-trade protection; real learner revert).
3. Build the **live data manager** (real-time feed + staleness/disconnect halts).
4. Obtain **`[VERIFIED]` exchange metadata** (META gate against real, reconciled contract specs).
5. Provision **real exchange credentials** and run a real **testnet** smoke end to end.
6. Run the full gate chain to 100% on the production deployment and grant the **`live_activation`
   sign-off** (a second operator). Live stays off until `TRADING_MODE=LIVE` + `APP_ENV=production`
   + `ENABLE_LIVE_TRADING=true` and the guard authorises each bounded order.
