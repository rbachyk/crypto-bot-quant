# Spec compliance vs AGENTS.md — status & known gaps

**Source of truth is [`AGENTS.md`](../AGENTS.md).** This file records, honestly, where the
implementation is complete and where it diverges or is incomplete, so that a green gate run is
not mistaken for full spec compliance or true live-readiness. Audited 2026-06-18.

## TL;DR

- **You can run everything locally on a Mac and pass the full gate chain** (`make run-all-gates`
  → 31/31 PASS incl. `LIVE`). The gate system and local setup are complete and consistent.
- **Gate-green is NOT the same as "ready for real money."** Several `LIVE`-gate criteria are
  operator-attested or run against synthetic/seeded data and PASS deterministically in this
  environment. And several AGENTS.md sections that the gates do **not** assert are incomplete
  (below). Real live trading additionally requires the items in "Before real live" at the end.

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

Priority order. "Safety-critical" = must fix before real money.

1. **[SAFETY] Regime detection is largely missing (Section 11).** There is no deterministic
   regime engine and no `regime_priority` config. The pipeline emits ad-hoc labels
   (`trend_up`/`range`, `high_vol_up`/`low_vol_down`) and **never** the `R*` codes, so the
   no-trade protection `NO_TRADE_REGIMES = {R8_DATA_UNSAFE, R7_TOXIC_EXECUTION, R4_HIGH_VOL_CHOP}`
   (`src/ranking/setup_quality.py:24`) **can never fire** — it is dead code against the real
   pipeline. (Toxic spread / stale data are still caught separately in execution revalidation, so
   there is partial overlapping protection, but the regime layer itself is inert.)
2. **[SAFETY] Learner `rollback.revert()` is freeze-only (Section 21.7).** `src/adaptation/rollback.py`
   freezes the learner but does not restore the last-good frozen config, cancel orders, or alert;
   the Phase-13 check that reports "revert path functional" only round-trips a snapshot. Misleading
   green. (Not reachable today — the learner is shadow-only and not wired to the trading path.)
3. **Live Data Manager missing (Section 8).** No websocket feed, stale-stream/disconnect detection,
   or ws-vs-REST cross-check. The live loop (`src/live/loop.py`) is snapshot **replay** only; a
   real-time feed must be built behind the existing `MarketFeed` Protocol before live operation.
   `candidate.data_fresh` is a static input, not produced by a staleness monitor.
4. **Explainability not persisted (Section 24).** `decision_log` is an in-memory dataclass
   serialized into the paper-session JSON, not a queryable DB table; the `TradeExplainability`
   schema ("no trade without it") is not implemented.
5. **Dashboard: 7 of 23 spec pages have no dedicated route (Section 25).** Missing: Data Coverage,
   Universe, Live Trading, Execution Quality, Risk, Online Learning, RL, Settings. Strategy/Regime/
   Session analytics are folded into one `/dashboard/analytics` page. Time-period selector and the
   persistent gate-status widget are on the performance pages but not literally every page; the 7
   entity-scoped time filters (by run / session / config / universe / strategy / model version) are
   not implemented.
6. **Anti-overfitting controls partial (Section 16).** Deflated Sharpe / multiple-testing
   correction, effective sample size, and purged+embargoed CV are absent (walk-forward + locked
   hold-out + stress + shuffle guard ARE present).
7. **Risk checklist partial (Section 17).** Liquidation-distance, margin-availability, weekly-loss
   limit and funding circuit-breaker are not implemented; circuit breakers are portfolio-level, not
   per-symbol.
8. **`LiveActivationRequest` typed schema missing (Section 27).** The activation request is a generic
   `Approval` row (`subject_type="live_activation"`) without the typed version fields.
9. **Reporting envelope not enforced (Section 34).** Reports lack the required
   methodology/limitations/recommendations/versions envelope; a few named reports (live, RL,
   online-learning, live-readiness, daily-review) have no dedicated generator.

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
