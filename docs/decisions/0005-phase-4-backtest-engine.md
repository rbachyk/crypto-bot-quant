# 0005 — Phase 4 backtest engine choices

Status: accepted · Date: 2026-06-17 · Phase: 4

Where AGENTS.md leaves a detail unspecified, the most capital-preserving and
honest-validation option is chosen and recorded here (AGENTS.md Conventions;
Priority Stack 3/5). Phase 4 delivers the event-based backtest engine, the
fee/slippage/funding cost models, risk + execution simulation, rejected-candidate
logging, the report generator, and the BT / WF / FEE / SLIP gates.

## A deterministic reference strategy + series, not a real strategy
The Roadmap ships validated trading strategies in Phase 5, but the BT/WF/FEE/SLIP
gates and the engine need *something* to run on now. Decision: a deterministic,
fully causal reference momentum strategy (`reference_momentum`,
`strategy_version: ref_bt_0001`) on a deterministic, offline reference series
(`src/backtest/reference.py`) with a **planted, causal** regime-switching trend
(a slow sinusoidal drift the past-only momentum rule can ride on both sides). This
exercises and proves the engine, walk-forward and stress machinery end-to-end
without claiming a real edge. The same series in `edge="noise"` mode (no drift,
i.i.d. returns) is the engine-level look-ahead guard: a causal strategy must show
~0 expectancy there. The fixture is never used for a live decision; it is a
labelled test fixture (Section 19). Real candidates flow through the identical
`run_engine` / walk-forward / stress paths in Phase 5.

## Look-ahead prevention is structural, not a post-hoc check
The strongest guard lives in the engine loop, not in a validator: a signal decided
on bar *k*'s close (feature row `decision_ts = (k+1)·iv`) can only fill at bar
*k+1*'s **open**. There is no code path by which a decision reads its own bar's
close or any future bar. Intrabar stop/take-profit use only the bar's own
high/low. This is asserted directly in `tests/test_backtest.py`
(`test_engine_fills_signal_at_next_bar_open_not_own_close`) and corroborated by
the noise-expectancy guard. Survivorship / future-universe leakage is prevented by
a point-in-time `activation_ts` per symbol (the reference universe activates SOL
partway through specifically to exercise it).

## Cost models tied to VERIFIED metadata; multipliers are the stress knobs
Fees come from the operator-`[VERIFIED]` `configs/metadata.yaml` (the META gate
guarantees they exist + are consistent); the YAML fallbacks apply only to a symbol
with no verified fee. Slippage = half-spread + a notional-vs-liquidity impact
term, always **adverse** to the taker (buys fill up, sells fill down). Funding is
charged to the open position at each funding timestamp (positive funding ⇒ longs
pay). `fee_multiplier` / `slippage_multiplier` / `funding_multiplier` are the
single knobs the FEE/SLIP stress runners turn (×2 fees, +50% slippage by default),
so the stressed run is the *same* backtest with multiplied costs — no parallel
code path.

## Risk simulation under-promises (full risk manager is Phase 6)
`src/backtest/risk.py` enforces the Section 17 per-trade sizing identity
`size = (equity × risk_pct) / |entry − stop|`, leverage-as-consequence
(`notional/equity` capped, never targeted), and the metadata
min-notional / min-order-size / lot-step gate — enough for an honest trade set and
rejected-candidate log. Portfolio heat, net-beta-to-BTC and correlation caps are
**deliberately not faked**; they land with the real risk manager in Phase 6
(`RISK` gate). The simulator rejects (and logs) the same candidates the real
manager would for the limits a single-position simulation can express.

## Walk-forward: notional train, real OOS folds, locked hold-out evaluated once
Strategy parameters are fixed in Phase 4 (real optimization is Phase 5), so each
fold is a pure out-of-sample evaluation on a disjoint, time-ordered segment and
the "train" portion is notional — this proves the *harness*. The most-recent
`holdout_frac` is a **locked hold-out**: untouched during all folds and evaluated
exactly once at the end (Section 16). WF passes only when ≥ `min_folds_passed`
folds clear every up-front kill-criterion AND the locked hold-out is positive net
of costs (edge stable across periods, not isolated to one). Re-based windows shift
all timestamps to a clean 0 origin while preserving causality (features for a bar
were still computed only from data at/before that bar's decision time).

## `backtest_version` is config-pinned; runs are content-addressed + indexed
All runtime behaviour is config-driven and versioned (Section 4):
`configs/backtest.yaml` pins `backtest_version: bt_0001` and every cost
assumption, fold layout, kill-criterion and stress multiplier — changing any is a
new `BACKTEST_VERSION`. Each run gets a content-addressed `run_id` (identical
inputs ⇒ identical id, idempotent) and an immutable `backtest_runs` index row
(Alembic migration 0004, idempotent `create_all(checkfirst=True)`); the full
Section 19 report (all metrics + breakdowns) is written to the regenerable reports
lake as JSON, mirroring the Phase 2/3 lake-vs-relational-index split (Appendix
B.4). Per-run `reports/backtest/` dumps are git-ignored like the earlier
`reports/data|gates` dumps.

## Capital-agnostic reporting
`initial_equity` is only a numeraire; results are reported as returns and
R-multiples, never as an absolute profit claim (Section 0 / output rules). The BT
gate's sanity arm rejects impossible results (runaway returns / unbounded
R-multiples) so a corrupt input can never read as a valid edge.
