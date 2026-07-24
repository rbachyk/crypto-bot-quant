"""Does cross-sectional DISPERSION predict whether a basket rebalance pays?

The hypothesis behind `min_score_gap`: a basket's edge is the score GAP between its long and
short sides, while the cost of forming it (fees + spread on every leg) is the same whatever that
gap is. So low-dispersion rebalances should pay full freight to harvest almost nothing — the
structural reason a carry/factor basket bleeds in quiet markets, and something that needs no
regime label to act on.

That is a HYPOTHESIS, not a finding. This script tests it on the real lake: it re-runs the
cross-sectional engine, records the score gap at every rebalance, and buckets the legs opened at
that rebalance by it. If the bottom gap bucket is where the losses are, a gate is justified and
the table says where to put it. If P&L is flat across buckets, dispersion does NOT predict and
gating on it is curve-fitting — in which case the honest move is to leave it off.

The second table is the decision table: for each candidate threshold it shows what you would have
skipped and what the remaining book would have earned, so the threshold is CHOSEN from evidence.
Legs are also broken down by entry regime, to show directly whether low-gap rebalances are the
R1_LOW_VOL_RANGE legs.

CAUTION: this measures in-sample over one window. A threshold picked here still has to clear
walk-forward + the hold-out + fee/slippage stress (`qbot promote-lake`) before it means anything.
Picking the bucket boundary that maximises P&L here is exactly the overfit the gates exist to
catch.

Run: docker compose exec worker-backtest python scripts/basket_dispersion_diagnostic.py
     docker compose exec worker-backtest python scripts/basket_dispersion_diagnostic.py \
         --strategy residual_momentum --timeframe 1h
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.backtest.config import load_backtest_config  # noqa: E402
from src.backtest.portfolio import CrossSectionalEngine  # noqa: E402
from src.data.config import load_data_config  # noqa: E402
from src.exchange.metadata import load_metadata_for  # noqa: E402
from src.strategies.candidates import build_strategy  # noqa: E402
from src.strategies.config import load_strategies_config  # noqa: E402


class _InstrumentedEngine(CrossSectionalEngine):
    """The real engine, plus a record of the score gap at each rebalance timestamp.

    Overriding ``_rebalance`` (which receives both ``ts`` and ``scores``) keeps the measured
    behaviour IDENTICAL to production — nothing about sizing, ranking or costs is re-implemented
    here, so the legs booked are the ones the strategy would really have booked."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.gap_at: dict[int, float] = {}

    def _rebalance(self, holdings, scores, bars_by_ts, rows_by_ts, by_symbol, ts, equity, result):
        self.gap_at[int(ts)] = self.score_gap(scores)
        return super()._rebalance(
            holdings, scores, bars_by_ts, rows_by_ts, by_symbol, ts, equity, result
        )


def _quantiles(values: list[float], n: int) -> list[float]:
    """n-1 interior cut points, so buckets hold roughly equal numbers of rebalances."""
    if len(values) < n:
        return []
    ordered = sorted(values)
    return [ordered[int(len(ordered) * i / n)] for i in range(1, n)]


def _bucket_of(gap: float, cuts: list[float]) -> int:
    return sum(1 for c in cuts if gap >= c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="funding_carry", help="cross-sectional candidate id")
    ap.add_argument("--config", default="configs/data.bybit.yaml")
    ap.add_argument("--timeframe", default="1h", help="decision timeframe (the validated one)")
    ap.add_argument("--buckets", type=int, default=5, help="equal-count gap buckets")
    args = ap.parse_args()

    data_cfg = load_data_config(args.config)
    sc = load_strategies_config()
    cand = sc.candidate(args.strategy)
    if cand is None:
        print(f"unknown candidate {args.strategy!r}")
        return 2
    strategy = build_strategy(cand, sc.strategy_version)
    if not getattr(strategy, "cross_sectional", False):
        print(f"{args.strategy!r} is not a cross-sectional (basket) strategy")
        return 2

    from src.backtest.service import build_lake_inputs
    from src.data.store import SeriesStore

    print(f"building lake inputs: {args.strategy} @ {args.timeframe} …")
    inputs = build_lake_inputs(
        SeriesStore(),
        exchange_id=data_cfg.exchange_id,
        symbols=data_cfg.active_symbols(),
        timeframe=args.timeframe,
        base_timeframe=data_cfg.base_timeframe,
        funding_timeframe=data_cfg.funding_timeframe,
        start_ms=data_cfg.window_start_ms,
        end_ms=data_cfg.window_end_ms,
        oi_timeframe=data_cfg.oi_grid,
    )
    if not inputs or all(not getattr(s, "bars", None) for s in inputs):
        print("no lake data — run `qbot download` first")
        return 2

    engine = _InstrumentedEngine(load_backtest_config(), load_metadata_for(data_cfg.exchange_id),
                                 strategy)
    if engine.min_score_gap > 0:
        print(f"NOTE: min_score_gap is already {engine.min_score_gap:g} — this run measures the "
              "ALREADY-GATED strategy. Set it to 0 to measure the ungated relationship.")
    result = engine.run(inputs)
    trades = result.trades
    if not trades:
        print("the basket booked no legs over this window — nothing to bucket")
        return 1

    gaps = sorted(engine.gap_at.values())
    print(f"\nrebalances: {len(gaps)}   legs: {len(trades)}")
    print(f"score gap  min {gaps[0]:.6f}  p25 {gaps[len(gaps)//4]:.6f}  "
          f"median {statistics.median(gaps):.6f}  p75 {gaps[3*len(gaps)//4]:.6f}  "
          f"max {gaps[-1]:.6f}")

    cuts = _quantiles(gaps, args.buckets)
    if not cuts:
        print("too few rebalances to bucket")
        return 1

    # ---- table 1: does the gap predict? ------------------------------- #
    by_bucket: dict[int, list] = defaultdict(list)
    unmatched = 0
    for t in trades:
        gap = engine.gap_at.get(int(t.entry_ts))
        if gap is None:
            unmatched += 1
            continue
        by_bucket[_bucket_of(gap, cuts)].append(t)

    edges = [f"<{cuts[0]:.5f}"] + [
        f"{cuts[i]:.5f}–{cuts[i + 1]:.5f}" for i in range(len(cuts) - 1)
    ] + [f">={cuts[-1]:.5f}"]
    print("\n--- legs bucketed by the score gap at their rebalance ---")
    print(f"{'gap bucket':>22} {'legs':>6} {'pnl':>10} {'exp_R':>8} {'win%':>6} "
          f"{'fees':>9} {'funding':>9}")
    for b in range(len(cuts) + 1):
        ts_ = by_bucket.get(b, [])
        if not ts_:
            print(f"{edges[b]:>22} {0:>6}")
            continue
        pnl = sum(t.pnl for t in ts_)
        exp_r = sum(t.pnl_r for t in ts_) / len(ts_)
        wins = sum(1 for t in ts_ if t.pnl > 0) / len(ts_)
        print(f"{edges[b]:>22} {len(ts_):>6} {pnl:>10.2f} {exp_r:>8.4f} {wins:>5.0%} "
              f"{sum(t.fee for t in ts_):>9.2f} {sum(t.funding for t in ts_):>9.2f}")
    if unmatched:
        print(f"({unmatched} legs had no recorded rebalance gap — carried from a prior basket)")

    # ---- table 2: what each threshold would have done ------------------ #
    print("\n--- if min_score_gap had been set to X (skip every rebalance below it) ---")
    print(f"{'min_score_gap':>14} {'rebal kept':>11} {'legs kept':>10} {'pnl kept':>10} "
          f"{'exp_R':>8} {'pnl skipped':>12}")
    total_pnl = sum(t.pnl for t in trades)
    for x in cuts:
        kept = [t for t in trades if (engine.gap_at.get(int(t.entry_ts), 0.0)) >= x]
        n_reb = sum(1 for g in gaps if g >= x)
        pnl_kept = sum(t.pnl for t in kept)
        exp_r = (sum(t.pnl_r for t in kept) / len(kept)) if kept else 0.0
        print(f"{x:>14.6f} {n_reb:>5}/{len(gaps):<5} {len(kept):>10} {pnl_kept:>10.2f} "
              f"{exp_r:>8.4f} {total_pnl - pnl_kept:>12.2f}")

    # ---- table 3: is low dispersion the same thing as R1? -------------- #
    print("\n--- entry regime × gap bucket (legs) ---")
    regimes = sorted({t.regime for t in trades})
    print(f"{'regime':>24} " + " ".join(f"{edges[b][:12]:>13}" for b in range(len(cuts) + 1)))
    for r in regimes:
        cells = []
        for b in range(len(cuts) + 1):
            n = sum(1 for t in by_bucket.get(b, []) if t.regime == r)
            cells.append(f"{n:>13}")
        print(f"{r:>24} " + " ".join(cells))

    print("\nRead it this way: a gate is justified only if the LOW-gap buckets carry the losses "
          "AND the pattern is monotonic-ish. A flat P&L across buckets means dispersion does not "
          "predict — leave min_score_gap at 0. Whatever you pick, it must clear walk-forward + "
          "hold-out + fee stress via `qbot promote-lake` before it is real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
