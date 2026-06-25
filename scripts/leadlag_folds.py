"""Diagnose lead_lag's weak folds (0 and 4) vs the strong ones: per-symbol / regime / win-loss."""
from __future__ import annotations

import glob
import pickle
from collections import defaultdict

from src.backtest.config import load_backtest_config
from src.backtest.engine import BacktestEngine
from src.exchange.metadata import load_metadata_config
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config
from src.strategies.lake_research import _decide_sides  # noqa: PLC2701

for p in sorted(glob.glob("/app/var/datalake/input_cache/*.pkl")):
    inp = pickle.load(open(p, "rb"))  # noqa: S301
    if inp and getattr(inp[0].frame, "timeframe", "") == "4h" and len(inp) >= 10:
        break

sc = load_strategies_config()
cfg = load_backtest_config()
meta = load_metadata_config()
cand = sc.candidate("lead_lag_xasset")

# Promoted side (same as validate): both-sides → side decision → promoted backtest.
both = build_strategy(cand, sc.strategy_version, cand.params)
full_both = BacktestEngine(cfg, meta, both).run(inp)
from src.backtest.metrics import build_report  # noqa: E402

sd = _decide_sides(build_report(full_both), sc.min_side_expectancy_r)
print(f"side decision: long={sd.allow_long} short={sd.allow_short}")
promoted = build_strategy(
    cand, sc.strategy_version, cand.params.with_sides(allow_long=sd.allow_long, allow_short=sd.allow_short)
)
res = BacktestEngine(cfg, meta, promoted).run(inp)
trades = res.trades

# Fold boundaries: holdout = last 20% of the decision timeline; 5 equal-time folds over first 80%.
all_ts = sorted({int(b["ts"]) for s in inp for b in s.bars})
lo, hi = all_ts[0], all_ts[-1]
span = hi - lo
ho_lo = lo + int(span * 0.8)
edges = [lo + int(span * 0.8 * k / 5) for k in range(6)]  # 5 folds across [lo, ho_lo]


def fold_of(ts):
    if ts >= ho_lo:
        return "HOLDOUT"
    for i in range(5):
        if edges[i] <= ts < edges[i + 1]:
            return f"fold{i}"
    return "fold4"


def stats(ts_list):
    if not ts_list:
        return "n=0"
    exp = sum(t.pnl_r for t in ts_list) / len(ts_list)
    wins = [t for t in ts_list if t.pnl > 0]
    wr = len(wins) / len(ts_list)
    aw = sum(t.pnl_r for t in wins) / len(wins) if wins else 0
    losses = [t for t in ts_list if t.pnl <= 0]
    al = sum(t.pnl_r for t in losses) / len(losses) if losses else 0
    return f"n={len(ts_list):5} exp={exp:+.4f} win={wr:.3f} avgW={aw:+.3f} avgL={al:+.3f}"


by_fold = defaultdict(list)
for t in trades:
    by_fold[fold_of(t.entry_ts)].append(t)

print("\n== per fold ==")
for f in ["fold0", "fold1", "fold2", "fold3", "fold4", "HOLDOUT"]:
    print(f"  {f:8} {stats(by_fold[f])}")

for weak in ("fold0", "fold4"):
    print(f"\n== {weak} by symbol (worst first) ==")
    bysym = defaultdict(list)
    for t in by_fold[weak]:
        bysym[t.symbol].append(t)
    rows = sorted(bysym.items(), key=lambda kv: sum(x.pnl_r for x in kv[1]) / len(kv[1]))
    for sym, ts in rows[:6] + rows[-3:]:
        print(f"  {sym:18} {stats(ts)}")
    print(f"  -- {weak} by regime --")
    byreg = defaultdict(list)
    for t in by_fold[weak]:
        byreg[t.regime].append(t)
    for reg, ts in sorted(byreg.items(), key=lambda kv: sum(x.pnl_r for x in kv[1]) / len(kv[1])):
        print(f"  {reg:22} {stats(ts)}")

print("\n== regime × fold expectancy grid (is any regime CONSISTENTLY bad?) ==")
regs = ["R1_LOW_VOL_RANGE", "R2_TREND", "R3_HIGH_VOL_EXPANSION", "R5_MARKET_WIDE_IMPULSE", "R6_LIQUIDATION_EVENT"]
folds_order = ["fold0", "fold1", "fold2", "fold3", "fold4", "HOLDOUT"]
print(f"  {'regime':24} " + " ".join(f"{f:>8}" for f in folds_order))
for reg in regs:
    cells = []
    for f in folds_order:
        ts = [t for t in by_fold[f] if t.regime == reg]
        cells.append(f"{(sum(x.pnl_r for x in ts)/len(ts)):+.3f}" if ts else "   .  ")
    print(f"  {reg:24} " + " ".join(f"{c:>8}" for c in cells))
