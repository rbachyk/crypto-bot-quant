"""Pressure-test the walk-forward gate: tabulate the sub-signals (per-fold dir/PF, pooled-OOS,
locked hold-out, deflated Sharpe) for a REAL-edge candidate (lead_lag), a marginal one (basis)
and a genuine NO-EDGE control (xsection). Any alternative criterion must still REJECT xsection."""
from __future__ import annotations

import glob
import pickle

from src.backtest.config import load_backtest_config
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import build_report
from src.backtest.walkforward import rebase_window
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
kc = cfg.walk_forward.kill_criteria


def _iv():
    best = None
    for s in inp:
        for a, b in zip(s.bars, s.bars[1:], strict=False):
            d = int(b["ts"] - a["ts"])
            if d > 0 and (best is None or d < best):
                best = d
    return best or 1


def promoted(cid):
    cand = sc.candidate(cid)
    both = build_report(BacktestEngine(cfg, meta, build_strategy(cand, sc.strategy_version, cand.params)).run(inp))
    sd = _decide_sides(both, sc.min_side_expectancy_r)
    return build_strategy(cand, sc.strategy_version, cand.params.with_sides(allow_long=sd.allow_long, allow_short=sd.allow_short))


iv = _iv()
with_bars = [s for s in inp if s.bars]
lo = min(s.bars[0]["ts"] for s in with_bars)
hi = max(s.bars[-1]["ts"] for s in with_bars) + iv
span = (hi - lo) // iv
ho_slots = int(span * cfg.walk_forward.holdout_frac)
test_end = lo + (span - ho_slots) * iv
fold_span = (test_end - lo) // cfg.walk_forward.folds

for cid in ("lead_lag_xasset", "basis_reversion", "xsection_rs"):
    strat = promoted(cid)
    fold_reps = []
    pooled = []
    for i in range(cfg.walk_forward.folds):
        flo = lo + i * fold_span
        fhi = lo + (i + 1) * fold_span if i < cfg.walk_forward.folds - 1 else test_end
        rep = build_report(BacktestEngine(cfg, meta, strat).run(rebase_window(inp, flo, fhi)))
        fold_reps.append(rep.payload)
    ho = build_report(BacktestEngine(cfg, meta, strat).run(rebase_window(inp, test_end, hi))).payload
    dir_pos = sum(1 for f in fold_reps if f["expectancy_r"] > 0)
    pf_pass = sum(1 for f in fold_reps if f["profit_factor"] >= kc.min_oos_profit_factor)
    full_pass = sum(
        1 for f in fold_reps
        if f["expectancy_r"] >= kc.min_oos_expectancy_r and f["profit_factor"] >= kc.min_oos_profit_factor
        and f["max_drawdown"] <= kc.max_oos_drawdown and f["trade_count"] >= kc.min_trades_per_fold
    )
    print(f"\n==== {cid} ====")
    print("  folds  exp_r : " + " ".join(f"{f['expectancy_r']:+.3f}" for f in fold_reps))
    print("  folds  PF    : " + " ".join(f"{f['profit_factor']:.2f}" for f in fold_reps))
    print(f"  dir-positive folds = {dir_pos}/5   PF>=1.10 folds = {pf_pass}/5   FULL-pass folds = {full_pass}/5")
    print(f"  HOLDOUT: exp_r={ho['expectancy_r']:+.4f} PF={ho['profit_factor']:.2f} "
          f"dd={ho['max_drawdown']:.2f} passes_kc={ho['expectancy_r']>=kc.min_oos_expectancy_r and ho['profit_factor']>=kc.min_oos_profit_factor and ho['max_drawdown']<=kc.max_oos_drawdown}")
