"""Sweep basis entry threshold (and exit band) — decide on the train FOLDS, glance at hold-out once."""
from __future__ import annotations

import glob
import pickle
from dataclasses import replace

from src.backtest.config import load_backtest_config
from src.exchange.metadata import load_metadata_config
from src.strategies.config import load_strategies_config
from src.strategies.lake_research import validate_candidate_on_lake

for p in sorted(glob.glob("/app/var/datalake/input_cache/*.pkl")):
    inp = pickle.load(open(p, "rb"))  # noqa: S301
    if inp and getattr(inp[0].frame, "timeframe", "") == "4h" and len(inp) >= 10:
        break

sc = load_strategies_config()
cfg = load_backtest_config()
meta = load_metadata_config()
base = sc.candidate("basis_reversion")


def run(threshold, exit_frac):
    extra = dict(base.params.extra)
    extra["premium_threshold"] = threshold
    extra["exit_premium_frac"] = exit_frac
    params = replace(base.params, extra=extra)
    cand = replace(base, params=params)
    v = validate_candidate_on_lake(cand, sc, cfg, meta, inp)
    wf = v.walk_forward
    folds = wf["folds"]
    mean_exp = sum(f["expectancy_r"] for f in folds) / len(folds)
    n_pos = sum(1 for f in folds if f["expectancy_r"] > 0)
    h = wf["holdout"]
    r = v.report
    return (
        f"thr={threshold:.4f} exit={exit_frac:.2f}: trades={r['trade_count']:5} "
        f"exp={r['expectancy_r']:+.4f} folds_passed={wf['folds_passed']}/5 "
        f"folds_pos={n_pos}/5 mean_fold={mean_exp:+.4f} "
        f"HOLDOUT(exp={h['expectancy_r']:+.4f} pf={h['profit_factor']:.2f} pass={h['passed']})"
    )


print("== entry-threshold sweep (exit_frac=0.25 fixed) ==")
for thr in (0.0015, 0.0020, 0.0025, 0.0030, 0.0040):
    print(run(thr, 0.25))

print("\n== exit OFF (-1.0) × threshold (decide on mean_fold + trade sufficiency) ==")
for thr in (0.0015, 0.0020, 0.0025):
    print(run(thr, -1.0))
