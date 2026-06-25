"""Load cached lake inputs directly (bypass the slow rebuild) and dump MAE/MFE + trade quality.

Runs the SAME validate_candidate_on_lake logic on pre-built SymbolInput pickles from the input
cache, so the metrics match the persisted validation reports — plus the new excursion block.
"""
from __future__ import annotations

import glob
import json
import pickle

from src.backtest.config import load_backtest_config
from src.exchange.metadata import load_metadata_config
from src.strategies.config import load_strategies_config
from src.strategies.lake_research import validate_candidate_on_lake

CACHE_GLOB = "/app/var/datalake/input_cache/*.pkl"
WANT_TF = "4h"


def _load_4h_inputs():
    for path in sorted(glob.glob(CACHE_GLOB)):
        with open(path, "rb") as fh:
            inputs = pickle.load(fh)  # noqa: S301 - our own cache
        if not inputs:
            continue
        tf = getattr(inputs[0].frame, "timeframe", "?")
        n = len(inputs)
        bars0 = len(inputs[0].bars)
        print(f"  {path.split('/')[-1]}: tf={tf} symbols={n} bars0={bars0}", flush=True)
        if tf == WANT_TF and n >= 10:
            print(f"  -> using {path.split('/')[-1]}", flush=True)
            return inputs
    return None


print("scanning input cache:", flush=True)
lake_inputs = _load_4h_inputs()
if lake_inputs is None:
    raise SystemExit("no 4h 20-symbol cache pkl found")

strat_cfg = load_strategies_config()
cfg = load_backtest_config()
meta = load_metadata_config()

print(f"\nstrategy_version={strat_cfg.strategy_version}\n", flush=True)
for cand in strat_cfg.enabled_candidates():
    v = validate_candidate_on_lake(cand, strat_cfg, cfg, meta, lake_inputs)
    r = v.report or {}
    wf = v.walk_forward or {}
    folds = wf.get("folds") or []
    out = {
        "candidate": v.candidate_id,
        "promoted": v.promoted,
        "shelved": v.shelved_reasons,
        "trades": r.get("trade_count"),
        "win_rate": r.get("win_rate"),
        "avg_win_r": r.get("avg_win_r"),
        "avg_loss_r": r.get("avg_loss_r"),
        "expectancy_r": r.get("expectancy_r"),
        "wf_folds_passed": wf.get("folds_passed"),
        "wf_fold_er": [round(f.get("expectancy_r", 0), 4) for f in folds],
        "holdout_er": round((wf.get("holdout") or {}).get("expectancy_r", 0), 4)
        if wf.get("holdout")
        else None,
        "mfe_wins": (r.get("excursion") or {}).get("avg_mfe_r_wins"),
        "capture": round(
            (r.get("avg_win_r") or 0) / ((r.get("excursion") or {}).get("avg_mfe_r_wins") or 1), 3
        ),
    }
    print(json.dumps(out), flush=True)
