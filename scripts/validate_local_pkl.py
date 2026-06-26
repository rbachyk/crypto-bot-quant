"""Validate candidates against a LOCAL cached-input pickle (copied out of the lakedata volume),
running on HOST cpu with the host's current code — no container rebuild, no contention with an
in-container build. Usage: python scripts/validate_local_pkl.py /tmp/lake_1h.pkl [cand_id ...]"""
from __future__ import annotations

import pickle
import sys

from src.backtest.config import load_backtest_config
from src.exchange.metadata import load_metadata_config
from src.strategies.config import load_strategies_config
from src.strategies.lake_research import validate_candidate_on_lake

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lake_1h.pkl"
only = set(sys.argv[2:])

with open(path, "rb") as fh:
    inp = pickle.load(fh)  # noqa: S301 - our own cache file
tf = getattr(inp[0].frame, "timeframe", "?")
print(f"loaded {path}: timeframe={tf} symbols={len(inp)} bars0={len(inp[0].bars)}\n", flush=True)

sc = load_strategies_config()
cfg = load_backtest_config()
meta = load_metadata_config()

for cand in sc.enabled_candidates():
    if only and cand.id not in only:
        continue
    v = validate_candidate_on_lake(cand, sc, cfg, meta, inp)
    wf = v.walk_forward or {}
    h = wf.get("holdout") or {}
    r = v.report or {}
    defl = (wf.get("overfitting") or {}).get("deflated_sharpe")
    print(
        f"{cand.id:20} lake_only={cand.lake_only!s:5} PROMOTED={v.promoted!s:5} "
        f"trades={r.get('trade_count')} exp={r.get('expectancy_r')} folds={wf.get('folds_passed')}/5 "
        f"deflated={defl} holdout(pf={h.get('profit_factor')} pass={h.get('passed')})",
        flush=True,
    )
    if v.shelved_reasons:
        print(f"  shelved: {v.shelved_reasons}", flush=True)
