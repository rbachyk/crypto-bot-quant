"""Localize where basis_reversion bleeds: per exit-reason / symbol / regime P&L, train vs hold-out."""
from __future__ import annotations

import glob
import pickle
from collections import defaultdict

from src.backtest.config import load_backtest_config
from src.backtest.engine import BacktestEngine
from src.exchange.metadata import load_metadata_config
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config

for p in sorted(glob.glob("/app/var/datalake/input_cache/*.pkl")):
    inp = pickle.load(open(p, "rb"))  # noqa: S301
    if inp and getattr(inp[0].frame, "timeframe", "") == "4h" and len(inp) >= 10:
        break

sc = load_strategies_config()
cfg = load_backtest_config()
meta = load_metadata_config()
cand = sc.candidate("basis_reversion")
strat = build_strategy(cand, sc.strategy_version, cand.params)
res = BacktestEngine(cfg, meta, strat).run(inp)
trades = res.trades


def _agg(label, key):
    buckets = defaultdict(list)
    for t in trades:
        buckets[key(t)].append(t)
    print(f"\n== by {label} ==")
    for k, ts in sorted(buckets.items(), key=lambda kv: sum(x.pnl_r for x in kv[1]) / len(kv[1])):
        exp = sum(x.pnl_r for x in ts) / len(ts)
        wr = sum(1 for x in ts if x.pnl > 0) / len(ts)
        mfe = sum(x.mfe_r for x in ts) / len(ts)
        print(f"  {str(k):20} n={len(ts):5} exp_r={exp:+.4f} win={wr:.3f} mfe={mfe:.3f}")


_agg("exit_reason", lambda t: t.exit_reason)
_agg("side", lambda t: "long" if t.side > 0 else "short")
_agg("regime", lambda t: t.regime)
_agg("symbol", lambda t: t.symbol)

# Train vs hold-out split: hold-out = the most-recent 20% of the decision timeline.
all_ts = sorted({int(b["ts"]) for s in inp for b in s.bars})
ho_lo = all_ts[int(len(all_ts) * 0.8)]
tr = [t for t in trades if t.entry_ts < ho_lo]
hd = [t for t in trades if t.entry_ts >= ho_lo]
for name, ts in (("TRAIN", tr), ("HOLDOUT", hd)):
    if not ts:
        continue
    exp = sum(x.pnl_r for x in ts) / len(ts)
    wr = sum(1 for x in ts if x.pnl > 0) / len(ts)
    print(f"\n{name}: n={len(ts)} exp_r={exp:+.4f} win={wr:.3f}")
