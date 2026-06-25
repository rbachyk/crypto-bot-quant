"""Does FUNDING alignment predict basis trade quality? Bin basis trades by carry alignment
(-side·funding_z > 0 = the position collects funding / faded a crowded side) and show win/exp.
If carry-positive trades clearly out-perform, a funding-confirmation entry filter is worth it."""
from __future__ import annotations

import glob
import pickle

from src.backtest.config import load_backtest_config
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import build_report
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
rows = {s.symbol: {int(r["decision_ts"]): r for r in s.frame.rows} for s in inp}

cand = sc.candidate("basis_reversion")
both = build_report(BacktestEngine(cfg, meta, build_strategy(cand, sc.strategy_version, cand.params)).run(inp))
sd = _decide_sides(both, sc.min_side_expectancy_r)
strat = build_strategy(cand, sc.strategy_version, cand.params.with_sides(allow_long=sd.allow_long, allow_short=sd.allow_short))
trades = BacktestEngine(cfg, meta, strat).run(inp).trades

# carry = -side * funding_z : > 0 means the position is paid funding (faded the crowded side).
pairs = []
for t in trades:
    r = rows.get(t.symbol, {}).get(t.entry_ts)
    if r is None:
        continue
    fz = float(r.get("funding_z", 0.0))
    pairs.append((-t.side * fz, t.pnl_r, t.side))

print(f"basis trades with funding_z: {len(pairs)}")


def stats(ps):
    if not ps:
        return "n=0"
    wr = sum(1 for _, p, _ in ps if p > 0) / len(ps)
    return f"n={len(ps):5} win={wr:.3f} exp_r={sum(p for _, p, _ in ps)/len(ps):+.4f}"


print("\n== by carry alignment (-side·funding_z), weak→strong ==")
ps = sorted(pairs, key=lambda x: x[0])
n = len(ps)
for i in range(5):
    chunk = ps[i * n // 5 : (i + 1) * n // 5]
    c = [x[0] for x in chunk]
    print(f"  bin{i} carry[{c[0]:+.2f}..{c[-1]:+.2f}] {stats(chunk)}")

print("\n== carry-positive vs carry-negative ==")
print(f"  carry>0 (paid):  {stats([x for x in pairs if x[0] > 0])}")
print(f"  carry<=0 (pays): {stats([x for x in pairs if x[0] <= 0])}")
for nm, sgn in (("short", -1), ("long", 1)):
    sp = [x for x in pairs if x[2] == sgn]
    pos = [x for x in sp if x[0] > 0]
    neg = [x for x in sp if x[0] <= 0]
    print(f"  [{nm}] carry>0 {stats(pos)}  | carry<=0 {stats(neg)}")
