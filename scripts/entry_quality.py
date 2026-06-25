"""Entry-quality diagnostic: for each strategy, bin trades by ENTRY SIGNAL STRENGTH and show
win rate / expectancy_r per bin. A rising win-rate-with-strength = the edge concentrates in
stronger signals (raise the threshold); a flat relationship = the entry trigger is noise."""
from __future__ import annotations

import glob
import pickle
from collections import defaultdict

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

# decision-time rows per symbol, keyed by decision_ts (for recomputing the entry signal per trade).
rows = {s.symbol: {int(r["decision_ts"]): r for r in s.frame.rows} for s in inp}
all_dts = sorted({d for m in rows.values() for d in m})


def promoted_trades(cid: str):
    cand = sc.candidate(cid)
    both = build_strategy(cand, sc.strategy_version, cand.params)
    full = BacktestEngine(cfg, meta, both).run(inp)
    sd = _decide_sides(build_report(full), sc.min_side_expectancy_r)
    strat = build_strategy(
        cand, sc.strategy_version,
        cand.params.with_sides(allow_long=sd.allow_long, allow_short=sd.allow_short),
    )
    return cand, strat, BacktestEngine(cfg, meta, strat).run(inp).trades, sd


def _xmean_cache():
    """ret_short cross-sectional mean per decision_ts (for xsection signal recompute)."""
    out = {}
    for dts in all_dts:
        vals = [float(rows[s][dts].get("ret_short", 0.0)) for s in rows if dts in rows[s]]
        out[dts] = sum(vals) / len(vals) if vals else 0.0
    return out


def signal_strength(cid: str, cand, t):
    r = rows.get(t.symbol, {}).get(t.entry_ts)
    if r is None:
        return None
    if cid == "basis_reversion":
        return abs(float(r.get("premium", 0.0)))
    if cid == "xsection_rs":
        return abs(float(r.get("ret_short", 0.0)) - _XMEAN[t.entry_ts])
    if cid == "lead_lag_xasset":
        leader = str(cand.fixture.values["leader"])
        lr = rows.get(leader, {}).get(t.entry_ts)
        return abs(float(lr.get("ret_1", 0.0))) if lr else None
    return None


def bins(pairs, nb=5):
    pairs = sorted(pairs, key=lambda x: x[0])
    n = len(pairs)
    for i in range(nb):
        chunk = pairs[i * n // nb : (i + 1) * n // nb]
        if not chunk:
            continue
        sg = [c[0] for c in chunk]
        prs = [c[1] for c in chunk]
        wr = sum(1 for x in prs if x > 0) / len(prs)
        exp = sum(prs) / len(prs)
        print(f"  bin{i} sig[{sg[0]:.4f}..{sg[-1]:.4f}] n={len(prs):5} win={wr:.3f} exp_r={exp:+.4f}")


_XMEAN = _xmean_cache()
for cid in ("lead_lag_xasset", "basis_reversion", "xsection_rs"):
    cand, strat, trades, sd = promoted_trades(cid)
    rep = build_report(BacktestEngine(cfg, meta, strat).run(inp))
    print(f"\n==== {cid} (long={sd.allow_long} short={sd.allow_short}) "
          f"exp={rep.payload['expectancy_r']:+.4f} win={rep.payload['win_rate']:.3f} "
          f"n={len(trades)} ====")
    pairs = [(signal_strength(cid, cand, t), t.pnl_r) for t in trades]
    pairs = [(s, p) for s, p in pairs if s is not None]
    print(" by entry signal strength (weak→strong):")
    bins(pairs)
    # per side
    for side_name, sgn in (("long", 1), ("short", -1)):
        sp = [(signal_strength(cid, cand, t), t.pnl_r) for t in trades if t.side == sgn]
        sp = [(s, p) for s, p in sp if s is not None]
        if sp:
            wr = sum(1 for _, p in sp if p > 0) / len(sp)
            print(f"  [{side_name}] n={len(sp):5} win={wr:.3f} exp={sum(p for _,p in sp)/len(sp):+.4f}")
