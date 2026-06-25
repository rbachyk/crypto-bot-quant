"""Test the xsection 'another approach': cross-sectional MEAN-REVERSION (long laggard / short
leader) vs the current MOMENTUM (long leader / short laggard). If the flip is clearly positive,
the relative-strength signal carries a reversion edge, not a continuation one — worth reimplementing."""
from __future__ import annotations

import glob
import pickle

from src.backtest.config import load_backtest_config
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import build_report
from src.exchange.metadata import load_metadata_config
from src.strategies.candidates import CrossSectionalRSStrategy, build_strategy
from src.strategies.config import load_strategies_config

for p in sorted(glob.glob("/app/var/datalake/input_cache/*.pkl")):
    inp = pickle.load(open(p, "rb"))  # noqa: S301
    if inp and getattr(inp[0].frame, "timeframe", "") == "4h" and len(inp) >= 10:
        break

sc = load_strategies_config()
cfg = load_backtest_config()
meta = load_metadata_config()
cand = sc.candidate("xsection_rs")


class FlippedXSection(CrossSectionalRSStrategy):
    """Cross-sectional MEAN-REVERSION: fade relative strength (short the leader, long the laggard)."""

    def evaluate_portfolio(self, symbol, row, peers):
        sig = super().evaluate_portfolio(symbol, row, peers)
        if sig is None:
            return None
        from dataclasses import replace as _r
        return _r(sig, side=-sig.side, reason="xsection mean-reversion (faded)")


def show(label, strat):
    r = build_report(BacktestEngine(cfg, meta, strat).run(inp)).payload
    sb = r["side_breakdown"]
    print(f"{label:28} n={r['trade_count']:5} exp={r['expectancy_r']:+.4f} win={r['win_rate']:.3f} "
          f"| long {sb['long']['expectancy_r']:+.4f}/{sb['long']['trades']} "
          f"short {sb['short']['expectancy_r']:+.4f}/{sb['short']['trades']}")


print("== xsection: momentum (current) vs mean-reversion (flipped) — both sides ==")
show("momentum (current)", build_strategy(cand, sc.strategy_version, cand.params))
show("mean-reversion (flipped)", FlippedXSection(cand, sc.strategy_version, cand.params))
