"""Correctness proof for the cross-sectional (basket) engine: a PLANTED funding-dispersion edge
must show up positive; a no-funding control must net ≈ 0. A new validation engine is exactly where
a subtle bug produces false promotes, so this is the guard before trusting any funding_carry result.
"""

from __future__ import annotations

from src.backtest.config import load_backtest_config
from src.backtest.engine import SymbolInput
from src.backtest.metrics import build_report
from src.backtest.portfolio import CrossSectionalEngine
from src.exchange.metadata import load_metadata_config
from src.features.pipeline import FeatureFrame
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config

IV = 60_000


def _sym(symbol: str, funding_z: float, funding_rate: float, n: int = 240) -> SymbolInput:
    """Flat price (so carry is the ONLY P&L), a constant funding_z, and a funding rate paid every
    8 bars. With the price flat, the basket's realized P&L is exactly funding collected − costs."""
    bars = [
        {"ts": k * IV, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e6}
        for k in range(n)
    ]
    rows = [
        {"ts": k * IV, "decision_ts": k * IV, "funding_z": funding_z, "atr_pct": 0.01,
         "session_code": 0}
        for k in range(n)
    ]
    frame = FeatureFrame(symbol=symbol, timeframe="1m", feature_names=["funding_z"], rows=rows)
    funding = [{"ts": k * IV, "funding_rate": funding_rate} for k in range(0, n, 8)]
    spread = [{"ts": k * IV, "spread_bps": 2.0} for k in range(n)]
    return SymbolInput(
        symbol=symbol, bars=bars, frame=frame, spread_samples=spread, funding_events=funding
    )


def _universe(aligned: bool) -> list[SymbolInput]:
    """10 symbols, funding_z spread −2..+2. EDGE: funding_rate aligns with funding_z (high funding_z
    = high positive rate ⇒ shorts collect; low = negative ⇒ longs collect). CONTROL: rate = 0."""
    out = []
    for i in range(10):
        fz = -2.0 + i * (4.0 / 9.0)
        rate = 0.001 * fz if aligned else 0.0
        out.append(_sym(f"S{i}/USDT:USDT", funding_z=fz, funding_rate=rate))
    return out


def _carry_strategy():
    sc = load_strategies_config()
    cand = sc.candidate("funding_carry")
    return build_strategy(cand, sc.strategy_version)


def test_cross_sectional_engine_harvests_a_planted_carry_edge():
    cfg = load_backtest_config()
    meta = load_metadata_config()
    strat = _carry_strategy()
    assert getattr(strat, "cross_sectional", False) is True

    def report(aligned):
        return build_report(CrossSectionalEngine(cfg, meta, strat).run(_universe(aligned))).payload

    edge = report(True)
    ctrl = report(False)

    # The basket is dollar-neutral: roughly equal long and short legs traded.
    sb = edge["side_breakdown"]
    assert sb["long"]["trades"] > 0 and sb["short"]["trades"] > 0

    # PLANTED carry shows up; the no-funding control nets ≈ 0 (only costs) and clearly worse.
    assert edge["expectancy_r"] > 0.0
    assert edge["net_pnl"] > 0.0
    assert ctrl["net_pnl"] <= 0.0  # no carry ⇒ costs only
    assert edge["net_pnl"] > ctrl["net_pnl"]


def test_cross_sectional_routing_via_run_engine():
    """run_engine dispatches a cross_sectional strategy to the basket engine (so walk-forward /
    stress route automatically), and a normal strategy still uses the per-trade engine."""
    from src.backtest.service import run_engine

    cfg = load_backtest_config()
    meta = load_metadata_config()
    run = run_engine(cfg, meta, _universe(aligned=True), strategy=_carry_strategy())
    assert run.report.trade_count > 0  # produced a basket report through run_engine
