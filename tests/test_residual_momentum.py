"""Offline proof for the beta-residual cross-sectional modes of the CrossSectionalEngine.

Two planted fixtures isolate the engine mechanics from the factor question:
* a COMMON market factor moves every symbol together (betas ≈ 1; raw-return dispersion is dominated
  by the factor — the reason plain raw momentum/reversion is dead), plus
* a per-symbol IDIOSYNCRATIC component that the engine must isolate by removing β·r_mkt.

``test_residual_momentum_*`` plants a persistent idiosyncratic TREND and confirms the promoted
``residual_momentum`` candidate longs the residual winners / shorts the losers and harvests it.
``test_residual_reversion_mode_*`` plants a mean-reverting idiosyncratic oscillation and confirms
the engine's ``residual_reversion`` mode (kept as a tested-negative control on the real lake) still
harvests reversion — so both rank modes are exercised, independent of which one has a live edge.
"""

from __future__ import annotations

import dataclasses
import math

from src.backtest.config import load_backtest_config
from src.backtest.engine import SymbolInput
from src.backtest.portfolio import CrossSectionalEngine
from src.exchange.metadata import load_metadata_config
from src.features.pipeline import FeatureFrame
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config

IV = 3_600_000  # 1h bars (matches the candidate's 24h rebalance → 24-bar cadence)
N = 360  # bars (≈ 15 days of 1h → many rebalances)
NSYM = 10
MKT_AMP = 0.03  # common-factor log-amplitude (cancels in a dollar-neutral basket)


def _market(k: int) -> float:
    return MKT_AMP * math.sin(2.0 * math.pi * k / 200.0)


def _build(logprice) -> list[SymbolInput]:
    syms = [f"S{i}/USDT:USDT" for i in range(NSYM)]
    inputs: list[SymbolInput] = []
    for i, s in enumerate(syms):
        bars = []
        for k in range(N):
            p = 100.0 * math.exp(logprice(i, k))
            ts = k * IV
            bars.append({"ts": ts, "open": p, "high": p, "low": p, "close": p, "volume": 1e6})
        frame = FeatureFrame(symbol=s, timeframe="1h", feature_names=[], rows=[
            {"ts": b["ts"], "decision_ts": b["ts"], "atr_pct": 0.01, "session_code": 0}
            for b in bars
        ])
        inputs.append(SymbolInput(
            symbol=s, bars=bars, frame=frame,
            spread_samples=[{"ts": k * IV, "spread_bps": 2.0} for k in range(N)],
            funding_events=[],
        ))
    return inputs


def _strategy(score_mode: str, signal_window: int = 24):
    sc = load_strategies_config()
    cand = sc.candidate("residual_momentum")
    extra = dict(cand.params.extra)
    extra.update(score_mode=score_mode, signal_window=float(signal_window))
    params = dataclasses.replace(cand.params, extra=extra)
    return build_strategy(cand, sc.strategy_version, params=params)


def test_residual_momentum_engine_harvests_idiosyncratic_trend():
    """Persistent idiosyncratic drift (half the universe up, half down) on top of the common factor:
    the momentum mode must long the up-drifters / short the down-drifters and profit on both."""
    drift = [(-1.0 + 2.0 * i / (NSYM - 1)) * 0.0015 for i in range(NSYM)]  # ±0.15%/bar, symmetric

    def logprice(i: int, k: int) -> float:
        return _market(k) + drift[i] * k  # common factor + idiosyncratic linear trend

    result = CrossSectionalEngine(
        load_backtest_config(), load_metadata_config(), _strategy("residual_momentum")
    ).run(_build(logprice))

    assert result.trades, "the residual-momentum engine should trade the basket"
    longs = [t for t in result.trades if t.side > 0]
    shorts = [t for t in result.trades if t.side < 0]
    assert longs and shorts, "dollar-neutral basket trades both sides"
    total = sum(t.pnl for t in result.trades)
    assert total > 0.0, f"planted idiosyncratic trend should be net positive, got {total:.4f}"


def test_residual_momentum_score_isolates_the_idiosyncratic_component():
    """With a common factor present, the momentum score must rank an up-drifting name above a
    down-drifting one (positive vs negative residual) — i.e. it strips β·r_mkt, not reads raw."""
    drift = [(-1.0 + 2.0 * i / (NSYM - 1)) * 0.0015 for i in range(NSYM)]

    def logprice(i: int, k: int) -> float:
        return _market(k) + drift[i] * k

    eng = CrossSectionalEngine(
        load_backtest_config(), load_metadata_config(), _strategy("residual_momentum")
    )
    inputs = _build(logprice)
    eng._grid_iv = IV
    eng._prepare_returns(inputs)
    ts = 60 * IV
    scores = {s.symbol: eng._residual_score(s.symbol, ts) for s in inputs}
    scores = {k: v for k, v in scores.items() if v is not None}
    assert len(scores) == NSYM
    # highest score (engine LONGs) = the strongest up-drifter S9; lowest (SHORTs) = S0.
    assert max(scores, key=scores.__getitem__) == "S9/USDT:USDT"
    assert min(scores, key=scores.__getitem__) == "S0/USDT:USDT"
    assert scores["S9/USDT:USDT"] > 0 > scores["S0/USDT:USDT"]


def test_residual_reversion_mode_harvests_reversion():
    """Control: the engine's residual_reversion mode (tested-negative on the real lake) still
    harvests a planted mean-reverting idiosyncratic oscillation — both rank modes are exercised."""
    period = 48
    phase = [2.0 * math.pi * i / NSYM for i in range(NSYM)]

    def logprice(i: int, k: int) -> float:
        return _market(k) + 0.05 * math.sin(2.0 * math.pi * k / period + phase[i])

    result = CrossSectionalEngine(
        load_backtest_config(), load_metadata_config(),
        _strategy("residual_reversion", signal_window=12),
    ).run(_build(logprice))

    assert result.trades
    longs = [t for t in result.trades if t.side > 0]
    shorts = [t for t in result.trades if t.side < 0]
    assert longs and shorts
    assert sum(t.pnl for t in result.trades) > 0.0
