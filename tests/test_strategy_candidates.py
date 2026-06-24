"""Candidate exit-geometry tests — volatility-scaled (ATR) stops/TPs.

The stop/TP fractions in configs/strategies.yaml are FLOORS; when an ATR multiplier is set the
effective geometry is ``max(floor, k × atr_pct)`` so it adapts to the decision timeframe (a fixed
1.2% stop is several 5m bars but only ~1 1h bar → noise stop-outs on coarser grids). These tests
prove the scaling widens on volatile bars, holds the floor on calm/low-vol bars (so the 5m
synthetic fixtures are unchanged), and never yields a degenerate sub-floor stop.
"""

from __future__ import annotations

from dataclasses import replace

from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config


def _cand(candidate_id: str):
    scfg = load_strategies_config()
    cand = scfg.candidate(candidate_id)
    assert cand is not None
    return cand, build_strategy(cand, scfg.strategy_version)


def test_atr_stop_widens_on_volatile_bars() -> None:
    cand, strat = _cand("basis_reversion")  # per-symbol family B
    thr = cand.params.extra["premium_threshold"]
    assert cand.params.atr_stop_mult > 0  # ATR scaling configured

    # High-vol bar: stop scales to atr_stop_mult × atr_pct (above the floor).
    sig = strat.evaluate({"premium": thr + 0.001, "atr_pct": 0.05})
    assert sig is not None
    assert sig.stop_frac == cand.params.atr_stop_mult * 0.05
    assert sig.stop_frac > cand.params.stop_frac
    # TP likewise scales (mean-reversion target grows with realized range).
    assert sig.tp_frac == cand.params.atr_tp_mult * 0.05


def test_atr_stop_holds_the_floor_on_calm_bars() -> None:
    """Low ATR ⇒ the fixed floor dominates, so the 5m fixtures behave exactly as before."""
    cand, strat = _cand("basis_reversion")
    thr = cand.params.extra["premium_threshold"]
    sig = strat.evaluate({"premium": thr + 0.001, "atr_pct": 0.001})  # 0.001×2.0 < 0.018 floor
    assert sig is not None
    assert sig.stop_frac == cand.params.stop_frac
    assert sig.tp_frac == cand.params.tp_frac


def test_missing_atr_falls_back_to_floor_never_sub_floor() -> None:
    """No atr_pct in the row ⇒ floor (never a near-zero stop, which would explode sizing)."""
    cand, strat = _cand("basis_reversion")
    thr = cand.params.extra["premium_threshold"]
    sig = strat.evaluate({"premium": thr + 0.001})  # atr_pct absent
    assert sig is not None
    assert sig.stop_frac == cand.params.stop_frac


def test_regime_gate_off_by_default_blocks_no_trade_and_restricts_allowlist() -> None:
    """The regime gate is opt-in: with neither knob set the strategy fires every bar (legacy).
    block_no_trade_regimes excludes the live safety regimes (R4 chop); a regimes allow-list trades
    ONLY the listed regimes. Regime is computed from decision-time features in the row."""
    scfg = load_strategies_config()
    cand = scfg.candidate("basis_reversion")
    thr = cand.params.extra["premium_threshold"]

    def row(**kw):  # premium fires the short side; regime features set the regime
        return {"premium": thr + 0.001, "atr_pct": 0.01, **kw}

    chop = row(atr_pct_rank=0.9, dir_efficiency=0.1)  # high vol + low dir-eff → R4_HIGH_VOL_CHOP
    rng = row(atr_pct_rank=0.1, dir_efficiency=0.1)  # → R1_LOW_VOL_RANGE
    trend = row(atr_pct_rank=0.1, dir_efficiency=0.5, trend_slope=0.001)  # → R2_TREND

    base = build_strategy(cand, scfg.strategy_version)  # default: no gating
    assert base.evaluate(chop) is not None and base.evaluate(rng) is not None

    blocked = build_strategy(
        cand, scfg.strategy_version, replace(cand.params, block_no_trade_regimes=True)
    )
    assert blocked.evaluate(chop) is None  # R4 is a no-trade regime
    assert blocked.evaluate(rng) is not None  # R1 is tradeable

    only_trend = build_strategy(
        cand, scfg.strategy_version, replace(cand.params, regimes=("R2_TREND",))
    )
    assert only_trend.evaluate(rng) is None  # R1 not in the allow-list
    assert only_trend.evaluate(trend) is not None  # R2 is


def test_momentum_tp_stays_unreachable_when_mult_zero() -> None:
    """Momentum candidates set atr_tp_mult=0, so the TP stays the unreachable fixed floor (the
    time-stop is the exit) even on very volatile bars — only the stop scales."""
    cand, strat = _cand("lead_lag_xasset")
    assert cand.params.atr_tp_mult == 0.0
    stop, tp = strat._exit_geometry({"atr_pct": 0.05})
    assert tp == cand.params.tp_frac  # 0.50, NOT scaled
    assert stop == cand.params.atr_stop_mult * 0.05  # stop scaled
