"""Candidate exit-geometry tests — volatility-scaled (ATR) stops/TPs.

The stop/TP fractions in configs/strategies.yaml are FLOORS; when an ATR multiplier is set the
effective geometry is ``max(floor, k × atr_pct)`` so it adapts to the decision timeframe (a fixed
1.2% stop is several 5m bars but only ~1 1h bar → noise stop-outs on coarser grids). These tests
prove the scaling widens on volatile bars, holds the floor on calm/low-vol bars (so the 5m
synthetic fixtures are unchanged), and never yields a degenerate sub-floor stop.
"""

from __future__ import annotations

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


def test_momentum_tp_stays_unreachable_when_mult_zero() -> None:
    """Momentum candidates set atr_tp_mult=0, so the TP stays the unreachable fixed floor (the
    time-stop is the exit) even on very volatile bars — only the stop scales."""
    cand, strat = _cand("lead_lag_xasset")
    assert cand.params.atr_tp_mult == 0.0
    stop, tp = strat._exit_geometry({"atr_pct": 0.05})
    assert tp == cand.params.tp_frac  # 0.50, NOT scaled
    assert stop == cand.params.atr_stop_mult * 0.05  # stop scaled
