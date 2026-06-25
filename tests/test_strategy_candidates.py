"""Candidate exit-geometry tests — volatility-scaled (ATR) stops/TPs.

The stop/TP fractions in configs/strategies.yaml are FLOORS; when an ATR multiplier is set the
effective geometry is ``max(floor, k × atr_pct)`` so it adapts to the decision timeframe (a fixed
1.2% stop is several 5m bars but only ~1 1h bar → noise stop-outs on coarser grids). These tests
prove the scaling widens on volatile bars, holds the floor on calm/low-vol bars (so the 5m
synthetic fixtures are unchanged), and never yields a degenerate sub-floor stop.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from src.backtest.strategy import PositionView
from src.strategies.candidates import build_strategy
from src.strategies.config import load_strategies_config


def _cand(candidate_id: str):
    scfg = load_strategies_config()
    cand = scfg.candidate(candidate_id)
    assert cand is not None
    return cand, build_strategy(cand, scfg.strategy_version)


def _pos(side: int) -> PositionView:
    return PositionView(side=side, entry_price=100.0, bars_held=1, regime="R1")


def _basis_with_exit_frac(exit_frac: float):
    """basis with the premium-reversion exit ENABLED at a given band (config ships it disabled)."""
    scfg = load_strategies_config()
    cand = scfg.candidate("basis_reversion")
    assert cand is not None
    extra = {**cand.params.extra, "exit_premium_frac": exit_frac}
    return cand, build_strategy(cand, scfg.strategy_version, replace(cand.params, extra=extra))


def test_momentum_tp_r_mult_sets_a_reachable_target_in_r() -> None:
    """When tp_r_mult > 0 the TP sits at exactly that many R (× the effective stop distance),
    overriding the unreachable momentum tp_frac — so the take-profit auto-scales with the stop."""
    cand, strat = _cand("lead_lag_xasset")
    assert cand.params.tp_r_mult > 0  # momentum now carries an R-target
    # Build a signal via the cross-asset path (leader move triggers a follower entry).
    leader = str(cand.fixture.values["leader"])
    follower = "ETH/USDT:USDT"
    thr = cand.params.extra["leader_ret_threshold"]
    row = {"atr_pct": 0.02}
    peers = {leader: {"ret_1": thr * 2, "atr_pct": 0.02}}
    sig = strat.evaluate_portfolio(follower, row, peers)
    assert sig is not None
    # stop = max(stop_frac floor, atr_stop_mult × atr); tp = tp_r_mult × stop (exactly that many R).
    expected_stop = max(cand.params.stop_frac, cand.params.atr_stop_mult * 0.02)
    assert sig.stop_frac == pytest.approx(expected_stop)
    assert sig.tp_frac == pytest.approx(cand.params.tp_r_mult * expected_stop)
    assert sig.trail_frac > 0  # trailing stop kept as the backstop


def test_basis_band_entry_skips_the_extreme_repricing_tail() -> None:
    """basis fades a dislocation only inside the reversion band [threshold, cap]: a moderate premium
    fires, but an EXTREME one (> premium_cap, i.e. one-way repricing) is skipped — both sides."""
    cand, strat = _cand("basis_reversion")
    thr = cand.params.extra["premium_threshold"]
    cap = cand.params.extra["premium_cap"]
    assert cap > thr  # band configured
    row = {"atr_pct": 0.01}
    # In-band dislocation fires (short on rich perp, long on cheap).
    assert strat.evaluate({"premium": (thr + cap) / 2, **row}).side == -1
    assert strat.evaluate({"premium": -(thr + cap) / 2, **row}).side == 1
    # Beyond the cap → no trade (extreme = repricing, not reversion).
    assert strat.evaluate({"premium": cap * 2, **row}) is None
    assert strat.evaluate({"premium": -cap * 2, **row}) is None


def test_basis_manage_exits_when_premium_reverts() -> None:
    """The manage hook (when ENABLED, exit_premium_frac ≥ 0) closes a faded position once the
    premium has reverted to the exit band — the family's real exit before the ATR TP/time-stop."""
    cand, strat = _basis_with_exit_frac(0.25)
    thr = cand.params.extra["premium_threshold"]
    exit_level = 0.25 * thr

    # SHORT faded a rich perp (premium ≥ +thr): still rich ⇒ hold; reverted under the band ⇒ exit.
    assert strat.manage({"premium": thr, "atr_pct": 0.01}, _pos(-1)) is None
    dec = strat.manage({"premium": exit_level - 1e-6, "atr_pct": 0.01}, _pos(-1))
    assert dec is not None and dec.reason == "premium_reverted"

    # LONG faded a cheap perp (premium ≤ −thr): exit once it has risen back above −exit_level.
    assert strat.manage({"premium": -thr, "atr_pct": 0.01}, _pos(1)) is None
    assert strat.manage({"premium": -exit_level + 1e-6, "atr_pct": 0.01}, _pos(1)) is not None


def test_basis_manage_exit_disabled_by_negative_sentinel() -> None:
    """A negative exit_premium_frac (the shipped config default) DISABLES the premium-reversion
    exit — manage never fires, so positions hold to TP/stop/time (the real-data A/B winner)."""
    cand, strat = _cand("basis_reversion")
    assert cand.params.extra["exit_premium_frac"] < 0  # shipped disabled on this snapshot
    assert strat.manage({"premium": 0.0, "atr_pct": 0.02}, _pos(-1)) is None
    assert strat.manage({"premium": 0.0, "atr_pct": 0.02}, _pos(1)) is None


def test_basis_manage_exit_offset_tracks_maker_config() -> None:
    """When enabled, the manage exit posts at the same volatility-scaled passive offset as the
    maker entry (limit_offset_atr_mult × atr_pct), so the exit limit mirrors the entry style."""
    cand, strat = _basis_with_exit_frac(0.25)
    dec = strat.manage({"premium": 0.0, "atr_pct": 0.02}, _pos(-1))  # 0 premium ⇒ fully reverted
    assert dec is not None
    assert dec.limit_offset_frac == cand.params.limit_offset_atr_mult * 0.02


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

    # Explicitly UNGATED (the shipped config now enables the safety gate, so override it here).
    ungated = replace(cand.params, block_no_trade_regimes=False, regimes=())
    base = build_strategy(cand, scfg.strategy_version, ungated)
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


def test_momentum_tp_is_a_reachable_r_multiple() -> None:
    """Momentum candidates now carry tp_r_mult > 0, so the TP is a REACHABLE target at
    tp_r_mult × the effective stop (that many R), overriding the legacy unreachable floor. The
    stop still scales with ATR; both move together so the target stays a fixed R-multiple."""
    cand, strat = _cand("lead_lag_xasset")
    assert cand.params.atr_tp_mult == 0.0
    assert cand.params.tp_r_mult > 0.0
    stop, tp = strat._exit_geometry({"atr_pct": 0.05})
    assert stop == max(cand.params.stop_frac, cand.params.atr_stop_mult * 0.05)  # stop scaled
    assert tp == pytest.approx(cand.params.tp_r_mult * stop)  # reachable R-target, not the floor
