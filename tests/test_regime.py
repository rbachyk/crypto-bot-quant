"""Section 11: deterministic regime detection + anti-whipsaw + no-trade wiring."""

from __future__ import annotations

from src.config.settings import REPO_ROOT
from src.ranking.setup_quality import NO_TRADE_REGIMES as RANKING_NO_TRADE
from src.regime import NO_TRADE_REGIMES, RegimeTracker, detect_regime, load_regime_config
from src.regime.detector import (
    R1_LOW_VOL_RANGE,
    R2_TREND,
    R3_HIGH_VOL_EXPANSION,
    R4_HIGH_VOL_CHOP,
    R5_MARKET_WIDE_IMPULSE,
    R6_LIQUIDATION_EVENT,
    R7_TOXIC_EXECUTION,
    R8_DATA_UNSAFE,
)


def _row(**over) -> dict:
    base = {
        "atr_pct_rank": 0.5,
        "dir_efficiency": 0.5,
        "trend_slope": 0.0,
        "vol_z": 0.0,
        "ret_1": 0.0,
    }
    base.update(over)
    return base


def test_data_unsafe_wins_everything() -> None:
    # Even with a toxic spread and high vol, R8 (highest priority) wins.
    assert detect_regime(_row(atr_pct_rank=0.95), spread_bps=99.0, data_ok=False) == R8_DATA_UNSAFE


def test_toxic_execution_when_spread_above_cap() -> None:
    assert detect_regime(_row(), spread_bps=30.0, data_ok=True) == R7_TOXIC_EXECUTION


def test_liquidation_spike_beats_expansion() -> None:
    # high vol + large single-bar move → R6 (ranked above R3/R4).
    r = detect_regime(_row(atr_pct_rank=0.9, dir_efficiency=0.7, ret_1=0.06), spread_bps=1.0)
    assert r == R6_LIQUIDATION_EVENT


def test_high_vol_expansion_vs_chop_split_on_directional_efficiency() -> None:
    assert (
        detect_regime(_row(atr_pct_rank=0.9, dir_efficiency=0.7), spread_bps=1.0)
        == R3_HIGH_VOL_EXPANSION
    )
    assert (
        detect_regime(_row(atr_pct_rank=0.9, dir_efficiency=0.2), spread_bps=1.0)
        == R4_HIGH_VOL_CHOP
    )


def test_market_wide_impulse() -> None:
    assert detect_regime(_row(vol_z=4.0), spread_bps=1.0) == R5_MARKET_WIDE_IMPULSE


def test_trend_and_default_range() -> None:
    assert detect_regime(_row(trend_slope=0.002, dir_efficiency=0.5), spread_bps=1.0) == R2_TREND
    assert detect_regime(_row(), spread_bps=1.0) == R1_LOW_VOL_RANGE


def test_no_trade_set_is_canonical_and_shared() -> None:
    assert {R8_DATA_UNSAFE, R7_TOXIC_EXECUTION, R4_HIGH_VOL_CHOP} == NO_TRADE_REGIMES
    # The ranking gate consumes the SAME object — detection and the gate cannot drift.
    assert RANKING_NO_TRADE is NO_TRADE_REGIMES


def test_priority_list_loaded_from_config() -> None:
    cfg = load_regime_config(str(REPO_ROOT / "configs" / "regime.yaml"))
    assert cfg.priority[0] == R8_DATA_UNSAFE  # safest first
    assert cfg.priority[-1] == R1_LOW_VOL_RANGE
    assert len(cfg.priority) == 8


def test_tracker_anti_whipsaw_for_tradeable_regimes() -> None:
    t = RegimeTracker(load_regime_config())  # min_persist_bars = 3
    trend = _row(trend_slope=0.002, dir_efficiency=0.5)
    assert t.update(_row(), spread_bps=1.0) == R1_LOW_VOL_RANGE
    # A single trending bar must NOT flip the regime (anti-whipsaw).
    assert t.update(trend, spread_bps=1.0) == R1_LOW_VOL_RANGE
    assert t.update(trend, spread_bps=1.0) == R1_LOW_VOL_RANGE
    assert t.update(trend, spread_bps=1.0) == R2_TREND  # persisted 3 bars → switch


def test_tracker_protective_regime_engages_immediately() -> None:
    t = RegimeTracker(load_regime_config())
    t.update(_row(), spread_bps=1.0)  # R1
    # A toxic-execution regime must engage on the very next bar (no persistence delay).
    assert t.update(_row(), spread_bps=30.0) == R7_TOXIC_EXECUTION
