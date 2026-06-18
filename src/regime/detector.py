"""Deterministic, per-symbol regime detection (AGENTS.md Section 11).

v1 is a pure function of decision-time features (no ML — the ML regime classifier stays
shadow-only until promoted). When several regime conditions match, the SAFEST wins, in the
single ``regime_priority`` order declared in ``configs/regime.yaml`` so detection and tests
share one source. Three regimes are no-trade protection (``R8_DATA_UNSAFE``,
``R7_TOXIC_EXECUTION``, ``R4_HIGH_VOL_CHOP``); the ranking setup-quality gate consumes
:data:`NO_TRADE_REGIMES` to block them.

:class:`RegimeTracker` adds the mandatory anti-whipsaw rule: a *tradeable* regime must persist
a minimum number of bars before the label switches, but a *protective* (no-trade) regime engages
immediately — safety is never delayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

REGIME_YAML = REPO_ROOT / "configs" / "regime.yaml"

# The eight v1 regimes (Rn is a stable id, not a rank).
R8_DATA_UNSAFE = "R8_DATA_UNSAFE"
R7_TOXIC_EXECUTION = "R7_TOXIC_EXECUTION"
R6_LIQUIDATION_EVENT = "R6_LIQUIDATION_EVENT"
R5_MARKET_WIDE_IMPULSE = "R5_MARKET_WIDE_IMPULSE"
R4_HIGH_VOL_CHOP = "R4_HIGH_VOL_CHOP"
R3_HIGH_VOL_EXPANSION = "R3_HIGH_VOL_EXPANSION"
R2_TREND = "R2_TREND"
R1_LOW_VOL_RANGE = "R1_LOW_VOL_RANGE"

REGIME_CODES = frozenset(
    {
        R8_DATA_UNSAFE,
        R7_TOXIC_EXECUTION,
        R6_LIQUIDATION_EVENT,
        R5_MARKET_WIDE_IMPULSE,
        R4_HIGH_VOL_CHOP,
        R3_HIGH_VOL_EXPANSION,
        R2_TREND,
        R1_LOW_VOL_RANGE,
    }
)

_DEFAULT_PRIORITY = (
    R8_DATA_UNSAFE,
    R7_TOXIC_EXECUTION,
    R6_LIQUIDATION_EVENT,
    R3_HIGH_VOL_EXPANSION,
    R4_HIGH_VOL_CHOP,
    R5_MARKET_WIDE_IMPULSE,
    R2_TREND,
    R1_LOW_VOL_RANGE,
)


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    priority: tuple[str, ...] = _DEFAULT_PRIORITY
    no_trade: frozenset[str] = frozenset({R8_DATA_UNSAFE, R7_TOXIC_EXECUTION, R4_HIGH_VOL_CHOP})
    high_vol_rank: float = 0.80
    trend_dir_efficiency: float = 0.55
    chop_dir_efficiency: float = 0.35
    toxic_spread_bps: float = 25.0
    impulse_vol_z: float = 3.0
    liquidation_ret: float = 0.05
    trend_slope_min: float = 0.0008
    min_persist_bars: int = 3
    _protective: frozenset[str] = field(default=frozenset(), repr=False)

    @property
    def protective(self) -> frozenset[str]:
        """Regimes that engage immediately (no anti-whipsaw delay)."""
        return self.no_trade | {R8_DATA_UNSAFE, R7_TOXIC_EXECUTION, R6_LIQUIDATION_EVENT}


@lru_cache
def load_regime_config(path: str | None = None) -> RegimeConfig:
    yaml_path = Path(path) if path else REGIME_YAML
    data = (yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}).get("regime", {})
    th = data.get("thresholds", {})
    priority = tuple(data.get("priority", _DEFAULT_PRIORITY))
    return RegimeConfig(
        priority=priority,
        no_trade=frozenset(
            data.get("no_trade", [R8_DATA_UNSAFE, R7_TOXIC_EXECUTION, R4_HIGH_VOL_CHOP])
        ),
        high_vol_rank=float(th.get("high_vol_rank", 0.80)),
        trend_dir_efficiency=float(th.get("trend_dir_efficiency", 0.55)),
        chop_dir_efficiency=float(th.get("chop_dir_efficiency", 0.35)),
        toxic_spread_bps=float(th.get("toxic_spread_bps", 25.0)),
        impulse_vol_z=float(th.get("impulse_vol_z", 3.0)),
        liquidation_ret=float(th.get("liquidation_ret", 0.05)),
        trend_slope_min=float(th.get("trend_slope_min", 0.0008)),
        min_persist_bars=int(data.get("min_persist_bars", 3)),
    )


# Re-exported for the ranking setup-quality gate (single source of truth).
NO_TRADE_REGIMES: frozenset[str] = load_regime_config().no_trade


def detect_regime(
    row: dict,
    *,
    spread_bps: float = 0.0,
    data_ok: bool = True,
    cfg: RegimeConfig | None = None,
) -> str:
    """Classify the decision-time regime; safest matching label wins (priority order)."""
    cfg = cfg or load_regime_config()
    atr_rank = float(row.get("atr_pct_rank", 0.0))
    dir_eff = float(row.get("dir_efficiency", 0.0))
    trend = abs(float(row.get("trend_slope", 0.0)))
    vol_z = float(row.get("vol_z", 0.0))
    ret_1 = abs(float(row.get("ret_1", 0.0)))
    high_vol = atr_rank >= cfg.high_vol_rank

    matches: set[str] = {R1_LOW_VOL_RANGE}  # the always-available default
    if not data_ok:
        matches.add(R8_DATA_UNSAFE)
    if spread_bps > cfg.toxic_spread_bps:
        matches.add(R7_TOXIC_EXECUTION)
    if high_vol and ret_1 >= cfg.liquidation_ret:
        matches.add(R6_LIQUIDATION_EVENT)
    if high_vol and dir_eff >= cfg.trend_dir_efficiency:
        matches.add(R3_HIGH_VOL_EXPANSION)
    if high_vol and dir_eff <= cfg.chop_dir_efficiency:
        matches.add(R4_HIGH_VOL_CHOP)
    if vol_z >= cfg.impulse_vol_z:
        matches.add(R5_MARKET_WIDE_IMPULSE)
    if trend >= cfg.trend_slope_min and dir_eff >= cfg.chop_dir_efficiency:
        matches.add(R2_TREND)

    for code in cfg.priority:
        if code in matches:
            return code
    return R1_LOW_VOL_RANGE


class RegimeTracker:
    """Stateful per-symbol detector with anti-whipsaw persistence (Section 11)."""

    def __init__(self, cfg: RegimeConfig | None = None) -> None:
        self.cfg = cfg or load_regime_config()
        self.current: str | None = None
        self._pending: str | None = None
        self._count = 0

    def update(self, row: dict, *, spread_bps: float = 0.0, data_ok: bool = True) -> str:
        detected = detect_regime(row, spread_bps=spread_bps, data_ok=data_ok, cfg=self.cfg)
        if self.current is None or detected == self.current:
            self._pending, self._count = None, 0
            self.current = detected
            return self.current
        # Protective regimes engage immediately; tradeable changes must persist.
        if detected in self.cfg.protective:
            self.current, self._pending, self._count = detected, None, 0
            return self.current
        if detected == self._pending:
            self._count += 1
        else:
            self._pending, self._count = detected, 1
        if self._count >= self.cfg.min_persist_bars:
            self.current, self._pending, self._count = detected, None, 0
        return self.current
