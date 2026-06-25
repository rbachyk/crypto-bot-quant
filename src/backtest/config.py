"""Loader for ``configs/backtest.yaml`` — the Backtest Engine contract.

Turns the YAML into typed, frozen dataclasses read by the engine, the cost
models, the risk + execution simulation, the walk-forward harness, the stress
runners and the BT/WF/FEE/SLIP gates, so every consumer shares one definition of
costs, fold layout, kill-criteria and stress multipliers (Section 4 config-driven
behaviour; Section 19 event-based backtest).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

BACKTEST_YAML = REPO_ROOT / "configs" / "backtest.yaml"


@dataclass(frozen=True, slots=True)
class AccountConfig:
    initial_equity: float = 100_000.0
    risk_pct: float = 0.005
    max_leverage: float = 10.0
    max_concurrent_per_symbol: int = 1
    max_concurrent_total: int = 5


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    fill: str = "next_bar_open"
    entry_order_type: str = "market"
    max_spread_bps: float = 25.0
    max_slippage_frac: float = 0.01


@dataclass(frozen=True, slots=True)
class CostConfig:
    fee_multiplier: float = 1.0
    fallback_maker_fee: float = 0.0002
    fallback_taker_fee: float = 0.00055
    slippage_multiplier: float = 1.0
    impact_coeff: float = 0.0
    min_half_spread_frac: float = 0.0001
    funding_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class ReferenceStrategyConfig:
    name: str = "reference_momentum"
    strategy_version: str = "ref_bt_0001"
    signal_threshold: float = 0.0008
    stop_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    min_stop_frac: float = 0.004
    hold_bars: int = 8
    allow_long: bool = True
    allow_short: bool = True


@dataclass(frozen=True, slots=True)
class ReferenceDataConfig:
    edge: str = "trend"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT:USDT"])
    bars: int = 3000
    timeframe: str = "5m"
    trend_drift: float = 0.0004
    trend_period_bars: int = 180
    base_sigma: float = 0.0016
    seed: str = "reference_bt"
    activation_bar: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KillCriteria:
    min_oos_expectancy_r: float = 0.03
    min_oos_profit_factor: float = 1.10
    max_oos_drawdown: float = 0.25
    min_folds_passed: int = 4
    min_trades_per_fold: int = 20
    # Multiple-testing-aware significance floor: the deflated Sharpe (PSR over the fold trials)
    # must clear this for the verdict to pass. 0.5 = "the edge is more-likely-than-not genuinely
    # positive after adjusting for the multiple folds". A no-edge strategy whose folds average
    # negative scores below 0.5; it is the neutral point of a probability, not a fitted number.
    min_deflated_sharpe: float = 0.5


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    folds: int = 5
    train_frac: float = 0.5
    holdout_frac: float = 0.2
    kill_criteria: KillCriteria = field(default_factory=KillCriteria)
    # How a FOLD is judged. The walk-forward asks two DIFFERENT questions: per-fold = is the edge
    # present across time (STABILITY)? hold-out = is it economically viable on never-seen data?
    # "directional" (default) tests fold STABILITY as expectancy_r > 0 (+ trade adequacy + the
    # drawdown risk cap) and reserves the economic magnitude bar (expectancy≥min, PF≥min) for the
    # locked hold-out — so a thin-but-real edge that is directionally positive in most folds and
    # clears the hold-out is not rejected for per-fold magnitude noise. "economic" (legacy) applies
    # the full economic kill-criteria to every fold AND the hold-out. The hold-out is ALWAYS judged
    # on the full economic criteria regardless; only the FOLDS' test changes.
    fold_criterion: str = "directional"


@dataclass(frozen=True, slots=True)
class StressConfig:
    fee_multiplier: float = 2.0
    slippage_multiplier: float = 1.5


@dataclass(frozen=True, slots=True)
class SanityConfig:
    max_abs_total_return: float = 100.0
    max_abs_bar_return: float = 0.5


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    backtest_version: str
    account: AccountConfig
    execution: ExecutionConfig
    costs: CostConfig
    reference_strategy: ReferenceStrategyConfig
    reference: ReferenceDataConfig
    walk_forward: WalkForwardConfig
    stress: StressConfig
    sanity: SanityConfig

    def with_cost_overrides(
        self,
        *,
        fee_multiplier: float | None = None,
        slippage_multiplier: float | None = None,
        funding_multiplier: float | None = None,
    ) -> BacktestConfig:
        """Return a copy with cost multipliers overridden (fee/slippage stress)."""
        costs = replace(
            self.costs,
            fee_multiplier=fee_multiplier
            if fee_multiplier is not None
            else self.costs.fee_multiplier,
            slippage_multiplier=slippage_multiplier
            if slippage_multiplier is not None
            else self.costs.slippage_multiplier,
            funding_multiplier=funding_multiplier
            if funding_multiplier is not None
            else self.costs.funding_multiplier,
        )
        return replace(self, costs=costs)


@lru_cache
def load_backtest_config(path: str | None = None) -> BacktestConfig:
    yaml_path = Path(path) if path else BACKTEST_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["backtest"]

    acc = data.get("account", {})
    ex = data.get("execution", {})
    co = data.get("costs", {})
    rs = data.get("reference_strategy", {})
    rd = data.get("reference", {})
    wf = data.get("walk_forward", {})
    kc = wf.get("kill_criteria", {})
    st = data.get("stress", {})
    sa = data.get("sanity", {})

    return BacktestConfig(
        backtest_version=str(data.get("backtest_version", "bt_0001")),
        account=AccountConfig(
            initial_equity=float(acc.get("initial_equity", 100_000.0)),
            risk_pct=float(acc.get("risk_pct", 0.005)),
            max_leverage=float(acc.get("max_leverage", 10.0)),
            max_concurrent_per_symbol=int(acc.get("max_concurrent_per_symbol", 1)),
            max_concurrent_total=int(acc.get("max_concurrent_total", 5)),
        ),
        execution=ExecutionConfig(
            fill=str(ex.get("fill", "next_bar_open")),
            entry_order_type=str(ex.get("entry_order_type", "market")),
            max_spread_bps=float(ex.get("max_spread_bps", 25.0)),
            max_slippage_frac=float(ex.get("max_slippage_frac", 0.01)),
        ),
        costs=CostConfig(
            fee_multiplier=float(co.get("fee_multiplier", 1.0)),
            fallback_maker_fee=float(co.get("fallback_maker_fee", 0.0002)),
            fallback_taker_fee=float(co.get("fallback_taker_fee", 0.00055)),
            slippage_multiplier=float(co.get("slippage_multiplier", 1.0)),
            impact_coeff=float(co.get("impact_coeff", 0.0)),
            min_half_spread_frac=float(co.get("min_half_spread_frac", 0.0001)),
            funding_multiplier=float(co.get("funding_multiplier", 1.0)),
        ),
        reference_strategy=ReferenceStrategyConfig(
            name=str(rs.get("name", "reference_momentum")),
            strategy_version=str(rs.get("strategy_version", "ref_bt_0001")),
            signal_threshold=float(rs.get("signal_threshold", 0.0008)),
            stop_atr_mult=float(rs.get("stop_atr_mult", 1.5)),
            tp_atr_mult=float(rs.get("tp_atr_mult", 3.0)),
            min_stop_frac=float(rs.get("min_stop_frac", 0.004)),
            hold_bars=int(rs.get("hold_bars", 8)),
            allow_long=bool(rs.get("allow_long", True)),
            allow_short=bool(rs.get("allow_short", True)),
        ),
        reference=ReferenceDataConfig(
            edge=str(rd.get("edge", "trend")),
            symbols=list(rd.get("symbols", ["BTC/USDT:USDT"])),
            bars=int(rd.get("bars", 3000)),
            timeframe=str(rd.get("timeframe", "5m")),
            trend_drift=float(rd.get("trend_drift", 0.0004)),
            trend_period_bars=int(rd.get("trend_period_bars", 180)),
            base_sigma=float(rd.get("base_sigma", 0.0016)),
            seed=str(rd.get("seed", "reference_bt")),
            activation_bar={str(k): int(v) for k, v in (rd.get("activation_bar") or {}).items()},
        ),
        walk_forward=WalkForwardConfig(
            folds=int(wf.get("folds", 5)),
            train_frac=float(wf.get("train_frac", 0.5)),
            holdout_frac=float(wf.get("holdout_frac", 0.2)),
            fold_criterion=str(wf.get("fold_criterion", "directional")),
            kill_criteria=KillCriteria(
                min_oos_expectancy_r=float(kc.get("min_oos_expectancy_r", 0.03)),
                min_oos_profit_factor=float(kc.get("min_oos_profit_factor", 1.10)),
                max_oos_drawdown=float(kc.get("max_oos_drawdown", 0.25)),
                min_folds_passed=int(kc.get("min_folds_passed", 4)),
                min_trades_per_fold=int(kc.get("min_trades_per_fold", 20)),
                min_deflated_sharpe=float(kc.get("min_deflated_sharpe", 0.5)),
            ),
        ),
        stress=StressConfig(
            fee_multiplier=float(st.get("fee_multiplier", 2.0)),
            slippage_multiplier=float(st.get("slippage_multiplier", 1.5)),
        ),
        sanity=SanityConfig(
            max_abs_total_return=float(sa.get("max_abs_total_return", 100.0)),
            max_abs_bar_return=float(sa.get("max_abs_bar_return", 0.5)),
        ),
    )
