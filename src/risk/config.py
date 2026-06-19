"""Loader for ``configs/risk.yaml`` — the Risk Manager contract (Section 17).

Turns the YAML into typed, frozen dataclasses read by the risk manager, the
circuit breakers and the RISK gate. The immutable :class:`RiskEnvelope` is built
through its clamping constructor so a config edit can only tighten it
(Section 2.2). ``base_risk_pct`` is likewise clamped to the envelope's
``max_risk_pct_per_trade`` at load — the risk_cap is the ceiling (Section 17).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT
from src.risk.envelope import RiskEnvelope

RISK_YAML = REPO_ROOT / "configs" / "risk.yaml"


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    consecutive_loss_limit: int = 4
    abnormal_slippage_cooldown: bool = True
    require_reconciled: bool = True
    # Section 17 additional breakers (capital-agnostic fractions).
    weekly_loss_limit: float = 0.08  # halt for the week at this fraction of weekly-peak equity
    per_symbol_loss_limit: float = 0.04  # per-symbol realized-loss halt (fraction of equity)
    funding_breaker_limit: float = 0.02  # cumulative funding paid halt (fraction of equity)
    min_liquidation_distance: float = 0.10  # entry refused if liq price is closer than this
    min_free_margin_frac: float = 0.20  # entry refused if free margin below this fraction


@dataclass(frozen=True, slots=True)
class RiskConfig:
    risk_policy_version: str
    envelope: RiskEnvelope
    base_risk_pct: float
    max_concurrent_total: int
    max_concurrent_per_symbol: int
    max_concurrent_per_regime: int
    breakers: BreakerConfig
    symbol_beta_to_btc: dict[str, float] = field(default_factory=dict)
    default_beta_to_btc: float = 1.0

    def beta_to_btc(self, symbol: str) -> float:
        return float(self.symbol_beta_to_btc.get(symbol, self.default_beta_to_btc))


@lru_cache
def load_risk_config(path: str | None = None) -> RiskConfig:
    yaml_path = Path(path) if path else RISK_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["risk"]

    envelope = RiskEnvelope.from_config(data.get("risk_envelope", {}))

    # base_risk_pct can never exceed the envelope's per-trade cap (Section 17:
    # "Max risk % of equity per trade (risk_cap)" is the ceiling).
    base_risk_pct = min(float(data.get("base_risk_pct", 0.005)), envelope.max_risk_pct_per_trade)

    br = data.get("breakers", {})
    breakers = BreakerConfig(
        consecutive_loss_limit=int(br.get("consecutive_loss_limit", 4)),
        abnormal_slippage_cooldown=bool(br.get("abnormal_slippage_cooldown", True)),
        require_reconciled=bool(br.get("require_reconciled", True)),
        weekly_loss_limit=float(br.get("weekly_loss_limit", 0.08)),
        per_symbol_loss_limit=float(br.get("per_symbol_loss_limit", 0.04)),
        funding_breaker_limit=float(br.get("funding_breaker_limit", 0.02)),
        min_liquidation_distance=float(br.get("min_liquidation_distance", 0.10)),
        min_free_margin_frac=float(br.get("min_free_margin_frac", 0.20)),
    )

    return RiskConfig(
        risk_policy_version=str(data.get("risk_policy_version", "risk_0001")),
        envelope=envelope,
        base_risk_pct=base_risk_pct,
        max_concurrent_total=int(data.get("max_concurrent_total", 5)),
        max_concurrent_per_symbol=int(data.get("max_concurrent_per_symbol", 1)),
        max_concurrent_per_regime=int(data.get("max_concurrent_per_regime", 3)),
        breakers=breakers,
        symbol_beta_to_btc={
            str(k): float(v) for k, v in (data.get("symbol_beta_to_btc") or {}).items()
        },
        default_beta_to_btc=float(data.get("default_beta_to_btc", 1.0)),
    )
