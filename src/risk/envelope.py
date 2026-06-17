"""The Immutable Risk Envelope (AGENTS.md Section 2.2).

The envelope is the box **no learner may ever modify**. These are version-
controlled constants; no deterministic optimizer, ML model, online learner or RL
policy may widen, disable or exceed them — at any maturity stage, ever. Learning
may only act *inside* this box (Section 2.2).

Two layers of protection:

* the values are loaded from ``configs/risk.yaml`` (version-controlled);
* :data:`HARD_CEILINGS` are enforced in code here regardless of config — a config
  edit can only make the envelope *tighter*, never looser than these ceilings.
  This is the Section 21.9 guarantee ("``risk_envelope`` … enforced in code
  regardless of config edits; the config cannot disable them").
"""

from __future__ import annotations

from dataclasses import dataclass

# Code-level ceilings the config may never exceed (capital protection, Priority
# Stack 1). A config edit that sets a looser value is clamped to these.
HARD_CEILINGS: dict[str, float] = {
    "max_leverage": 10.0,
    "max_risk_pct_per_trade": 0.02,
    "portfolio_heat_cap": 0.10,
    "net_beta_btc_cap": 0.60,
    "daily_loss_limit": 0.10,
    "max_drawdown_limit": 0.25,
}


@dataclass(frozen=True, slots=True)
class RiskEnvelope:
    """The immutable risk envelope (Section 2.2). Frozen + clamped at load."""

    max_leverage: float
    max_risk_pct_per_trade: float
    portfolio_heat_cap: float
    net_beta_btc_cap: float
    daily_loss_limit: float
    max_drawdown_limit: float

    @classmethod
    def from_config(cls, raw: dict) -> RiskEnvelope:
        """Build from config, clamping every field to its hard code ceiling.

        ``min(config, ceiling)`` means a config can only tighten the envelope;
        it can never widen it past the code-level ceiling (Section 2.2 / 21.9).
        """
        return cls(
            max_leverage=_clamp("max_leverage", raw.get("max_leverage")),
            max_risk_pct_per_trade=_clamp(
                "max_risk_pct_per_trade", raw.get("max_risk_pct_per_trade")
            ),
            portfolio_heat_cap=_clamp("portfolio_heat_cap", raw.get("portfolio_heat_cap")),
            net_beta_btc_cap=_clamp("net_beta_btc_cap", raw.get("net_beta_btc_cap")),
            daily_loss_limit=_clamp("daily_loss_limit", raw.get("daily_loss_limit")),
            max_drawdown_limit=_clamp("max_drawdown_limit", raw.get("max_drawdown_limit")),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "max_leverage": self.max_leverage,
            "max_risk_pct_per_trade": self.max_risk_pct_per_trade,
            "portfolio_heat_cap": self.portfolio_heat_cap,
            "net_beta_btc_cap": self.net_beta_btc_cap,
            "daily_loss_limit": self.daily_loss_limit,
            "max_drawdown_limit": self.max_drawdown_limit,
        }


def _clamp(field: str, value: object) -> float:
    ceiling = HARD_CEILINGS[field]
    if not isinstance(value, (int, float)) or value <= 0:
        # Missing/invalid → fall back to the capital-preserving ceiling.
        return ceiling
    return min(float(value), ceiling)
