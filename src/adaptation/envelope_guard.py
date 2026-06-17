"""Immutable-envelope enforcement (AGENTS.md Section 2.2, 21.6).

The :func:`enforce` function is the last line of defence before any learner
action reaches the Risk Layer. It checks the *immutable* envelope constants
loaded from ``configs/risk.yaml`` against the proposed :class:`~.action_space.BoundedAction`
and either clamps, rejects, or passes the action.

Hard invariants that CANNOT be overridden by any config edit (Section 2.2):
  - ``size_bucket`` ≤ 1.0
  - ``take=True`` cannot resurrect a hard-blocked candidate
  - Actions referencing disabled/unvalidated strategies → reject
  - Stops, leverage, heat, beta, breaker thresholds, and no-trade regimes are
    NEVER registered as tunable and will trigger a reject if present

The envelope constants (max_leverage, max_risk_pct_per_trade, etc.) are loaded
fresh from the risk config on every call to avoid caching a stale value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.adaptation.action_space import BoundedAction

# Paths are resolved relative to the repo root (not this file).
_RISK_YAML: Path | None = None


def _risk_yaml_path() -> Path:
    global _RISK_YAML  # noqa: PLW0603
    if _RISK_YAML is None:
        from src.config.settings import REPO_ROOT

        _RISK_YAML = REPO_ROOT / "configs" / "risk.yaml"
    return _RISK_YAML


@dataclass(frozen=True)
class RiskEnvelope:
    """Immutable copy of the risk envelope constants (Section 2.2)."""

    max_leverage: float
    max_risk_pct_per_trade: float
    portfolio_heat_cap: float
    net_beta_btc_cap: float
    daily_loss_limit: float
    max_drawdown_limit: float


def _load_envelope() -> RiskEnvelope:
    path = _risk_yaml_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    env = raw["risk"]["risk_envelope"]
    return RiskEnvelope(
        max_leverage=float(env["max_leverage"]),
        max_risk_pct_per_trade=float(env["max_risk_pct_per_trade"]),
        portfolio_heat_cap=float(env["portfolio_heat_cap"]),
        net_beta_btc_cap=float(env["net_beta_btc_cap"]),
        daily_loss_limit=float(env["daily_loss_limit"]),
        max_drawdown_limit=float(env["max_drawdown_limit"]),
    )


# Params that are hard-blocked from ever being registered as tunable.
# A param_nudge key matching any of these → reject (Section 21.6 hard invariants).
_FORBIDDEN_TUNABLES: frozenset[str] = frozenset(
    {
        "max_leverage",
        "max_risk_pct_per_trade",
        "portfolio_heat_cap",
        "net_beta_btc_cap",
        "daily_loss_limit",
        "max_drawdown_limit",
        "stop_price",
        "stop_frac",
        "leverage",
        "heat_cap",
        "beta_cap",
        "drawdown_limit",
        "daily_loss",
    }
)


@dataclass
class GuardResult:
    """Result of :func:`enforce`."""

    action: BoundedAction
    clamped_fields: list[str]
    rejected: bool
    rejection_reason: str | None


def enforce(
    action: BoundedAction,
    *,
    active_strategies: set[str] | None = None,
    envelope: RiskEnvelope | None = None,
) -> GuardResult:
    """Enforce the immutable envelope on a :class:`BoundedAction`.

    Must be called AFTER :func:`~src.adaptation.action_space.validate`.

    ``active_strategies``: the set of currently enabled, validated strategies.
    If provided, any ``strategy_weights`` key not in this set → reject.

    ``envelope``: optional pre-loaded envelope (for testing); if None the
    envelope is loaded fresh from ``configs/risk.yaml`` on every call.
    """
    import copy

    env = envelope or _load_envelope()
    act = copy.deepcopy(action)
    clamped: list[str] = []

    # --- size_bucket hard cap (Section 21.6) --------------------------------- #
    if act.size_bucket > 1.0:
        act.size_bucket = 1.0
        clamped.append("size_bucket")

    # size_bucket × max_risk_pct_per_trade must not exceed the per-trade cap.
    # At size_bucket ≤ 1.0 and max_risk_pct_per_trade = risk_cap this is always
    # satisfied by construction; we check explicitly to catch edge-case bugs.
    effective_risk = act.size_bucket * env.max_risk_pct_per_trade
    if effective_risk > env.max_risk_pct_per_trade:
        # Clamp size_bucket so it no longer breaches (should be unreachable given
        # size_bucket ≤ 1.0, but kept as a belt-and-suspenders guard).
        act.size_bucket = 1.0
        clamped.append("size_bucket")

    # --- strategy_weights — only active, validated strategies allowed --------- #
    if active_strategies is not None:
        bad = set(act.strategy_weights) - active_strategies
        if bad:
            return GuardResult(
                action=act,
                clamped_fields=clamped,
                rejected=True,
                rejection_reason=(
                    f"strategy_weights references disabled/unvalidated strategies: {sorted(bad)}"
                ),
            )

    # --- param_nudges — forbidden envelope params must not appear ------------ #
    forbidden_hit = set(act.param_nudges) & _FORBIDDEN_TUNABLES
    if forbidden_hit:
        return GuardResult(
            action=act,
            clamped_fields=clamped,
            rejected=True,
            rejection_reason=(
                f"param_nudges attempts to touch forbidden envelope params: "
                f"{sorted(forbidden_hit)}"
            ),
        )

    # --- take=True cannot resurrect a candidate the deterministic system     --- #
    # did not produce, nor one a hard blocker rejected.  This is checked in the  #
    # controller at decision time (we cannot verify absence without the live      #
    # candidate list here), so we only flag mode mismatch.                        #
    if act.take and act.mode not in ("SHADOW", "RECOMMEND", "LIVE_BOUNDED"):
        return GuardResult(
            action=act,
            clamped_fields=clamped,
            rejected=True,
            rejection_reason=f"take=True with unknown mode={act.mode!r}",
        )

    return GuardResult(
        action=act,
        clamped_fields=clamped,
        rejected=False,
        rejection_reason=None,
    )
