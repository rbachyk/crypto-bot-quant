"""The Risk Manager (AGENTS.md Section 17) — absolute authority over every order.

It approves, rejects, or resizes every candidate. No strategy, ML model, learner
or RL policy may bypass it (Section 17). Sizing is deterministic in v1:

    size = (equity × risk_pct) / |entry − stop|

with leverage a *consequence* of ``notional/equity`` capped by the envelope, never
a target (Section 2.2). It then enforces the portfolio-level caps — **heat**
(Σ open risk) and net **beta-to-BTC** — and the circuit breakers. ``risk_pct`` can
never exceed the envelope's ``max_risk_pct_per_trade`` (the risk_cap): that is
guaranteed at config-load (clamped) and re-asserted here.

The decision vocabulary is approve / resize / reject / block:

* **block** — a portfolio breaker fired (kill switch, daily-loss, drawdown,
  reconciliation, cooldown); no new entries at all (Section 17 circuit breakers).
* **reject** — this specific candidate is not allowed (min-notional, concurrency,
  beta cap, conflict).
* **resize** — approved at a *smaller* size to fit the leverage or heat cap.
* **approve** — approved at the full deterministic size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.exchange.metadata import MetadataConfig
from src.killswitch import KillSwitch
from src.ranking.candidate import Candidate
from src.risk.breakers import BreakerInputs, CircuitBreakers
from src.risk.config import RiskConfig
from src.risk.portfolio import PortfolioState


@dataclass(frozen=True, slots=True)
class AccountState:
    """Everything the risk manager needs about the account at decision time."""

    portfolio: PortfolioState
    breakers: BreakerInputs
    unknown_order_present: bool = False  # reconciliation found a foreign order
    # Free (available) margin on the account, when known (real venue). When provided, the
    # pre-trade free-margin blocker enforces the minimum free-margin buffer (Section 17); None
    # (paper / no account data) skips that check.
    free_margin: float | None = None
    # The would-be liquidation price for a new position on the candidate's symbol, when the venue
    # can estimate it (real venue). When provided, the pre-trade liquidation-distance blocker
    # refuses an entry whose liquidation sits too close; None (paper) skips the check.
    liquidation_price: float | None = None


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    action: str  # "approve" | "resize" | "reject" | "block"
    qty: float = 0.0
    notional: float = 0.0
    leverage: float = 0.0
    risk_amount: float = 0.0
    risk_pct_used: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)
    blocker: str | None = None

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "action": self.action,
            "qty": self.qty,
            "notional": self.notional,
            "leverage": self.leverage,
            "risk_amount": self.risk_amount,
            "risk_pct_used": self.risk_pct_used,
            "reasons": list(self.reasons),
            "blocker": self.blocker,
        }


def _round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    # floor (not int() truncation) so this matches the order builder's quantisation exactly —
    # a divergence could push a risk-approved size below min-notional at build time (Section 18).
    return math.floor(qty / step) * step


class RiskManager:
    """Deterministic per-trade sizing + portfolio caps + breakers (Section 17)."""

    def __init__(
        self,
        cfg: RiskConfig,
        meta: MetadataConfig,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.cfg = cfg
        self.meta = meta
        self.envelope = cfg.envelope
        self.breakers = CircuitBreakers(cfg, kill_switch=kill_switch)

    # ------------------------------------------------------------------ #
    def evaluate(self, candidate: Candidate, state: AccountState) -> RiskDecision:
        equity = state.portfolio.equity
        port = state.portfolio

        # 0) Portfolio circuit breakers (Section 17): a tripped breaker halts ALL
        #    new entries — this is a block, not a per-candidate reject.
        verdict = self.breakers.evaluate(state.breakers)
        if verdict.tripped:
            return RiskDecision(False, "block", blocker=verdict.reason, reasons=(verdict.reason,))

        # 1) Order-ownership / reconciliation conflict (Section 7/17): an unknown
        #    order means we cannot trust exchange state → halt new entries.
        if state.unknown_order_present:
            return RiskDecision(
                False,
                "block",
                blocker="unknown_order_conflict",
                reasons=("unknown_order_conflict",),
            )

        if equity <= 0:
            return RiskDecision(False, "reject", reasons=("non_positive_equity",), blocker=None)
        if candidate.stop_frac <= 0 or candidate.entry_price <= 0:
            return RiskDecision(False, "reject", reasons=("invalid_stop_or_price",))

        # 2) Concurrency caps (Section 17 portfolio-level).
        if port.count_for_symbol(candidate.symbol) >= self.cfg.max_concurrent_per_symbol:
            return RiskDecision(False, "reject", reasons=("open_position_conflict",))
        if port.count() >= self.cfg.max_concurrent_total:
            return RiskDecision(False, "reject", reasons=("max_concurrent_total",))
        if port.count_for_regime(candidate.regime) >= self.cfg.max_concurrent_per_regime:
            return RiskDecision(False, "reject", reasons=("max_concurrent_per_regime",))

        # 3) Deterministic per-trade sizing (Section 17). Each portfolio cap below
        #    can only REDUCE the size (resize down); the binding one wins.
        risk_pct = min(self.cfg.base_risk_pct, self.envelope.max_risk_pct_per_trade)
        stop_distance = candidate.entry_price * candidate.stop_frac
        target_risk = equity * risk_pct
        qty = target_risk / stop_distance
        reasons: list[str] = []
        action = "approve"

        # 3a) Leverage as a CONSEQUENCE, capped by the envelope (Section 2.2).
        max_notional = equity * self.envelope.max_leverage
        if qty * candidate.entry_price > max_notional:
            qty = max_notional / candidate.entry_price
            action = "resize"
            reasons.append("leverage_capped")

        # 3b) Heat cap (Σ open risk): resize so the new risk fits remaining heat.
        existing_heat = port.heat()
        remaining_heat = self.envelope.portfolio_heat_cap - existing_heat
        if remaining_heat <= 0:
            return RiskDecision(False, "reject", reasons=("portfolio_heat_cap_full",))
        max_qty_heat = (remaining_heat * equity) / stop_distance
        if qty > max_qty_heat:
            qty = max_qty_heat
            action = "resize"
            reasons.append("heat_capped")

        # 3c) Net beta-to-BTC cap (Section 2.2): resize so the marginal beta
        #     exposure keeps |net beta| within the cap; reject if there is no
        #     headroom in this direction at all.
        beta = self.cfg.beta_to_btc(candidate.symbol)
        b = candidate.side * beta / equity  # signed net-beta per unit notional
        if b != 0.0:
            current_net = port.net_beta()
            cap = self.envelope.net_beta_btc_cap
            bound = (cap - current_net) / b if b > 0 else (-cap - current_net) / b
            if bound <= 0:
                return RiskDecision(
                    False,
                    "reject",
                    reasons=(f"net_beta_cap_full(net={current_net:.4f}, cap={cap})",),
                )
            max_qty_beta = bound / candidate.entry_price
            if qty > max_qty_beta:
                qty = max_qty_beta
                action = "resize"
                reasons.append("beta_capped")

        # 3d) Quantise to the symbol's lot/qty step, then re-derive notional/risk.
        spec = self.meta.spec(candidate.symbol)
        fields = spec.fields if spec is not None else {}
        qty_step = float(fields.get("qty_step", fields.get("lot_size", 0.0)) or 0.0)
        min_order_size = float(fields.get("min_order_size", 0.0) or 0.0)
        min_notional = float(fields.get("min_notional", 0.0) or 0.0)
        qty = _round_step(qty, qty_step)
        notional = qty * candidate.entry_price
        risk_amount = qty * stop_distance

        # 3e) Min-notional / min-order-size gate (Section 17: below → no-trade).
        if qty <= 0 or qty < min_order_size:
            return RiskDecision(
                False,
                "reject",
                reasons=(f"below_min_order_size(qty={qty:.8g}<{min_order_size})",),
            )
        if notional < min_notional:
            return RiskDecision(
                False,
                "reject",
                reasons=(f"below_min_notional(notional={notional:.8g}<{min_notional})",),
            )

        # 4) Final envelope assertions (hard ceilings; Section 2.2).
        projected_net_beta = port.net_beta() + candidate.side * beta * notional / equity
        if abs(projected_net_beta) > self.envelope.net_beta_btc_cap + 1e-9:
            return RiskDecision(False, "reject", reasons=("net_beta_cap_breach",))
        leverage = notional / equity
        if leverage > self.envelope.max_leverage + 1e-9:
            return RiskDecision(False, "reject", reasons=("leverage_exceeds_envelope",))

        # 4a) Pre-trade liquidation-distance blocker (Section 17). Only when the venue supplies the
        #     would-be liquidation price (a real account): refuse an entry whose liquidation sits
        #     closer than min_liquidation_distance.
        if state.liquidation_price is not None and not self.breakers.liquidation_distance_ok(
            candidate.entry_price, state.liquidation_price, candidate.side
        ):
            return RiskDecision(False, "reject", reasons=("liquidation_too_close",))

        # 4b) Pre-trade free-margin blocker (Section 17). Only when the account's free margin is
        #     known (a real venue): refuse if posting this order's margin would breach the minimum
        #     free-margin buffer.
        if state.free_margin is not None:
            required_margin = notional / self.envelope.max_leverage
            if not self.breakers.margin_available(required_margin, state.free_margin, equity):
                return RiskDecision(False, "reject", reasons=("insufficient_free_margin",))

        risk_pct_used = risk_amount / equity
        if risk_pct_used > self.envelope.max_risk_pct_per_trade + 1e-9:
            return RiskDecision(False, "reject", reasons=("risk_pct_exceeds_envelope",))

        if action == "approve":
            reasons.append("within_all_limits")
        return RiskDecision(
            approved=True,
            action=action,
            qty=qty,
            notional=notional,
            leverage=leverage,
            risk_amount=risk_amount,
            risk_pct_used=risk_pct_used,
            reasons=tuple(reasons),
        )
