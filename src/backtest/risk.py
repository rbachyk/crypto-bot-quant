"""Risk simulation for the backtest (AGENTS.md Section 17, capital-agnostic).

A deterministic, simplified stand-in for the full Phase 6 risk manager: enough to
size positions honestly and reject candidates the real risk manager would reject,
so the backtest's trade set and rejected-candidate log are realistic. It enforces
the Section 17 per-trade sizing identity and the limits a single-position
simulation can express:

    size = (equity × risk_pct) / |entry − stop|

with leverage-as-consequence (``notional/equity``) capped by the envelope and the
metadata min-notional / min-order-size / lot-step gate. Full portfolio heat,
net-beta and correlation caps land with the real risk manager in Phase 6; this
module deliberately under-promises rather than fake them.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.config import AccountConfig
from src.exchange.metadata import MetadataConfig


@dataclass(frozen=True, slots=True)
class SizingResult:
    approved: bool
    qty: float = 0.0
    notional: float = 0.0
    leverage: float = 0.0
    risk_amount: float = 0.0
    reason: str = ""


def _round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return (int(qty / step)) * step


class RiskSimulator:
    """Deterministic per-trade sizing + hard gates (Section 17)."""

    def __init__(self, account: AccountConfig, meta: MetadataConfig) -> None:
        self.account = account
        self.meta = meta

    def size(
        self,
        symbol: str,
        *,
        equity: float,
        entry_price: float,
        stop_frac: float,
        risk_scale: float = 1.0,
    ) -> SizingResult:
        """``risk_scale`` (per-strategy, default 1.0) scales the per-trade risk DOWN for a
        strategy that is too volatile at the account standard — a hot momentum edge can be sized
        at e.g. 0.4× so its drawdown fits the risk envelope without changing its expectancy_r
        (R is sizing-invariant). Clamped to (0, 1]: a strategy can scale down, never UP past the
        account ``risk_pct`` (so it can never loosen the per-trade risk control)."""
        if equity <= 0:
            return SizingResult(False, reason="non_positive_equity")
        if stop_frac <= 0 or entry_price <= 0:
            return SizingResult(False, reason="invalid_stop_or_price")
        risk_scale = min(1.0, max(0.0, risk_scale))

        spec = self.meta.spec(symbol)
        fields = spec.fields if spec is not None else {}
        qty_step = float(fields.get("qty_step", fields.get("lot_size", 0.0)) or 0.0)
        min_order_size = float(fields.get("min_order_size", 0.0) or 0.0)
        min_notional = float(fields.get("min_notional", 0.0) or 0.0)
        meta_leverage = fields.get("max_leverage", self.account.max_leverage)
        max_leverage = min(
            self.account.max_leverage,
            float(meta_leverage or self.account.max_leverage),
        )

        # Section 17 per-trade sizing: size = (equity × risk_pct × risk_scale) / |entry − stop|.
        risk_amount = equity * self.account.risk_pct * risk_scale
        stop_distance = entry_price * stop_frac
        qty = risk_amount / stop_distance

        # Leverage-as-consequence: cap notional so notional/equity <= max_leverage.
        notional = qty * entry_price
        max_notional = equity * max_leverage
        if notional > max_notional:
            qty = max_notional / entry_price  # reduce size to respect the cap
            notional = qty * entry_price

        qty = _round_step(qty, qty_step)
        notional = qty * entry_price

        if qty <= 0 or qty < min_order_size:
            return SizingResult(
                False, reason=f"below_min_order_size(qty={qty:.8g}<{min_order_size})"
            )
        if notional < min_notional:
            return SizingResult(
                False, reason=f"below_min_notional(notional={notional:.8g}<{min_notional})"
            )

        leverage = notional / equity
        return SizingResult(
            approved=True,
            qty=qty,
            notional=notional,
            leverage=leverage,
            risk_amount=qty * stop_distance,
        )
