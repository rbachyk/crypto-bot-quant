"""Orders + the order builder (AGENTS.md Section 18, EXEC gate).

The order builder turns an approved candidate (+ its :class:`RiskDecision` size)
into a complete **bracket**: the entry order plus the exchange-resident stop and,
depending on the exit geometry, an exchange-resident take-profit or an
exchange-native trailing stop (Section 12/18). It respects the symbol's verified
tick size, lot/qty step and minimum notional (Section 18 "order builder respects
tick/lot/min-notional"), and stamps every leg with the ownership prefix +
provenance tags (Section 7). Stops/TP are attached **at entry** so a position is
never left without exchange-side protection (Section 2.2).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from src.exchange.metadata import VerifiedSpec
from src.execution.config import ExecutionPolicyConfig
from src.execution.ownership import OwnershipPolicy
from src.ranking.candidate import Candidate
from src.risk.manager import RiskDecision

# A take-profit distance at/above this fraction encodes "no fixed TP" (momentum,
# Section 12: the tail is the edge) — the builder uses a trailing stop instead.
NO_FIXED_TP_FRAC = 0.5

BUY = "buy"
SELL = "sell"


class OrderType(str, enum.Enum):
    """Order types (mapped to the venue's verified ``supported_order_types``)."""

    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"  # maker-only limit
    REDUCE_ONLY = "reduce_only"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TRAILING_STOP = "trailing_stop"


@dataclass(slots=True)
class Order:
    """A single order leg (entry / stop / take-profit / trailing)."""

    client_id: str
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    order_type: OrderType
    role: str = "entry"  # entry | stop | take_profit | trailing
    price: float | None = None
    stop_price: float | None = None
    trail_offset: float | None = None  # fraction of price (native trailing)
    reduce_only: bool = False
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "role": self.role,
            "price": self.price,
            "stop_price": self.stop_price,
            "trail_offset": self.trail_offset,
            "reduce_only": self.reduce_only,
            "tags": dict(self.tags),
        }


@dataclass(slots=True)
class OrderPlan:
    """An entry bracket: entry + exchange-resident stop + (TP or trailing)."""

    symbol: str
    side: int
    qty: float
    entry: Order
    stop: Order
    take_profit: Order | None = None
    trailing: Order | None = None

    def legs(self) -> list[Order]:
        return [o for o in (self.entry, self.stop, self.take_profit, self.trailing) if o]

    @property
    def has_exchange_side_stop(self) -> bool:
        return self.stop is not None and self.stop.reduce_only

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": "long" if self.side > 0 else "short",
            "qty": self.qty,
            "legs": [o.to_dict() for o in self.legs()],
        }


@dataclass(slots=True)
class BuildResult:
    ok: bool
    plan: OrderPlan | None = None
    reason: str = ""


def _round_to_tick(price: float, tick: float, *, side_up: bool) -> float:
    if tick <= 0:
        return price
    n = price / tick
    rounded = math.ceil(n) if side_up else math.floor(n)
    return rounded * tick


def _round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


class OrderBuilder:
    """Builds tick/lot/min-notional-respecting bracket orders (Section 18)."""

    def __init__(
        self,
        cfg: ExecutionPolicyConfig,
        ownership: OwnershipPolicy,
    ) -> None:
        self.cfg = cfg
        self.ownership = ownership

    def supported_order_types(self, spec: VerifiedSpec) -> set[str]:
        ot = spec.fields.get("order_types")
        return set(ot) if isinstance(ot, list) else set()

    def build(
        self,
        candidate: Candidate,
        decision: RiskDecision,
        spec: VerifiedSpec,
        *,
        entry_style: str | None = None,
    ) -> BuildResult:
        if not decision.approved or decision.qty <= 0:
            return BuildResult(False, reason="risk_not_approved")

        f = spec.fields
        tick = float(f.get("tick_size", 0.0) or 0.0)
        qty_step = float(f.get("qty_step", f.get("lot_size", 0.0)) or 0.0)
        min_order_size = float(f.get("min_order_size", 0.0) or 0.0)
        min_notional = float(f.get("min_notional", 0.0) or 0.0)

        qty = _round_step(decision.qty, qty_step)
        if qty <= 0 or qty < min_order_size:
            return BuildResult(False, reason=f"qty_below_min({qty}<{min_order_size})")

        long = candidate.side > 0
        entry_side = BUY if long else SELL
        exit_side = SELL if long else BUY

        # Entry style: explicit override, else the strategy's own maker flag, else config default
        # (parity with the backtest, where maker entries rest at a passive limit).
        style = entry_style or ("maker" if candidate.maker else self.cfg.default_entry_style)
        entry_type = OrderType.MARKET if style == "taker" else OrderType.POST_ONLY

        if entry_type is OrderType.MARKET:
            # Taker: snapped with adverse rounding (pay up to enter).
            entry_price = _round_to_tick(candidate.entry_price, tick, side_up=long)
        else:
            # Maker passive limit: post limit_offset_frac INSIDE the reference (buy below / sell
            # above) and snap to the more-passive tick — mirrors the backtest maker fill price.
            raw_limit = candidate.entry_price * (1.0 - candidate.side * candidate.limit_offset_frac)
            entry_price = _round_to_tick(raw_limit, tick, side_up=not long)
        if entry_price * qty < min_notional:
            return BuildResult(
                False, reason=f"notional_below_min({entry_price * qty:.8g}<{min_notional})"
            )
        entry = Order(
            client_id=self.ownership.new_client_id("entry"),
            symbol=candidate.symbol,
            side=entry_side,
            qty=qty,
            order_type=entry_type,
            role="entry",
            price=None if entry_type is OrderType.MARKET else entry_price,
            tags=self.ownership.tags(),
        )

        # Exchange-resident stop (reduce-only) attached AT ENTRY (Section 2.2).
        stop_price = _round_to_tick(candidate.stop_price, tick, side_up=not long)
        stop = Order(
            client_id=self.ownership.new_client_id("stop"),
            symbol=candidate.symbol,
            side=exit_side,
            qty=qty,
            order_type=OrderType.STOP_MARKET,
            role="stop",
            stop_price=stop_price,
            reduce_only=True,
            tags=self.ownership.tags(parent_id=entry.client_id),
        )

        no_fixed_tp = candidate.tp_frac >= NO_FIXED_TP_FRAC
        take_profit: Order | None = None
        trailing: Order | None = None

        # Trailing stop: parity with the backtest uses the STRATEGY'S OWN offset (atr_trail_mult×
        # atr, carried on the candidate), falling back to the config offset. Floored at the stop
        # so the trail never sits tighter than the hard stop. Exchange-native (survives downtime).
        trail_off = max(candidate.trail_frac, self.cfg.trailing_offset_frac)
        if trail_off > 0 or no_fixed_tp:
            offset = max(trail_off, candidate.stop_frac)
            trailing = Order(
                client_id=self.ownership.new_client_id("trail"),
                symbol=candidate.symbol,
                side=exit_side,
                qty=qty,
                order_type=OrderType.TRAILING_STOP,
                role="trailing",
                trail_offset=offset,
                reduce_only=True,
                tags=self.ownership.tags(parent_id=entry.client_id),
            )
        # Take-profit: armed when the TP is REACHABLE (not the unreachable momentum sentinel) and
        # enabled. A momentum candidate now carries BOTH a reachable R-target TP AND a trailing
        # stop — Bybit holds stopLoss + takeProfit + trailingStop on the position simultaneously, so
        # whichever triggers first exits, matching the backtest's stop/TP/trail OR-of-exits.
        if not no_fixed_tp and self.cfg.attach_take_profit:
            tp_price = _round_to_tick(candidate.tp_price, tick, side_up=long)
            take_profit = Order(
                client_id=self.ownership.new_client_id("tp"),
                symbol=candidate.symbol,
                side=exit_side,
                qty=qty,
                order_type=OrderType.TAKE_PROFIT_MARKET,
                role="take_profit",
                # A take-profit-market is a TRIGGER order: ``stop_price`` is the trigger the
                # venue arms (mirrors the stop leg). ``price`` is kept equal for any consumer
                # that reads the target level. Both must be set or the venue layer (which
                # attaches ``takeProfit`` off ``stop_price``) would silently drop the TP.
                price=tp_price,
                stop_price=tp_price,
                reduce_only=True,
                tags=self.ownership.tags(parent_id=entry.client_id),
            )

        plan = OrderPlan(
            symbol=candidate.symbol,
            side=candidate.side,
            qty=qty,
            entry=entry,
            stop=stop,
            take_profit=take_profit,
            trailing=trailing,
        )
        return BuildResult(True, plan=plan)
