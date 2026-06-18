"""Simulated execution venue (AGENTS.md Section 18, EXEC gate).

A deterministic, offline stand-in for the exchange used for paper trading and the
EXEC/ORDER-OWN gates. It models the venue behaviours the execution adapter must
support: **atomic** bracket placement (entry + exchange-resident stop + TP/native
trailing registered in one call), partial fills, cancel and cancel/replace, order
status, slippage measurement (expected vs actual fill price), and the reconcilable
state the bot checks against (Section 17). It can be injected with *foreign*
orders/positions (no ownership prefix) to exercise unknown-order detection.

A real venue (ccxt + native SDK) swaps in behind the same surface in a later
phase; nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.exchange.metadata import MetadataConfig
from src.execution.order import BUY, SELL, Order, OrderPlan, OrderType


@dataclass(slots=True)
class Fill:
    """A realised fill with the Section 18 execution-quality fields."""

    client_id: str
    symbol: str
    side: str
    qty: float
    expected_price: float
    actual_price: float
    fee: float
    maker: bool
    latency_ms: float
    slippage_frac: float  # |actual − expected| / expected
    slippage_cost: float  # |actual − expected| × qty
    spread_bps_at_order: float
    signal_age_ms: float
    order_type: str

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "expected_price": self.expected_price,
            "actual_price": self.actual_price,
            "fee": self.fee,
            "maker": self.maker,
            "latency_ms": self.latency_ms,
            "slippage_frac": self.slippage_frac,
            "slippage_cost": self.slippage_cost,
            "spread_bps_at_order": self.spread_bps_at_order,
            "signal_age_ms": self.signal_age_ms,
            "order_type": self.order_type,
        }


@dataclass(slots=True)
class VenuePosition:
    symbol: str
    side: int
    qty: float
    entry_price: float
    stop_order_id: str | None = None
    tp_order_id: str | None = None
    trail_order_id: str | None = None
    owned: bool = True  # carries the bot's ownership prefix

    def has_exchange_side_stop(self) -> bool:
        return self.stop_order_id is not None or self.trail_order_id is not None


@dataclass(slots=True)
class BracketResult:
    fill: Fill
    position: VenuePosition
    resting_order_ids: list[str] = field(default_factory=list)
    fully_filled: bool = True
    remaining_qty: float = 0.0


@runtime_checkable
class Venue(Protocol):
    """The execution surface the engine drives — one contract for simulated and live.

    A live (ccxt/testnet) venue swaps in behind this Protocol; the ExecutionEngine,
    paper engine and live loop never depend on the concrete venue (Section 18)."""

    open_orders: dict[str, Order]
    positions: dict[str, VenuePosition]

    def place_bracket(
        self,
        plan: OrderPlan,
        *,
        ref_price: float,
        realized_slippage_frac: float,
        latency_ms: float,
        spread_bps: float = 0.0,
        signal_age_ms: float = 0.0,
        fill_ratio: float = 1.0,
    ) -> BracketResult: ...

    def cancel(self, client_id: str, *, owned_only: bool = True) -> bool: ...

    def cancel_replace(self, client_id: str, new_order: Order) -> str | None: ...

    def order_status(self, client_id: str) -> str: ...

    def emergency_close_all(self, *, confirm: bool) -> int: ...

    def snapshot(self) -> dict[str, object]: ...


class SimulatedVenue:
    """Deterministic paper venue (Section 18)."""

    def __init__(self, cfg_meta: MetadataConfig) -> None:
        self.meta = cfg_meta
        self.open_orders: dict[str, Order] = {}
        self.positions: dict[str, VenuePosition] = {}
        self.fills: list[Fill] = []
        self.cancelled: set[str] = set()
        self._replace_counter = 0

    # -- fees ------------------------------------------------------------ #
    def _fee(self, symbol: str, notional: float, *, maker: bool) -> float:
        spec = self.meta.spec(symbol)
        if spec is None:
            return 0.0
        key = "maker_fee" if maker else "taker_fee"
        rate = spec.fields.get(key, 0.0)
        return notional * float(rate if isinstance(rate, (int, float)) else 0.0)

    # -- placement ------------------------------------------------------- #
    def place_bracket(
        self,
        plan: OrderPlan,
        *,
        ref_price: float,
        realized_slippage_frac: float,
        latency_ms: float,
        spread_bps: float = 0.0,
        signal_age_ms: float = 0.0,
        fill_ratio: float = 1.0,
    ) -> BracketResult:
        """Fill the entry and ATOMICALLY register the stop + TP/trailing legs.

        The position is never created without its exchange-resident protection
        (Section 2.2): the stop (and TP or native trailing) are registered in the
        same call that opens the position.
        """
        entry = plan.entry
        maker = entry.order_type in (OrderType.POST_ONLY, OrderType.LIMIT)
        # Maker rests at its limit price (no slippage); taker pays adverse slippage.
        if maker:
            actual_price = float(entry.price) if entry.price is not None else ref_price
            slip = 0.0
        else:
            direction = 1.0 if entry.side == BUY else -1.0
            actual_price = ref_price * (1.0 + direction * realized_slippage_frac)
            slip = realized_slippage_frac

        filled_qty = entry.qty * max(0.0, min(1.0, fill_ratio))
        notional = filled_qty * actual_price
        fee = self._fee(plan.symbol, notional, maker=maker)
        # Realised slippage is measured against the reference price the bot
        # expected to fill at (Section 18 "expected price" vs "actual price").
        expected_price = float(entry.price) if entry.price is not None else ref_price
        fill = Fill(
            client_id=entry.client_id,
            symbol=plan.symbol,
            side=entry.side,
            qty=filled_qty,
            expected_price=expected_price,
            actual_price=actual_price,
            fee=fee,
            maker=maker,
            latency_ms=latency_ms,
            slippage_frac=slip,
            slippage_cost=abs(actual_price - ref_price) * filled_qty,
            spread_bps_at_order=spread_bps,
            signal_age_ms=signal_age_ms,
            order_type=entry.order_type.value,
        )
        self.fills.append(fill)

        position = VenuePosition(
            symbol=plan.symbol,
            side=plan.side,
            qty=filled_qty,
            entry_price=actual_price,
            owned=True,
        )
        resting: list[str] = []
        # Only a real (non-zero) fill creates a position and its atomic protection legs.
        # A zero fill must NOT register a qty-0 phantom position in the book (it would
        # pollute reconciliation/heat) nor rest reduce-only legs for a position that
        # doesn't exist — the entry simply rests below as a working order.
        if filled_qty > 0:
            if plan.stop is not None:
                self.open_orders[plan.stop.client_id] = plan.stop
                position.stop_order_id = plan.stop.client_id
                resting.append(plan.stop.client_id)
            if plan.take_profit is not None:
                self.open_orders[plan.take_profit.client_id] = plan.take_profit
                position.tp_order_id = plan.take_profit.client_id
                resting.append(plan.take_profit.client_id)
            if plan.trailing is not None:
                self.open_orders[plan.trailing.client_id] = plan.trailing
                position.trail_order_id = plan.trailing.client_id
                resting.append(plan.trailing.client_id)
            self.positions[plan.symbol] = position

        fully = fill_ratio >= 1.0
        remaining = entry.qty - filled_qty
        if not fully:
            # The unfilled remainder rests as a working entry order.
            self.open_orders[entry.client_id] = entry
            resting.append(entry.client_id)
        return BracketResult(
            fill=fill,
            position=position,
            resting_order_ids=resting,
            fully_filled=fully,
            remaining_qty=remaining,
        )

    # -- order management ------------------------------------------------ #
    def order_status(self, client_id: str) -> str:
        if client_id in self.open_orders:
            return "open"
        if client_id in self.cancelled:
            return "cancelled"
        return "unknown"

    def cancel(self, client_id: str, *, owned_only: bool = True) -> bool:
        """Cancel a resting order. By default only the bot's own orders (Section 7)."""
        order = self.open_orders.get(client_id)
        if order is None:
            return False
        if owned_only and not _is_owned(order):
            return False
        del self.open_orders[client_id]
        self.cancelled.add(client_id)
        return True

    def cancel_replace(self, client_id: str, new_order: Order) -> str | None:
        """Atomically cancel ``client_id`` and place ``new_order`` (Section 18)."""
        if not self.cancel(client_id):
            return None
        self.open_orders[new_order.client_id] = new_order
        self._replace_counter += 1
        return new_order.client_id

    # -- foreign state injection (ORDER-OWN tests) ----------------------- #
    def inject_foreign_order(self, order: Order) -> None:
        """Inject an order with no ownership prefix (a manual / other-bot order)."""
        self.open_orders[order.client_id] = order

    def inject_foreign_position(self, symbol: str, side: int, qty: float, price: float) -> None:
        self.positions[symbol] = VenuePosition(
            symbol=symbol, side=side, qty=qty, entry_price=price, owned=False
        )

    # -- emergency close (Section 7/35) ---------------------------------- #
    def emergency_close_all(self, *, confirm: bool) -> int:
        """Close all positions + cancel all orders. REQUIRES explicit confirmation."""
        if not confirm:
            raise PermissionError("emergency_close_all requires explicit confirmation (Section 7)")
        n = len(self.positions) + len(self.open_orders)
        self.positions.clear()
        self.open_orders.clear()
        return n

    # -- reconciliation snapshot ----------------------------------------- #
    def snapshot(self) -> dict[str, object]:
        return {
            "orders": dict(self.open_orders),
            "positions": dict(self.positions),
        }


def _is_owned(order: Order) -> bool:
    # An order placed through the OrderBuilder carries provenance tags; a foreign
    # order injected for testing carries none.
    return bool(order.tags.get("bot_instance_id"))


__all__ = ["Fill", "VenuePosition", "BracketResult", "SimulatedVenue", "Venue", "BUY", "SELL"]
