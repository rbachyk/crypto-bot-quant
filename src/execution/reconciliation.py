"""Startup / periodic reconciliation (AGENTS.md Section 7 & 17).

On every start (and periodically) the bot reconciles all positions and orders with
the exchange. Any order or position not confidently attributable to this instance
— i.e. lacking the ownership prefix, or present on the venue but absent from the
bot's own book — means the bot must **halt new entries and alert** (Section 7).
The bot must never touch foreign orders except in explicit emergency-close mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.execution.order import Order
from src.execution.ownership import OwnershipPolicy
from src.execution.venue import VenuePosition
from src.monitoring import Alert, AlertSeverity, AlertSink, get_alert_sink


@dataclass(frozen=True, slots=True)
class ReconResult:
    ok: bool
    halt_required: bool
    unknown_orders: tuple[str, ...] = field(default_factory=tuple)
    unknown_positions: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "halt_required": self.halt_required,
            "unknown_orders": list(self.unknown_orders),
            "unknown_positions": list(self.unknown_positions),
            "detail": self.detail,
        }


class Reconciler:
    """Detects foreign / unattributable orders and positions (Section 7)."""

    def __init__(self, ownership: OwnershipPolicy, alert_sink: AlertSink | None = None) -> None:
        self.ownership = ownership
        self.alert_sink = alert_sink or get_alert_sink()

    def reconcile(
        self,
        orders: dict[str, Order],
        positions: dict[str, VenuePosition],
        *,
        known_order_ids: set[str] | None = None,
        known_position_symbols: set[str] | None = None,
        environment: str = "local",
    ) -> ReconResult:
        known_orders = known_order_ids if known_order_ids is not None else set()
        known_positions = known_position_symbols if known_position_symbols is not None else set()

        # An order is foreign if it lacks our prefix, OR is unknown to our book.
        unknown_orders = tuple(
            sorted(
                oid
                for oid, order in orders.items()
                if not self.ownership.is_own(order.client_id) or oid not in known_orders
            )
        )
        # A position is foreign if it is not owned, OR its symbol is unknown to us.
        unknown_positions = tuple(
            sorted(
                sym for sym, pos in positions.items() if not pos.owned or sym not in known_positions
            )
        )

        halt = bool(unknown_orders or unknown_positions)
        if halt:
            detail = (
                f"unknown_orders={list(unknown_orders)} unknown_positions={list(unknown_positions)}"
            )
            self.alert_sink.send(
                Alert(
                    title="reconciliation: unknown order/position detected",
                    severity=AlertSeverity.CRITICAL,
                    component="execution",
                    environment=environment,
                    recommended_action=(
                        f"Halt new entries; investigate foreign order/position ({detail}); never "
                        "touch it except via audited emergency-close (Section 7)."
                    ),
                )
            )
            return ReconResult(
                ok=False,
                halt_required=True,
                unknown_orders=unknown_orders,
                unknown_positions=unknown_positions,
                detail=detail,
            )
        return ReconResult(ok=True, halt_required=False, detail="all orders/positions attributable")
