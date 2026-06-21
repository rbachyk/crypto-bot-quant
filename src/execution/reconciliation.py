"""Startup / periodic reconciliation (AGENTS.md Section 7 & 17).

On every start (and periodically) the bot reconciles all positions and orders with
the exchange. Any order or position not confidently attributable to this instance
— i.e. lacking the ownership prefix, or present on the venue but absent from the
bot's own book — means the bot must **halt new entries and alert** (Section 7).
The bot must never touch foreign orders except in explicit emergency-close mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

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


class _ExchangeBook(Protocol):
    """The slice of a venue used for startup reconciliation (CcxtLiveVenue implements it)."""

    def fetch_open_orders(self) -> dict[str, Order]: ...
    def fetch_exchange_positions(self) -> dict[str, VenuePosition]: ...


@dataclass(frozen=True, slots=True)
class StartupReconResult:
    """The outcome of reconciling the REAL exchange book against this bot at startup."""

    ok: bool
    halt_required: bool
    owned_orders: tuple[str, ...] = field(default_factory=tuple)
    owned_positions: tuple[str, ...] = field(default_factory=tuple)
    foreign_orders: tuple[str, ...] = field(default_factory=tuple)
    foreign_positions: tuple[str, ...] = field(default_factory=tuple)
    environment: str = "local"
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "halt_required": self.halt_required,
            "owned_orders": list(self.owned_orders),
            "owned_positions": list(self.owned_positions),
            "foreign_orders": list(self.foreign_orders),
            "foreign_positions": list(self.foreign_positions),
            "environment": self.environment,
            "detail": self.detail,
        }

    def report(self) -> str:
        """Human-readable startup reconciliation report (for the dashboard / logs)."""
        verdict = "HALT" if self.halt_required else "OK"
        lines = [
            f"Startup reconciliation [{self.environment}] — {verdict}",
            f"  owned orders adopted     : {len(self.owned_orders)} {list(self.owned_orders)}",
            f"  owned positions adopted  : {len(self.owned_positions)} "
            f"{list(self.owned_positions)}",
            f"  FOREIGN orders           : {len(self.foreign_orders)} {list(self.foreign_orders)}",
            f"  FOREIGN positions        : {len(self.foreign_positions)} "
            f"{list(self.foreign_positions)}",
        ]
        if self.halt_required:
            lines.append(
                "  → new entries HALTED: foreign/manual order or position present. Do not "
                "trade until the book is clean or the foreign item is audited (Section 7)."
            )
        return "\n".join(lines)


def reconcile_startup(
    venue: Any,
    ownership: OwnershipPolicy,
    *,
    environment: str = "local",
    alert_sink: AlertSink | None = None,
    adopt: bool = True,
) -> StartupReconResult:
    """Reconcile the REAL exchange book against this bot at startup (Section 7).

    Pulls live open orders + positions from the exchange (via the venue), classifies each as
    owned (carries our prefix) or foreign/manual, and **halts new entries** if any foreign
    item exists. Owned items are adopted into the venue's mirror so the per-tick risk/recon
    checks see real exposure. A venue without the fetch hooks (the offline SimulatedVenue used
    by paper) is a no-op clean book — there is no real exchange to reconcile against.
    """
    if not hasattr(venue, "fetch_open_orders") or not hasattr(venue, "fetch_exchange_positions"):
        return StartupReconResult(
            ok=True, halt_required=False, environment=environment,
            detail="no real exchange book (offline venue) — nothing to reconcile",
        )

    exch_orders: dict[str, Order] = venue.fetch_open_orders()
    exch_positions: dict[str, VenuePosition] = venue.fetch_exchange_positions()

    owned_orders = sorted(oid for oid, o in exch_orders.items() if ownership.is_own(o.client_id))
    foreign_orders = sorted(oid for oid in exch_orders if oid not in set(owned_orders))
    owned_positions = sorted(sym for sym, p in exch_positions.items() if p.owned)
    foreign_positions = sorted(sym for sym in exch_positions if sym not in set(owned_positions))

    # Adopt owned items into the venue mirror so risk/recon see the real open exposure.
    if adopt:
        for oid in owned_orders:
            venue.open_orders[oid] = exch_orders[oid]
        for sym in owned_positions:
            venue.positions[sym] = exch_positions[sym]

    halt = bool(foreign_orders or foreign_positions)
    detail = (
        f"owned_orders={owned_orders} owned_positions={owned_positions} "
        f"foreign_orders={foreign_orders} foreign_positions={foreign_positions}"
    )
    if halt:
        (alert_sink or get_alert_sink()).send(
            Alert(
                title="startup reconciliation: foreign order/position on exchange",
                severity=AlertSeverity.CRITICAL,
                component="execution",
                environment=environment,
                recommended_action=(
                    f"Halt new entries; a foreign/manual order or position is present "
                    f"({detail}); never touch it except via audited emergency-close (Section 7)."
                ),
            )
        )
    return StartupReconResult(
        ok=not halt,
        halt_required=halt,
        owned_orders=tuple(owned_orders),
        owned_positions=tuple(owned_positions),
        foreign_orders=tuple(foreign_orders),
        foreign_positions=tuple(foreign_positions),
        environment=environment,
        detail=detail,
    )
