"""Real ccxt-backed execution venue — TESTNET by default (AGENTS.md Section 18).

The live counterpart of :class:`~src.execution.venue.SimulatedVenue`, behind the same
:class:`~src.execution.venue.Venue` Protocol. It places a bracket **atomically** —
the entry order carries its exchange-resident stop-loss (and take-profit) via ccxt's
unified ``stopLoss`` / ``takeProfit`` params — so a position is never opened without
protection (Section 2.2). Every order carries the bot's ownership prefix as
``clientOrderId`` (Section 7).

Safety, by construction:
* ``settings.exchange_env`` selects the Bybit environment — ``testnet`` (separate test network),
  ``demo`` (mainnet data + a virtual-funds demo account, ``api-demo.bybit.com``), or ``live``
  (real money). testnet and demo are different endpoints with different keys; only ``live`` is
  real money (``is_live``). Defaults to testnet; no real funds unless ``EXCHANGE_ENV=live``.
* Refuses to construct without API credentials (no anonymous trading).
* For a **live** (mainnet, real-money) environment it refuses to place any order unless
  an injected activation guard authorises it (wired by M8's LiveActivationGuard). Testnet
  needs no guard — it cannot move real money.

Tests inject a fake client, so nothing here needs the network or real keys.
"""

from __future__ import annotations

import contextlib
from typing import Any, Protocol

from src.config import Settings, get_settings
from src.exchange.metadata import MetadataConfig
from src.execution.order import BUY, Order, OrderPlan, OrderType
from src.execution.venue import BracketResult, Fill, Venue, VenuePosition

# Order types that rest as maker (no taker slippage).
_MAKER_TYPES = (OrderType.POST_ONLY, OrderType.LIMIT)

VALID_EXCHANGE_ENVS = ("live", "testnet", "demo")


def apply_exchange_env(ex: Any, exchange_env: str) -> None:
    """Point a ccxt client at the right Bybit environment (Section 6).

    Bybit has THREE distinct environments — they are not interchangeable (different
    endpoints, different API keys):

    * ``live``    — real-money mainnet (``api.bybit.com``); no change to the client.
    * ``testnet`` — the separate **test network** (``testnet.bybit.com``); virtual funds,
      its own keys. ``set_sandbox_mode(True)``.
    * ``demo``    — Bybit **demo trading**: mainnet market data + a virtual-funds demo
      account (``api-demo.bybit.com``); demo keys from the main site. ``enable_demo_trading``.
    """
    if exchange_env == "demo":
        if not hasattr(ex, "enable_demo_trading"):
            raise ValueError(
                "this ccxt build has no demo-trading support; upgrade ccxt or use testnet"
            )
        ex.enable_demo_trading(True)
    elif exchange_env != "live" and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)  # testnet — no real funds


class LiveOrderGuard(Protocol):
    """Authorises (or refuses) a real-money order. Injected by M8."""

    def allow_live_order(self, plan: OrderPlan) -> tuple[bool, str]: ...


class CcxtLiveVenue:
    """Real venue (default: Bybit testnet) behind the Venue Protocol."""

    def __init__(
        self,
        meta: MetadataConfig,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        guard: LiveOrderGuard | None = None,
    ) -> None:
        self.meta = meta
        self.settings = settings or get_settings()
        self.exchange_id = self.settings.exchange_id
        self.exchange_env = self.settings.exchange_env  # "live" | "testnet" | "demo"
        self._guard = guard
        self.open_orders: dict[str, Order] = {}
        self.positions: dict[str, VenuePosition] = {}
        self.fills: list[Fill] = []
        self.cancelled: set[str] = set()

        if client is not None:
            self._ex = client
        else:
            if not (self.settings.exchange_api_key and self.settings.exchange_api_secret):
                raise ValueError(
                    "CcxtLiveVenue requires EXCHANGE_API_KEY/SECRET (no anonymous trading)"
                )
            import ccxt

            klass = getattr(ccxt, self.exchange_id)
            self._ex = klass(
                {
                    "apiKey": self.settings.exchange_api_key,
                    "secret": self.settings.exchange_api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap"},
                }
            )
            apply_exchange_env(self._ex, self.exchange_env)

    @property
    def is_live(self) -> bool:
        """Real-money mainnet — NOT testnet and NOT demo (both use virtual funds)."""
        return self.exchange_env == "live"

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
        """Place the entry with its stop (+TP) attached atomically, then mirror state.

        On a real-money venue the activation guard must authorise the order first; a
        denial raises (the engine surfaces it as a non-placement) — we never silently
        trade live."""
        if self.is_live:
            allowed, reason = self._authorise(plan)
            if not allowed:
                raise PermissionError(f"live order refused by activation guard: {reason}")

        self._ensure_tradable_metadata(plan.symbol)

        entry = plan.entry
        maker = entry.order_type in _MAKER_TYPES
        order_type = "limit" if maker else "market"
        price = float(entry.price) if entry.price is not None else None

        params: dict[str, Any] = {"clientOrderId": entry.client_id}
        # Atomic exchange-resident protection attached to the entry (Section 2.2).
        sl_trigger = _leg_trigger(plan.stop)
        if sl_trigger is not None:
            params["stopLoss"] = {"triggerPrice": sl_trigger, "type": "market"}
        tp_trigger = _leg_trigger(plan.take_profit)
        if tp_trigger is not None:
            params["takeProfit"] = {"triggerPrice": tp_trigger, "type": "market"}
        # Momentum / no-fixed-TP exits arm an exchange-native trailing stop instead of a
        # fixed TP. The initial stop above is always present, so the position is never
        # unprotected even if the venue ignores the trailing param.
        if plan.trailing is not None and plan.trailing.trail_offset:
            params["trailingPercent"] = float(plan.trailing.trail_offset) * 100.0

        resp = self._ex.create_order(plan.symbol, order_type, entry.side, entry.qty, price, params)

        avg = _num(resp.get("average")) or _num(resp.get("price")) or ref_price
        filled = _num(resp.get("filled"))
        filled_qty = filled if filled is not None else entry.qty * max(0.0, min(1.0, fill_ratio))
        fee = _num((resp.get("fee") or {}).get("cost")) or 0.0
        expected = price if price is not None else ref_price
        slip_frac = 0.0 if maker else realized_slippage_frac
        fill = Fill(
            client_id=entry.client_id,
            symbol=plan.symbol,
            side=entry.side,
            qty=filled_qty,
            expected_price=expected,
            actual_price=avg,
            fee=fee,
            maker=maker,
            latency_ms=latency_ms,
            slippage_frac=slip_frac,
            slippage_cost=abs(avg - ref_price) * filled_qty,
            spread_bps_at_order=spread_bps,
            signal_age_ms=signal_age_ms,
            order_type=entry.order_type.value,
        )
        self.fills.append(fill)

        # The stop is attached to the position on the exchange; record a marker id so
        # has_exchange_side_stop() is True (the Section 2.2 invariant the engine checks).
        position = VenuePosition(
            symbol=plan.symbol,
            side=plan.side,
            qty=filled_qty,
            entry_price=avg,
            # Only a FILLED position carries an exchange-side stop/TP marker. A zero-fill (e.g. a
            # resting maker entry) must NOT report has_exchange_side_stop()==True, or the engine
            # would treat an unfilled order as an executed, protected position (Section 2.2).
            stop_order_id=(
                f"{entry.client_id}:sl" if (filled_qty > 0 and plan.stop is not None) else None
            ),
            tp_order_id=(
                f"{entry.client_id}:tp"
                if (filled_qty > 0 and plan.take_profit is not None)
                else None
            ),
            owned=True,
        )
        resting: list[str] = []
        if filled_qty > 0:
            self.positions[plan.symbol] = position
        fully = filled_qty >= entry.qty - 1e-12
        if not fully:
            self.open_orders[entry.client_id] = entry
            resting.append(entry.client_id)
        return BracketResult(
            fill=fill,
            position=position,
            resting_order_ids=resting,
            fully_filled=fully,
            remaining_qty=max(0.0, entry.qty - filled_qty),
        )

    def _authorise(self, plan: OrderPlan) -> tuple[bool, str]:
        if self._guard is None:
            return False, "no activation guard configured for a live venue"
        return self._guard.allow_live_order(plan)

    def _ensure_tradable_metadata(self, symbol: str) -> None:
        """Refuse to place an order without verified, exchange-matched metadata (Section 6).

        The last line of defence before an order leaves for the exchange: if the loaded
        metadata is for a different venue, is unverified (operator review pending), or the
        symbol's spec is missing/incomplete/contradictory, we BLOCK rather than size/route an
        order on a placeholder spec."""
        blocker = self.meta.tradable_blocker(symbol, exchange_id=self.exchange_id)
        if blocker is not None:
            raise PermissionError(f"order blocked — unverified exchange metadata: {blocker}")

    # -- order management ------------------------------------------------ #
    def order_status(self, client_id: str) -> str:
        if client_id in self.open_orders:
            return "open"
        if client_id in self.cancelled:
            return "cancelled"
        return "unknown"

    def cancel(self, client_id: str, *, owned_only: bool = True) -> bool:
        order = self.open_orders.get(client_id)
        if order is None:
            return False
        if owned_only and not order.tags.get("bot_instance_id"):
            return False
        with contextlib.suppress(Exception):  # already gone / race; treat as cancelled
            self._ex.cancel_order(client_id, order.symbol, {"clientOrderId": client_id})
        del self.open_orders[client_id]
        self.cancelled.add(client_id)
        return True

    def cancel_replace(self, client_id: str, new_order: Order) -> str | None:
        if not self.cancel(client_id):
            return None
        self.place_order(new_order)
        return new_order.client_id

    def place_order(self, order: Order) -> None:
        """Place a single (non-bracket) order — used by cancel/replace."""
        self._ensure_tradable_metadata(order.symbol)
        otype = "limit" if order.order_type in _MAKER_TYPES else "market"
        price = float(order.price) if order.price is not None else None
        self._ex.create_order(
            order.symbol, otype, order.side, order.qty, price, {"clientOrderId": order.client_id}
        )
        self.open_orders[order.client_id] = order

    # -- reconciliation -------------------------------------------------- #
    def fetch_open_orders(self) -> dict[str, Order]:
        """Live resting orders on the exchange, keyed by clientOrderId (for startup
        reconciliation vs the bot's mirror, Section 7). Orders whose clientOrderId lacks
        the bot's ownership prefix are foreign/manual and must halt new entries."""
        out: dict[str, Order] = {}
        for o in self._ex.fetch_open_orders() or []:
            info = o.get("info") or {}
            cid = str(o.get("clientOrderId") or info.get("clientOrderId") or o.get("id") or "")
            if not cid:
                continue
            side = str(o.get("side") or "").lower()
            out[cid] = Order(
                client_id=cid,
                symbol=str(o.get("symbol") or ""),
                side="buy" if side == "buy" else "sell",
                qty=_num(o.get("amount")) or 0.0,
                order_type=OrderType.LIMIT,
                price=_num(o.get("price")),
                # Only orders carrying our prefix are tagged as ours; the reconciler keys
                # ownership off is_own(client_id), so an empty tag dict here is deliberate.
                tags={"bot_instance_id": self.settings.bot_instance_id}
                if cid.startswith(self.settings.order_client_id_prefix)
                else {},
            )
        return out

    def fetch_exchange_positions(self) -> dict[str, VenuePosition]:
        """Live exchange positions (for reconciliation vs the bot's mirror, Section 7)."""
        out: dict[str, VenuePosition] = {}
        for p in self._ex.fetch_positions() or []:
            qty = _num(p.get("contracts")) or 0.0
            if qty <= 0:
                continue
            sym = str(p.get("symbol"))
            cid = str((p.get("info") or {}).get("clientOrderId") or "")
            out[sym] = VenuePosition(
                symbol=sym,
                side=1 if str(p.get("side")) == "long" else -1,
                qty=qty,
                entry_price=_num(p.get("entryPrice")) or 0.0,
                owned=cid.startswith(self.settings.order_client_id_prefix),
            )
        return out

    def emergency_close_all(self, *, confirm: bool) -> int:
        if not confirm:
            raise PermissionError("emergency_close_all requires explicit confirmation (Section 7)")
        n = 0
        for sym, pos in list(self.positions.items()):
            close_side = "sell" if pos.side > 0 else "buy"
            with contextlib.suppress(Exception):
                self._ex.create_order(
                    sym, "market", close_side, pos.qty, None, {"reduceOnly": True}
                )
            n += 1
        for cid, order in list(self.open_orders.items()):
            with contextlib.suppress(Exception):
                self._ex.cancel_order(cid, order.symbol, {"clientOrderId": cid})
            n += 1
        self.positions.clear()
        self.open_orders.clear()
        return n

    def snapshot(self) -> dict[str, object]:
        return {"orders": dict(self.open_orders), "positions": dict(self.positions)}


def _num(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _leg_trigger(leg: Order | None) -> float | None:
    """Trigger price for a protective leg. Prefers ``stop_price`` (the trigger field) but
    falls back to ``price`` so a take-profit built with only a target price still attaches —
    a missing trigger would silently drop exchange-side protection (Section 2.2)."""
    if leg is None:
        return None
    trigger = leg.stop_price if leg.stop_price is not None else leg.price
    return float(trigger) if trigger is not None else None


def get_venue(
    meta: MetadataConfig,
    settings: Settings | None = None,
    *,
    live: bool = False,
    client: Any | None = None,
    guard: LiveOrderGuard | None = None,
) -> Venue:
    """Return the execution venue. Default is the offline SimulatedVenue (paper).

    ``live=True`` opts into the real ccxt venue (testnet by default via
    ``settings.exchange_env``); this is an explicit choice by the live loop, never the
    default. Real-money (mainnet) placement additionally requires the activation guard.
    """
    from src.execution.venue import SimulatedVenue

    settings = settings or get_settings()
    if not live:
        return SimulatedVenue(meta)
    return CcxtLiveVenue(meta, settings, client=client, guard=guard)


__all__ = ["CcxtLiveVenue", "LiveOrderGuard", "get_venue", "BUY"]
