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

import time
from typing import Any, Protocol

import structlog

from src.config import Settings, get_settings
from src.exchange.metadata import MetadataConfig
from src.execution.config import load_execution_config
from src.execution.order import BUY, Order, OrderPlan, OrderType
from src.execution.ownership import MAX_CLIENT_ID_LEN
from src.execution.venue import BracketResult, Fill, Venue, VenuePosition

_log = structlog.get_logger("execution.live_venue")

# Order types that rest as maker (no taker slippage).
_MAKER_TYPES = (OrderType.POST_ONLY, OrderType.LIMIT)

# Protective (conditional trigger) order types — must NEVER be sent as plain
# market/limit orders (they would execute immediately and flatten the position).
_TRIGGER_TYPES = (OrderType.STOP_MARKET, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_MARKET)

# ccxt order statuses after which an order can no longer fill.
_TERMINAL_STATUSES = frozenset({"closed", "canceled", "cancelled", "rejected", "expired"})

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
    elif exchange_env == "testnet":
        # Must explicitly switch to the sandbox. If the ccxt build lacks sandbox support we RAISE
        # rather than silently fall through to the LIVE mainnet endpoints with testnet keys.
        if not hasattr(ex, "set_sandbox_mode"):
            raise ValueError(
                "this ccxt build has no sandbox support; cannot route testnet safely "
                "(it would otherwise hit the live mainnet endpoints)"
            )
        ex.set_sandbox_mode(True)  # testnet — no real funds
    # live → mainnet, no switch.


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
        fill_timeout_s: float | None = None,
        fill_poll_interval_s: float | None = None,
    ) -> None:
        self.meta = meta
        self.settings = settings or get_settings()
        self.exchange_id = self.settings.exchange_id
        self.exchange_env = self.settings.exchange_env  # "live" | "testnet" | "demo"
        self._guard = guard
        # Entry-fill observation window (Section 18 / execution.yaml): how long to poll
        # the exchange order status before cancelling an unfilled remainder.
        ecfg = load_execution_config()
        self.fill_timeout_s = (
            float(fill_timeout_s) if fill_timeout_s is not None else ecfg.entry_fill_timeout_s
        )
        self.fill_poll_interval_s = (
            float(fill_poll_interval_s)
            if fill_poll_interval_s is not None
            else ecfg.entry_fill_poll_interval_s
        )
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
        trade live.

        The entry fill is OBSERVED via order-status polling (see
        :meth:`_observe_entry_fill`), never assumed from the create response. A partial
        fill books only the filled qty — the attached stopLoss/takeProfit are
        position-level on Bybit, so once the remainder is cancelled the protective legs
        cover exactly the filled position. A zero fill returns a clean no-position
        result (qty-0, no protection markers) and registers nothing in the mirror."""
        if self.is_live:
            allowed, reason = self._authorise(plan)
            if not allowed:
                raise PermissionError(f"live order refused by activation guard: {reason}")

        self._ensure_tradable_metadata(plan.symbol)

        entry = plan.entry
        post_only = entry.order_type is OrderType.POST_ONLY
        maker_intent = entry.order_type in _MAKER_TYPES
        order_type = "limit" if maker_intent else "market"
        price = float(entry.price) if entry.price is not None else None

        params: dict[str, Any] = {"clientOrderId": entry.client_id}
        if post_only:
            # H8: maker intent is EXCHANGE-ENFORCED — the real ccxt-unified post-only flag,
            # so a limit that would cross the book is rejected by Bybit instead of silently
            # filling as taker while being recorded maker with zero slippage.
            params["postOnly"] = True
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

        filled_qty = 0.0
        avg_obs: float | None = None
        fee = 0.0
        entry_still_resting = False
        try:
            resp = self._ex.create_order(
                plan.symbol, order_type, entry.side, entry.qty, price, params
            )
        except Exception as exc:
            if not (post_only and _is_post_only_rejection(exc)):
                raise
            # The post-only entry would have crossed the book → the exchange rejected it.
            # That is a clean NO-FILL (maker_first semantics), or the escalation trigger
            # for passive_then_taker below — never a silent taker fill.
            _log.info(
                "post_only_rejected_for_crossing",
                symbol=plan.symbol, client_id=entry.client_id, error=str(exc),
            )
        else:
            # OBSERVE the fill (never assume it): Bybit's create response omits fill fields
            # for a resting order, so we poll the order status and book only what actually
            # filled; any unfilled remainder is cancelled at the window end. ``fill_ratio``
            # is a simulated-venue knob and is deliberately ignored here.
            filled_qty, avg_obs, fee, entry_still_resting = self._observe_entry_fill(
                resp, plan.symbol, entry
            )

        # passive_then_taker (Section 18): the passive leg was rejected for crossing or is
        # (partially) unfilled at the observation-window end (remainder already cancelled +
        # verified) → escalate the remaining qty ONCE to a taker market order. maker_first
        # never escalates. An unconfirmable remainder-cancel blocks escalation (the passive
        # order may still fill — escalating could double the position).
        taker_qty = 0.0
        taker_avg: float | None = None
        if (
            post_only
            and plan.entry_style == "passive_then_taker"
            and not entry_still_resting
            and filled_qty < entry.qty - 1e-12
        ):
            taker_qty, taker_avg, taker_fee = self._escalate_entry_taker(
                plan, entry, entry.qty - filled_qty, params
            )
            fee += taker_fee

        total_qty = filled_qty + taker_qty
        # Average fill price is the observed one; fall back to the limit price only when
        # the venue reported a filled qty but no average (then the reference price).
        maker_px = avg_obs if avg_obs else (price if price is not None else ref_price)
        if taker_qty > 0:
            taker_px = taker_avg if taker_avg else ref_price
            avg = (maker_px * filled_qty + taker_px * taker_qty) / total_qty
        else:
            avg = maker_px
        expected = price if price is not None else ref_price
        # H8: execution-quality fields come from the OBSERVED fill, never assumed. With
        # post-only exchange-enforced, a resting fill IS maker; any taker escalation makes
        # the fill non-maker. Slippage is measured actual-vs-expected, not modelled.
        maker = maker_intent and taker_qty <= 0.0
        slip_frac = abs(avg - expected) / expected if (total_qty > 0 and expected > 0) else 0.0
        fill = Fill(
            client_id=entry.client_id,
            symbol=plan.symbol,
            side=entry.side,
            qty=total_qty,
            expected_price=expected,
            actual_price=avg,
            fee=fee,
            maker=maker,
            latency_ms=latency_ms,
            slippage_frac=slip_frac,
            slippage_cost=abs(avg - ref_price) * total_qty,
            spread_bps_at_order=spread_bps,
            signal_age_ms=signal_age_ms,
            order_type="market" if (taker_qty > 0 and filled_qty <= 0) else entry.order_type.value,
        )
        self.fills.append(fill)
        filled_qty = total_qty

        # ADVISORY post-fill stop check (Section 2.2). A market fill's attached SL commonly hasn't
        # propagated to the position read yet, so an immediate "no stop" is NOT proof of rejection
        # — downgrading here would false-alarm and refuse healthy orders. So we keep the optimistic
        # marker and only LOG when not positively confirmed; the per-tick reconciliation (which
        # re-reads with time elapsed) makes the authoritative unprotected determination + alert. A
        # zero-fill carries no marker.
        protected = self._confirm_exchange_stop(plan.symbol) if (filled_qty > 0) else None
        has_stop = filled_qty > 0 and plan.stop is not None
        has_tp = filled_qty > 0 and plan.take_profit is not None
        if filled_qty > 0 and plan.stop is not None and protected is False:
            _log.warning("stop_not_yet_confirmed_at_placement", symbol=plan.symbol)
        position = VenuePosition(
            symbol=plan.symbol,
            side=plan.side,
            qty=filled_qty,
            entry_price=avg,
            stop_order_id=f"{entry.client_id}:sl" if has_stop else None,
            tp_order_id=f"{entry.client_id}:tp" if has_tp else None,
            owned=True,
        )
        resting: list[str] = []
        if filled_qty > 0:
            self.positions[plan.symbol] = position
        fully = filled_qty >= entry.qty - 1e-12
        # The unfilled remainder was cancelled at the observation-window end; the entry
        # is only kept as a tracked resting order if that cancel could NOT be confirmed
        # (so reconciliation still sees it — never silently dropped).
        if not fully and entry_still_resting:
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

    def _observe_entry_fill(
        self, resp: dict, symbol: str, entry: Order
    ) -> tuple[float, float | None, float, bool]:
        """Observe the entry fill on the exchange — never assume it (Section 18).

        Returns ``(filled_qty, avg_price, fee, entry_still_resting)``. The create
        response is trusted only when it positively reports a terminal fill (a full
        fill, or a terminal status with the filled qty present). Otherwise the order
        status is polled (light backoff) until it goes terminal or the configured
        window elapses; at window end the unfilled remainder is CANCELLED and the
        cancel is VERIFIED with a final fetch — a race-fill discovered there is still
        booked. ``entry_still_resting`` is True only when the cancel could not be
        confirmed, so the caller keeps tracking the order instead of assuming it died.
        """
        qty = float(entry.qty)
        filled = _num(resp.get("filled"))
        avg = _num(resp.get("average")) or _num(resp.get("price"))
        fee = _num((resp.get("fee") or {}).get("cost"))
        status = str(resp.get("status") or "").lower()
        if filled is not None and (filled >= qty - 1e-12 or status in _TERMINAL_STATUSES):
            return min(filled, qty), avg, fee or 0.0, False

        order_id = str(resp.get("id") or "") or entry.client_id
        fetch_params = {"clientOrderId": entry.client_id}

        def _absorb(o: dict) -> None:
            nonlocal filled, avg, fee, status
            f = _num(o.get("filled"))
            if f is not None:
                filled = f
            avg = _num(o.get("average")) or avg
            fee = _num((o.get("fee") or {}).get("cost")) or fee
            status = str(o.get("status") or status or "").lower()

        deadline = time.monotonic() + max(0.0, self.fill_timeout_s)
        interval = max(0.01, self.fill_poll_interval_s)
        while True:
            try:
                _absorb(self._ex.fetch_order(order_id, symbol, fetch_params) or {})
            except Exception as exc:  # noqa: BLE001 - a poll hiccup: keep polling to the window end
                _log.warning("entry_fill_poll_failed", symbol=symbol, error=str(exc))
            if status in _TERMINAL_STATUSES or (filled is not None and filled >= qty - 1e-12):
                return min(filled or 0.0, qty), avg, fee or 0.0, False
            if time.monotonic() >= deadline:
                break
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
            interval = min(interval * 1.5, 1.0)

        # Window elapsed with the order still working: cancel the remainder, then VERIFY.
        cancel_error: str | None = None
        try:
            self._ex.cancel_order(order_id, symbol, fetch_params)
        except Exception as exc:  # noqa: BLE001 - verified below; never assume the cancel worked
            cancel_error = str(exc)
        try:
            _absorb(self._ex.fetch_order(order_id, symbol, fetch_params) or {})
        except Exception as exc:  # noqa: BLE001
            _log.warning("entry_cancel_verify_failed", symbol=symbol, error=str(exc))
        still_resting = status not in _TERMINAL_STATUSES
        if still_resting:
            _log.error(
                "entry_cancel_unconfirmed",
                symbol=symbol,
                client_id=entry.client_id,
                status=status or "unknown",
                error=cancel_error,
            )
        else:
            _log.info(
                "entry_remainder_cancelled",
                symbol=symbol,
                client_id=entry.client_id,
                filled=filled or 0.0,
                requested=qty,
            )
        return min(filled or 0.0, qty), avg, fee or 0.0, still_resting

    def _escalate_entry_taker(
        self, plan: OrderPlan, entry: Order, qty: float, base_params: dict[str, Any]
    ) -> tuple[float, float | None, float]:
        """One-shot taker escalation for ``passive_then_taker`` (Section 18 / H8).

        The passive post-only leg was rejected for crossing or unfilled at the observation
        window end (its remainder already cancelled + verified), so the remaining qty is sent
        as a market order carrying the same attached SL/TP/trailing params (position-level on
        Bybit, so the escalated fill is protected too) under a derived — still prefix-owned —
        clientOrderId, and its fill is OBSERVED like any entry. Returns ``(filled_qty,
        avg_price, fee)``; a failed escalation is a logged no-fill, never a crash (the engine
        records ``entry_unfilled``)."""
        esc_id = entry.client_id[: MAX_CLIENT_ID_LEN - 1] + "T"
        params = {k: v for k, v in base_params.items() if k != "postOnly"}
        params["clientOrderId"] = esc_id
        try:
            resp = self._ex.create_order(plan.symbol, "market", entry.side, qty, None, params)
        except Exception as exc:  # noqa: BLE001 - escalation failure = clean no-fill, not a crash
            _log.error("taker_escalation_failed", symbol=plan.symbol, error=str(exc))
            return 0.0, None, 0.0
        esc_order = Order(
            client_id=esc_id,
            symbol=entry.symbol,
            side=entry.side,
            qty=qty,
            order_type=OrderType.MARKET,
            role="entry",
            tags=dict(entry.tags),
        )
        filled, avg, fee, still_resting = self._observe_entry_fill(resp, plan.symbol, esc_order)
        if still_resting:
            # Cancel of the escalated remainder unconfirmed → keep it tracked (Section 7).
            self.open_orders[esc_id] = esc_order
        _log.info(
            "entry_escalated_to_taker",
            symbol=plan.symbol, requested=qty, filled=filled, client_id=esc_id,
        )
        return filled, avg, fee

    def _confirm_exchange_stop(self, symbol: str) -> bool | None:
        """Confirm the exchange-side stop right after placement (Section 2.2).

        Returns True if the exchange position carries a stop, False if the position is present
        but UNPROTECTED (the stop param was rejected), or None if the position isn't visible yet
        (fill latency) — in which case the caller keeps the optimistic marker and the per-tick
        reconciliation re-verifies. Best-effort: any fetch error returns None (don't false-fail)."""
        try:
            pos = self.fetch_exchange_positions().get(symbol)
        except Exception:  # noqa: BLE001 - a fetch hiccup must not fail the placement
            return None
        if pos is None:
            return None
        return pos.has_exchange_side_stop()

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
        """Cancel a resting order — VERIFIED, never assumed (M15).

        A cancel call that errors does NOT mean the order is gone: the status is re-fetched,
        and only a terminal status lets the order leave the mirror. An order the exchange
        still lists stays tracked (reconciliation keeps seeing it) and this returns False —
        the bot never mutates its book on an unconfirmed cancel."""
        order = self.open_orders.get(client_id)
        if order is None:
            return False
        if owned_only and not order.tags.get("bot_instance_id"):
            return False
        try:
            self._ex.cancel_order(client_id, order.symbol, {"clientOrderId": client_id})
        except Exception as exc:  # noqa: BLE001 - verify below; the order may already be gone
            if not self._order_is_terminal(client_id, order.symbol):
                _log.error(
                    "cancel_unconfirmed",
                    client_id=client_id, symbol=order.symbol, error=str(exc),
                )
                return False  # still live on the exchange → keep tracking it
        del self.open_orders[client_id]
        self.cancelled.add(client_id)
        return True

    def _order_is_terminal(self, client_id: str, symbol: str) -> bool:
        """Re-fetch an order's status after a failed cancel: True only when the exchange
        positively reports a terminal status (filled/cancelled/rejected/expired). Any fetch
        error → False (assume still live; never drop an order we can't account for)."""
        try:
            o = self._ex.fetch_order(client_id, symbol, {"clientOrderId": client_id}) or {}
        except Exception:  # noqa: BLE001 - can't verify → treat as still live
            return False
        return str(o.get("status") or "").lower() in _TERMINAL_STATUSES

    def cancel_replace(self, client_id: str, new_order: Order) -> str | None:
        if not self.cancel(client_id):
            return None
        self.place_order(new_order)
        return new_order.client_id

    def place_order(self, order: Order) -> None:
        """Place a single (non-bracket) order — used by cancel/replace of PROTECTIVE legs only.

        Refuses an ``entry`` order: an entry must go through :meth:`place_bracket` so its
        exchange-resident stop is attached atomically (Section 2.2). A bare entry placed here
        could fill with no protection.

        A protective leg (stop / take-profit) is sent as a proper CONDITIONAL order —
        trigger price via ccxt's unified ``stopLossPrice`` / ``takeProfitPrice`` params,
        always ``reduceOnly`` — never as a plain market/limit order, which would execute
        IMMEDIATELY at market and flatten the position (or open opposite exposure)."""
        if order.role == "entry":
            raise ValueError(
                "place_order cannot place an entry (no attached stop); use place_bracket so "
                "protection is attached atomically (Section 2.2)"
            )
        self._ensure_tradable_metadata(order.symbol)
        if order.order_type is OrderType.TRAILING_STOP:
            # There is no standalone trailing-stop order here: the trailing stop is
            # attached at entry (``trailingPercent`` in place_bracket) and lives on the
            # position. Degrading it to a market order would flatten the position.
            raise NotImplementedError(
                "standalone trailing-stop replacement is not supported on the live venue; "
                "the trailing stop is attached at entry and rides the position"
            )
        params: dict[str, Any] = {"clientOrderId": order.client_id}
        if order.order_type in _TRIGGER_TYPES:
            trigger = _leg_trigger(order)
            if trigger is None:
                raise ValueError(
                    f"protective leg {order.client_id} ({order.order_type.value}) has no "
                    "trigger price — refusing to send it as an immediate market order"
                )
            key = (
                "takeProfitPrice"
                if order.order_type is OrderType.TAKE_PROFIT_MARKET
                else "stopLossPrice"
            )
            params[key] = trigger
            # A protective leg must only ever REDUCE the position (Section 2.2).
            params["reduceOnly"] = True
            otype = "limit" if order.order_type is OrderType.STOP_LIMIT else "market"
            price = (
                float(order.price)
                if (order.order_type is OrderType.STOP_LIMIT and order.price is not None)
                else None
            )
        else:
            if order.reduce_only:
                params["reduceOnly"] = True
            otype = "limit" if order.order_type in _MAKER_TYPES else "market"
            price = float(order.price) if order.price is not None else None
        self._ex.create_order(order.symbol, otype, order.side, order.qty, price, params)
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

    def _fetch_balance(self) -> dict:
        try:
            return self._ex.fetch_balance() or {}
        except Exception:  # noqa: BLE001 - a balance hiccup must not crash the loop
            return {}

    def fetch_free_margin(self) -> float | None:
        """Free (available) margin in the quote currency, for the pre-trade free-margin blocker
        (Section 17). Returns None if the balance can't be read (the blocker then skips)."""
        bal = self._fetch_balance()
        quote = "USDT"
        free = (bal.get("free") or {}).get(quote)
        if free is None:
            free = ((bal.get(quote) or {}) if isinstance(bal.get(quote), dict) else {}).get("free")
        return _num(free)

    def fetch_account_equity(self) -> float | None:
        """Total account equity in the quote currency (the REAL account value). Fed to the circuit
        breakers in live/demo so the daily-loss + max-drawdown breakers reason over real equity —
        the engine's simulated exit path is inert when exits happen exchange-side, so without this
        a losing live session would never trip them. None if unreadable (breakers fall back)."""
        bal = self._fetch_balance()
        quote = "USDT"
        total = (bal.get("total") or {}).get(quote)
        if total is None:
            raw = bal.get(quote)
            total = raw.get("total") if isinstance(raw, dict) else None
        return _num(total)

    def fetch_exchange_positions(self) -> dict[str, VenuePosition]:
        """Live exchange positions (for reconciliation vs the bot's mirror, Section 7).

        Reads the REAL exchange-side protection (stopLoss / takeProfit / trailingStop) off each
        position so ``has_exchange_side_stop()`` reflects what the venue actually holds — an
        adopted or reconciled position with NO stop is correctly reported unprotected (Section 2.2),
        instead of trusting a self-minted marker."""
        out: dict[str, VenuePosition] = {}
        for p in self._ex.fetch_positions() or []:
            qty = _num(p.get("contracts")) or 0.0
            if qty <= 0:
                continue
            sym = str(p.get("symbol"))
            info = p.get("info") or {}
            cid = str(info.get("clientOrderId") or "")
            owned = cid.startswith(self.settings.order_client_id_prefix)
            sl = _num(p.get("stopLossPrice")) or _num(info.get("stopLoss"))
            tp = _num(p.get("takeProfitPrice")) or _num(info.get("takeProfit"))
            trail = _num(info.get("trailingStop"))
            out[sym] = VenuePosition(
                symbol=sym,
                side=1 if str(p.get("side")) == "long" else -1,
                qty=qty,
                entry_price=_num(p.get("entryPrice")) or 0.0,
                stop_order_id=f"{sym}:sl" if (sl and sl > 0) else None,
                tp_order_id=f"{sym}:tp" if (tp and tp > 0) else None,
                trail_order_id=f"{sym}:trail" if (trail and trail > 0) else None,
                owned=owned,
            )
        return out

    def fetch_exit_fill(
        self, symbol: str, side: int, *, since_ts: int | None = None
    ) -> tuple[float, float] | None:
        """Observed exit price + fee for a position that closed on the exchange (H7).

        Sources the account's OWN executions via ccxt ``fetch_my_trades`` (Bybit:
        /v5/execution/list) — the most reliable Bybit-compatible record of SL/TP/trailing
        and reduce-only market-close fills. Close-side executions since ``since_ts`` (the
        entry time) are aggregated to ``(vwap_price, total_fee)``; a position closed in
        several partial executions books its true volume-weighted exit. Returns None when
        no close-side execution is visible (or the fetch fails) — the caller falls back to
        a mark price with a loud log."""
        close_side = "sell" if side > 0 else "buy"
        try:
            trades = self._ex.fetch_my_trades(symbol, since_ts, 100) or []
        except Exception as exc:  # noqa: BLE001 - a history fetch error → let the caller fall back
            _log.warning("exit_fill_fetch_failed", symbol=symbol, error=str(exc))
            return None
        qty_sum = 0.0
        px_qty = 0.0
        fee_sum = 0.0
        for t in trades:
            if str(t.get("side") or "").lower() != close_side:
                continue
            amt = _num(t.get("amount")) or 0.0
            px = _num(t.get("price")) or 0.0
            if amt <= 0 or px <= 0:
                continue
            qty_sum += amt
            px_qty += px * amt
            fee_sum += _num((t.get("fee") or {}).get("cost")) or 0.0
        if qty_sum <= 0:
            return None
        return px_qty / qty_sum, fee_sum

    def close_position(self, symbol: str) -> bool:
        """Reduce-only market close of ONE owned position (bot-side time-stop) + cancel its legs.

        M15: a failed close is a FAILED close — the position stays tracked (the caller retries
        next tick) and the error is surfaced, instead of dropping the mirror as if flat while a
        real position keeps running on the exchange."""
        pos = self.positions.get(symbol)
        if pos is None:
            return False
        close_side = "sell" if pos.side > 0 else "buy"
        try:
            self._ex.create_order(symbol, "market", close_side, pos.qty, None, {"reduceOnly": True})
        except Exception as exc:  # noqa: BLE001 - surface + keep tracking; never fake success
            _log.error("close_position_failed", symbol=symbol, error=str(exc))
            return False
        for cid in [c for c, o in self.open_orders.items() if o.symbol == symbol]:
            # Verified cancel: an unconfirmed leg cancel keeps the order tracked (Section 7).
            self.cancel(cid, owned_only=False)
        self.positions.pop(symbol, None)
        return True

    def emergency_close_all(self, *, confirm: bool) -> int:
        """Flatten the REAL exchange book (M14): reduce-only close every exchange-listed
        position and cancel every resting order carrying our ownership prefix — fetched from
        the exchange, not the in-memory mirror, so it still works after a restart (when the
        mirror is empty but real positions remain). Mirror cleanup happens per item AFTER its
        exchange call succeeds; failures are logged and keep the item tracked. Returns the
        number of successful close/cancel actions."""
        if not confirm:
            raise PermissionError("emergency_close_all requires explicit confirmation (Section 7)")
        n = 0
        # Positions: the exchange book is authoritative; merge the mirror in case the fetch
        # fails (fall back to closing at least what we know about).
        targets: dict[str, VenuePosition] = dict(self.positions)
        try:
            targets.update(self.fetch_exchange_positions())
        except Exception:  # noqa: BLE001 - emergency close must still act on the mirror
            _log.error("emergency_close_fetch_positions_failed", exc_info=True)
        for sym, pos in targets.items():
            close_side = "sell" if pos.side > 0 else "buy"
            try:
                self._ex.create_order(
                    sym, "market", close_side, pos.qty, None, {"reduceOnly": True}
                )
            except Exception as exc:  # noqa: BLE001 - keep closing the rest
                _log.error("emergency_close_position_failed", symbol=sym, error=str(exc))
                continue
            self.positions.pop(sym, None)
            n += 1
        # Orders: cancel every OWNED (prefix-carrying) resting order on the exchange.
        prefix = self.settings.order_client_id_prefix
        order_targets: dict[str, str] = {cid: o.symbol for cid, o in self.open_orders.items()}
        try:
            for o in self._ex.fetch_open_orders() or []:
                info = o.get("info") or {}
                cid = str(o.get("clientOrderId") or info.get("clientOrderId") or "")
                if cid.startswith(prefix):
                    order_targets[cid] = str(o.get("symbol") or "")
        except Exception:  # noqa: BLE001 - fall back to cancelling the mirrored orders
            _log.error("emergency_close_fetch_orders_failed", exc_info=True)
        for cid, sym in order_targets.items():
            try:
                self._ex.cancel_order(cid, sym, {"clientOrderId": cid})
            except Exception as exc:  # noqa: BLE001 - keep cancelling the rest
                _log.error("emergency_close_cancel_failed", client_id=cid, error=str(exc))
                continue
            self.open_orders.pop(cid, None)
            self.cancelled.add(cid)
            n += 1
        return n

    def snapshot(self) -> dict[str, object]:
        return {"orders": dict(self.open_orders), "positions": dict(self.positions)}


def _num(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_post_only_rejection(exc: Exception) -> bool:
    """Whether an order-create error is the exchange rejecting a POST-ONLY order that would
    have crossed the book (taken liquidity). ccxt's typed exception is preferred; the message
    heuristics cover fakes/older builds and Bybit's raw wording. Anything else (auth, network,
    bad params) is NOT a crossing rejection and must propagate."""
    try:
        import ccxt

        if isinstance(exc, ccxt.OrderImmediatelyFillable):
            return True
    except Exception:  # noqa: BLE001 - ccxt absent/old → fall through to message heuristics
        pass
    msg = str(exc).lower()
    return (
        "post only" in msg
        or "post-only" in msg
        or "postonly" in msg
        or "immediately" in msg  # "order would be filled immediately" / ImmediatelyFillable
    )


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
