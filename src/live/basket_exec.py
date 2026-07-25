"""Real-venue execution for cross-sectional (basket) strategies — the DEMO path.

Until now a basket (funding_carry / residual_momentum) could only paper-trade: the
:class:`~src.backtest.portfolio.CrossSectionalEngine` models every leg fill from the bar
(``_open_leg`` / ``_close_leg``), and the per-symbol live loop refuses to drive a
cross-sectional strategy at all (``src/paper/lake.py`` filters ``cross_sectional`` out).
This module closes that gap.

**The exchange nets, so the bot must too.** An exchange holds ONE position per symbol. Two
basket strategies that both trade BTC do not get a position each — their legs collapse into a
single net position. A per-strategy mirror that assumes otherwise is wrong from the first
overlap: it will size against a book it does not own, and a whole-symbol close by one strategy
silently flattens the other's leg. That is the "shared capital allocator / aggregate-exposure
layer" AGENTS Section 17 names as REQUIRED before several strategies share one account, and
:class:`NetPositionManager` is it. Every strategy states its own desired position; the manager
owns the aggregate, and translates each request into the exact position DELTA to send. One demo
account runs any number of baskets.

**Observed fills are authoritative.** The engine's modelled price is only the order's reference;
a leg is booked at the price/qty/fee the venue actually reported. A partial fill books the
partial (and the manager records the partial, so the aggregate stays truthful); a rejected or
unfilled leg opens no leg at all.

**Protection (Section 2.2) vs the hedge.** A basket leg deliberately has no protective stop
between rebalances — a stop would knife the hedge and leave the remaining legs naked, which is
exactly why the backtest holds them delta-neutral to the cadence. But an unprotected position on
a real venue is also what Section 2.2 forbids: if this process dies, the legs sit there with
nothing exchange-side to close them. The compromise is a **disaster stop**: a deliberately-wide
exchange-resident stop (``basket_disaster_stop_frac``, e.g. 25% adverse) that normal basket
behaviour never reaches, so the hedge is untouched in operation, but a disconnected bot cannot
leak an unbounded loss. It is set on the NET position (Bybit's stop is position-level, which is
the correct granularity here) and refreshed whenever the net grows. 0 disables it — a
deliberate, logged deviation.

DEMO/TESTNET ONLY by construction: nothing here enables live trading. The venue it is handed
refuses real-money placement unless the full activation chain holds (Section 35).
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

from src.config import Settings
from src.exchange.metadata import MetadataConfig
from src.execution.order import Order, OrderPlan, OrderType
from src.execution.ownership import OwnershipPolicy

_log = structlog.get_logger("live.basket_exec")

#: Positions smaller than this (in contracts) are treated as flat — floating-point residue from
#: repeated partial fills must not register as an open position forever.
_DUST = 1e-12


@dataclass(frozen=True, slots=True)
class LegFill:
    """What the venue actually did with one basket leg — the booked truth for that leg."""

    qty: float
    price: float
    fee: float
    maker: bool
    slippage_cost: float = 0.0


class BasketExecutor(Protocol):
    """The execution surface :class:`CrossSectionalEngine` calls when a leg must be REAL.

    Both methods return ``None`` to mean "nothing happened" — the engine then opens no leg
    (open) or keeps the leg and retries next cadence (close). Never fabricate a fill here: a
    booked leg that does not exist on the exchange is the one failure this path must not have.
    """

    def open_leg(
        self, *, symbol: str, side: int, qty: float, ref_price: float, ts: int
    ) -> LegFill | None: ...

    def close_leg(
        self, *, symbol: str, side: int, qty: float, ref_price: float, entry_ts: int
    ) -> LegFill | None: ...


@dataclass
class _Intent:
    """One strategy's desired signed position in one symbol (contracts; + long, − short)."""

    qty: float = 0.0


class NetPositionManager:
    """Owns the aggregate exchange position; turns per-strategy intents into position deltas.

    The invariant: ``sum of every strategy's desired qty in a symbol == the exchange position in
    that symbol``. Each :meth:`apply` sends exactly the delta needed to keep it, so two strategies
    on one account never fight over a shared position and never flatten each other.
    """

    def __init__(
        self,
        venue: Any,
        meta: MetadataConfig,
        settings: Settings,
        *,
        disaster_stop_frac: float = 0.25,
        on_event: Any | None = None,
        scope_symbols: Any | None = None,
    ) -> None:
        self.venue = venue
        self.meta = meta
        self.settings = settings
        self.ownership = OwnershipPolicy(settings)
        self.disaster_stop_frac = max(0.0, float(disaster_stop_frac))
        self.on_event = on_event
        # The symbols this session OWNS on the account (account partitioning). Reconcile, stray-
        # flatten and the session-end backstop act ONLY on these — a position in ANY other symbol
        # belongs to another of our strategies (e.g. basis_reversion on its reserved symbols) and
        # must never be touched. None = own the whole account (a solo basket session, the old
        # exclusive behaviour). This is the guarantee that makes co-existence on one account safe.
        self.scope_symbols: set[str] | None = (
            set(scope_symbols) if scope_symbols is not None else None
        )
        self._desired: dict[tuple[str, str], _Intent] = {}
        # Execution-quality record: every real fill, so a session can compare what the demo venue
        # actually did against what the backtest cost model assumed.
        self.fills: list[dict] = []
        self.rejected: list[dict] = []
        if self.disaster_stop_frac <= 0:
            _log.warning(
                "basket_legs_unprotected",
                hint="basket_demo.disaster_stop_frac=0 — the net position carries NO "
                "exchange-resident stop (Section 2.2 deviation); a dead bot leaves it open",
            )

    # -- aggregate state -------------------------------------------------- #
    def in_scope(self, symbol: str) -> bool:
        """Whether this session is allowed to touch ``symbol`` — always True for a solo session
        (no scope), else only symbols in its partition. A position outside the scope is another
        strategy's and is invisible to this manager's reconcile/flatten."""
        return self.scope_symbols is None or symbol in self.scope_symbols

    def net(self, symbol: str) -> float:
        """The aggregate desired position across all strategies — what the exchange should hold."""
        total = sum(i.qty for (_s, sym), i in self._desired.items() if sym == symbol)
        return 0.0 if abs(total) < _DUST else total

    def strategy_qty(self, strategy_id: str, symbol: str) -> float:
        intent = self._desired.get((strategy_id, symbol))
        return intent.qty if intent is not None else 0.0

    def symbols(self) -> set[str]:
        return {sym for (_s, sym), i in self._desired.items() if abs(i.qty) >= _DUST}

    def net_positions(self) -> dict[str, float]:
        return {sym: self.net(sym) for sym in self.symbols()}

    # -- the one mutating operation --------------------------------------- #
    def apply(
        self, strategy_id: str, symbol: str, delta: float, *, ref_price: float, ts: int
    ) -> LegFill | None:
        """Move ``strategy_id``'s position in ``symbol`` by ``delta`` (signed) and send the order.

        Returns the observed fill (qty always POSITIVE — the caller knows its own direction), or
        None when nothing was executed. The strategy's recorded intent moves by what actually
        FILLED, never by what was asked for."""
        if abs(delta) < _DUST or ref_price <= 0:
            return None
        old_net = self.net(symbol)
        new_net = old_net + delta
        side = 1 if delta > 0 else -1
        qty = abs(delta)
        # A delta that shrinks |net| without crossing zero is a pure reduction — the exchange
        # accepts it as reduce-only, which is the safer form (it can never accidentally open
        # opposite exposure off a stale mirror). Anything that grows the book or crosses through
        # zero must NOT be reduce-only: the exchange would reject it.
        reduce_only = abs(new_net) < abs(old_net) and (
            new_net == 0.0 or (new_net > 0) == (old_net > 0)
        )
        # The disaster stop rides the NET position, so it is (re)attached only when the net grows
        # — a reducing order neither needs nor should move it.
        attach_stop = (
            self.disaster_stop_frac > 0 and abs(new_net) > abs(old_net) and abs(new_net) >= _DUST
        )
        plan = self._build_plan(
            symbol=symbol, side=side, qty=qty, ref_price=ref_price,
            reduce_only=reduce_only,
            stop_for_net=(1 if new_net > 0 else -1) if attach_stop else 0,
        )
        t0 = time.monotonic()
        try:
            res = self.venue.place_bracket(
                plan,
                ref_price=ref_price,
                realized_slippage_frac=0.0,  # the real venue measures its own; never modelled
                latency_ms=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - a rejected leg must not kill the basket loop
            _log.error(
                "basket_delta_order_failed", symbol=symbol, strategy=strategy_id,
                delta=delta, reduce_only=reduce_only, error=str(exc),
            )
            self.rejected.append(
                {"symbol": symbol, "strategy": strategy_id, "delta": delta, "ts": ts,
                 "error": str(exc)}
            )
            self._event(f"order REJECTED {symbol} Δ{delta:+g} ({strategy_id}): {exc}")
            return None
        latency_ms = (time.monotonic() - t0) * 1000.0
        fill = res.fill
        if fill.qty <= 0:
            _log.warning(
                "basket_delta_no_fill", symbol=symbol, strategy=strategy_id, requested=qty
            )
            self._event(f"NOT filled {symbol} Δ{delta:+g} ({strategy_id})")
            return None
        if fill.qty < qty - _DUST:
            # Partial: record what filled. The basket is then a touch unbalanced — the engine's
            # net-tilt warning covers it; chasing the remainder would chase a moving book.
            _log.warning(
                "basket_delta_partial_fill", symbol=symbol, strategy=strategy_id,
                requested=qty, filled=fill.qty,
            )
            self._event(f"PARTIAL {symbol}: {fill.qty:g}/{qty:g} ({strategy_id})")
        filled_signed = side * float(fill.qty)
        self._desired.setdefault((strategy_id, symbol), _Intent()).qty += filled_signed
        self.fills.append({
            "symbol": symbol, "strategy": strategy_id, "ts": ts,
            "delta": filled_signed, "reduce_only": reduce_only,
            "qty": fill.qty, "ref_price": ref_price, "fill_price": fill.actual_price,
            "fee": fill.fee, "maker": fill.maker, "slippage_frac": fill.slippage_frac,
            "latency_ms": latency_ms, "net_after": self.net(symbol),
        })
        return LegFill(
            qty=float(fill.qty),
            price=float(fill.actual_price),
            fee=float(fill.fee),
            maker=bool(fill.maker),
            slippage_cost=float(fill.slippage_cost),
        )

    # -- reconciliation ---------------------------------------------------- #
    def reconcile(self) -> dict[str, dict]:
        """Compare the aggregate intent against the REAL exchange book, per symbol.

        Returns ``{symbol: {"expected": x, "actual": y}}`` for every symbol that disagrees — a
        drift the caller must resolve so ``book == intent`` holds again.

        The basket demo session owns the account EXCLUSIVELY (the pre-flight requires a flat book
        and forbids a second session), so ``_desired`` is the complete source of truth for what
        SHOULD be on the book. We therefore reconcile against intent and do NOT trust the venue's
        per-position ``owned`` flag: Bybit's position read does not echo the opening order's
        clientOrderId, so every real position — including our own legs — comes back ``owned=False``.
        Keying off it produced a permanent false 'foreign position' CRITICAL and, worse, filtered
        the real book out of the comparison so genuine drift went undetected."""
        try:
            book = self.venue.fetch_exchange_positions()
        except Exception as exc:  # noqa: BLE001 - a book read hiccup is not a trading failure
            _log.warning("basket_recon_fetch_failed", error=str(exc))
            return {}
        # ONLY our partition's symbols. A position in any other symbol is another of our strategies'
        # (basis_reversion et al.) — recognized as ours, never reconciled or flattened here.
        actual = {
            sym: (p.side * p.qty)
            for sym, p in book.items()
            if p.qty > 0 and self.in_scope(sym)
        }
        drift: dict[str, dict] = {}
        for sym in set(actual) | self.symbols():
            if not self.in_scope(sym):
                continue
            expected = self.net(sym)
            got = actual.get(sym, 0.0)
            if abs(expected - got) > max(_DUST, abs(expected) * 1e-6, abs(got) * 1e-6):
                drift[sym] = {"expected": expected, "actual": got}
        return drift

    def flatten_stray(self, symbol: str, book_qty: float, ref_price: float) -> bool:
        """Reduce-only close of a position on the book that our intent does NOT account for.

        A mis-observed fill (the venue reported qty 0 but the order actually opened a position) or
        any exposure on this exclusively-owned account that ``_desired`` did not put there. Closing
        it restores ``book == intent`` instead of leaving it to sit — the SOL orphan this method
        exists for accrued for 12h before the session ended and left it open. ``book_qty`` is
        SIGNED (+long / −short); the close is the opposite side, reduce-only. Best-effort: returns
        True on a confirmed close, False otherwise (the next reconcile retries)."""
        if abs(book_qty) < _DUST:
            return True
        if not self.in_scope(symbol):
            # Defensive: never flatten a symbol outside our partition — it is another strategy's.
            _log.error("basket_flatten_stray_out_of_scope", symbol=symbol)
            return False
        # close_book_position closes what the REAL book holds even when it is absent from the
        # mirror (the mis-observed-fill case); plain close_position only touches the mirror.
        close = getattr(self.venue, "close_book_position", None) or getattr(
            self.venue, "close_position", None
        )
        if not callable(close):
            return False
        try:
            ok = bool(close(symbol))
        except Exception as exc:  # noqa: BLE001 - surface + let the next tick retry
            _log.error("basket_flatten_stray_failed", symbol=symbol, error=str(exc))
            self._event(f"flatten of stray {symbol} FAILED: {exc} — will retry")
            return False
        if ok:
            _log.warning(
                "basket_flattened_stray", symbol=symbol, book_qty=book_qty,
                hint="position on the book with no matching intent (mis-observed fill?) — "
                "closed reduce-only to restore book==intent",
            )
            self._event(f"flattened stray {symbol} ({book_qty:+g}) — book/intent realigned")
        return ok

    def drop_symbol(self, symbol: str) -> dict[str, float]:
        """Forget every strategy's intent in ``symbol`` (the exchange no longer holds it).

        Returns ``{strategy_id: qty}`` of what was dropped, so the caller can book each affected
        strategy's leg at the observed exit instead of vaporising its P&L."""
        dropped: dict[str, float] = {}
        for key in [k for k in self._desired if k[1] == symbol]:
            qty = self._desired.pop(key).qty
            if abs(qty) >= _DUST:
                dropped[key[0]] = qty
        return dropped

    def observed_exit(self, symbol: str, side: int, since_ts: int, fallback: float) -> tuple[
        float, float
    ]:
        """Real exit VWAP + fee from the account's own executions; the fallback mark is used (with
        the caller's loud log) when the venue shows us no close-side execution."""
        fetch = getattr(self.venue, "fetch_exit_fill", None)
        if callable(fetch):
            try:
                observed = fetch(symbol, side, since_ts=since_ts)
            except Exception as exc:  # noqa: BLE001 - fall back to the mark
                _log.warning("basket_exit_fill_error", symbol=symbol, error=str(exc))
                observed = None
            if observed is not None:
                return float(observed[0]), float(observed[1])
        _log.warning(
            "basket_exit_fill_unobserved", symbol=symbol,
            hint="no close-side execution visible — booked at the reference mark",
        )
        return float(fallback), 0.0

    # -- plan building ----------------------------------------------------- #
    def _build_plan(
        self,
        *,
        symbol: str,
        side: int,
        qty: float,
        ref_price: float,
        reduce_only: bool,
        stop_for_net: int,
    ) -> OrderPlan:
        """One position delta as an OrderPlan.

        Always a MARKET order: a basket MUST rebalance, and the netting manager's whole job is
        that the exchange position equals the aggregate intent — a passive order that may not
        fill would break that invariant for an unbounded time. (The backtest's maker-rebalance
        saving is therefore NOT modelled on demo; the session's execution summary reports the
        real taker cost, which is the honest comparison.)

        ``stop_for_net`` is the SIGN of the net position to protect (0 = attach no stop). The
        engine already rounded ``qty`` to the venue's lot step and checked min-order/min-notional,
        so no re-rounding here — a second pass could push a just-above-minimum leg back under it.
        """
        entry = Order(
            client_id=self.ownership.new_client_id("entry"),
            symbol=symbol,
            side="buy" if side > 0 else "sell",
            qty=qty,
            order_type=OrderType.MARKET,
            role="entry",
            reduce_only=reduce_only,
            tags=self.ownership.tags(),
        )
        stop: Order | None = None
        if stop_for_net != 0:
            stop_price = ref_price * (1.0 - stop_for_net * self.disaster_stop_frac)
            stop = Order(
                client_id=self.ownership.new_client_id("stop"),
                symbol=symbol,
                side="sell" if stop_for_net > 0 else "buy",
                qty=qty,
                order_type=OrderType.STOP_MARKET,
                role="stop",
                stop_price=self._round_tick(symbol, stop_price),
                reduce_only=True,
                tags=self.ownership.tags(parent_id=entry.client_id),
            )
        return OrderPlan(symbol=symbol, side=side, qty=qty, entry=entry, stop=stop)

    def _round_tick(self, symbol: str, price: float) -> float:
        spec = self.meta.spec(symbol)
        tick = float((spec.fields.get("tick_size", 0.0) if spec is not None else 0.0) or 0.0)
        if tick <= 0:
            return price
        return math.floor(price / tick) * tick

    def _event(self, message: str) -> None:
        if self.on_event is not None:
            with contextlib.suppress(Exception):  # reporting must never break execution
                self.on_event(message)

    # -- reporting ---------------------------------------------------------- #
    def execution_summary(self) -> dict:
        """Realized execution quality of this session's REAL fills — the whole point of running a
        basket on demo: does the backtest's cost model survive contact with the venue?"""
        n = len(self.fills)
        return {
            "orders_filled": n,
            "rejected": len(self.rejected),
            "avg_slippage_frac": (sum(f["slippage_frac"] for f in self.fills) / n) if n else 0.0,
            "maker_fill_rate": (sum(1 for f in self.fills if f["maker"]) / n) if n else 0.0,
            "total_fees": sum(f["fee"] for f in self.fills),
            "open_net_positions": self.net_positions(),
        }


@dataclass
class LiveBasketExecutor:
    """One strategy's view of the shared :class:`NetPositionManager` — a :class:`BasketExecutor`.

    Opening a long leg is ``+qty`` to this strategy's intent; closing it is ``−qty``. What the
    exchange receives is the aggregate delta, so another strategy holding the opposite side of
    the same symbol is netted rather than fought over."""

    manager: NetPositionManager
    strategy_id: str = "basket"
    entry_ts: dict[str, int] = field(default_factory=dict)

    def open_leg(
        self, *, symbol: str, side: int, qty: float, ref_price: float, ts: int
    ) -> LegFill | None:
        fill = self.manager.apply(
            self.strategy_id, symbol, side * qty, ref_price=ref_price, ts=ts
        )
        if fill is not None:
            self.entry_ts[symbol] = ts
        return fill

    def close_leg(
        self, *, symbol: str, side: int, qty: float, ref_price: float, entry_ts: int
    ) -> LegFill | None:
        """Close by sending this strategy's own delta — NEVER a whole-symbol close.

        ``venue.close_position`` flattens everything the account holds in the symbol, which on a
        shared account would silently close another strategy's leg too."""
        return self.manager.apply(
            self.strategy_id, symbol, -side * qty, ref_price=ref_price, ts=entry_ts
        )

    # Convenience pass-throughs so a single-strategy session needs no manager reference.
    @property
    def fills(self) -> list[dict]:
        return self.manager.fills

    @property
    def rejected(self) -> list[dict]:
        return self.manager.rejected

    def execution_summary(self) -> dict:
        return self.manager.execution_summary()


def build_executor(
    venue: Any,
    meta: MetadataConfig,
    settings: Settings,
    *,
    strategy_id: str = "basket",
    disaster_stop_frac: float = 0.25,
    on_event: Any | None = None,
) -> LiveBasketExecutor:
    """A single-strategy executor with its own manager (the one-basket demo session)."""
    manager = NetPositionManager(
        venue, meta, settings, disaster_stop_frac=disaster_stop_frac, on_event=on_event
    )
    return LiveBasketExecutor(manager, strategy_id=strategy_id)


__all__ = [
    "BasketExecutor",
    "LegFill",
    "LiveBasketExecutor",
    "NetPositionManager",
    "build_executor",
]
