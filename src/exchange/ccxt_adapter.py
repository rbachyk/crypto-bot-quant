"""Real ccxt-backed exchange adapter (AGENTS.md Section 6, Appendix C).

Implements :class:`~src.exchange.adapter.ExchangeAdapter` against a live exchange (default Bybit,
USDT-margined linear perps) for the read surface needed before live: symbol list + contract
metadata. Fetched metadata is flagged ``UNVERIFIED`` — an operator must reconcile it against
current exchange docs and the META gate before it can drive live trading (Section 6 workflow).
Public REST only; no API keys required. Order placement / account data are a later milestone.
"""

from __future__ import annotations

from typing import Any

from src.exchange.adapter import ExchangeAdapter, SymbolMetadata


def _decimals(tick: float | None) -> int | None:
    """Decimal places implied by a tick/step size (0.1 -> 1, 0.001 -> 3, 1.0 -> 0)."""
    if tick is None or tick <= 0:
        return None
    text = f"{tick:.12f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


class CcxtExchangeAdapter(ExchangeAdapter):
    """Real exchange adapter (default: Bybit swaps) behind the ExchangeAdapter interface."""

    def __init__(self, exchange_id: str = "bybit", client: Any | None = None) -> None:
        self.exchange_id = exchange_id
        self._ex: Any
        if client is not None:
            self._ex = client
        else:
            import ccxt

            klass = getattr(ccxt, exchange_id)
            self._ex = klass({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self._markets: dict | None = None

    def _markets_loaded(self) -> dict:
        if self._markets is None:
            self._markets = self._ex.load_markets()
        return self._markets

    def fetch_symbols(self) -> list[str]:
        """Active USDT-margined linear perpetuals, in ccxt unified form (BASE/USDT:USDT)."""
        return sorted(
            sym
            for sym, m in self._markets_loaded().items()
            if m.get("swap") and m.get("linear") and m.get("settle") == "USDT" and m.get("active")
        )

    def _increment_and_decimals(self, prec_val: Any) -> tuple[float | None, int | None]:
        """Convert a ccxt ``precision`` value to (increment_size, decimal_places), honoring the
        exchange's ``precisionMode`` (L-R). In TICK_SIZE mode the value IS the increment (Bybit);
        in DECIMAL_PLACES mode it is a DIGIT COUNT, so the increment is 10**-value — the old code
        assumed TICK_SIZE and would record e.g. tick_size=2.0 on a DECIMAL_PLACES venue."""
        if prec_val is None:
            return None, None
        try:
            import ccxt

            decimal_places = int(getattr(ccxt, "DECIMAL_PLACES", 2))
        except Exception:  # noqa: BLE001 - ccxt absent (injected fake client) → assume TICK_SIZE
            decimal_places = 2
        mode = getattr(self._ex, "precisionMode", None)
        if mode == decimal_places:
            d = int(prec_val)
            return (10.0 ** (-d) if d >= 0 else None), max(0, d)
        # TICK_SIZE (default / Bybit) — and SIGNIFICANT_DIGITS, which has no fixed tick: treat the
        # value as the increment and derive decimals from it (operator review is the backstop).
        tick = float(prec_val)
        return (tick if tick > 0 else None), _decimals(tick)

    def fetch_metadata(self, symbol: str) -> SymbolMetadata:
        m = self._markets_loaded().get(symbol)
        if m is None:
            raise KeyError(f"unknown symbol: {symbol}")
        prec = m.get("precision") or {}
        limits = m.get("limits") or {}
        tick_size, price_precision = self._increment_and_decimals(prec.get("price"))
        step_size, _ = self._increment_and_decimals(prec.get("amount"))
        lev_max = (limits.get("leverage") or {}).get("max")
        funding_min = (m.get("info") or {}).get("fundingInterval")
        return SymbolMetadata(
            symbol=symbol,
            tick_size=tick_size,
            lot_size=step_size,
            qty_step=step_size,
            price_precision=price_precision,
            min_order_size=(limits.get("amount") or {}).get("min"),
            min_notional=(limits.get("cost") or {}).get("min"),
            max_leverage=int(lev_max) if lev_max else None,
            maker_fee=m.get("maker"),
            taker_fee=m.get("taker"),
            funding_interval_hours=int(funding_min) // 60 if funding_min else None,
            status="trading" if m.get("active") else "inactive",
            verification_status="UNVERIFIED",
            raw={
                "source": self.exchange_id,
                "note": "fetched from exchange; verify against current docs before live",
                "info": m.get("info", {}),
            },
        )

    def ping(self) -> bool:
        try:
            self._markets_loaded()
            return True
        except Exception:  # noqa: BLE001
            return False
