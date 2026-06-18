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

    def fetch_metadata(self, symbol: str) -> SymbolMetadata:
        m = self._markets_loaded().get(symbol)
        if m is None:
            raise KeyError(f"unknown symbol: {symbol}")
        prec = m.get("precision") or {}
        limits = m.get("limits") or {}
        tick = prec.get("price")
        step = prec.get("amount")
        lev_max = (limits.get("leverage") or {}).get("max")
        funding_min = (m.get("info") or {}).get("fundingInterval")
        return SymbolMetadata(
            symbol=symbol,
            tick_size=float(tick) if tick is not None else None,
            lot_size=float(step) if step is not None else None,
            qty_step=float(step) if step is not None else None,
            price_precision=_decimals(tick),
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
