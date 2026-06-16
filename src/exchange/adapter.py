"""Exchange adapter skeleton (AGENTS.md Section 6).

Phase 1 ships the interface plus a deterministic, offline ``Skeleton`` adapter
that fabricates a small set of metadata records flagged ``[UNVERIFIED]`` (no
live trading with unverified metadata — Section 2.1). Real venues are wired via
ccxt + a native SDK fallback in later phases; nothing here touches the network.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(slots=True)
class SymbolMetadata:
    """Contract metadata for one symbol (subset; Section 6).

    Every value is ``[UNVERIFIED]`` until an operator confirms it against
    exchange docs and the ``META`` gate passes (Section 6 workflow).
    """

    symbol: str
    tick_size: float | None = None
    lot_size: float | None = None
    qty_step: float | None = None
    price_precision: int | None = None
    min_order_size: float | None = None
    min_notional: float | None = None
    max_leverage: int | None = None
    maker_fee: float | None = None
    taker_fee: float | None = None
    funding_interval_hours: int | None = None
    status: str = "trading"
    verification_status: str = "UNVERIFIED"
    raw: dict = field(default_factory=dict)


class ExchangeAdapter(abc.ABC):
    """Abstract exchange adapter. The only path to the venue (Section 6)."""

    exchange_id: str

    @abc.abstractmethod
    def fetch_symbols(self) -> list[str]:
        """Return the list of tradable perpetual-futures symbols."""

    @abc.abstractmethod
    def fetch_metadata(self, symbol: str) -> SymbolMetadata:
        """Return contract metadata for one symbol (flagged UNVERIFIED)."""

    @abc.abstractmethod
    def ping(self) -> bool:
        """Lightweight reachability check for health/monitoring."""


class SkeletonExchangeAdapter(ExchangeAdapter):
    """Offline, deterministic adapter used for Phase 1 wiring and tests.

    It returns a fixed universe with ``[UNVERIFIED]`` metadata and never makes
    network calls, so infrastructure can be exercised without exchange access.
    """

    def __init__(self, exchange_id: str = "skeleton") -> None:
        self.exchange_id = exchange_id
        self._symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

    def fetch_symbols(self) -> list[str]:
        return list(self._symbols)

    def fetch_metadata(self, symbol: str) -> SymbolMetadata:
        if symbol not in self._symbols:
            raise KeyError(f"unknown symbol: {symbol}")
        # Values intentionally left None / UNVERIFIED — placeholders only.
        return SymbolMetadata(
            symbol=symbol,
            status="trading",
            verification_status="UNVERIFIED",
            raw={"source": "skeleton", "note": "placeholder metadata; verify before live"},
        )

    def ping(self) -> bool:
        return True


def get_adapter(exchange_id: str | None = None) -> ExchangeAdapter:
    """Return an exchange adapter.

    Phase 1 always returns the offline skeleton; later phases select a real
    ccxt-backed adapter by ``exchange_id``.
    """
    return SkeletonExchangeAdapter(exchange_id or "skeleton")
