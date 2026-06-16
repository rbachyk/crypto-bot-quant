"""Exchange adapter layer (AGENTS.md Section 6).

All exchange-specific logic is isolated behind :class:`ExchangeAdapter`. No
strategy, feature, risk, or execution code may call exchange APIs directly.
"""

from src.exchange.adapter import (
    ExchangeAdapter,
    SkeletonExchangeAdapter,
    SymbolMetadata,
    get_adapter,
)

__all__ = [
    "ExchangeAdapter",
    "SkeletonExchangeAdapter",
    "SymbolMetadata",
    "get_adapter",
]
