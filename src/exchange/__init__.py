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
from src.exchange.metadata import (
    REQUIRED_FIELDS,
    MetadataConfig,
    VerifiedSpec,
    load_metadata_config,
    sync_verified_metadata,
)

__all__ = [
    "REQUIRED_FIELDS",
    "ExchangeAdapter",
    "MetadataConfig",
    "SkeletonExchangeAdapter",
    "SymbolMetadata",
    "VerifiedSpec",
    "get_adapter",
    "load_metadata_config",
    "sync_verified_metadata",
]
