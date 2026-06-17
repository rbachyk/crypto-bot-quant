"""Operator-verified exchange metadata (AGENTS.md Section 6, META gate).

The exchange adapter only ever returns ``[UNVERIFIED]`` metadata fetched from
the venue (``adapter.fetch_metadata``). The Metadata Verification Workflow
(Section 6) then has an operator review each contract spec against the venue's
authoritative reference and record the result as ``[VERIFIED]``. This module is
that recorded, versioned output: it loads ``configs/metadata.yaml`` and upserts
``[VERIFIED]`` :class:`~src.db.models.ExchangeMetadata` rows the META gate reads.

No live trading occurs with ``[UNVERIFIED]`` metadata (Section 2.1); the META
gate fails any active symbol whose metadata is missing, incomplete,
contradictory, ``[UNVERIFIED]``, or of a stale (non-current) version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from src.config.settings import REPO_ROOT
from src.db.models import ExchangeMetadata, VerificationStatus

METADATA_YAML = REPO_ROOT / "configs" / "metadata.yaml"

# Fields every active symbol must carry for META to pass (Appendix A META:
# tick, lot, min-notional, fees, funding, leverage, order types — plus the
# Section 6 adapter responsibilities needed to size/route an order safely).
REQUIRED_FIELDS: tuple[str, ...] = (
    "status",
    "contract_type",
    "quote_currency",
    "tick_size",
    "lot_size",
    "qty_step",
    "price_precision",
    "min_order_size",
    "min_notional",
    "max_leverage",
    "margin_mode",
    "position_mode",
    "maker_fee",
    "taker_fee",
    "funding_interval_hours",
    "order_types",
)

_VALID_FUNDING_INTERVALS = {1, 2, 4, 8, 12, 24}


@dataclass(frozen=True, slots=True)
class VerifiedSpec:
    """A single symbol's verified contract spec."""

    symbol: str
    fields: dict

    def missing_fields(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if self.fields.get(f) in (None, "", [])]

    def contradictions(self) -> list[str]:
        """Internal-consistency checks (Appendix A META 'contradictory values')."""
        f = self.fields
        out: list[str] = []
        for key in ("tick_size", "lot_size", "qty_step", "min_order_size", "min_notional"):
            val = f.get(key)
            if val is not None and not (isinstance(val, (int, float)) and val > 0):
                out.append(f"{key} must be > 0 (got {val})")
        lev = f.get("max_leverage")
        if lev is not None and not (isinstance(lev, int) and lev >= 1):
            out.append(f"max_leverage must be an int >= 1 (got {lev})")
        maker, taker = f.get("maker_fee"), f.get("taker_fee")
        if isinstance(maker, (int, float)) and maker < 0:
            out.append(f"maker_fee must be >= 0 (got {maker})")
        if isinstance(taker, (int, float)) and taker < 0:
            out.append(f"taker_fee must be >= 0 (got {taker})")
        if isinstance(maker, (int, float)) and isinstance(taker, (int, float)) and taker < maker:
            out.append(f"taker_fee ({taker}) must be >= maker_fee ({maker})")
        fih = f.get("funding_interval_hours")
        if fih is not None and fih not in _VALID_FUNDING_INTERVALS:
            out.append(f"funding_interval_hours {fih} not in {sorted(_VALID_FUNDING_INTERVALS)}")
        if f.get("status") not in (None, "trading"):
            out.append(f"status is {f.get('status')!r}, not 'trading'")
        ot = f.get("order_types")
        if ot is not None and not (isinstance(ot, list) and ot):
            out.append("order_types must be a non-empty list")
        return out

    def is_complete_and_consistent(self) -> bool:
        return not self.missing_fields() and not self.contradictions()


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    exchange_id: str
    metadata_version: str
    verified_against: str
    verified_at: str
    supported_order_types: list[str]
    specs: dict[str, VerifiedSpec] = field(default_factory=dict)

    def symbols(self) -> list[str]:
        return list(self.specs.keys())

    def spec(self, symbol: str) -> VerifiedSpec | None:
        return self.specs.get(symbol)


@lru_cache
def load_metadata_config(path: str | None = None) -> MetadataConfig:
    yaml_path = Path(path) if path else METADATA_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["metadata"]
    order_types = list(data.get("supported_order_types", []))
    specs: dict[str, VerifiedSpec] = {}
    for symbol, spec in (data.get("symbols") or {}).items():
        fields = dict(spec)
        # Default each symbol's order types to the venue-wide set unless overridden.
        fields.setdefault("order_types", list(order_types))
        specs[symbol] = VerifiedSpec(symbol=symbol, fields=fields)
    return MetadataConfig(
        exchange_id=str(data["exchange_id"]),
        metadata_version=str(data["metadata_version"]),
        verified_against=str(data.get("verified_against", "")),
        verified_at=str(data.get("verified_at", "")),
        supported_order_types=order_types,
        specs=specs,
    )


def sync_verified_metadata(session: Session, cfg: MetadataConfig | None = None) -> int:
    """Upsert ``[VERIFIED]`` metadata rows from config (the operator-review step).

    Idempotent: re-running with the same ``metadata_version`` updates the
    existing rows in place. Returns the number of symbols written.
    """
    cfg = cfg or load_metadata_config()
    written = 0
    for symbol, spec in cfg.specs.items():
        row = (
            session.query(ExchangeMetadata)
            .filter_by(
                exchange_id=cfg.exchange_id,
                symbol=symbol,
                metadata_version=cfg.metadata_version,
            )
            .one_or_none()
        )
        raw = {
            **spec.fields,
            "verified_against": cfg.verified_against,
            "verified_at": cfg.verified_at,
        }
        if row is None:
            session.add(
                ExchangeMetadata(
                    exchange_id=cfg.exchange_id,
                    symbol=symbol,
                    metadata_version=cfg.metadata_version,
                    verification_status=VerificationStatus.VERIFIED,
                    source="operator_verified",
                    raw=raw,
                )
            )
        else:
            row.verification_status = VerificationStatus.VERIFIED
            row.source = "operator_verified"
            row.raw = raw
        written += 1
    return written
