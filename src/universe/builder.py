"""Universe builder skeleton (AGENTS.md Section 9).

Phase 1 provides the structural skeleton: given an exchange adapter, produce a
*versioned* universe snapshot whose members default to ``research_only`` (no
symbol becomes tradable until it passes the universe/data/metadata gates in
later phases — Section 9). Real filters (liquidity, history length, spread,
listing age, …) are added in Phase 3.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.db.models import SymbolStatus, UniverseMember, UniverseVersion
from src.exchange import ExchangeAdapter, get_adapter


class UniverseBuilder:
    """Builds versioned, persisted universe snapshots."""

    def __init__(self, adapter: ExchangeAdapter | None = None) -> None:
        self.adapter = adapter or get_adapter()

    def new_version_id(self) -> str:
        ts = datetime.now(UTC).strftime("%Y_%m_%d_%H%M%S")
        return f"univ_{ts}"

    def build(
        self,
        session: Session,
        *,
        version: str | None = None,
        note: str = "skeleton build (Phase 1): all members research_only",
    ) -> UniverseVersion:
        """Create and persist a new universe version from the adapter symbols.

        No symbol is marked ``active`` in Phase 1; filtering/promotion is
        deferred to the Universe gate (Section 9, Phase 3).
        """
        version = version or self.new_version_id()
        uv = UniverseVersion(
            version=version,
            exchange_id=self.adapter.exchange_id,
            criteria={"phase": 1, "filters_applied": []},
            note=note,
        )
        session.add(uv)
        session.flush()

        for symbol in self.adapter.fetch_symbols():
            session.add(
                UniverseMember(
                    universe_version=version,
                    symbol=symbol,
                    status=SymbolStatus.RESEARCH_ONLY,
                    reason="phase-1 skeleton: filters not yet applied",
                )
            )
        return uv
