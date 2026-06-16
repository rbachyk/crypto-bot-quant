"""Exchange-adapter and universe-builder skeleton tests (Sections 6, 9)."""

from __future__ import annotations

from src.db.base import session_scope
from src.db.models import SymbolStatus, UniverseVersion
from src.exchange import SkeletonExchangeAdapter, get_adapter
from src.universe import UniverseBuilder

from tests.conftest import requires_db


def test_adapter_returns_unverified_metadata() -> None:
    adapter = get_adapter()
    symbols = adapter.fetch_symbols()
    assert symbols
    meta = adapter.fetch_metadata(symbols[0])
    # No live trading with unverified metadata (Section 2.1): skeleton is UNVERIFIED.
    assert meta.verification_status == "UNVERIFIED"
    assert adapter.ping() is True


@requires_db
def test_universe_builder_persists_research_only_members() -> None:
    builder = UniverseBuilder(SkeletonExchangeAdapter())
    with session_scope() as session:
        uv = builder.build(session, version="univ_test_phase1")
        version = uv.version

    with session_scope() as session:
        stored = session.get(UniverseVersion, version)
        assert stored is not None
        assert stored.members
        # Phase 1: nothing is tradable until later gates promote it (Section 9).
        assert all(m.status is SymbolStatus.RESEARCH_ONLY for m in stored.members)
        # cleanup
        session.delete(stored)
