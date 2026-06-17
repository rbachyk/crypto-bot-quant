"""Dynamic universe tests: filters, versioning, membership history (Section 9)."""

from __future__ import annotations

from src.config import get_settings
from src.db.base import session_scope
from src.db.models import SymbolStatus, UniverseChange, UniverseVersion
from src.universe import (
    UniverseConfig,
    UniverseFilterEvaluator,
    UniverseFilters,
    UniverseManager,
    latest_active_symbols,
)
from src.universe.filters import SymbolMetaView

from tests._data_helpers import populate, small_cfg
from tests.conftest import requires_db

SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT")


def _verified_view() -> SymbolMetaView:
    return SymbolMetaView(
        verified=True,
        status="trading",
        contract_type="perpetual",
        quote_currency="USDT",
        has_funding=True,
        has_open_interest=True,
    )


def _evaluator(tmp_path, filters: UniverseFilters | None = None):
    from tests._data_helpers import fresh_store

    data_cfg = small_cfg(symbols=SYMBOLS, timeframes=("1m", "5m"), hours=24)
    store = fresh_store(tmp_path)
    populate(store, data_cfg)
    uni_cfg = UniverseConfig(
        exchange_id="skeleton",
        universe_version="univ_test",
        candidates=list(SYMBOLS),
        eval_timeframe="1m",
        filters=filters or UniverseFilters(),
    )
    return UniverseFilterEvaluator(store, data_cfg, uni_cfg)


def test_all_filters_pass_makes_symbol_active(tmp_path) -> None:
    ev = _evaluator(tmp_path)
    out = ev.evaluate("BTC/USDT:USDT", _verified_view())
    assert out.passed_all, out.reason()
    assert out.status is SymbolStatus.ACTIVE
    assert out.metrics["history_bars"] == 1440


def test_soft_failure_is_research_only(tmp_path) -> None:
    # An impossibly high notional floor is a soft (quality) failure.
    ev = _evaluator(tmp_path, UniverseFilters(min_daily_notional_usd=1e30))
    out = ev.evaluate("BTC/USDT:USDT", _verified_view())
    assert not out.passed_all
    assert out.status is SymbolStatus.RESEARCH_ONLY


def test_unverified_metadata_is_quarantined(tmp_path) -> None:
    ev = _evaluator(tmp_path)
    view = SymbolMetaView(
        verified=False,
        status="trading",
        contract_type="perpetual",
        quote_currency="USDT",
        has_funding=True,
        has_open_interest=True,
    )
    out = ev.evaluate("BTC/USDT:USDT", view)
    assert not out.passed_all
    # A hard (metadata-safety) failure quarantines the symbol.
    assert out.status is SymbolStatus.QUARANTINED


@requires_db
def test_manager_builds_versioned_universe_with_change_log() -> None:
    settings = get_settings()
    with session_scope() as session:
        result = UniverseManager(settings=settings).build(session)
        version = result.version
        active = result.active_symbols

    assert active, "expected at least one active symbol"
    with session_scope() as session:
        uv = session.get(UniverseVersion, version)
        assert uv is not None
        assert uv.criteria.get("phase") == 3
        assert "filter_report" in uv.criteria
        members = {m.symbol: m.status for m in uv.members}
        assert any(s is SymbolStatus.ACTIVE for s in members.values())
        changes = session.query(UniverseChange).filter_by(universe_version=version).count()
        assert changes >= 1  # entering/leaving history recorded
        assert sorted(latest_active_symbols(session)) == sorted(active)


@requires_db
def test_manager_rebuild_is_idempotent() -> None:
    with session_scope() as session:
        first = UniverseManager().build(session)
    with session_scope() as session:
        second = UniverseManager().build(session)
    # Same membership => same content-addressed version, no duplicate row.
    assert first.version == second.version
    assert second.created is False
