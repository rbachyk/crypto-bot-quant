"""Exchange-metadata verification tests (Section 6, META gate)."""

from __future__ import annotations

from src.db.base import session_scope
from src.db.models import ExchangeMetadata, VerificationStatus
from src.exchange.metadata import (
    REQUIRED_FIELDS,
    VerifiedSpec,
    load_metadata_config,
    sync_verified_metadata,
)

from tests.conftest import requires_db


def test_config_specs_complete_and_consistent() -> None:
    cfg = load_metadata_config()
    assert cfg.metadata_version == "meta_0001"
    assert cfg.symbols()
    for symbol in cfg.symbols():
        spec = cfg.spec(symbol)
        assert spec is not None
        assert spec.missing_fields() == [], f"{symbol} missing {spec.missing_fields()}"
        assert spec.contradictions() == [], f"{symbol} contradictions {spec.contradictions()}"
        assert spec.is_complete_and_consistent()
        # Order types are required by META.
        assert spec.fields["order_types"]


def test_missing_field_is_detected() -> None:
    fields = dict.fromkeys(REQUIRED_FIELDS, 1)
    del fields["tick_size"]
    spec = VerifiedSpec("X/USDT:USDT", fields)
    assert "tick_size" in spec.missing_fields()
    assert not spec.is_complete_and_consistent()


def test_contradictions_are_detected() -> None:
    base = {
        "status": "trading",
        "contract_type": "perpetual",
        "quote_currency": "USDT",
        "tick_size": 0.1,
        "lot_size": 0.1,
        "qty_step": 0.1,
        "price_precision": 1,
        "min_order_size": 0.1,
        "min_notional": 5.0,
        "max_leverage": 10,
        "margin_mode": "isolated",
        "position_mode": "one_way",
        "maker_fee": 0.001,
        "taker_fee": 0.0001,  # taker < maker => contradiction
        "funding_interval_hours": 8,
        "order_types": ["limit"],
    }
    spec = VerifiedSpec("X/USDT:USDT", base)
    assert any("taker_fee" in c for c in spec.contradictions())

    bad_tick = {**base, "taker_fee": 0.001, "tick_size": 0.0}
    assert any("tick_size" in c for c in VerifiedSpec("Y", bad_tick).contradictions())

    bad_fund = {**base, "taker_fee": 0.001, "funding_interval_hours": 7}
    assert any("funding_interval_hours" in c for c in VerifiedSpec("Z", bad_fund).contradictions())


@requires_db
def test_sync_writes_verified_rows_idempotently() -> None:
    cfg = load_metadata_config()
    with session_scope() as session:
        # Clean slate for the current version.
        session.query(ExchangeMetadata).filter_by(
            exchange_id=cfg.exchange_id, metadata_version=cfg.metadata_version
        ).delete()

    with session_scope() as session:
        n1 = sync_verified_metadata(session, cfg)
    with session_scope() as session:
        n2 = sync_verified_metadata(session, cfg)  # idempotent
        rows = (
            session.query(ExchangeMetadata)
            .filter_by(exchange_id=cfg.exchange_id, metadata_version=cfg.metadata_version)
            .all()
        )
    assert n1 == n2 == len(cfg.symbols())
    assert len(rows) == len(cfg.symbols())  # no duplicates on re-sync
    assert all(r.verification_status is VerificationStatus.VERIFIED for r in rows)
