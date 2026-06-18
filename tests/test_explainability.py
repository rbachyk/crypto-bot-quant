"""Section 24: TradeExplainability guard + decision_log / trade_explainability persistence."""

from __future__ import annotations

import dataclasses

import pytest
from sqlalchemy import select
from src.data.config import DataConfig, ValidationThresholds
from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    OPEN_INTEREST,
    SPREAD,
    SeriesKey,
    timeframe_ms,
)
from src.data.source import DeterministicSource
from src.data.store import SeriesStore
from src.db.base import session_scope
from src.db.models import DecisionLog, TradeExplainabilityRow
from src.explainability import ExplainabilityError, TradeExplainability, write_decision_log
from src.paper.engine import PaperTradingEngine
from src.paper.lake import build_lake_paper_inputs
from src.paper.report import build_paper_report
from src.paper.run import persist_paper_session


def _complete() -> TradeExplainability:
    return TradeExplainability(
        trade_id="t1",
        symbol="BTC/USDT:USDT",
        strategy_id="basis_reversion",
        setup_type="reversion",
        regime="R2_TREND",
        signal_features={"atr_pct": 0.01},
        expected_edge_after_costs=10.0,
        expected_fees=1.0,
        expected_slippage=0.5,
        stop_price=49_000.0,
        execution_route="taker",
        risk_approved=True,
        risk_reason="approved",
        config_version="cfg_0001",
        universe_version="u_1",
        why_selected="ranked top",
        invalidation_conditions=["stop_hit"],
    )


def test_complete_schema_passes() -> None:
    assert _complete().ensure_complete().trade_id == "t1"


@pytest.mark.parametrize(
    "patch",
    [
        {"why_selected": ""},
        {"regime": ""},
        {"stop_price": 0.0},
        {"signal_features": {}},
        {"invalidation_conditions": []},
        {"universe_version": "  "},
    ],
)
def test_incomplete_schema_blocks_the_trade(patch) -> None:
    bad = dataclasses.replace(_complete(), **patch)
    with pytest.raises(ExplainabilityError, match="trade not taken"):
        bad.ensure_complete()


def test_write_decision_log_persists_row() -> None:
    write_decision_log(
        symbol="BTC/USDT:USDT",
        strategy="s",
        action="reject",
        reason="no_trade_regime",
        rejected_alternatives=[{"symbol": "ETH/USDT:USDT", "reason": "lower_score"}],
        config_version="cfg_0001",
        session_id="dlog_test_session",
    )
    with session_scope() as s:
        row = (
            s.execute(select(DecisionLog).where(DecisionLog.session_id == "dlog_test_session"))
            .scalars()
            .first()
        )
        assert row is not None and row.action == "reject"
        assert row.rejected_alternatives[0]["symbol"] == "ETH/USDT:USDT"


def _seed(store, start, end):
    src = DeterministicSource("bybit")
    for dt, tf in (
        (OHLCV, "5m"),
        (MARK, "5m"),
        (INDEX, "5m"),
        (SPREAD, "5m"),
        (OPEN_INTEREST, "1h"),
        (FUNDING, "8h"),
    ):
        k = SeriesKey("bybit", dt, "BTC/USDT:USDT", tf)
        store.write(k, src.fetch(k, start, end))


def test_paper_session_persists_decision_logs_and_explainability(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms("5m")
    _seed(store, start, end)
    cfg = DataConfig(
        exchange_id="bybit",
        data_version="t",
        symbols=["BTC/USDT:USDT"],
        timeframes=["5m"],
        base_timeframe="5m",
        funding_interval_hours=8,
        required_series=[OHLCV, MARK, INDEX, FUNDING, OPEN_INTEREST, SPREAD],
        window_start_ms=start,
        window_end_ms=end,
        thresholds=ValidationThresholds(),
        oi_timeframe="1h",
    )
    inputs, _, _ = build_lake_paper_inputs(
        cfg, timeframe="5m", symbols=["BTC/USDT:USDT"], store=store
    )

    engine = PaperTradingEngine()
    session = engine.new_session("explain_test_session")
    engine.process_candidates(inputs, session)
    assert session.executed_count > 0
    persist_paper_session(session, build_paper_report(session))

    with session_scope() as s:
        dlogs = (
            s.execute(select(DecisionLog).where(DecisionLog.session_id == session.session_id))
            .scalars()
            .all()
        )
        explains = (
            s.execute(
                select(TradeExplainabilityRow).where(
                    TradeExplainabilityRow.session_id == session.session_id
                )
            )
            .scalars()
            .all()
        )
    assert len(dlogs) >= session.executed_count  # one per decision (incl. executes)
    assert len(explains) == session.executed_count  # one explainability per executed trade
    assert all(e.payload.get("why_selected") for e in explains)
    assert all(e.regime.startswith("R") for e in explains)  # Section-11 R-code regime
