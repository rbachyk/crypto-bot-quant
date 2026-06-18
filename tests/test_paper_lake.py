"""M4: real-data (replay) paper sessions built from lake data.

Seeds a SeriesStore from the deterministic source, then proves the candidate stream
is derived from the real feature frame and runs through the actual paper pipeline
(ranking → risk → execution → SimulatedVenue) producing PaperTrades — no fabricated
candidates. DB persistence is covered by run_lake_paper_session in the e2e path; here
we keep it hermetic (no DB) by exercising the engine directly.
"""

from __future__ import annotations

import math

import pytest
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
from src.paper.engine import PaperTradingEngine
from src.paper.lake import build_lake_paper_inputs

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
BASE = "5m"
OI_TF = "1h"
FUND = "8h"


def _data_cfg(start: int, end: int):
    from src.data.config import DataConfig, ValidationThresholds

    return DataConfig(
        exchange_id=EX,
        data_version="t",
        symbols=[SYM],
        timeframes=[TF],
        base_timeframe=BASE,
        funding_interval_hours=8,
        required_series=[OHLCV, MARK, INDEX, FUNDING, OPEN_INTEREST, SPREAD],
        window_start_ms=start,
        window_end_ms=end,
        thresholds=ValidationThresholds(),
        oi_timeframe=OI_TF,
    )


def _seed(store: SeriesStore, start: int, end: int) -> None:
    src = DeterministicSource(EX)
    for dt, tf in (
        (OHLCV, TF),
        (MARK, BASE),
        (INDEX, BASE),
        (SPREAD, BASE),
        (OPEN_INTEREST, OI_TF),
        (FUNDING, FUND),
    ):
        key = SeriesKey(EX, dt, SYM, tf)
        store.write(key, src.fetch(key, start, end))


def test_build_lake_paper_inputs_from_real_frame(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)

    inputs, strat_id, version = build_lake_paper_inputs(
        cfg, timeframe=TF, symbols=[SYM], store=store
    )
    assert strat_id == "reference_momentum" and version
    assert inputs, "the strategy should fire on the seeded series"
    for pin in inputs:
        c = pin.candidate
        assert c.side in (1, -1)
        assert c.entry_price > 0
        assert c.symbol == SYM
        # features come from the REAL decision-time row, not fabricated constants.
        assert set(c.features) == {"atr_pct", "premium", "funding_z"}
        assert math.isfinite(pin.exit_move_frac)
        assert c.spread_bps > 0  # estimated spread from the lake


def test_lake_paper_inputs_run_through_paper_pipeline(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)
    inputs, _, _ = build_lake_paper_inputs(cfg, timeframe=TF, symbols=[SYM], store=store)

    engine = PaperTradingEngine()
    session = engine.new_session("lake_test")
    engine.process_candidates(inputs, session)
    # Every candidate is either executed or rejected (full pipeline ran on real candidates).
    assert session.executed_count + session.rejected_count == len(inputs)
    assert session.executed_count > 0
    for t in session.trades:
        assert t.has_exchange_side_stop  # bracket attached at entry
        assert t.spread_bps_at_entry > 0


def test_lake_paper_rejects_portfolio_strategy(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 50 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)
    with pytest.raises(ValueError, match="cross-asset|portfolio"):
        build_lake_paper_inputs(
            cfg, timeframe=TF, symbols=[SYM], candidate_id="lead_lag_xasset", store=store
        )


def test_ml_shadow_scores_real_lake_candidates(tmp_path) -> None:
    """Real-data ML shadow: the meta-labeler scores REAL lake candidates, applied=False."""
    from src.ml import ShadowPredictor
    from src.ml.config import load_ml_config
    from src.ml.labels import build_reference_dataset, train_test_split

    store = SeriesStore(tmp_path)
    start, end = 0, 400 * timeframe_ms(TF)
    _seed(store, start, end)
    cfg = _data_cfg(start, end)
    inputs, _, _ = build_lake_paper_inputs(cfg, timeframe=TF, symbols=[SYM], store=store)
    candidates = [pin.candidate for pin in inputs]
    assert candidates

    predictor = ShadowPredictor.from_config(load_ml_config())
    train_samples, _ = train_test_split(build_reference_dataset(seed=42), seed=42)
    predictor.train(train_samples)
    result = predictor.run(candidates, write_to_db=False)
    assert result.applied is False  # shadow-only: never influences trading
    assert len(result.bundles) == len(candidates)
