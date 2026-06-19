"""Real-data strategy validation (Section 13) — promote on REAL downloaded data, not fixtures.

Seeds a SeriesStore from the deterministic source (a stand-in for a downloaded snapshot, same
canonical schema) and runs the real-data validation harness end to end, proving it composes the
same gates (backtest + side decision + walk-forward + fee/slippage stress) over lake inputs and
returns well-formed promote/shelve verdicts — without touching fixtures or the network.
"""

from __future__ import annotations

import pytest
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
from src.strategies.lake_research import validate_all_on_lake
from src.strategies.research import CandidateValidation

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
OI_TF = "1h"
FUND = "8h"


def _seed_lake(store: SeriesStore, start: int, end: int) -> None:
    src = DeterministicSource(EX)
    for dt, tf in (
        (OHLCV, TF), (MARK, TF), (INDEX, TF), (SPREAD, TF),
        (OPEN_INTEREST, OI_TF), (FUNDING, FUND),
    ):
        key = SeriesKey(EX, dt, SYM, tf)
        store.write(key, src.fetch(key, start, end))


def _cfg(start: int, end: int) -> DataConfig:
    return DataConfig(
        exchange_id=EX, data_version="t", symbols=[SYM], timeframes=[TF], base_timeframe=TF,
        funding_interval_hours=8,
        required_series=[OHLCV, MARK, INDEX, FUNDING, OPEN_INTEREST, SPREAD],
        window_start_ms=start, window_end_ms=end, thresholds=ValidationThresholds(),
        oi_timeframe=OI_TF,
    )


def test_validate_all_on_lake_returns_wellformed_verdicts(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    iv = timeframe_ms(TF)
    h1 = timeframe_ms("1h")
    start = (1_700_000_000_000 // h1) * h1
    end = start + 600 * iv
    _seed_lake(store, start, end)

    verdicts = validate_all_on_lake(_cfg(start, end), timeframe=TF, symbols=[SYM], store=store)
    assert verdicts, "every enabled candidate gets a verdict"
    for v in verdicts:
        assert isinstance(v, CandidateValidation)
        assert v.status in ("promoted", "shelved")
        assert isinstance(v.promoted, bool)
        # real-data validation drops the synthetic noise-control step
        assert "skipped" in v.noise_control
        if not v.promoted:
            assert v.shelved_reasons  # a shelve always explains why


def test_validate_all_on_lake_errors_without_data(tmp_path) -> None:
    store = SeriesStore(tmp_path)  # empty store → no bars
    start, end = 0, 600 * timeframe_ms(TF)
    with pytest.raises(ValueError, match="no real data in the lake"):
        validate_all_on_lake(_cfg(start, end), timeframe=TF, symbols=[SYM], store=store)
