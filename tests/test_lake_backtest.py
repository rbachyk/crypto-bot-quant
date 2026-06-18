"""M2: real-data backtest inputs from the Parquet lake.

Populates a SeriesStore from the deterministic source (a stand-in for downloaded
exchange data — same canonical schema), then proves ``build_lake_inputs`` reads
it back through the ONE feature pipeline into engine inputs and that the engine
runs on them. Also covers the per-series OI grid override (``oi_timeframe``).
"""

from __future__ import annotations

import pytest
from src.backtest.config import load_backtest_config
from src.backtest.service import build_lake_inputs, lake_candidate_strategy, run_engine
from src.config.settings import REPO_ROOT
from src.data.config import DataConfig, ValidationThresholds, load_data_config
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
from src.exchange.metadata import load_metadata_config
from src.features.pipeline import StoreReader

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
BASE = "5m"
OI_TF = "1h"
FUND = "8h"


def _seed_lake(store: SeriesStore, start: int, end: int) -> None:
    """Write every required series into the store on its proper grid."""
    src = DeterministicSource(EX)
    for data_type, tf in (
        (OHLCV, TF),
        (MARK, BASE),
        (INDEX, BASE),
        (SPREAD, BASE),
        (OPEN_INTEREST, OI_TF),
        (FUNDING, FUND),
    ):
        key = SeriesKey(EX, data_type, SYM, tf)
        store.write(key, src.fetch(key, start, end))


def test_build_lake_inputs_round_trip_and_engine_run(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 200 * timeframe_ms(TF)  # ~16.6h of 5m bars
    _seed_lake(store, start, end)

    inputs = build_lake_inputs(
        store,
        exchange_id=EX,
        symbols=[SYM],
        timeframe=TF,
        base_timeframe=BASE,
        funding_timeframe=FUND,
        start_ms=start,
        end_ms=end,
        oi_timeframe=OI_TF,
    )
    assert len(inputs) == 1
    si = inputs[0]
    assert si.symbol == SYM
    assert si.bars and si.frame.rows
    # spread sampled on the 5m base grid; funding on the 8h grid (3 events in 16.6h).
    assert len(si.spread_samples) == 200
    assert len(si.funding_events) == 3
    assert all("spread_bps" in s for s in si.spread_samples)
    # features computed (oi_change is derived from the 1h OI series without error).
    assert "oi_change" in si.frame.feature_names

    # The engine runs end-to-end on real-shaped lake inputs and yields a report.
    result = run_engine(load_backtest_config(), load_metadata_config(), inputs, label="lake_test")
    assert result.report.payload["label"] == "lake_test"
    assert result.report.trade_count >= 0


def test_lake_inputs_rebased_to_zero_based_and_engine_trades(tmp_path) -> None:
    """Regression: real lake data carries absolute epoch ts; the engine indexes bars
    0-based (entry_bar = decision_ts // iv). build_lake_inputs must rebase, else every
    signal maps past the end of the bars array and NOTHING ever trades."""
    store = SeriesStore(tmp_path)
    iv = timeframe_ms(TF)
    h1 = timeframe_ms("1h")
    start = (1_700_000_000_000 // h1) * h1  # realistic recent epoch, 1h+5m aligned
    end = start + 400 * iv
    _seed_lake(store, start, end)

    inputs = build_lake_inputs(
        store,
        exchange_id=EX,
        symbols=[SYM],
        timeframe=TF,
        base_timeframe=BASE,
        funding_timeframe=FUND,
        start_ms=start,
        end_ms=end,
        oi_timeframe=OI_TF,
    )
    si = inputs[0]
    assert si.bars[0]["ts"] == 0  # rebased to a 0-based grid
    assert si.bars[1]["ts"] - si.bars[0]["ts"] == iv
    # The engine's entry-bar mapping must land inside the bars array for every row.
    assert all(r["decision_ts"] // iv < len(si.bars) for r in si.frame.rows)
    # Signals therefore reach the engine and execute (the bug produced zero trades).
    result = run_engine(load_backtest_config(), load_metadata_config(), inputs, label="rebase")
    assert result.report.trade_count > 0


def test_build_lake_inputs_skips_symbols_without_history(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 50 * timeframe_ms(TF)
    _seed_lake(store, start, end)
    inputs = build_lake_inputs(
        store,
        exchange_id=EX,
        symbols=[SYM, "GHOST/USDT:USDT"],  # GHOST has no data in the lake
        timeframe=TF,
        base_timeframe=BASE,
        funding_timeframe=FUND,
        start_ms=start,
        end_ms=end,
        oi_timeframe=OI_TF,
    )
    assert [si.symbol for si in inputs] == [SYM]


def test_store_reader_reads_oi_on_its_own_grid(tmp_path) -> None:
    store = SeriesStore(tmp_path)
    start, end = 0, 24 * timeframe_ms("1h")
    _seed_lake(store, start, end)
    reader = StoreReader(store, EX, TF, BASE, FUND, start, end, oi_timeframe=OI_TF)
    oi = reader.series(SYM, OPEN_INTEREST)
    assert oi, "OI must resolve via the 1h grid"
    assert all(r["ts"] % timeframe_ms(OI_TF) == 0 for r in oi)
    # A reader left on the base grid finds nothing (OI was written at 1h only).
    base_reader = StoreReader(store, EX, TF, BASE, FUND, start, end)
    assert base_reader.series(SYM, OPEN_INTEREST) == []


# --------------------------------------------------------------------------- #
# oi_timeframe config plumbing                                                 #
# --------------------------------------------------------------------------- #
def _cfg(oi_timeframe: str | None) -> DataConfig:
    return DataConfig(
        exchange_id=EX,
        data_version="t",
        symbols=[SYM],
        timeframes=[TF],
        base_timeframe=BASE,
        funding_interval_hours=8,
        required_series=[OHLCV, OPEN_INTEREST],
        window_start_ms=0,
        window_end_ms=timeframe_ms(TF),
        thresholds=ValidationThresholds(),
        oi_timeframe=oi_timeframe,
    )


def test_oi_grid_defaults_to_base() -> None:
    cfg = _cfg(None)
    assert cfg.oi_grid == BASE
    oi_key = next(k for k in cfg.required_keys(SYM) if k.data_type == OPEN_INTEREST)
    assert oi_key.timeframe == BASE


def test_oi_grid_override() -> None:
    cfg = _cfg(OI_TF)
    assert cfg.oi_grid == OI_TF
    oi_key = next(k for k in cfg.required_keys(SYM) if k.data_type == OPEN_INTEREST)
    assert oi_key.timeframe == OI_TF


def test_bybit_config_file_uses_1h_oi() -> None:
    cfg = load_data_config(str(REPO_ROOT / "configs" / "data.bybit.yaml"))
    assert cfg.exchange_id == "bybit"
    assert cfg.oi_grid == "1h"
    oi_key = next(k for k in cfg.required_keys(SYM) if k.data_type == OPEN_INTEREST)
    assert oi_key.timeframe == "1h"


def test_lake_candidate_strategy_builds_real_strategy() -> None:
    """Real-data backtests can run the configured research library (families A/B/G)."""
    strat, sid, version = lake_candidate_strategy("basis_reversion")
    assert sid == "basis_reversion"
    assert version
    assert hasattr(strat, "evaluate")  # family B is a per-row Strategy


def test_lake_candidate_strategy_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown candidate strategy"):
        lake_candidate_strategy("does_not_exist")
