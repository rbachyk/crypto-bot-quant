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


def test_engine_trades_symbol_listed_mid_window(tmp_path) -> None:
    """Regression: a contract listed AFTER the window start (e.g. SOL/ETH perps vs a multi-year
    BTC window) has its first candle at a large grid slot, NOT array index 0, after rebasing to
    the window start. The engine must index bars by grid slot (ts // iv), else every signal for
    such a symbol maps past its short bars array and it produces ZERO trades — the bug that
    shelved all candidates on the 5-year real-data run. Here data exists only over the SECOND
    half of the requested window, so after rebase bars[0] sits ~300 slots in."""
    store = SeriesStore(tmp_path)
    iv = timeframe_ms(TF)
    listing = 300 * iv  # the contract "lists" 300 slots into the window
    window_start = 0
    window_end = listing + 400 * iv  # 400 bars of real history after listing
    _seed_lake(store, listing, window_end)  # NOTHING before the listing slot

    inputs = build_lake_inputs(
        store,
        exchange_id=EX,
        symbols=[SYM],
        timeframe=TF,
        base_timeframe=BASE,
        funding_timeframe=FUND,
        start_ms=window_start,
        end_ms=window_end,
        oi_timeframe=OI_TF,
    )
    si = inputs[0]
    # The leading absence is preserved: bars start ~300 slots in, not at array-index-0 ts.
    assert si.bars[0]["ts"] == listing  # rebased by window_start (0) ⇒ unchanged here
    assert si.bars[0]["ts"] // iv >= 300
    # Grid-indexed engine still matches the mid-window signals to their bars and trades.
    result = run_engine(load_backtest_config(), load_metadata_config(), inputs, label="midwindow")
    assert result.report.trade_count > 0


def _lake_kw(start, end):
    return {
        "exchange_id": EX, "symbols": [SYM], "timeframe": TF, "base_timeframe": BASE,
        "funding_timeframe": FUND, "start_ms": start, "end_ms": end, "oi_timeframe": OI_TF,
    }


def test_build_lake_inputs_cache_hit_skips_feature_rebuild(tmp_path, monkeypatch) -> None:
    """The build (parquet read + feature compute + rebase) is deterministic, so a repeat run over
    the SAME lake snapshot must load the persisted inputs and NOT recompute features (the part that
    costs hours). Proven by making compute_features explode on the second call."""
    store = SeriesStore(tmp_path)
    iv = timeframe_ms(TF)
    start, end = 0, 200 * iv
    _seed_lake(store, start, end)

    first = build_lake_inputs(store, **_lake_kw(start, end))  # builds + persists
    assert (tmp_path / "input_cache").exists() and list((tmp_path / "input_cache").glob("*.pkl"))

    import src.backtest.service as svc

    def _boom(*_a, **_k):
        raise AssertionError("compute_features must NOT run on a cache hit")

    monkeypatch.setattr(svc, "compute_features", _boom)
    # Same data, and a LATER end_ms (mimicking the as_of:now window ticking) — still a hit, because
    # the key omits end_ms and uses the data fingerprint.
    second = build_lake_inputs(store, **{**_lake_kw(start, end + 10 * iv)})
    assert len(first) == len(second) == 1
    assert [b["ts"] for b in first[0].bars] == [b["ts"] for b in second[0].bars]
    assert first[0].frame.feature_names == second[0].frame.feature_names


def test_build_lake_inputs_cache_invalidates_when_data_changes(tmp_path) -> None:
    """Appending real bars rewrites the parquet → the data fingerprint changes → the stale cache is
    NOT served; the build picks up the new data."""
    store = SeriesStore(tmp_path)
    iv = timeframe_ms(TF)
    _seed_lake(store, 0, 200 * iv)
    n_first = len(build_lake_inputs(store, **_lake_kw(0, 200 * iv))[0].bars)

    _seed_lake(store, 200 * iv, 260 * iv)  # 60 more real bars → fingerprint changes
    rebuilt = build_lake_inputs(store, **_lake_kw(0, 260 * iv))
    assert len(rebuilt[0].bars) > n_first  # rebuilt on the new data, not the stale 200-bar cache


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


def test_prebuild_timeframes_default_and_override() -> None:
    base = {
        "exchange_id": EX, "data_version": "t", "symbols": [SYM], "timeframes": ["5m", "1h", "4h"],
        "base_timeframe": BASE, "funding_interval_hours": 8, "required_series": [OHLCV],
        "window_start_ms": 0, "window_end_ms": timeframe_ms(TF),
        "thresholds": ValidationThresholds(),
    }
    assert DataConfig(**base).prebuild_timeframes == ["5m", "1h", "4h"]  # default = all
    assert DataConfig(**base, prebuild_input_timeframes=["4h"]).prebuild_timeframes == ["4h"]


def test_bybit_config_prebuilds_4h_only() -> None:
    cfg = load_data_config(str(REPO_ROOT / "configs" / "data.bybit.yaml"))
    assert cfg.prebuild_timeframes == ["4h"]  # 1h/5m are opt-in (slow on 20 symbols)


def test_prewarm_input_cache_builds_and_persists(tmp_path) -> None:
    """Pre-building at download time populates the cache (per-symbol progress) so a later
    validation/backtest is an instant load instead of an ~hours rebuild."""
    from src.backtest.service import prewarm_input_cache

    store = SeriesStore(tmp_path)
    start, end = 0, 200 * timeframe_ms(TF)
    _seed_lake(store, start, end)
    cfg = DataConfig(
        exchange_id=EX, data_version="t", symbols=[SYM], timeframes=["5m", "1h", "4h"],
        base_timeframe=BASE, funding_interval_hours=8,
        required_series=[OHLCV, MARK, INDEX, FUNDING], window_start_ms=start, window_end_ms=end,
        thresholds=ValidationThresholds(), oi_timeframe=OI_TF, prebuild_input_timeframes=["5m"],
    )
    seen: list = []
    shapes = prewarm_input_cache(cfg, store, progress=lambda tf, d, t, s: seen.append((tf, d, t)))
    assert shapes == {"5m": 1}  # only the prebuild timeframe was built, one symbol with data
    assert list((tmp_path / "input_cache").glob("*.pkl"))  # cache persisted
    assert seen and seen[-1] == ("5m", 1, 1)  # per-symbol progress reached completion


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


def test_bybit_config_is_multiyear_without_oi_spread() -> None:
    """The bybit config spans years for a real validation sample, so it EXCLUDES open_interest
    and spread (Bybit serves those only for recent days); oi_grid stays 1h for when OI is used."""
    cfg = load_data_config(str(REPO_ROOT / "configs" / "data.bybit.yaml"))
    assert cfg.exchange_id == "bybit"
    assert cfg.oi_grid == "1h"
    types = {k.data_type for k in cfg.required_keys(SYM)}
    assert OPEN_INTEREST not in types and "spread" not in types  # excluded for multi-year history
    assert {"ohlcv", "mark", "index", "funding"} <= types  # the long-history series remain
    years = (cfg.window_end_ms - cfg.window_start_ms) / 1000 / 86400 / 365
    assert years > 3  # a real sample, not a few days


def test_lake_candidate_strategy_builds_real_strategy() -> None:
    """Real-data backtests can run the configured research library (families A/B/G)."""
    strat, sid, version = lake_candidate_strategy("basis_reversion")
    assert sid == "basis_reversion"
    assert version
    assert hasattr(strat, "evaluate")  # family B is a per-row Strategy


def test_lake_candidate_strategy_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown candidate strategy"):
        lake_candidate_strategy("does_not_exist")
