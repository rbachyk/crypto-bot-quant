"""Shared builders for the Phase 2 data-platform tests."""

from __future__ import annotations

from pathlib import Path

from src.data.config import DataConfig, ValidationThresholds
from src.data.schema import parse_utc_ms
from src.data.source import DeterministicSource, get_data_source
from src.data.store import SeriesStore

# A funding-boundary anchor so an 8h-multiple window contains funding points.
ANCHOR = "2026-06-01T00:00:00Z"


def small_cfg(
    *,
    symbols: tuple[str, ...] = ("BTC/USDT:USDT",),
    timeframes: tuple[str, ...] = ("5m",),
    hours: int = 8,
    insufficient: tuple[str, ...] = (),
    thresholds: ValidationThresholds | None = None,
) -> DataConfig:
    end = parse_utc_ms(ANCHOR)
    start = end - hours * 3_600_000
    return DataConfig(
        exchange_id="skeleton",
        data_version="data_test",
        symbols=list(symbols),
        timeframes=list(timeframes),
        base_timeframe="5m",
        funding_interval_hours=8,
        required_series=["ohlcv", "mark", "index", "funding", "open_interest", "spread"],
        window_start_ms=start,
        window_end_ms=end,
        insufficient_history=list(insufficient),
        thresholds=thresholds or ValidationThresholds(),
    )


def fresh_store(tmp_path: Path) -> SeriesStore:
    return SeriesStore(tmp_path / "lake")


def populate(store: SeriesStore, cfg: DataConfig, source: DeterministicSource | None = None) -> int:
    src = source or get_data_source(cfg.exchange_id)
    written = 0
    for symbol in cfg.active_symbols():
        if not src.has_symbol(symbol):
            continue
        for key in cfg.required_keys(symbol):
            written += store.write(key, src.fetch(key, cfg.window_start_ms, cfg.window_end_ms))
    return written
