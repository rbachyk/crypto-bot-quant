"""Data-quality validation (Section 23 / DQ gate)."""

from __future__ import annotations

from dataclasses import replace

from src.data.schema import (
    FUNDING,
    MARK,
    OHLCV,
    SPREAD,
    SeriesKey,
)
from src.data.source import DeterministicSource
from src.data.validation import DataValidator

from tests._data_helpers import fresh_store, populate, small_cfg


def _critical_checks(store, cfg) -> set[str]:
    report = DataValidator(store, cfg).validate()
    return {v.check for v in report.critical}


def test_clean_data_passes(tmp_path) -> None:
    cfg = small_cfg(symbols=("BTC/USDT:USDT", "ETH/USDT:USDT"))
    store = fresh_store(tmp_path)
    populate(store, cfg)
    report = DataValidator(store, cfg).validate()
    assert report.passed
    assert report.violations == []
    assert report.series_validated == 2 * len(cfg.required_keys(cfg.symbols[0]))


def test_missing_candles_is_critical(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    store.delete_range(key, cfg.window_start_ms + iv, cfg.window_start_ms + 3 * iv)
    assert "missing_candles" in _critical_checks(store, cfg)


def test_missing_candles_within_tolerance_is_a_warning_not_critical(tmp_path) -> None:
    """A few scattered missing candles within max_unfilled_gap_bars are a non-blocking WARNING —
    the snapshot stays valid (the multi-year-history use case). ohlcv-only config so the gap does
    not also trip the cross-series mark/index alignment check."""
    base = _single_ohlcv_cfg()
    cfg = replace(base, thresholds=replace(base.thresholds, max_unfilled_gap_bars=5))
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    store.delete_range(key, cfg.window_start_ms + iv, cfg.window_start_ms + 3 * iv)  # 2 missing
    report = DataValidator(store, cfg).validate()
    assert report.passed  # 2 missing <= tolerance 5 → not critical
    assert any(v.check == "missing_candles" and v.severity == "warning" for v in report.violations)


def test_impossible_prices_is_critical(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    ts = cfg.window_start_ms
    store.delete_range(key, ts, ts + key.interval_ms)
    # high < low — physically impossible candle.
    store.write(key, [{"ts": ts, "open": 100, "high": 90, "low": 110, "close": 100, "volume": 1}])
    assert "impossible_prices" in _critical_checks(store, cfg)


def test_extreme_gap_is_warning_not_critical(tmp_path) -> None:
    """A >50% close-to-close move is real crypto market data (a flash crash / liquidation cascade),
    NOT a data defect — flagged as a WARNING for review, never a critical that blocks the snapshot.
    Genuinely-broken bars (h<l, non-positive) stay critical (see the impossible-prices test).
    """
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    ts = cfg.window_start_ms + key.interval_ms
    store.delete_range(key, ts, ts + key.interval_ms)
    # A ~10x close vs neighbours -> >50% close-to-close move (a real flash-crash-shaped spike).
    base = store.read(key)[0]["close"]
    big = base * 10
    store.write(
        key,
        [{"ts": ts, "open": big, "high": big * 1.001,
          "low": big * 0.999, "close": big, "volume": 1}],
    )
    report = DataValidator(store, cfg).validate()
    assert "extreme_gaps" not in _critical_checks(store, cfg)  # not a blocking critical
    assert any(v.check == "extreme_gaps" and v.severity == "warning" for v in report.violations)


def test_one_tick_ohlc_glitch_is_tolerated(tmp_path) -> None:
    """A 1-tick rounding glitch (low a hair above open) is a feed artifact, not 'impossible prices'
    — it must NOT block the snapshot. (Regression for real Bybit bars failing validation.)"""
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    ts = cfg.window_start_ms
    store.delete_range(key, ts, ts + key.interval_ms)
    # low (100.01) sits a hair ABOVE open (100.0) — a 0.01% rounding glitch, prices otherwise sane.
    store.write(
        key,
        [{"ts": ts, "open": 100.0, "high": 100.05, "low": 100.01, "close": 100.02, "volume": 1}],
    )
    assert "impossible_prices" not in _critical_checks(store, cfg)


def test_abnormal_spread_is_critical(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, SPREAD, cfg.symbols[0], cfg.base_timeframe)
    ts = cfg.window_start_ms
    store.delete_range(key, ts, ts + key.interval_ms)
    store.write(key, [{"ts": ts, "bid": 99, "ask": 101, "spread": 2, "spread_bps": 5000.0}])
    assert "abnormal_spread" in _critical_checks(store, cfg)


def test_funding_misalignment_is_critical(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, FUNDING, cfg.symbols[0], cfg.funding_timeframe)
    off_grid = cfg.window_start_ms + 60_000  # 1 minute past an hour boundary — corrupt
    store.write(key, [{"ts": off_grid, "funding_rate": 0.0001, "funding_interval_hours": 8}])
    assert "funding_alignment" in _critical_checks(store, cfg)


def test_denser_funding_settlements_are_events_not_violations(tmp_path) -> None:
    """A symbol whose funding interval tightened (Bybit: 8h → 4h/2h/1h on volatile contracts)
    settles OFF the nominal 8h grid but ON whole hours — those are settlement EVENTS, not data
    defects: no alignment/ordering/duplicate/missing criticals (H10 regression)."""
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, FUNDING, cfg.symbols[0], cfg.funding_timeframe)
    denser = cfg.window_start_ms + 4 * 3_600_000  # 4h after the 8h settlement, hour-aligned
    store.write(key, [{"ts": denser, "funding_rate": 0.0002, "funding_interval_hours": 4}])
    report = DataValidator(store, cfg).validate()
    assert report.passed
    assert "funding_alignment" not in {v.check for v in report.violations}


def test_markindex_misalignment_is_critical(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    mark = SeriesKey(cfg.exchange_id, MARK, cfg.symbols[0], cfg.base_timeframe)
    # Drop a mark sample so mark timestamps no longer align with the perp grid.
    store.delete_range(mark, cfg.window_start_ms, cfg.window_start_ms + mark.interval_ms)
    assert "markindex_alignment" in _critical_checks(store, cfg)


def test_markindex_close_stamp_listing_edge_is_not_misalignment(tmp_path) -> None:
    """A perp LISTING inside the window legitimately has its first mark/index sample one base
    interval after its first bar: samples are stamped at the kline CLOSE, and no kline exists
    before the listing bar to close at its open. That single leading offset must not trip the
    alignment check (real misalignment — interior holes — stays critical, see above)."""
    from src.data.schema import INDEX

    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    iv = SeriesKey(cfg.exchange_id, MARK, cfg.symbols[0], cfg.base_timeframe).interval_ms
    listing = cfg.window_start_ms + 4 * iv  # perp lists mid-window
    for dt in ("ohlcv", MARK, INDEX, "open_interest", "spread"):
        tf = "5m" if dt == "ohlcv" else cfg.base_timeframe
        store.delete_range(SeriesKey(cfg.exchange_id, dt, cfg.symbols[0], tf),
                           cfg.window_start_ms, listing)
    for dt in (MARK, INDEX):
        # First close-stamped sample lands one interval AFTER the listing bar's open.
        store.delete_range(SeriesKey(cfg.exchange_id, dt, cfg.symbols[0], cfg.base_timeframe),
                           listing, listing + iv)
    assert "markindex_alignment" not in _critical_checks(store, cfg)


# --------------------------------------------------------------------------- #
# L15: clock drift measured against the EXCHANGE server time, or honestly N/A  #
# --------------------------------------------------------------------------- #
class _ClockSource(DeterministicSource):
    """Offline source that also reports a (fake) exchange server time."""

    def __init__(self, offset_ms: float) -> None:
        super().__init__("skeleton")
        self._offset_ms = offset_ms

    def server_time_ms(self) -> int | None:
        import time

        return int(time.time() * 1000 + self._offset_ms)


def test_clock_drift_vs_server_time_breach_is_critical(tmp_path) -> None:
    cfg = small_cfg()  # tolerance 2.0s
    store = fresh_store(tmp_path)
    populate(store, cfg)
    report = DataValidator(store, cfg, source=_ClockSource(offset_ms=10_000)).validate()
    assert "clock_drift" in {v.check for v in report.critical}


def test_clock_drift_within_tolerance_passes(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    report = DataValidator(store, cfg, source=_ClockSource(offset_ms=0.0)).validate()
    assert "clock_drift" not in {v.check for v in report.violations}
    assert "clock_drift" not in report.not_applicable  # genuinely measured, not skipped


def test_clock_drift_not_applicable_without_server_clock(tmp_path) -> None:
    """No external clock (offline source / no source): the check must be recorded as NOT
    APPLICABLE — neither a fake pass (the old wall-vs-monotonic no-op) nor a violation."""
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    for validator in (
        DataValidator(store, cfg),  # no source at all
        DataValidator(store, cfg, source=DeterministicSource("skeleton")),  # offline source
    ):
        report = validator.validate()
        assert "clock_drift" not in {v.check for v in report.violations}
        assert "clock_drift" in report.not_applicable
        assert report.to_dict()["not_applicable"]["clock_drift"]


class _FakeStore:
    """Returns canned rows for one key (to exercise checks the append-only store
    structurally prevents: out-of-order and duplicate timestamps)."""

    def __init__(self, key: SeriesKey, rows: list[dict]) -> None:
        self._key, self._rows = key, rows

    def read(self, key, start=None, end=None):  # type: ignore[no-untyped-def]
        return list(self._rows) if key == self._key else []

    def timestamps(self, key, start, end):  # type: ignore[no-untyped-def]
        return {r["ts"] for r in self.read(key, start, end)}


def _single_ohlcv_cfg():  # type: ignore[no-untyped-def]
    cfg = small_cfg()
    return replace(cfg, required_series=["ohlcv"])


def _ohlcv(ts: int) -> dict:
    return {"ts": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}


def test_out_of_order_timestamps_is_critical() -> None:
    cfg = _single_ohlcv_cfg()
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    s = cfg.window_start_ms
    # Cover the grid but hand the validator rows out of order.
    grid = [s + i * iv for i in range((cfg.window_end_ms - s) // iv)]
    rows = [_ohlcv(ts) for ts in grid]
    rows[0], rows[1] = rows[1], rows[0]
    report = DataValidator(_FakeStore(key, rows), cfg).validate()
    assert "ordering" in {v.check for v in report.critical}


def test_duplicate_timestamps_is_critical() -> None:
    cfg = _single_ohlcv_cfg()
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    s = cfg.window_start_ms
    grid = [s + i * iv for i in range((cfg.window_end_ms - s) // iv)]
    rows = [_ohlcv(ts) for ts in grid] + [_ohlcv(s)]  # duplicate first ts
    report = DataValidator(_FakeStore(key, rows), cfg).validate()
    assert "duplicates" in {v.check for v in report.critical}


def test_future_timestamps_is_critical() -> None:
    cfg = _single_ohlcv_cfg()
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    future = ((2_000_000_000_000 // iv) + 1) * iv  # year ~2033, on-grid
    cfg = replace(cfg, window_start_ms=future, window_end_ms=future + 2 * iv)
    rows = [_ohlcv(future)]
    report = DataValidator(_FakeStore(key, rows), cfg).validate()
    assert "future_timestamps" in {v.check for v in report.critical}
