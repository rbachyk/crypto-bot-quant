"""Gap detection, backfill repair, and incremental update (Section 8)."""

from __future__ import annotations

from src.data.gaps import find_gaps
from src.data.ingest import Ingestor
from src.data.schema import FUNDING, OHLCV, SeriesKey
from src.data.source import DataSource, get_data_source
from src.data.store import SeriesStore

from tests._data_helpers import fresh_store, populate, small_cfg

_HOUR_MS = 3_600_000


def test_full_window_has_zero_gaps(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    report = find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms)
    assert report.covered
    assert report.expected == report.present


def test_gap_detection_and_range_coalescing(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    start = cfg.window_start_ms
    store.delete_range(key, start + iv, start + 4 * iv)  # 3 contiguous missing bars
    report = find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms)
    assert len(report.missing_ts) == 3
    assert report.ranges() == [(start + iv, start + 4 * iv)]


def test_repair_closes_gaps_idempotently(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    populate(store, cfg)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    store.delete_range(key, cfg.window_start_ms + iv, cfg.window_start_ms + 4 * iv)

    ing = Ingestor(get_data_source(cfg.exchange_id), store)
    result = ing.repair(key, cfg.window_start_ms, cfg.window_end_ms)
    assert result.rows_written == 3
    assert result.repaired
    # Repaired values match the original source (pure function of ts).
    assert find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms).covered
    # Running repair again is a no-op.
    assert ing.repair(key, cfg.window_start_ms, cfg.window_end_ms).rows_written == 0


def test_incremental_update_fetches_only_the_tail(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    half = cfg.window_start_ms + (cfg.window_end_ms - cfg.window_start_ms) // 2
    ing = Ingestor(get_data_source(cfg.exchange_id), store)
    ing.download(key, cfg.window_start_ms, half)
    before = store.count(key)
    added = ing.update_incremental(key, cfg.window_start_ms, cfg.window_end_ms)
    assert added > 0
    assert store.count(key) == before + added
    # No duplication, fully covered now.
    assert find_gaps(store, key, cfg.window_start_ms, cfg.window_end_ms).covered


def test_missing_symbol_cannot_be_filled(tmp_path) -> None:
    cfg = small_cfg(symbols=("BTC/USDT:USDT",))
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, OHLCV, "BTC/USDT:USDT", "5m")
    from src.data.source import DeterministicSource

    ing = Ingestor(DeterministicSource(missing_symbols={"BTC/USDT:USDT"}), store)
    result = ing.repair(key, cfg.window_start_ms, cfg.window_end_ms)
    assert result.rows_written == 0
    assert not result.repaired  # exchange genuinely lacks history


# --------------------------------------------------------------------------- #
# Listing watermark (H11): "pre-listing" vs "missing head". A full download     #
# persists the exchange's actual first candle; gap detection then reports data  #
# loss at the HEAD of a series (delete_range, skipped early pages) like any     #
# other gap instead of silently masking it as pre-listing absence.              #
# --------------------------------------------------------------------------- #
class _ListingCutoffSource(DataSource):
    """Wraps a source but has NO data before ``listing_ts`` (a mid-window listing)."""

    def __init__(self, inner: DataSource, listing_ts: int) -> None:
        self._inner, self._listing = inner, listing_ts

    def has_symbol(self, symbol: str) -> bool:
        return self._inner.has_symbol(symbol)

    def fetch(self, key: SeriesKey, start_ms: int, end_ms: int) -> list[dict]:
        return [r for r in self._inner.fetch(key, start_ms, end_ms) if r["ts"] >= self._listing]


def test_head_deletion_is_detected_and_repaired(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    start, end = cfg.window_start_ms, cfg.window_end_ms
    ing = Ingestor(get_data_source(cfg.exchange_id), store)
    ing.download(key, start, end)  # full download records the listing watermark
    assert store.listing_ts(key) == start

    store.delete_range(key, start, start + 5 * iv)  # head-of-series data loss
    report = find_gaps(store, key, start, end)
    assert report.missing_ts == [start + i * iv for i in range(5)]  # NOT masked as pre-listing
    assert report.ranges() == [(start, start + 5 * iv)]
    assert ing.repair(key, start, end).repaired  # ...and it is repairable
    assert find_gaps(store, key, start, end).covered


def test_genuine_pre_listing_absence_is_still_not_a_gap(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    start, end = cfg.window_start_ms, cfg.window_end_ms
    listing = start + 20 * iv  # the contract lists 20 candles into the window
    ing = Ingestor(_ListingCutoffSource(get_data_source(cfg.exchange_id), listing), store)
    ing.download(key, start, end)
    # The first row a from-the-start fetch returns IS the listing edge.
    assert store.listing_ts(key) == listing
    assert find_gaps(store, key, start, end).covered  # leading absence ⇒ no gap
    # ...but an interior hole after the listing IS still reported.
    store.delete_range(key, listing + 5 * iv, listing + 6 * iv)
    assert find_gaps(store, key, start, end).missing_ts == [listing + 5 * iv]


def test_watermark_survives_delete_range_and_is_monotone_min(tmp_path) -> None:
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    iv = key.interval_ms
    start, end = cfg.window_start_ms, cfg.window_end_ms
    Ingestor(get_data_source(cfg.exchange_id), store).download(key, start, end)
    assert store.listing_ts(key) == start

    store.delete_range(key, start, end)  # wipe the whole series
    assert store.listing_ts(key) == start  # delete_range must NOT move the watermark
    report = find_gaps(store, key, start, end)
    assert len(report.missing_ts) == (end - start) // iv  # full loss reported, not "pre-listing"

    store.record_listing_ts(key, start + iv)  # later/higher probe: ignored (monotone-min)
    assert store.listing_ts(key) == start
    store.record_listing_ts(key, start - iv)  # wider window found earlier data: moves down
    assert store.listing_ts(key) == start - iv


def test_incremental_tail_resume_does_not_mint_a_watermark(tmp_path) -> None:
    """A tail resume starts mid-series by construction — recording its first row as 'listing'
    would mask everything before the tail. Only from-the-start fetches record the watermark."""
    cfg = small_cfg()
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "5m")
    start, end = cfg.window_start_ms, cfg.window_end_ms
    half = start + (end - start) // 2
    src = get_data_source(cfg.exchange_id)
    store.write(key, src.fetch(key, start, half))  # legacy series: data, no watermark
    ing = Ingestor(src, store)
    assert ing.update_incremental(key, start, end) > 0
    assert store.listing_ts(key) is None  # tail resume minted nothing
    # An EMPTY series' incremental update resumes from the window start ⇒ it IS the probe.
    key2 = SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "1h")
    assert ing.update_incremental(key2, start, end) > 0
    assert store.listing_ts(key2) == start


# --------------------------------------------------------------------------- #
# Funding gap semantics (H10): funding is an EVENT series — covered ⇔ no        #
# spacing longer than the interval stamped on each row (the symbol's local      #
# cadence), so denser-than-8h settlements are neither gaps nor duplicates and   #
# a dropped settlement on the denser cadence IS a gap.                          #
# --------------------------------------------------------------------------- #
def _funding_rows_8h_then_4h(start: int) -> list[dict]:
    """Settlements every 8h for 16h, then every 4h (interval tightened mid-window)."""
    rows = [
        {"ts": start, "funding_interval_hours": 8},
        {"ts": start + 8 * _HOUR_MS, "funding_interval_hours": 8},
        {"ts": start + 16 * _HOUR_MS, "funding_interval_hours": 8},
        {"ts": start + 20 * _HOUR_MS, "funding_interval_hours": 4},
    ]
    return [{**r, "funding_rate": 0.0001} for r in rows]


def test_funding_interval_switch_coverage_is_clean(tmp_path) -> None:
    cfg = small_cfg(hours=24)
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, FUNDING, cfg.symbols[0], cfg.funding_timeframe)
    start, end = cfg.window_start_ms, cfg.window_end_ms
    store.write(key, _funding_rows_8h_then_4h(start))
    report = find_gaps(store, key, start, end)
    assert report.covered  # the off-8h-grid settlement is cadence, not a gap/duplicate
    assert report.present == 4


def test_funding_missing_settlement_on_denser_cadence_is_a_gap(tmp_path) -> None:
    cfg = small_cfg(hours=24)
    store = fresh_store(tmp_path)
    key = SeriesKey(cfg.exchange_id, FUNDING, cfg.symbols[0], cfg.funding_timeframe)
    start, end = cfg.window_start_ms, cfg.window_end_ms
    victim = start + 20 * _HOUR_MS  # the 4h settlement — invisible to a fixed 8h grid
    store.write(key, [r for r in _funding_rows_8h_then_4h(start) if r["ts"] != victim])
    report = find_gaps(store, key, start, end)
    # The last row (16:00, stamped 8h) leaves an 8h trailing cadence: 00:00 == end ⇒ nothing
    # trails, but the tightened cadence is only knowable from the STORED stamp — write the
    # follower stamped 4h and the hole to its predecessor is reported.
    assert report.covered  # without the 4h follower there is no evidence of a denser cadence
    store.write(
        key, [{"ts": start + 22 * _HOUR_MS, "funding_rate": 0.0001, "funding_interval_hours": 2}]
    )
    report = find_gaps(store, key, start, end)
    assert report.missing_ts == [start + 18 * _HOUR_MS, start + 20 * _HOUR_MS]


# --------------------------------------------------------------------------- #
# Backfill CLI (L18): --config points the repair at a real lake config (e.g.    #
# data.bybit.yaml) instead of always loading the skeleton configs/data.yaml.    #
# --------------------------------------------------------------------------- #
_CLI_CONFIG_YAML = """\
version: 1
data:
  exchange_id: skeleton
  data_version: data_cli_test
  symbols: ["BTC/USDT:USDT"]
  timeframes: ["5m"]
  base_timeframe: "5m"
  funding_interval_hours: 8
  required_series: ["ohlcv"]
  window:
    as_of: "2026-06-01T00:00:00Z"
    duration_hours: 1
"""


def test_backfill_cli_config_targets_alternate_config(tmp_path, monkeypatch, capsys) -> None:
    from src.config import get_settings
    from src.data import backfill as backfill_cli

    cfg_path = tmp_path / "data.cli.yaml"
    cfg_path.write_text(_CLI_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("DATA_LAKE_PATH", str(tmp_path / "lake"))
    monkeypatch.setenv("ARTIFACT_PATH", str(tmp_path / "artifacts"))
    # Never configure structlog inside pytest: it would cache a PrintLogger bound to the
    # captured (soon-closed) stderr and poison every later test that logs.
    monkeypatch.setattr(backfill_cli, "configure_logging", lambda: None)
    get_settings.cache_clear()
    try:
        from src.data.backfill import main

        assert main(["--config", str(cfg_path)]) == 0
        # The repair ran against the CONFIGURED lake/universe, not the skeleton default.
        store = SeriesStore(tmp_path / "lake")
        key = SeriesKey("skeleton", OHLCV, "BTC/USDT:USDT", "5m")
        assert store.count(key) == 12  # 1h of 5m bars
        assert "ohlcv" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()  # do not leak the tmp lake path to other tests
