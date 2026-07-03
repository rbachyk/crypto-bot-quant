"""Data-config window resolution + snapshot determinism (Appendix B.5).

Under ``as_of: now`` the window advances hourly by design (fresh data). Pinning ``as_of_ms``
freezes the window so the snapshot id is reproducible across runs — these lock that in."""

from __future__ import annotations

from src.data.config import load_data_config
from src.data.snapshot import _deterministic_snapshot_id, series_checksums
from src.data.store import SeriesStore

_CFG = "configs/data.bybit.yaml"
_PINNED = 1_700_000_400_000  # an arbitrary fixed instant (not on the hour grid)


def test_pinned_as_of_window_is_deterministic() -> None:
    a = load_data_config(_CFG, as_of_ms=_PINNED)
    b = load_data_config(_CFG, as_of_ms=_PINNED)
    # Same pin → identical window across calls (independent of wall-clock).
    assert (a.window_start_ms, a.window_end_ms) == (b.window_start_ms, b.window_end_ms)
    assert a.window_end_ms % 3_600_000 == 0  # snapped to the hour grid for determinism


def test_series_end_is_clamped_per_timeframe_so_only_closed_bars_are_required() -> None:
    """The hour-grid window end guarantees closure only for OHLCV timeframes ≤ 1h. Coarser
    grids (4h here) must have their effective end clamped down to their OWN grid, or coverage
    would demand a bar that may still be forming (which the source correctly refuses to
    serve). Close-stamped samples (mark/index/spread) and funding/OI events are realized AT
    their stamp, so their end is the raw window end."""
    from src.data.schema import FUNDING, MARK, OHLCV, TIMEFRAME_MS, SeriesKey

    cfg = load_data_config(_CFG, as_of_ms=_PINNED)
    end = cfg.window_end_ms
    assert end % TIMEFRAME_MS["4h"] != 0  # pinned instant is NOT on the 4h grid (the hard case)

    ohlcv_4h = cfg.series_end_ms(SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], "4h"))
    assert ohlcv_4h == (end // TIMEFRAME_MS["4h"]) * TIMEFRAME_MS["4h"]  # clamped down
    assert end - TIMEFRAME_MS["4h"] < ohlcv_4h < end
    # ≤ 1h grids divide the hour-grid end exactly — no clamp.
    for tf in ("5m", "1h"):
        assert cfg.series_end_ms(SeriesKey(cfg.exchange_id, OHLCV, cfg.symbols[0], tf)) == end
    # Non-OHLCV series keep the raw end (their stamps are realized at stamp time). An 8h clamp
    # on funding would wrongly drop an already-settled funding event near the window end.
    assert cfg.series_end_ms(SeriesKey(cfg.exchange_id, MARK, cfg.symbols[0], "5m")) == end
    assert cfg.series_end_ms(SeriesKey(cfg.exchange_id, FUNDING, cfg.symbols[0], "8h")) == end


def test_snapshot_id_is_stable_for_a_pinned_window(tmp_path) -> None:
    cfg = load_data_config(_CFG, as_of_ms=_PINNED)
    store = SeriesStore(tmp_path)  # empty store → empty-but-deterministic checksums
    id1 = _deterministic_snapshot_id(cfg, series_checksums(store, cfg))
    id2 = _deterministic_snapshot_id(cfg, series_checksums(store, cfg))
    assert id1 == id2  # reproducible: id depends only on (pinned window, series content)
    assert id1.startswith(cfg.data_version)
