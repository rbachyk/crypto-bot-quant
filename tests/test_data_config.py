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


def test_snapshot_id_is_stable_for_a_pinned_window(tmp_path) -> None:
    cfg = load_data_config(_CFG, as_of_ms=_PINNED)
    store = SeriesStore(tmp_path)  # empty store → empty-but-deterministic checksums
    id1 = _deterministic_snapshot_id(cfg, series_checksums(store, cfg))
    id2 = _deterministic_snapshot_id(cfg, series_checksums(store, cfg))
    assert id1 == id2  # reproducible: id depends only on (pinned window, series content)
    assert id1.startswith(cfg.data_version)
