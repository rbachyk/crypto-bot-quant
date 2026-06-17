"""Feature pipeline tests: Parity Rule, no look-ahead, reproducibility, leakage."""

from __future__ import annotations

from src.data.schema import timeframe_ms
from src.features import (
    FEATURE_NAMES,
    FeatureStore,
    SyntheticReader,
    causal_invariance_violations,
    compute_features,
    expectancy_z,
    forward_labels,
    has_nan_or_inf,
    load_feature_config,
    synthetic_leakage_report,
)
from src.features.pipeline import StoreReader

from tests._data_helpers import fresh_store, populate, small_cfg

SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT")


def _store_reader(tmp_path, data_cfg, feat_cfg):
    store = fresh_store(tmp_path)
    populate(store, data_cfg)
    reader = StoreReader(
        store,
        data_cfg.exchange_id,
        feat_cfg.timeframe,
        data_cfg.base_timeframe,
        data_cfg.funding_timeframe,
        data_cfg.window_start_ms,
        data_cfg.window_end_ms,
    )
    return store, reader


def test_features_are_finite_and_aligned(tmp_path) -> None:
    feat_cfg = load_feature_config()
    data_cfg = small_cfg(symbols=SYMBOLS, timeframes=("5m",), hours=24)
    _, reader = _store_reader(tmp_path, data_cfg, feat_cfg)
    frame = compute_features("BTC/USDT:USDT", reader, feat_cfg)
    assert frame.rows
    assert frame.feature_names == list(FEATURE_NAMES)
    assert not has_nan_or_inf(frame)
    iv = timeframe_ms(feat_cfg.timeframe)
    for r in frame.rows:
        assert r["decision_ts"] == r["ts"] + iv
        assert r["ts"] % iv == 0


def test_features_are_reproducible(tmp_path) -> None:
    feat_cfg = load_feature_config()
    data_cfg = small_cfg(symbols=SYMBOLS, timeframes=("5m",), hours=24)
    _, reader = _store_reader(tmp_path, data_cfg, feat_cfg)
    a = compute_features("BTC/USDT:USDT", reader, feat_cfg)
    b = compute_features("BTC/USDT:USDT", reader, feat_cfg)
    assert a.checksum() == b.checksum()


def test_no_lookahead_causal_invariance(tmp_path) -> None:
    feat_cfg = load_feature_config()
    data_cfg = small_cfg(symbols=SYMBOLS, timeframes=("5m",), hours=24)
    _, reader = _store_reader(tmp_path, data_cfg, feat_cfg)
    # Recomputing any row from future-truncated raw data must match exactly.
    violations = causal_invariance_violations("BTC/USDT:USDT", reader, feat_cfg)
    assert violations == []


def test_truncation_test_catches_a_leaky_feature(tmp_path) -> None:
    """Positive control: a compute that peeks one bar ahead MUST be flagged."""
    feat_cfg = load_feature_config()
    data_cfg = small_cfg(symbols=SYMBOLS, timeframes=("5m",), hours=24)
    _, reader = _store_reader(tmp_path, data_cfg, feat_cfg)

    def leaky_compute(symbol, rdr, cfg):
        # Same pipeline, then corrupt ret_1 to read the NEXT bar's close (the
        # canonical look-ahead bug). Under future truncation the next bar is
        # absent, so the row changes -> the harness must catch it.
        frame = compute_features(symbol, rdr, cfg)
        bars = rdr.ohlcv(symbol)
        idx_by_ts = {b["ts"]: i for i, b in enumerate(bars)}
        for row in frame.rows:
            i = idx_by_ts[row["ts"]]
            if i + 1 < len(bars):
                row["ret_1"] = bars[i + 1]["close"] / bars[i]["close"] - 1.0
        return frame

    violations = causal_invariance_violations(
        "BTC/USDT:USDT", reader, feat_cfg, compute_fn=leaky_compute
    )
    assert violations, "leaky compute was not caught by the causal-invariance test"


def test_forward_labels() -> None:
    closes = [10.0, 11.0, 12.0, 13.0]
    labels = forward_labels(closes, horizon=1)
    assert labels[0] == 11.0 / 10.0 - 1.0
    assert labels[-1] is None  # no future for the last bar


def test_synthetic_leakage_expectancy_near_zero() -> None:
    feat_cfg = load_feature_config()
    report = synthetic_leakage_report(feat_cfg)
    assert report["passed"], report
    assert abs(report["z"]) <= feat_cfg.leakage.max_synthetic_expectancy_z


def test_expectancy_z_flags_perfect_leak() -> None:
    """A signal equal to the sign of the future label has huge expectancy z."""
    feat_cfg = load_feature_config()
    reader = SyntheticReader(feat_cfg, 2000)
    frame = compute_features("SYNTH/USDT:USDT", reader, feat_cfg)
    labels = forward_labels(frame.closes(), feat_cfg.label_horizon)
    leaky_signals = [1.0 if (lab or 0.0) > 0 else -1.0 for lab in labels]
    stats = expectancy_z(leaky_signals, labels)
    assert abs(stats["z"]) > feat_cfg.leakage.max_synthetic_expectancy_z


def test_feature_store_build_is_reproducible(tmp_path) -> None:
    feat_cfg = load_feature_config()
    data_cfg = small_cfg(symbols=SYMBOLS, timeframes=("5m",), hours=24)
    store = fresh_store(tmp_path)
    populate(store, data_cfg)

    fstore = FeatureStore(data_cfg=data_cfg, feat_cfg=feat_cfg)
    fstore.store = store
    fstore.root = tmp_path / "features"

    a = fstore.build(list(SYMBOLS), dataset_version="data_test_abc")
    b = fstore.build(list(SYMBOLS), dataset_version="data_test_abc")
    assert a.feature_snapshot_id == b.feature_snapshot_id
    assert a.checksum == b.checksum
    assert a.created is True and b.created is False  # immutable reuse
    assert (fstore.root / a.feature_snapshot_id / "manifest.json").exists()
    assert a.total_rows > 0
