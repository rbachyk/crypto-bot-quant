"""Feature pipeline — ONE causal code path for backtest and live (Section 10).

The Parity Rule (Section 10): there is a single feature-computation code path;
the only thing that differs between backtest and live is the *data-reading
adapter* (:class:`FeatureDataReader`). Every feature obeys the decision-time
rule — a feature row for closed bar ``k`` (the most recent fully-closed candle,
whose close time is the decision time ``t_k + interval``) is computed from
**only** bars ``0..k`` and point-in-time samples with ``ts <= t_k + interval``.
The forward-return label is future-only and is NEVER a feature input; it exists
solely for the leakage test.

Because every feature for bar ``k`` references only inputs available at the
decision time, truncating all future data leaves the row unchanged — the
property the FEAT gate verifies (no look-ahead) and the reason the feature store
is byte-reproducible from a dataset snapshot.
"""

from __future__ import annotations

import abc
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    OPEN_INTEREST,
    SeriesKey,
    timeframe_ms,
)
from src.data.store import SeriesStore
from src.features.config import FeatureConfig

# Stable feature order (the matrix column order). Grouped per Section 10:
# Market / Derivatives / Context. Cross-asset & Execution groups are deferred to
# later phases (they need multi-symbol joins / live fill data) — see decision doc.
FEATURE_NAMES: tuple[str, ...] = (
    # Market
    "ret_1",
    "ret_short",
    "rv_short",
    "atr_pct",
    "atr_pct_rank",
    "dir_efficiency",
    "trend_slope",
    "vol_z",
    # Derivatives
    "premium",
    "funding_rate",
    "funding_z",
    "oi_change",
    # Context
    "hour_utc",
    "is_weekend",
    "session_code",
    "pre_funding",
)

_ROUND = 10  # decimals features are rounded to (deterministic canonical form)


# --------------------------------------------------------------------------- #
# Data-reading adapters (the ONLY backtest/live difference — the Parity Rule)  #
# --------------------------------------------------------------------------- #
class FeatureDataReader(abc.ABC):
    @abc.abstractmethod
    def ohlcv(self, symbol: str) -> list[dict]:
        """Closed OHLCV bars at the feature timeframe, sorted by ts."""

    @abc.abstractmethod
    def series(self, symbol: str, data_type: str) -> list[dict]:
        """Point-in-time samples for ``data_type``, sorted by ts."""


class StoreReader(FeatureDataReader):
    """Reads features' raw inputs from the Parquet series store over a window."""

    def __init__(
        self,
        store: SeriesStore,
        exchange_id: str,
        timeframe: str,
        base_timeframe: str,
        funding_timeframe: str,
        start_ms: int,
        end_ms: int,
        *,
        oi_timeframe: str | None = None,
    ) -> None:
        self.store = store
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.base_timeframe = base_timeframe
        self.funding_timeframe = funding_timeframe
        # OI may live on its own (coarser) grid; defaults to the base grid.
        self.oi_timeframe = oi_timeframe or base_timeframe
        self.start_ms = start_ms
        self.end_ms = end_ms

    def _tf_for(self, data_type: str) -> str:
        if data_type == FUNDING:
            return self.funding_timeframe
        if data_type == OPEN_INTEREST:
            return self.oi_timeframe
        return self.base_timeframe

    def ohlcv(self, symbol: str) -> list[dict]:
        key = SeriesKey(self.exchange_id, OHLCV, symbol, self.timeframe)
        return self.store.read(key, self.start_ms, self.end_ms)

    def series(self, symbol: str, data_type: str) -> list[dict]:
        key = SeriesKey(self.exchange_id, data_type, symbol, self._tf_for(data_type))
        return self.store.read(key, self.start_ms, self.end_ms)


class TruncatedReader(FeatureDataReader):
    """Wraps a reader, hiding all data after a cutoff (for the no-look-ahead test).

    OHLCV is capped to bars with ``ts <= max_ohlcv_ts``; point-in-time series to
    samples with ``ts <= max_series_ts``. A correctly causal pipeline computes
    the row for the last retained bar identically with or without future data.
    """

    def __init__(self, base: FeatureDataReader, max_ohlcv_ts: int, max_series_ts: int) -> None:
        self.base = base
        self.max_ohlcv_ts = max_ohlcv_ts
        self.max_series_ts = max_series_ts

    def ohlcv(self, symbol: str) -> list[dict]:
        return [r for r in self.base.ohlcv(symbol) if r["ts"] <= self.max_ohlcv_ts]

    def series(self, symbol: str, data_type: str) -> list[dict]:
        return [r for r in self.base.series(symbol, data_type) if r["ts"] <= self.max_series_ts]


# --------------------------------------------------------------------------- #
# Feature frame                                                                #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FeatureFrame:
    symbol: str
    timeframe: str
    feature_names: list[str]
    rows: list[dict] = field(default_factory=list)

    def matrix(self) -> list[list[float]]:
        return [[r[name] for name in self.feature_names] for r in self.rows]

    def closes(self) -> list[float]:
        return [r["close"] for r in self.rows]

    def canonical(self) -> str:
        """Deterministic JSON used for the reproducibility checksum."""
        payload = [
            {
                "ts": r["ts"],
                "decision_ts": r["decision_ts"],
                **{name: round(float(r[name]), _ROUND) for name in self.feature_names},
            }
            for r in self.rows
        ]
        return json.dumps(
            {"symbol": self.symbol, "timeframe": self.timeframe, "rows": payload},
            sort_keys=True,
            separators=(",", ":"),
        )

    def checksum(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Numeric helpers (pure, causal)                                               #
# --------------------------------------------------------------------------- #
def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _percentile_rank(window: list[float], value: float) -> float:
    if not window:
        return 0.0
    below = sum(1 for v in window if v <= value)
    return below / len(window)


def _ols_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    return num / denom


def _asof(
    samples: list[dict], field_name: str, decision_ts: int
) -> tuple[float | None, float | None]:
    """Return (current, previous) values of the last two samples with ts <= decision_ts."""
    cur: float | None = None
    prev: float | None = None
    for row in samples:  # samples are sorted by ts
        if row["ts"] > decision_ts:
            break
        prev = cur
        cur = float(row[field_name])
    return cur, prev


def _asof_history(samples: list[dict], field_name: str, decision_ts: int) -> list[float]:
    return [float(r[field_name]) for r in samples if r["ts"] <= decision_ts]


# --------------------------------------------------------------------------- #
# Feature computation (single code path)                                       #
# --------------------------------------------------------------------------- #
def compute_features(symbol: str, reader: FeatureDataReader, cfg: FeatureConfig) -> FeatureFrame:
    bars = reader.ohlcv(symbol)
    iv = timeframe_ms(cfg.timeframe)
    mark = reader.series(symbol, MARK)
    index = reader.series(symbol, INDEX)
    funding = reader.series(symbol, FUNDING)
    oi = reader.series(symbol, OPEN_INTEREST)

    n = len(bars)
    ts = [b["ts"] for b in bars]
    close = [b["close"] for b in bars]
    high = [b["high"] for b in bars]
    low = [b["low"] for b in bars]
    volume = [b["volume"] for b in bars]
    logret = [0.0] + [
        math.log(close[i] / close[i - 1]) if close[i - 1] > 0 else 0.0 for i in range(1, n)
    ]
    true_range = [high[0] - low[0]] + [
        max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        for i in range(1, n)
    ]

    short = cfg.windows.short
    long_w = cfg.windows.long
    rank_w = cfg.windows.rank
    atr_pct_history: list[float] = []

    frame = FeatureFrame(symbol=symbol, timeframe=cfg.timeframe, feature_names=list(FEATURE_NAMES))
    for k in range(n):
        # atr% history is maintained causally for every bar with enough data.
        if k >= short:
            atr = sum(true_range[k - short + 1 : k + 1]) / short
            atr_pct_k = atr / close[k] if close[k] > 0 else 0.0
            atr_pct_history.append(atr_pct_k)
        if k < cfg.warmup:
            continue

        decision_ts = ts[k] + iv  # close time of bar k = decision time

        # --- Market features (bars 0..k) ---
        ret_1 = close[k] / close[k - 1] - 1.0
        ret_short = close[k] / close[k - short] - 1.0
        rv_short = _std(logret[k - short + 1 : k + 1])
        atr = sum(true_range[k - short + 1 : k + 1]) / short
        atr_pct = atr / close[k] if close[k] > 0 else 0.0
        rank_window = atr_pct_history[-rank_w:] if len(atr_pct_history) > 1 else atr_pct_history
        atr_pct_rank = _percentile_rank(rank_window, atr_pct)
        path = sum(abs(close[j] - close[j - 1]) for j in range(k - short + 1, k + 1))
        dir_efficiency = abs(close[k] - close[k - short]) / path if path > 0 else 0.0
        slope = _ols_slope(close[k - long_w + 1 : k + 1])
        mean_close = sum(close[k - long_w + 1 : k + 1]) / long_w
        trend_slope = slope / mean_close if mean_close > 0 else 0.0
        vol_window = volume[k - short + 1 : k + 1]
        vol_std = _std(vol_window)
        vol_mean = sum(vol_window) / len(vol_window)
        vol_z = (volume[k] - vol_mean) / vol_std if vol_std > 0 else 0.0

        # --- Derivatives (point-in-time samples with ts <= decision_ts) ---
        mark_now, _ = _asof(mark, "mark_price", decision_ts)
        index_now, _ = _asof(index, "index_price", decision_ts)
        if mark_now is not None and index_now is not None and index_now != 0.0:
            premium = (mark_now - index_now) / index_now
        else:
            premium = 0.0
        fund_now, _ = _asof(funding, "funding_rate", decision_ts)
        funding_rate = fund_now if fund_now is not None else 0.0
        fund_hist = _asof_history(funding, "funding_rate", decision_ts)
        if len(fund_hist) >= 3:
            f_std = _std(fund_hist)
            f_mean = sum(fund_hist) / len(fund_hist)
            funding_z = (funding_rate - f_mean) / f_std if f_std > 0 else 0.0
        else:
            funding_z = 0.0
        oi_now, oi_prev = _asof(oi, "open_interest", decision_ts)
        if oi_now is not None and oi_prev is not None and oi_prev != 0.0:
            oi_change = oi_now / oi_prev - 1.0
        else:
            oi_change = 0.0

        # --- Context (decision time) ---
        dt = datetime.fromtimestamp(decision_ts / 1000, tz=UTC)
        hour_utc = float(dt.hour)
        is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
        session_code = float(_session_code(dt.hour))
        funding_iv_ms = 8 * 3_600_000  # perpetual funding interval (Section 8)
        ms_into = decision_ts % funding_iv_ms
        pre_funding = 1.0 if (funding_iv_ms - ms_into) <= iv else 0.0

        frame.rows.append(
            {
                "ts": ts[k],
                "decision_ts": decision_ts,
                "close": close[k],
                "ret_1": ret_1,
                "ret_short": ret_short,
                "rv_short": rv_short,
                "atr_pct": atr_pct,
                "atr_pct_rank": atr_pct_rank,
                "dir_efficiency": dir_efficiency,
                "trend_slope": trend_slope,
                "vol_z": vol_z,
                "premium": premium,
                "funding_rate": funding_rate,
                "funding_z": funding_z,
                "oi_change": oi_change,
                "hour_utc": hour_utc,
                "is_weekend": is_weekend,
                "session_code": session_code,
                "pre_funding": pre_funding,
            }
        )
    return frame


def _session_code(hour_utc: int) -> int:
    """Coarse trading session bucket (Section 10 Context group)."""
    if 0 <= hour_utc < 8:
        return 0  # Asia
    if 8 <= hour_utc < 16:
        return 1  # Europe
    return 2  # US


def has_nan_or_inf(frame: FeatureFrame) -> bool:
    for row in frame.rows:
        for name in frame.feature_names:
            val = row[name]
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return True
    return False
