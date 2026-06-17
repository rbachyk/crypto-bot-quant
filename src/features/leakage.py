"""Leakage & look-ahead test harness (AGENTS.md Section 16, FEAT gate).

Two independent guards on the feature pipeline:

1. **No look-ahead (causal invariance):** recompute a feature row from raw inputs
   truncated to the decision time; a causal pipeline yields the identical row, so
   any mismatch is leakage. This is the strongest reproducibility check — it runs
   the whole pipeline on truncated raw data and compares.

2. **Synthetic-data expectancy ~0:** build features + a forward-return label on a
   synthetic series with NO real structure (i.i.d. zero-mean returns). A past-only
   signal must be uncorrelated with the future label, so expectancy ≈ 0. If a
   feature secretly carried the future, the signal would correlate with the label
   even on noise → non-zero expectancy → FAIL (Appendix A FEAT: "synthetic data
   yields non-zero expectancy" ⇒ leakage).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

from src.data.schema import timeframe_ms
from src.features.config import FeatureConfig
from src.features.pipeline import (
    FeatureDataReader,
    FeatureFrame,
    TruncatedReader,
    compute_features,
)

ComputeFn = Callable[[str, FeatureDataReader, FeatureConfig], FeatureFrame]

_FUNDING_IV_MS = 8 * 3_600_000


def _unit(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


# --------------------------------------------------------------------------- #
# Labels & expectancy                                                          #
# --------------------------------------------------------------------------- #
def forward_labels(closes: list[float], horizon: int) -> list[float | None]:
    """Future return over ``horizon`` bars (None where the future is unavailable)."""
    n = len(closes)
    out: list[float | None] = []
    for k in range(n):
        j = k + horizon
        out.append(closes[j] / closes[k] - 1.0 if j < n and closes[k] > 0 else None)
    return out


def expectancy_z(signals: list[float], labels: list[float | None]) -> dict:
    """Expectancy of ``signal * forward_return`` and its z-score vs zero."""
    products = [
        s * lab for s, lab in zip(signals, labels, strict=True) if lab is not None and s != 0.0
    ]
    n = len(products)
    if n < 2:
        return {"expectancy": 0.0, "stderr": 0.0, "z": 0.0, "n": n}
    mean = sum(products) / n
    var = sum((p - mean) ** 2 for p in products) / (n - 1)
    stderr = math.sqrt(var / n)
    z = mean / stderr if stderr > 0 else 0.0
    return {"expectancy": mean, "stderr": stderr, "z": z, "n": n}


def momentum_signals(frame: FeatureFrame) -> list[float]:
    """A representative past-only signal: the sign of short-horizon momentum."""
    return [_sign(r["ret_short"]) for r in frame.rows]


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


# --------------------------------------------------------------------------- #
# Synthetic source (no real structure)                                         #
# --------------------------------------------------------------------------- #
class SyntheticReader(FeatureDataReader):
    """Offline reader serving an i.i.d. random-walk series with valid derivatives.

    Returns are a deterministic pseudo-random function of bar index only (no
    dependence on the future), so a causal pipeline has zero predictive edge.
    """

    def __init__(self, cfg: FeatureConfig, n_bars: int, seed: str = "synthetic") -> None:
        self.cfg = cfg
        self.seed = seed
        self.iv = timeframe_ms(cfg.timeframe)
        self._bars = self._build_bars(n_bars)
        self._mark, self._index, self._oi, self._spread = self._build_point_in_time()
        self._funding = self._build_funding(n_bars)

    def _build_bars(self, n_bars: int) -> list[dict]:
        bars: list[dict] = []
        price = 100.0
        for i in range(n_bars):
            r = (_unit(self.seed, "ret", i) - 0.5) * 0.004  # zero-mean i.i.d.
            prev = price
            price = max(price * (1.0 + r), 1e-6)
            hi = max(prev, price) * (1.0 + _unit(self.seed, "hi", i) * 0.001)
            lo = min(prev, price) * (1.0 - _unit(self.seed, "lo", i) * 0.001)
            vol = 100.0 + _unit(self.seed, "vol", i) * 50.0
            bars.append(
                {
                    "ts": i * self.iv,
                    "open": prev,
                    "high": hi,
                    "low": lo,
                    "close": price,
                    "volume": vol,
                }
            )
        return bars

    def _build_point_in_time(self) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        mark, index, oi, spread = [], [], [], []
        for b in self._bars:
            ts, c = b["ts"], b["close"]
            mark.append(
                {"ts": ts, "mark_price": c * (1.0 + (_unit(self.seed, "mk", ts) - 0.5) * 6e-4)}
            )
            index.append(
                {"ts": ts, "index_price": c * (1.0 + (_unit(self.seed, "ix", ts) - 0.5) * 4e-4)}
            )
            oi.append({"ts": ts, "open_interest": 1e6 * (1.0 + _unit(self.seed, "oi", ts))})
            frac = 0.0002 + _unit(self.seed, "sp", ts) * 0.0008
            spread.append(
                {
                    "ts": ts,
                    "bid": c * (1 - frac / 2),
                    "ask": c * (1 + frac / 2),
                    "spread": c * frac,
                    "spread_bps": frac * 1e4,
                }
            )
        return mark, index, oi, spread

    def _build_funding(self, n_bars: int) -> list[dict]:
        span = n_bars * self.iv
        funding: list[dict] = []
        ts = 0
        while ts < span:
            funding.append(
                {
                    "ts": ts,
                    "funding_rate": (_unit(self.seed, "fr", ts) - 0.5) * 0.001,
                    "funding_interval_hours": 8,
                }
            )
            ts += _FUNDING_IV_MS
        return funding

    def ohlcv(self, symbol: str) -> list[dict]:
        return list(self._bars)

    def series(self, symbol: str, data_type: str) -> list[dict]:
        return {
            "mark": self._mark,
            "index": self._index,
            "open_interest": self._oi,
            "spread": self._spread,
            "funding": self._funding,
        }.get(data_type, [])


def synthetic_leakage_report(cfg: FeatureConfig) -> dict:
    """Build features+labels on synthetic noise and measure signal expectancy."""
    reader = SyntheticReader(cfg, cfg.leakage.synthetic_bars)
    frame = compute_features("SYNTH/USDT:USDT", reader, cfg)
    labels = forward_labels(frame.closes(), cfg.label_horizon)
    signals = momentum_signals(frame)
    stats = expectancy_z(signals, labels)
    passed = abs(stats["z"]) <= cfg.leakage.max_synthetic_expectancy_z
    return {
        **stats,
        "max_abs_z": cfg.leakage.max_synthetic_expectancy_z,
        "passed": passed,
        "rows": len(frame.rows),
    }


# --------------------------------------------------------------------------- #
# Causal invariance (no look-ahead)                                            #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CausalViolation:
    symbol: str
    bar_ts: int
    feature: str
    full_value: float
    truncated_value: float


def causal_invariance_violations(
    symbol: str,
    reader: FeatureDataReader,
    cfg: FeatureConfig,
    sample_indices: list[int] | None = None,
    tol: float = 1e-9,
    compute_fn: ComputeFn = compute_features,
) -> list[CausalViolation]:
    """Recompute sampled rows from future-truncated inputs; report any drift.

    ``compute_fn`` defaults to the production pipeline; tests pass a deliberately
    leaky compute to prove this harness actually catches look-ahead."""
    full = compute_fn(symbol, reader, cfg)
    if not full.rows:
        return []
    if sample_indices is None:
        # A spread of rows: first valid, last, and evenly-spaced interior rows.
        m = len(full.rows)
        sample_indices = sorted({0, m - 1, *[round(m * f) for f in (0.25, 0.5, 0.75)]})
        sample_indices = [i for i in sample_indices if 0 <= i < m]

    violations: list[CausalViolation] = []
    for idx in sample_indices:
        row = full.rows[idx]
        treader = TruncatedReader(reader, max_ohlcv_ts=row["ts"], max_series_ts=row["decision_ts"])
        tframe = compute_fn(symbol, treader, cfg)
        if not tframe.rows:
            continue
        last = tframe.rows[-1]
        if last["ts"] != row["ts"]:
            violations.append(
                CausalViolation(symbol, row["ts"], "decision_ts", row["ts"], last["ts"])
            )
            continue
        for name in full.feature_names:
            if abs(float(last[name]) - float(row[name])) > tol:
                violations.append(
                    CausalViolation(symbol, row["ts"], name, float(row[name]), float(last[name]))
                )
    return violations
