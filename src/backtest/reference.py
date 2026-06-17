"""Deterministic reference market data for the Phase 4 engine self-test.

The backtest engine, walk-forward and stress machinery (and the BT/WF/FEE/SLIP
gates) need *something* to run on before Phase 5 ships validated strategies. This
module fabricates a fully reproducible, offline (no-network) market series with
two modes:

* ``edge="trend"`` — returns follow an AR(1) process (``r_t = φ·r_{t-1} + ε_t``),
  a genuine, **causal** momentum edge a past-only strategy can capture. This is
  what lets the engine demonstrate a positive net-of-cost expectancy that
  survives walk-forward, ×2 fees and +50% slippage.
* ``edge="noise"`` — ``φ = 0``: i.i.d. zero-mean returns with no structure. A
  causal strategy must show ~0 expectancy here — the engine-level look-ahead /
  leakage guard (the mirror of the FEAT gate's synthetic test).

The series is NOT real market data and is never used for live decisions; it is a
labelled test fixture (Section 19 backtest must be event-based + leakage-free).
Each symbol is seeded independently so per-symbol breakdowns are meaningful and a
point-in-time universe (``activation_bar``) can be exercised.
"""

from __future__ import annotations

import hashlib
import math

from src.backtest.config import ReferenceDataConfig
from src.data.schema import FUNDING, INDEX, MARK, OPEN_INTEREST, SPREAD, timeframe_ms
from src.features.pipeline import FeatureDataReader

_FUNDING_IV_MS = 8 * 3_600_000


def _unit(*parts: object) -> float:
    """Deterministic pseudo-random uniform in [0, 1) from a stable hash."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class ReferenceReader(FeatureDataReader):
    """Offline reader serving a deterministic AR(1) (or noise) series per symbol."""

    def __init__(self, symbol: str, cfg: ReferenceDataConfig) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self.iv = timeframe_ms(cfg.timeframe)
        self.seed = f"{cfg.seed}:{symbol}"
        self.is_trend = cfg.edge == "trend"
        # A per-symbol phase offset so symbols are correlated-but-distinct trends
        # (meaningful per-symbol breakdowns; no two symbols are identical).
        self.phase = _unit(self.seed, "phase") * cfg.trend_period_bars
        self._bars = self._build_bars(cfg.bars)
        self._mark, self._index, self._oi, self._spread = self._build_point_in_time()
        self._funding = self._build_funding(cfg.bars)

    # -- generation ------------------------------------------------------ #
    def _build_bars(self, n_bars: int) -> list[dict]:
        bars: list[dict] = []
        price = 100.0
        period = max(2, self.cfg.trend_period_bars)
        for i in range(n_bars):
            eps = (_unit(self.seed, "ret", i) - 0.5) * 2.0 * self.cfg.base_sigma
            # Causal, regime-switching drift: a slow sinusoid that the momentum
            # strategy can ride on both sides. The drift at bar i depends only on
            # i (no future dependence), so the pipeline stays look-ahead-free.
            drift = 0.0
            if self.is_trend:
                drift = self.cfg.trend_drift * math.sin(2.0 * math.pi * (i + self.phase) / period)
            r = drift + eps
            prev = price
            price = max(price * (1.0 + r), 1e-6)
            # Intrabar range scales with the bar's own move plus a small wick, so
            # stops/take-profits realistically trigger inside a bar.
            wick = (0.25 + _unit(self.seed, "wk", i)) * abs(r) * price + price * 1e-4
            hi = max(prev, price) + wick
            lo = max(min(prev, price) - wick, 1e-6)
            vol = 5000.0 + _unit(self.seed, "vol", i) * 5000.0
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
                {"ts": ts, "mark_price": c * (1.0 + (_unit(self.seed, "mk", ts) - 0.5) * 4e-4)}
            )
            index.append(
                {"ts": ts, "index_price": c * (1.0 + (_unit(self.seed, "ix", ts) - 0.5) * 3e-4)}
            )
            oi.append({"ts": ts, "open_interest": 1e7 * (1.0 + _unit(self.seed, "oi", ts))})
            frac = 0.0002 + _unit(self.seed, "sp", ts) * 0.0006
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
                    "funding_rate": (_unit(self.seed, "fr", ts) - 0.5) * 0.0006,
                    "funding_interval_hours": 8,
                }
            )
            ts += _FUNDING_IV_MS
        return funding

    # -- FeatureDataReader ----------------------------------------------- #
    def ohlcv(self, symbol: str) -> list[dict]:
        return list(self._bars)

    def series(self, symbol: str, data_type: str) -> list[dict]:
        return {
            MARK: self._mark,
            INDEX: self._index,
            OPEN_INTEREST: self._oi,
            SPREAD: self._spread,
            FUNDING: self._funding,
        }.get(data_type, [])

    # -- helpers for the engine ------------------------------------------ #
    def spread_bps_at(self, decision_ts: int) -> float:
        """Last spread sample with ts <= decision_ts (modelled spread at decision)."""
        bps = 2.0
        for row in self._spread:
            if row["ts"] > decision_ts:
                break
            bps = float(row["spread_bps"])
        return bps

    def funding_events(self) -> list[dict]:
        return list(self._funding)
