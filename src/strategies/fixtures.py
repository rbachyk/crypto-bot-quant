"""Deterministic reference fixtures for the Phase 5 research candidates.

There is no live/real market data offline, so — exactly as the Phase 4 engine
self-test does — each family is validated on a fully reproducible, no-network
series that PLANTS the family's hypothesised causal structure (Section 12). The
strategy code is real; the data is a labelled synthetic fixture. Every fixture is
strictly causal (a feature row for bar ``k`` depends only on data at or before the
decision time ``(k+1)·iv``), so a past-only strategy can capture the planted edge
and the no-look-ahead guards hold.

Each fixture also has a ``noise`` control (``edge=False``) with the structure
removed: a causal strategy must show ~0 expectancy there — the engine-level
look-ahead / leakage guard (mirrors the FEAT gate's synthetic test).

Builders return ``list[SymbolInput]`` produced through the ONE feature pipeline
(the Parity Rule, Section 10) — the identical path live/paper/backtest use.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from src.backtest.engine import SymbolInput
from src.data.schema import FUNDING, INDEX, MARK, OPEN_INTEREST, SPREAD, timeframe_ms
from src.features.config import FeatureConfig, load_feature_config
from src.features.pipeline import FeatureDataReader, compute_features
from src.strategies.config import CandidateConfig

_FUNDING_IV_MS = 8 * 3_600_000


def _unit(*parts: object) -> float:
    """Deterministic pseudo-random uniform in [0, 1) from a stable hash."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _sym(*parts: object) -> float:
    """Symmetric deterministic pseudo-random in [-1, 1)."""
    return (_unit(*parts) - 0.5) * 2.0


# --------------------------------------------------------------------------- #
# Generic reader over precomputed bars + point-in-time series                 #
# --------------------------------------------------------------------------- #
class _FixtureReader(FeatureDataReader):
    """Serves precomputed OHLCV + point-in-time series for one symbol."""

    def __init__(
        self,
        bars: list[dict],
        mark: list[dict],
        index: list[dict],
        oi: list[dict],
        spread: list[dict],
        funding: list[dict],
    ) -> None:
        self._bars = bars
        self._mark = mark
        self._index = index
        self._oi = oi
        self._spread = spread
        self._funding = funding

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


def _ohlcv_from_returns(seed: str, returns: list[float], iv: int, base: float = 100.0) -> list[dict]:
    """Build OHLCV bars from per-bar returns with realistic intrabar wicks.

    ``returns[i]`` is bar ``i``'s close-over-close return (``returns[0]`` ignored).
    Wicks scale with the bar's own move so stops/TPs realistically trigger inside
    a bar (the engine manages exits against high/low).
    """
    bars: list[dict] = []
    price = base
    for i, r in enumerate(returns):
        prev = price
        price = max(prev * (1.0 + r), 1e-6)
        wick = (0.25 + _unit(seed, "wk", i)) * abs(r) * price + price * 1e-4
        hi = max(prev, price) + wick
        lo = max(min(prev, price) - wick, 1e-6)
        vol = 5000.0 + _unit(seed, "vol", i) * 5000.0
        bars.append(
            {"ts": i * iv, "open": prev, "high": hi, "low": lo, "close": price, "volume": vol}
        )
    return bars


def _point_in_time(
    seed: str, bars: list[dict], premium: list[float] | None = None
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Mark/index/OI/spread samples aligned to each bar.

    When ``premium`` is given (Family B), ``mark = index·(1+premium[i])`` so the
    decision-time ``premium`` feature equals the planted deviation; otherwise
    mark/index carry only tiny independent noise (premium is not that family's
    edge).
    """
    mark, index, oi, spread = [], [], [], []
    for i, b in enumerate(bars):
        ts, c = b["ts"], b["close"]
        idx_px = c * (1.0 + _sym(seed, "ix", ts) * 1.5e-4)
        if premium is not None:
            mk_px = idx_px * (1.0 + premium[i])
        else:
            mk_px = c * (1.0 + _sym(seed, "mk", ts) * 1.5e-4)
        index.append({"ts": ts, "index_price": idx_px})
        mark.append({"ts": ts, "mark_price": mk_px})
        oi.append({"ts": ts, "open_interest": 1e7 * (1.0 + _unit(seed, "oi", ts))})
        frac = 0.0002 + _unit(seed, "sp", ts) * 0.0006
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


def _funding(seed: str, n_bars: int, iv: int) -> list[dict]:
    span = n_bars * iv
    out: list[dict] = []
    ts = 0
    while ts < span:
        out.append(
            {
                "ts": ts,
                "funding_rate": _sym(seed, "fr", ts) * 0.0003,
                "funding_interval_hours": 8,
            }
        )
        ts += _FUNDING_IV_MS
    return out


# --------------------------------------------------------------------------- #
# Family A — Cross-Asset Lead-Lag                                              #
# --------------------------------------------------------------------------- #
def _leader_returns(seed: str, n: int, drift: float, period: int, sigma: float) -> list[float]:
    import math

    phase = _unit(seed, "phase") * period
    rets = [0.0]
    for i in range(1, n):
        d = drift * math.sin(2.0 * math.pi * (i + phase) / max(2, period))
        rets.append(d + _sym(seed, "lret", i) * sigma)
    return rets


def _lead_lag_returns(cand: CandidateConfig, edge: bool) -> dict[str, list[float]]:
    f = cand.fixture
    v = f.values
    n = f.bars
    leader = str(v["leader"])
    followers = [str(s) for s in v["followers"]]
    lret = _leader_returns(
        f.seed, n, float(v["leader_drift"]), int(v["leader_period_bars"]), float(v["leader_sigma"])
    )
    out: dict[str, list[float]] = {leader: lret}
    lag_beta = float(v["lag_beta"]) if edge else 0.0
    fsig = float(v["follower_sigma"])
    for fo in followers:
        rets = [0.0]
        for i in range(1, n):
            lagged = lag_beta * lret[i - 1]  # follower follows the leader's PREVIOUS bar
            rets.append(lagged + _sym(f.seed, fo, "n", i) * fsig)
        out[fo] = rets
    return out


# --------------------------------------------------------------------------- #
# Family B — Perpetual Premium / Basis Mean Reversion                         #
# --------------------------------------------------------------------------- #
def _basis_series(cand: CandidateConfig, edge: bool) -> dict[str, tuple[list[float], list[float]]]:
    """Per symbol: (returns, premium). r[m] = -kappa(sign)·d[m] + noise; premium = d."""
    f = cand.fixture
    v = f.values
    n = f.bars
    symbols = [str(s) for s in v["symbols"]]
    rho = float(v["rho"])
    shock = float(v["shock_sigma"])
    kappa_rich = float(v["kappa_rich"]) if edge else 0.0
    kappa_cheap = float(v["kappa_cheap"]) if edge else 0.0
    noise = float(v["noise_sigma"])
    out: dict[str, tuple[list[float], list[float]]] = {}
    for sym in symbols:
        d = 0.0
        prem = [0.0]
        rets = [0.0]
        # d[0] established for bar 0; bar 0 return is 0 (base price).
        d = shock * _sym(f.seed, sym, "d", 0)
        prem[0] = d
        for i in range(1, n):
            d = rho * d + shock * _sym(f.seed, sym, "d", i)
            prem.append(d)
            kappa = kappa_rich if d > 0 else kappa_cheap
            rets.append(-kappa * d + _sym(f.seed, sym, "rn", i) * noise)
        out[sym] = (rets, prem)
    return out


# --------------------------------------------------------------------------- #
# Family G — Cross-Sectional Relative Strength / Dispersion                    #
# --------------------------------------------------------------------------- #
def _xsection_returns(cand: CandidateConfig, edge: bool) -> dict[str, list[float]]:
    """r_i[m] = market[m] + idio_i[m] + noise; idio is a persistent AR(1) (the edge)."""
    f = cand.fixture
    v = f.values
    n = f.bars
    symbols = [str(s) for s in v["symbols"]]
    msig = float(v["market_sigma"])
    rho = float(v["idio_rho"]) if edge else 0.0
    shock = float(v["idio_shock"]) if edge else 0.0
    noise = float(v["idio_noise"])
    market = [0.0] + [_sym(f.seed, "mkt", i) * msig for i in range(1, n)]
    idio = {s: 0.0 for s in symbols}
    out: dict[str, list[float]] = {s: [0.0] for s in symbols}
    for i in range(1, n):
        for s in symbols:
            idio[s] = rho * idio[s] + shock * _sym(f.seed, s, "idio", i)
            out[s].append(market[i] + idio[s] + _sym(f.seed, s, "n", i) * noise)
    return out


# --------------------------------------------------------------------------- #
# Builder                                                                      #
# --------------------------------------------------------------------------- #
def _feature_config(timeframe: str) -> FeatureConfig:
    base = load_feature_config()
    if base.timeframe == timeframe:
        return base
    return replace(base, timeframe=timeframe)


def _build_inputs(
    cand: CandidateConfig,
    returns_by_symbol: dict[str, list[float]],
    premium_by_symbol: dict[str, list[float]] | None,
) -> list[SymbolInput]:
    iv = timeframe_ms(cand.fixture.timeframe)
    feat_cfg = _feature_config(cand.fixture.timeframe)
    inputs: list[SymbolInput] = []
    for symbol, rets in returns_by_symbol.items():
        seed = f"{cand.fixture.seed}:{symbol}"
        bars = _ohlcv_from_returns(seed, rets, iv)
        premium = premium_by_symbol.get(symbol) if premium_by_symbol else None
        mark, index, oi, spread = _point_in_time(seed, bars, premium=premium)
        funding = _funding(seed, cand.fixture.bars, iv)
        reader = _FixtureReader(bars, mark, index, oi, spread, funding)
        frame = compute_features(symbol, reader, feat_cfg)
        inputs.append(
            SymbolInput(
                symbol=symbol,
                bars=bars,
                frame=frame,
                spread_samples=[{"ts": s["ts"], "spread_bps": s["spread_bps"]} for s in spread],
                funding_events=[
                    {"ts": fnd["ts"], "funding_rate": fnd["funding_rate"]} for fnd in funding
                ],
                activation_ts=0,
            )
        )
    return inputs


_INPUTS_CACHE: dict[tuple[str, bool], list[SymbolInput]] = {}


def build_candidate_inputs(cand: CandidateConfig, *, edge: bool = True) -> list[SymbolInput]:
    """Per-symbol engine inputs for a candidate's fixture (edge or noise control).

    Memoized: the fixture is fully deterministic and the engine treats inputs as
    read-only, so repeated gate / walk-forward / stress runs reuse one build.
    """
    key = (cand.id, edge)
    cached = _INPUTS_CACHE.get(key)
    if cached is not None:
        return cached

    if cand.family == "A":
        inputs = _build_inputs(cand, _lead_lag_returns(cand, edge), None)
    elif cand.family == "B":
        series = _basis_series(cand, edge)
        returns = {s: r for s, (r, _p) in series.items()}
        premium = {s: p for s, (_r, p) in series.items()}
        inputs = _build_inputs(cand, returns, premium)
    elif cand.family == "G":
        inputs = _build_inputs(cand, _xsection_returns(cand, edge), None)
    else:
        raise ValueError(f"no fixture for family {cand.family!r}")

    _INPUTS_CACHE[key] = inputs
    return inputs
