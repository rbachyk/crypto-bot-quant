"""Unit tests for the real ccxt-backed data source + exchange adapter.

A fake ccxt client is injected via the ``client=`` constructor arg so the whole
surface (pagination, grid-alignment, dedup, schema mapping) is covered without
any network. The offline ``skeleton`` default is exercised elsewhere; here we
prove the live path maps an exchange's responses into the canonical schema.
"""

from __future__ import annotations

import math

import pytest
from src.data.ccxt_source import CcxtDataSource
from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    OPEN_INTEREST,
    SPREAD,
    TIMEFRAME_MS,
    SeriesKey,
)
from src.exchange.ccxt_adapter import CcxtExchangeAdapter

_PAGE_CAP = 3  # force multi-page pagination regardless of the requested limit


class FakeBybit:
    """Minimal deterministic stand-in for a ccxt exchange client."""

    def __init__(self) -> None:
        self.markets = {
            "BTC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "settle": "USDT",
                "active": True,
                "precision": {"price": 0.1, "amount": 0.001},
                "limits": {
                    "leverage": {"max": 100},
                    "amount": {"min": 0.001},
                    "cost": {"min": 5.0},
                },
                "maker": 0.0002,
                "taker": 0.00055,
                "info": {"fundingInterval": 480},  # minutes -> 8h
            },
            "ETH/USDT:USDT": {
                "swap": True,
                "linear": True,
                "settle": "USDT",
                "active": True,
                "precision": {"price": 0.01, "amount": 0.01},
                "limits": {"leverage": {"max": 50}},
                "maker": 0.0002,
                "taker": 0.00055,
                "info": {"fundingInterval": 480},
            },
            "BTC/USD:BTC": {  # inverse — must be filtered out
                "swap": True,
                "linear": False,
                "settle": "BTC",
                "active": True,
            },
            "DOGE/USDT:USDT": {  # inactive — must be filtered out
                "swap": True,
                "linear": True,
                "settle": "USDT",
                "active": False,
            },
        }

    def load_markets(self) -> dict:
        return self.markets

    def parse_timeframe(self, timeframe: str) -> int:
        return TIMEFRAME_MS[timeframe] // 1000

    def fetch_ohlcv(self, symbol, timeframe, since=0, limit=1000, params=None):
        params = params or {}
        iv = TIMEFRAME_MS[timeframe]
        start = ((since + iv - 1) // iv) * iv  # snap up to grid
        # mark/index use a recognisably different price so we can assert mapping.
        bump = {"mark": 1.0, "index": 2.0}.get(params.get("price"), 0.0)
        out = []
        for i in range(_PAGE_CAP):
            ts = start + i * iv
            close = 100.0 + bump + (ts / iv)
            out.append([ts, close - 0.5, close + 1.0, close - 1.0, close, 10.0 + i])
        return out

    def fetch_funding_rate_history(self, symbol, since=0, limit=200):
        iv = TIMEFRAME_MS["8h"]
        start = ((since + iv - 1) // iv) * iv
        return [
            {"timestamp": start + i * iv, "fundingRate": 0.0001 * (i + 1)} for i in range(_PAGE_CAP)
        ]

    def fetch_open_interest_history(self, symbol, timeframe, since=0, limit=200):
        iv = TIMEFRAME_MS[timeframe]
        start = ((since + iv - 1) // iv) * iv
        return [
            {"timestamp": start + i * iv, "openInterestAmount": 1_000_000.0 + i}
            for i in range(_PAGE_CAP)
        ]


@pytest.fixture
def source() -> CcxtDataSource:
    return CcxtDataSource("bybit", client=FakeBybit())


def _window(timeframe: str, n: int) -> tuple[int, int]:
    iv = TIMEFRAME_MS[timeframe]
    return 0, n * iv


# --------------------------------------------------------------------------- #
# CcxtDataSource                                                               #
# --------------------------------------------------------------------------- #
def test_has_symbol(source: CcxtDataSource) -> None:
    assert source.has_symbol("BTC/USDT:USDT")
    assert not source.has_symbol("NOPE/USDT:USDT")


def test_ohlcv_paginates_and_aligns(source: CcxtDataSource) -> None:
    start, end = _window("1m", 10)
    key = SeriesKey("bybit", OHLCV, "BTC/USDT:USDT", "1m")
    rows = source.fetch(key, start, end)
    iv = TIMEFRAME_MS["1m"]
    assert [r["ts"] for r in rows] == list(range(0, end, iv))  # 10 rows, multi-page
    assert all(set(r) == set(key.columns) for r in rows)
    assert all(r["ts"] % iv == 0 for r in rows)
    assert rows == sorted(rows, key=lambda r: r["ts"])


def test_ohlcv_excludes_out_of_range(source: CcxtDataSource) -> None:
    iv = TIMEFRAME_MS["1m"]
    key = SeriesKey("bybit", OHLCV, "BTC/USDT:USDT", "1m")
    rows = source.fetch(key, iv, 5 * iv)  # [1m, 5m)
    assert [r["ts"] for r in rows] == [iv, 2 * iv, 3 * iv, 4 * iv]


def test_mark_and_index_use_close(source: CcxtDataSource) -> None:
    start, end = _window("5m", 4)
    mark = source.fetch(SeriesKey("bybit", MARK, "BTC/USDT:USDT", "5m"), start, end)
    index = source.fetch(SeriesKey("bybit", INDEX, "BTC/USDT:USDT", "5m"), start, end)
    assert all("mark_price" in r for r in mark)
    assert all("index_price" in r for r in index)
    # mark gets +1.0 bump, index +2.0 (per FakeBybit) at ts=0 -> close 100/101/102.
    assert mark[0]["mark_price"] == pytest.approx(101.0)
    assert index[0]["index_price"] == pytest.approx(102.0)


def test_funding_rows(source: CcxtDataSource) -> None:
    iv = TIMEFRAME_MS["8h"]
    key = SeriesKey("bybit", FUNDING, "BTC/USDT:USDT", "8h")
    rows = source.fetch(key, 0, 3 * iv)
    assert [r["ts"] for r in rows] == [0, iv, 2 * iv]
    assert all(r["funding_interval_hours"] == 8 for r in rows)
    assert rows[0]["funding_rate"] == pytest.approx(0.0001)


def test_open_interest_rows(source: CcxtDataSource) -> None:
    start, end = _window("5m", 3)
    key = SeriesKey("bybit", OPEN_INTEREST, "BTC/USDT:USDT", "5m")
    rows = source.fetch(key, start, end)
    assert [r["ts"] for r in rows] == [0, TIMEFRAME_MS["5m"], 2 * TIMEFRAME_MS["5m"]]
    assert all(r["open_interest"] >= 1_000_000.0 for r in rows)


def test_spread_is_estimated(source: CcxtDataSource) -> None:
    start, end = _window("5m", 3)
    key = SeriesKey("bybit", SPREAD, "BTC/USDT:USDT", "5m")
    rows = source.fetch(key, start, end)
    assert all(set(r) == set(key.columns) for r in rows)
    for r in rows:
        assert r["ask"] > r["bid"] > 0
        assert math.isclose(r["spread"], r["ask"] - r["bid"], rel_tol=1e-9)
        assert r["spread_bps"] == pytest.approx(5.0)  # default conservative estimate


def test_spread_bps_configurable() -> None:
    src = CcxtDataSource("bybit", estimated_spread_bps=12.0, client=FakeBybit())
    iv = TIMEFRAME_MS["5m"]
    rows = src.fetch(SeriesKey("bybit", SPREAD, "BTC/USDT:USDT", "5m"), 0, iv)
    assert rows[0]["spread_bps"] == pytest.approx(12.0)


def test_ping(source: CcxtDataSource) -> None:
    assert source.ping() is True


# --------------------------------------------------------------------------- #
# CcxtExchangeAdapter                                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def adapter() -> CcxtExchangeAdapter:
    return CcxtExchangeAdapter("bybit", client=FakeBybit())


def test_fetch_symbols_filters_to_active_usdt_linear(adapter: CcxtExchangeAdapter) -> None:
    assert adapter.fetch_symbols() == ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def test_fetch_metadata_maps_fields(adapter: CcxtExchangeAdapter) -> None:
    md = adapter.fetch_metadata("BTC/USDT:USDT")
    assert md.tick_size == pytest.approx(0.1)
    assert md.qty_step == pytest.approx(0.001)
    assert md.price_precision == 1
    assert md.max_leverage == 100
    assert md.min_notional == pytest.approx(5.0)
    assert md.maker_fee == pytest.approx(0.0002)
    assert md.taker_fee == pytest.approx(0.00055)
    assert md.funding_interval_hours == 8
    assert md.status == "trading"
    assert md.verification_status == "UNVERIFIED"  # never auto-verified


def test_fetch_metadata_unknown_symbol(adapter: CcxtExchangeAdapter) -> None:
    with pytest.raises(KeyError):
        adapter.fetch_metadata("NOPE/USDT:USDT")


def test_adapter_ping(adapter: CcxtExchangeAdapter) -> None:
    assert adapter.ping() is True
