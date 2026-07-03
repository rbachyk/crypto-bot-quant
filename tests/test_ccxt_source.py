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


def test_mark_and_index_use_close_stamped_at_kline_close(source: CcxtDataSource) -> None:
    """A kline's close value is observable at the kline CLOSE, so the sample at ``ts`` carries
    the close of the kline that OPENED at ``ts - iv`` (never the bar closing after ``ts``)."""
    iv = TIMEFRAME_MS["5m"]
    start, end = _window("5m", 4)
    mark = source.fetch(SeriesKey("bybit", MARK, "BTC/USDT:USDT", "5m"), start, end)
    index = source.fetch(SeriesKey("bybit", INDEX, "BTC/USDT:USDT", "5m"), start, end)
    assert all("mark_price" in r for r in mark)
    assert all("index_price" in r for r in index)
    # No kline exists before open=0, so the first stamp is its close time ``iv`` (not 0).
    assert [r["ts"] for r in mark] == [iv, 2 * iv, 3 * iv]
    assert [r["ts"] for r in index] == [iv, 2 * iv, 3 * iv]
    # mark gets +1.0 bump, index +2.0 (per FakeBybit): kline open=0 closes 101/102, stamped iv.
    assert mark[0]["mark_price"] == pytest.approx(101.0)
    assert index[0]["index_price"] == pytest.approx(102.0)
    # And the value at each stamp is the PREVIOUS kline's close — never the one closing later.
    assert mark[1]["mark_price"] == pytest.approx(102.0)  # close of kline open=iv


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
    iv = TIMEFRAME_MS["5m"]
    start, end = _window("5m", 3)
    key = SeriesKey("bybit", SPREAD, "BTC/USDT:USDT", "5m")
    rows = source.fetch(key, start, end)
    # Estimated from each kline close ⇒ stamped at the kline CLOSE time, like mark/index.
    assert [r["ts"] for r in rows] == [iv, 2 * iv]
    assert all(set(r) == set(key.columns) for r in rows)
    for r in rows:
        assert r["ask"] > r["bid"] > 0
        assert math.isclose(r["spread"], r["ask"] - r["bid"], rel_tol=1e-9)
        assert r["spread_bps"] == pytest.approx(5.0)  # default conservative estimate


def test_spread_bps_configurable() -> None:
    src = CcxtDataSource("bybit", estimated_spread_bps=12.0, client=FakeBybit())
    iv = TIMEFRAME_MS["5m"]
    rows = src.fetch(SeriesKey("bybit", SPREAD, "BTC/USDT:USDT", "5m"), 0, 2 * iv)
    assert rows[0]["spread_bps"] == pytest.approx(12.0)


def test_ping(source: CcxtDataSource) -> None:
    assert source.ping() is True


# --------------------------------------------------------------------------- #
# Closed-bar guard: the venue also serves the STILL-FORMING kline; storing it   #
# would freeze a partial candle forever (append-only keep-first dedup), so the  #
# fetch layer must drop any kline whose close time has not passed — for every   #
# kline-derived series, regardless of the caller's window math (C1 regression). #
# --------------------------------------------------------------------------- #
class _FormingBarClient:
    """Serves three 4h klines, the LAST being the current still-forming bar."""

    def __init__(self, opens: list[int]) -> None:
        self.opens = opens

    def load_markets(self) -> dict:
        return {"BTC/USDT:USDT": {}}

    def fetch_ohlcv(self, symbol, timeframe, since=0, limit=1000, params=None):  # noqa: A002
        return [[o, 100.0, 101.0, 99.0, 100.5 + o, 10.0] for o in self.opens if o >= since]


def _forming_4h_window() -> tuple[list[int], int, int]:
    """Three 4h opens ending at the CURRENT (still-forming) bar + a window reaching into it."""
    import time

    iv = TIMEFRAME_MS["4h"]
    cur_open = (int(time.time() * 1000) // iv) * iv  # closes in the future
    opens = [cur_open - 2 * iv, cur_open - iv, cur_open]
    return opens, opens[0], cur_open + iv


def test_still_forming_4h_kline_is_not_emitted() -> None:
    opens, start, end = _forming_4h_window()
    src = CcxtDataSource("bybit", client=_FormingBarClient(opens))
    rows = src.fetch(SeriesKey("bybit", OHLCV, "BTC/USDT:USDT", "4h"), start, end)
    # Only the two CLOSED bars come through; the in-progress bar is dropped at the fetch layer.
    assert [r["ts"] for r in rows] == opens[:2]


def test_still_forming_4h_kline_is_never_stored(tmp_path) -> None:
    """End-to-end belt: ingesting a window that reaches into the forming bar must not freeze a
    partial candle in the append-only store (incremental update would never re-read it)."""
    from src.data.ingest import Ingestor
    from src.data.store import SeriesStore

    opens, start, end = _forming_4h_window()
    src = CcxtDataSource("bybit", client=_FormingBarClient(opens))
    store = SeriesStore(tmp_path / "lake")
    key = SeriesKey("bybit", OHLCV, "BTC/USDT:USDT", "4h")
    assert Ingestor(src, store).download(key, start, end) == 2
    assert opens[2] not in store.timestamps(key, start, end)  # the forming bar is absent
    # latest_ts stays on the last CLOSED bar, so a later incremental update re-fetches the
    # in-progress bar once it has closed instead of freezing the partial version forever.
    assert store.latest_ts(key) == opens[1]


def test_forming_kline_guard_applies_to_kline_derived_point_series() -> None:
    """mark (and index/spread) samples come from klines too: no sample may be stamped at a time
    that has not been reached — the forming kline contributes nothing."""
    import time

    opens, start, end = _forming_4h_window()
    src = CcxtDataSource("bybit", client=_FormingBarClient(opens))
    rows = src.fetch(SeriesKey("bybit", MARK, "BTC/USDT:USDT", "4h"), start, end)
    # Closed klines only: opens[0]/[1] close at opens[1]/[2] — those are the stamps.
    assert [r["ts"] for r in rows] == [opens[1], opens[2]]
    assert all(r["ts"] <= int(time.time() * 1000) for r in rows)  # nothing from the future


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


# --------------------------------------------------------------------------- #
# Rate-limit resilience: Bybit returns retCode 10006 ("Too many visits") under  #
# burst load; the source must back off + retry instead of failing the download. #
# --------------------------------------------------------------------------- #
class RateLimitExceeded(Exception):
    """Stands in for ccxt.RateLimitExceeded (matched by class name AND message in _call)."""


class _FlakyClient:
    """Raises a rate-limit error the first ``fail_times`` ohlcv calls, then serves one bar."""

    def __init__(self, fail_times: int) -> None:
        self._left = fail_times
        self.calls = 0

    def load_markets(self) -> dict:
        return {"BTC/USDT:USDT": {}}

    def parse_timeframe(self, tf: str) -> int:
        return TIMEFRAME_MS[tf] // 1000

    def fetch_ohlcv(self, symbol, timeframe, since=0, limit=1000, params=None):  # noqa: A002
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise RateLimitExceeded(
                'bybit {"retCode":10006,"retMsg":"Too many visits. Exceeded the API Rate Limit."}'
            )
        return [[0, 1.0, 1.0, 1.0, 1.0, 1.0]] if since == 0 else []


def test_call_backs_off_and_retries_then_succeeds() -> None:
    client = _FlakyClient(fail_times=3)
    src = CcxtDataSource(
        "bybit", client=client, max_retries=5, retry_base_sec=0.0, retry_max_sec=0.0
    )
    rows = src.fetch(SeriesKey("bybit", OHLCV, "BTC/USDT:USDT", "5m"), 0, 5 * TIMEFRAME_MS["5m"])
    assert client.calls >= 4  # 3 rate-limit failures retried, then a success
    assert rows  # data came through after the backoff


class _SystemBusyClient:
    """Raises a GENERIC exception carrying a Bybit retCode (10016 system busy) — i.e. a throttle
    ccxt wrapped as a non-RateLimit type, classified retryable by message, not type."""

    def __init__(self, fail_times: int) -> None:
        self._left = fail_times
        self.calls = 0

    def load_markets(self) -> dict:
        return {"BTC/USDT:USDT": {}}

    def parse_timeframe(self, tf: str) -> int:
        return TIMEFRAME_MS[tf] // 1000

    def fetch_ohlcv(self, symbol, timeframe, since=0, limit=1000, params=None):  # noqa: A002
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise Exception('bybit {"retCode":10016,"retMsg":"System busy, please try again"}')
        return [[0, 1.0, 1.0, 1.0, 1.0, 1.0]] if since == 0 else []


def test_call_retries_bybit_throttle_by_message_not_type() -> None:
    client = _SystemBusyClient(fail_times=2)
    src = CcxtDataSource(
        "bybit", client=client, max_retries=5, retry_base_sec=0.0, retry_max_sec=0.0
    )
    rows = src.fetch(SeriesKey("bybit", OHLCV, "BTC/USDT:USDT", "5m"), 0, 5 * TIMEFRAME_MS["5m"])
    assert client.calls >= 3 and rows  # 10016 retried despite the generic exception type


def test_call_exhausts_retries_then_raises() -> None:
    client = _FlakyClient(fail_times=999)  # never recovers
    src = CcxtDataSource("bybit", client=client, max_retries=2, retry_base_sec=0.0)
    with pytest.raises(RateLimitExceeded):
        src.fetch(SeriesKey("bybit", OHLCV, "BTC/USDT:USDT", "5m"), 0, TIMEFRAME_MS["5m"])
    assert client.calls == 3  # initial attempt + 2 retries


def test_call_reraises_non_retryable_immediately() -> None:
    class _Boom(Exception):
        pass

    class _Client:
        def parse_timeframe(self, tf: str) -> int:
            return TIMEFRAME_MS[tf] // 1000

        def fetch_ohlcv(self, *a, **k):
            raise _Boom("a real bug, not a rate limit")

    src = CcxtDataSource("bybit", client=_Client(), max_retries=5, retry_base_sec=0.0)
    with pytest.raises(_Boom):
        src.fetch(SeriesKey("bybit", OHLCV, "BTC/USDT:USDT", "5m"), 0, TIMEFRAME_MS["5m"])
