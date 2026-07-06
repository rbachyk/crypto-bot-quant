"""Section 8: live data manager — staleness, disconnect, reconnect backfill, ws/REST, halts."""

from __future__ import annotations

from src.live.data_manager import LiveDataManager

IV = 60_000  # 1m
SYMS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]


class FakeSource:
    def __init__(self) -> None:
        self._connected = True
        self.bars: dict[str, tuple[int, dict]] = {}
        self.backfill_rows: list[dict] = []

    def connected(self) -> bool:
        return self._connected

    def latest_bar(self, symbol):
        return self.bars.get(symbol)

    def backfill(self, symbol, since_ms, end_ms):
        return [r for r in self.backfill_rows if since_ms <= r["ts"] < end_ms]

    def push(self, symbol, ts):
        self.bars[symbol] = (ts, {"ts": ts, "close": 100.0})


def _mgr(src) -> LiveDataManager:
    return LiveDataManager(src, SYMS, interval_ms=IV, stale_after_intervals=2)


def test_get_data_source_routes_testnet_seed_to_sandbox() -> None:
    """REGRESSION (R2): the seed/PIT/backfill REST source must read the SAME environment as the
    streaming feed — testnet→sandbox — so a testnet session doesn't seed mainnet history and then
    advance on testnet bars (a mainnet→testnet seam across the feature lookback). demo/live stay
    mainnet."""
    from src.data.source import get_data_source

    tn = get_data_source("bybit", exchange_env="testnet")._ex
    assert tn.urls["api"] == tn.urls["test"]  # testnet/sandbox endpoints
    for env in ("demo", "live"):
        ex = get_data_source("bybit", exchange_env=env)._ex
        assert ex.urls["api"] != ex.urls["test"]  # mainnet seed data


def test_polling_source_routes_testnet_to_sandbox_demo_stays_mainnet() -> None:
    """REGRESSION (E1): the REST data feed applies the exchange env so TESTNET reads testnet klines
    (matching the venue its orders hit), while DEMO/LIVE keep MAINNET data — Bybit's demo endpoint
    serves no public klines (the silent tick-0 bug), so demo trading is mainnet-data by design."""
    from src.live.data_manager import CcxtPollingSource

    tn = CcxtPollingSource("bybit", "5m", exchange_env="testnet")._src._ex
    assert tn.urls["api"] == tn.urls["test"]  # routed to the testnet/sandbox endpoints

    for env in ("demo", "live"):
        ex = CcxtPollingSource("bybit", "5m", exchange_env=env)._src._ex
        assert ex.urls["api"] != ex.urls["test"]  # mainnet data preserved


def test_polling_source_returns_last_closed_bar_not_forming() -> None:
    """REGRESSION: CcxtPollingSource.latest_bar must return the last CLOSED bar, never the
    still-forming candle. The forming candle's open-ts is < now so it passes the fetch ts<end
    filter; returning it fed a partial bar (and a decision_ts in the future) into the live pipeline,
    freezing each symbol on its bar-open data for the whole period."""
    import time as _t

    from src.data.schema import timeframe_ms
    from src.live.data_manager import CcxtPollingSource

    tf = "5m"
    iv = timeframe_ms(tf)
    now = int(_t.time() * 1000)
    grid = (now // iv) * iv
    closed_ts, forming_ts = grid - iv, grid  # last closed, current forming (open <= now < close)

    class _FakeSrc:
        def fetch(self, key, start, end):  # type: ignore[no-untyped-def]
            return [
                {"ts": closed_ts, "open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 1},
                {"ts": forming_ts, "open": 2, "high": 2, "low": 2, "close": 2.0, "volume": 2},
            ]

    src = CcxtPollingSource.__new__(CcxtPollingSource)  # bypass ccxt client construction
    src._src, src._ex, src.timeframe = _FakeSrc(), "bybit", tf
    ts, bar = src.latest_bar("BTC/USDT:USDT")
    assert ts == closed_ts and bar["close"] == 1.0  # forming bar dropped


def test_fresh_then_stale_when_stream_stops() -> None:
    src = FakeSource()
    mgr = _mgr(src)
    for s in SYMS:
        src.push(s, 1_000_000)
    h = mgr.poll(1_000_000)
    assert set(h.fresh) == set(SYMS) and not h.exchange_halt

    # No new bars for > 2 intervals → all stale → exchange-wide halt.
    h2 = mgr.poll(1_000_000 + 3 * IV)
    assert set(h2.stale) == set(SYMS)
    assert h2.exchange_halt and "stale" in h2.reason
    assert not mgr.is_fresh("BTC/USDT:USDT")


def test_disconnect_halts_all() -> None:
    src = FakeSource()
    mgr = _mgr(src)
    src.push("BTC/USDT:USDT", 1_000_000)
    src._connected = False
    h = mgr.poll(1_000_000)
    assert not h.connected and h.exchange_halt and "disconnected" in h.reason
    assert mgr.stale_symbols() == set(SYMS)


def test_reconnect_backfill_recovers_freshness() -> None:
    src = FakeSource()
    mgr = _mgr(src)
    src.push("BTC/USDT:USDT", 1_000_000)
    mgr.poll(1_000_000)
    # Gap, then reconnect with REST backfill of the missed bars.
    src.backfill_rows = [{"ts": 1_000_000 + IV}, {"ts": 1_000_000 + 2 * IV}]
    recovered = mgr.backfill_after_reconnect("BTC/USDT:USDT", 1_000_000 + 3 * IV)
    assert [int(r["ts"]) for r in recovered] == [1_000_000 + IV, 1_000_000 + 2 * IV]
    assert mgr.is_fresh("BTC/USDT:USDT")


def test_ws_rest_divergence_flags_integrity_halt() -> None:
    src = FakeSource()
    mgr = LiveDataManager(src, SYMS, interval_ms=IV, ws_rest_tol_bps=10.0)
    for s in SYMS:
        src.push(s, 1_000_000)
    assert mgr.compare_ws_rest("BTC/USDT:USDT", 100.0, 100.05) is True  # 5 bps ok
    assert mgr.compare_ws_rest("BTC/USDT:USDT", 100.0, 101.0) is False  # 100 bps diverges
    h = mgr.poll(1_000_000)
    assert h.exchange_halt and "integrity" in h.reason
    assert not mgr.is_fresh("BTC/USDT:USDT")


def test_partial_staleness_does_not_halt_all() -> None:
    src = FakeSource()
    mgr = _mgr(src)
    src.push("BTC/USDT:USDT", 2_000_000)
    src.push("ETH/USDT:USDT", 2_000_000)
    mgr.poll(2_000_000)
    # Only BTC keeps streaming; ETH goes stale.
    src.push("BTC/USDT:USDT", 2_000_000 + 3 * IV)
    h = mgr.poll(2_000_000 + 3 * IV)
    assert h.fresh == ["BTC/USDT:USDT"] and h.stale == ["ETH/USDT:USDT"]
    assert not h.exchange_halt  # some symbols still fresh → no exchange-wide halt


def test_poll_bails_mid_cycle_when_should_stop_fires() -> None:
    """A Stop pressed while the exchange is slow must be honored mid-poll — poll checks should_stop
    BETWEEN per-symbol fetches, so a many-symbol cycle can't sit through every fetch before the
    loop's top-level cancel check runs (the freeze that made stuck sessions un-cancellable)."""
    fetched: list[str] = []

    class RecordingSource(FakeSource):
        def latest_bar(self, symbol):
            fetched.append(symbol)
            return super().latest_bar(symbol)

    src = RecordingSource()
    for s in SYMS:
        src.push(s, 1_000_000)
    mgr = _mgr(src)
    # Stop is already requested → poll must fetch NO symbols and return immediately.
    mgr.poll(1_000_000, should_stop=lambda: True)
    assert fetched == []


def test_poll_tolerates_a_symbol_fetch_error_without_killing_the_session() -> None:
    """A transient per-symbol fetch failure is caught → that symbol goes stale (not fresh) and the
    exception does NOT propagate, so one bad REST call degrades to a visible halt-and-sleep instead
    of freezing/killing a live session. Other symbols still update."""

    class FlakySource(FakeSource):
        def latest_bar(self, symbol):
            if symbol == "BTC/USDT:USDT":
                raise TimeoutError("simulated hung fetch")
            return super().latest_bar(symbol)

    src = FlakySource()
    src.push("ETH/USDT:USDT", 1_000_000)
    mgr = _mgr(src)
    h = mgr.poll(1_000_000)  # must not raise
    assert "ETH/USDT:USDT" in h.fresh  # healthy symbol still advanced
    assert "BTC/USDT:USDT" in h.stale  # the failing fetch left BTC stale
    assert not mgr.is_fresh("BTC/USDT:USDT")
