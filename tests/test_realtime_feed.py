"""Section 35: real-time live-loop mode — candidates driven from the live stream.

Offline + deterministic: the rolling window is seeded from DeterministicSource and the live
stream is a scripted FeedSource delivering successive new bars — no network or keys.
"""

from __future__ import annotations

from src.data.config import DataConfig, ValidationThresholds
from src.data.schema import OHLCV, SeriesKey, timeframe_ms
from src.data.source import DeterministicSource
from src.live.loop import LiveLoop
from src.live.realtime import LiveCandidateFeed

EX = "bybit"
SYM = "BTC/USDT:USDT"
TF = "5m"
IV = timeframe_ms(TF)
SEED_END = 300 * IV  # seed window is [0, SEED_END); new bars arrive at/after SEED_END


def _cfg() -> DataConfig:
    return DataConfig(
        exchange_id=EX,
        data_version="t",
        symbols=[SYM],
        timeframes=[TF],
        base_timeframe=TF,
        funding_interval_hours=8,
        required_series=[OHLCV],
        window_start_ms=0,
        window_end_ms=SEED_END,
        thresholds=ValidationThresholds(),
        oi_timeframe="1h",
    )


class ScriptedFeedSource:
    """Delivers successive new closed bars (then holds the last) — a deterministic live stream."""

    def __init__(self, bars: list[dict]) -> None:
        self._seq = [(int(b["ts"]), b) for b in bars]
        self._i = 0

    def connected(self) -> bool:
        return True

    def latest_bar(self, symbol):
        if self._i >= len(self._seq):
            return self._seq[-1] if self._seq else None
        item = self._seq[self._i]
        self._i += 1
        return item

    def backfill(self, *a, **k):
        return []


def _new_bars(n: int = 24) -> list[dict]:
    src = DeterministicSource(EX)
    return src.fetch(SeriesKey(EX, OHLCV, SYM, TF), SEED_END, SEED_END + n * IV)


def _feed(max_groups: int = 24) -> LiveCandidateFeed:
    return LiveCandidateFeed(
        _cfg(),
        feed_source=ScriptedFeedSource(_new_bars()),
        rest_source=DeterministicSource(EX),
        timeframe=TF,
        symbols=[SYM],
        seed_end_ms=SEED_END,
        max_groups=max_groups,
    )


def test_live_feed_yields_well_formed_candidates_from_stream() -> None:
    groups = list(_feed().groups())
    assert groups, "the strategy should fire on at least one streamed bar"
    for decision_ts, grp in groups:
        assert len(grp) == 1
        cand = grp[0].candidate
        assert cand.symbol == SYM
        assert cand.regime.startswith("R")  # Section-11 R-code regime
        assert cand.entry_price > 0
        assert grp[0].exit_move_frac == 0.0  # live exits are exchange-side (bracket SL/TP)
        assert int(cand.decision_ts) == decision_ts


def test_feed_refreshes_point_in_time_on_cadence() -> None:
    """Funding/OI/spread are re-fetched on the wall-clock cadence so funding_z (and the carry) don't
    FREEZE at seed time — the bug behind funding_carry never turning over. Throttled in between."""
    import time

    class _Counting(DeterministicSource):
        def __init__(self, ex: str) -> None:
            super().__init__(ex)
            self.fetches = 0

        def fetch(self, *a, **k):  # type: ignore[no-untyped-def]
            self.fetches += 1
            return super().fetch(*a, **k)

    src = _Counting(EX)
    feed = LiveCandidateFeed(
        _cfg(), feed_source=ScriptedFeedSource(_new_bars()), rest_source=src,
        timeframe=TF, symbols=[SYM], seed_end_ms=SEED_END,
    )
    feed.seed()
    after_seed = src.fetches
    feed._last_pit_refresh = int(time.time() * 1000)  # just refreshed → throttled
    feed._maybe_refresh_point_in_time()
    assert src.fetches == after_seed  # no extra fetch within the cadence
    feed._last_pit_refresh = 0  # cadence elapsed
    feed._maybe_refresh_point_in_time()
    assert src.fetches > after_seed  # re-fetched the point-in-time series


def test_live_feed_emits_per_cycle_heartbeat() -> None:
    """on_cycle fires every poll cycle (signal or not), so a quiet selective strategy is visibly
    alive on the dashboard instead of frozen at tick 0."""
    beats: list[dict] = []
    feed = LiveCandidateFeed(
        _cfg(),
        feed_source=ScriptedFeedSource(_new_bars()),
        rest_source=DeterministicSource(EX),
        timeframe=TF,
        symbols=[SYM],
        seed_end_ms=SEED_END,
        max_groups=5,
        on_cycle=beats.append,
    )
    list(feed.groups())
    assert beats, "the feed must emit a per-cycle heartbeat"
    assert {"cycles", "advanced", "signals", "last_ts"} <= set(beats[-1])
    assert beats[-1]["cycles"] >= 1


def test_held_positions_remark_on_signalless_bar() -> None:
    """REGRESSION: a held position must re-price on EVERY new bar, not only when the strategy
    signals again. The feed interleaves an empty (signal-less) group per advanced bar and the loop
    re-marks open positions on it — so a slow-timeframe position's unrealized P&L tracks the latest
    close instead of freezing for hours between signals (the lead_lag '0.00, not changing' symptom).
    """
    from src.risk.portfolio import Position

    class _SignallessBarFeed:
        """Yields one bar that produced no signal — an empty group, exactly what the live feed
        emits when a new bar closes but no strategy fires."""

        def groups(self):
            yield (1_000, [])

    loop = LiveLoop(mode="paper")
    loop.engine._open_positions["ETH/USDT:USDT"] = Position(
        symbol="ETH/USDT:USDT", side=1, qty=2.0, entry_price=100.0, risk_amount=10.0,
        beta_to_btc=1.0, regime="R1",
    )
    loop.engine._position_meta["ETH/USDT:USDT"] = ("lead_lag_xasset", 0)

    captured: list[list[dict]] = []
    result = loop.run(
        _SignallessBarFeed(), session_name="remark",
        on_positions=lambda _sid, pos: captured.append(pos),
        price_of=lambda _s: 110.0,
    )

    assert not result.ticks  # an empty bar is NOT a signal tick
    assert captured, "held positions must re-mark on a signal-less bar"
    pos = captured[-1][0]
    assert pos["mark_price"] == 110.0 and pos["unrealized_pnl"] == 20.0  # +1 × (110-100) × 2


def test_live_loop_runs_the_realtime_feed() -> None:
    result = LiveLoop(mode="paper").run(_feed(), session_name="rt")
    assert result.ticks  # processed live decision times
    assert result.executed > 0
    assert result.executed + result.rejected == sum(t.candidates for t in result.ticks)
    assert result.session.session_id.startswith("paper:")


class FlakyFeedSource:
    """Delivers bars but raises a transient error on every other call — simulating the
    intermittent REST/websocket faults that accumulate over a multi-day session."""

    def __init__(self, bars: list[dict]) -> None:
        self._seq = [(int(b["ts"]), b) for b in bars]
        self._i = 0
        self._call = 0

    def connected(self) -> bool:
        return True

    def latest_bar(self, symbol):
        self._call += 1
        if self._call % 2 == 0:
            raise ConnectionError("transient exchange disconnect")
        if self._i >= len(self._seq):
            return self._seq[-1] if self._seq else None
        item = self._seq[self._i]
        self._i += 1
        return item

    def backfill(self, *a, **k):
        return []


def test_transient_feed_errors_do_not_kill_the_stream() -> None:
    """A feed source that intermittently raises must NOT end the session — the regression for
    the multi-day stop where one unguarded latest_bar() exception killed the whole loop."""
    feed = LiveCandidateFeed(
        _cfg(),
        feed_source=FlakyFeedSource(_new_bars(24)),
        rest_source=DeterministicSource(EX),
        timeframe=TF,
        symbols=[SYM],
        seed_end_ms=SEED_END,
        poll_sec=0.1,  # continuous → keeps polling through the transient errors
        max_groups=6,  # the stream survives the raised errors and still reaches the cap
    )
    groups = list(feed.groups())
    assert len(groups) == 6  # delivered despite a transient error on every other poll


def test_continuous_session_resumes_after_data_integrity_halt() -> None:
    """A continuous session pauses (does not end) while data integrity is down, then resumes —
    a transient exchange-wide halt must not silently end a multi-day run (Section 8)."""
    from src.live.data_manager import DataHealth

    class _FlappingDataManager:
        """Halts for the first few polls, then recovers."""

        def __init__(self) -> None:
            self.polls = 0

        def poll(self, now_ms):
            self.polls += 1
            halted = self.polls <= 3  # down for 3 cycles, then healthy
            return DataHealth(
                ts=now_ms, connected=not halted, exchange_halt=halted,
                reason="blip" if halted else "",
            )

        def is_fresh(self, sym):
            return True

    feed = LiveCandidateFeed(
        _cfg(),
        feed_source=ScriptedFeedSource(_new_bars(8)),
        rest_source=DeterministicSource(EX),
        timeframe=TF,
        symbols=[SYM],
        data_manager=_FlappingDataManager(),
        seed_end_ms=SEED_END,
        poll_sec=1.0,  # continuous → waits through the halt instead of ending
        max_groups=3,
    )
    groups = list(feed.groups())
    assert len(groups) == 3  # resumed and produced candidates after the halt cleared


def test_continuous_feed_stops_on_should_stop() -> None:
    """A continuous (poll_sec>0) session must terminate promptly when Stop is requested —
    otherwise it would wait for new bars forever and never honour the dashboard Stop button."""
    calls = {"n": 0}

    def _stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 5  # allow a few cycles, then stop

    feed = LiveCandidateFeed(
        _cfg(),
        feed_source=ScriptedFeedSource(_new_bars(4)),
        rest_source=DeterministicSource(EX),
        timeframe=TF,
        symbols=[SYM],
        seed_end_ms=SEED_END,
        poll_sec=2.0,  # continuous: would otherwise wait for new bars
        should_stop=_stop,
        max_groups=None,  # unbounded → only Stop ends it
    )
    groups = list(feed.groups())  # must return (not hang)
    assert isinstance(groups, list)
    assert calls["n"] > 5  # the stop predicate was polled and eventually halted the stream
