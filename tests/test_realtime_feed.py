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


def test_live_loop_runs_the_realtime_feed() -> None:
    result = LiveLoop(mode="paper").run(_feed(), session_name="rt")
    assert result.ticks  # processed live decision times
    assert result.executed > 0
    assert result.executed + result.rejected == sum(t.candidates for t in result.ticks)
    assert result.session.session_id.startswith("paper:")
