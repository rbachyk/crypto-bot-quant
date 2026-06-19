"""Real-time live-loop mode (AGENTS.md Section 35).

Drives the candidate stream from the LIVE market feed instead of snapshot replay. A rolling
per-symbol bar window (seeded by REST, then advanced from the websocket/poll
:class:`~src.live.data_manager.FeedSource`) is run through the SAME feature pipeline (the
Parity Rule) and strategy on every newly-closed bar; each signal becomes a candidate via the
shared :func:`~src.paper.lake.build_candidate`. Stale symbols are skipped and an exchange-wide
data-integrity failure stops the stream (Section 8). Real exits are exchange-side (the bracket's
SL/TP), so live candidates carry ``exit_move_frac=0`` — the venue manages the exit, not a
forward-looked move.

The feed satisfies the loop's ``MarketFeed`` Protocol, so :class:`~src.live.loop.LiveLoop` runs
it unchanged (paper / testnet / live venue, all gated). Tests inject a scripted feed source +
an offline seed source, so the whole path runs with no network or keys.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator

from src.config import Settings, get_settings
from src.data.config import DataConfig
from src.data.schema import (
    FUNDING,
    INDEX,
    MARK,
    OHLCV,
    OPEN_INTEREST,
    SPREAD,
    SeriesKey,
    timeframe_ms,
)
from src.data.source import DataSource
from src.features.pipeline import FeatureDataReader, compute_features
from src.paper.engine import PaperCandidateInput
from src.paper.lake import build_candidate
from src.regime.detector import load_regime_config

_POINT_IN_TIME = (MARK, INDEX, FUNDING, OPEN_INTEREST, SPREAD)


class RollingReader(FeatureDataReader):
    """In-memory FeatureDataReader over a rolling per-symbol window (live mode)."""

    def __init__(self, max_bars: int = 600) -> None:
        self._max = max_bars
        self._ohlcv: dict[str, deque] = {}
        self._series: dict[str, dict[str, list]] = {}

    def seed_ohlcv(self, symbol: str, bars: list[dict]) -> None:
        self._ohlcv[symbol] = deque(bars, maxlen=self._max)

    def append_bar(self, symbol: str, bar: dict) -> None:
        self._ohlcv.setdefault(symbol, deque(maxlen=self._max)).append(bar)

    def set_series(self, symbol: str, data_type: str, rows: list[dict]) -> None:
        self._series.setdefault(symbol, {})[data_type] = rows

    def ohlcv(self, symbol: str) -> list[dict]:
        return list(self._ohlcv.get(symbol, ()))

    def series(self, symbol: str, data_type: str) -> list[dict]:
        return self._series.get(symbol, {}).get(data_type, [])


class LiveCandidateFeed:
    """A ``MarketFeed`` that yields candidate groups from the live stream (Section 35)."""

    def __init__(
        self,
        data_cfg: DataConfig,
        *,
        feed_source,
        rest_source: DataSource | None = None,
        timeframe: str | None = None,
        symbols: list[str] | None = None,
        candidate_id: str | None = None,
        strategies: list[tuple] | None = None,
        data_manager=None,
        settings: Settings | None = None,
        window_bars: int = 300,
        max_groups: int | None = None,
        poll_sec: float = 0.0,
        seed_end_ms: int | None = None,
        equity: float = 10_000.0,
        should_stop=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.data_cfg = data_cfg
        self.feed_source = feed_source
        self.rest_source = rest_source
        self.data_manager = data_manager
        self.timeframe = timeframe or data_cfg.base_timeframe
        self.symbols = symbols or data_cfg.active_symbols()
        self.window_bars = window_bars
        self.max_groups = max_groups
        # poll_sec > 0 makes this a CONTINUOUS stream: when no symbol has a new closed bar it
        # waits and re-polls (a real demo/live session), instead of returning after one pass.
        self.poll_sec = poll_sec
        self.seed_end_ms = seed_end_ms
        self.equity = equity
        self._should_stop = should_stop  # polled during the wait so Stop is responsive
        self._reader = RollingReader(max_bars=window_bars * 2)

        from src.backtest.config import load_backtest_config
        from src.backtest.service import (
            _lake_feature_config,
            lake_candidate_strategy,
            make_strategy,
        )
        from src.strategies.promotion import is_strategy_promoted

        # The active strategy ensemble. Per-row strategies (reference / family B) implement
        # ``evaluate(row)``; cross-asset strategies (families A/G) implement
        # ``evaluate_portfolio(symbol, row, peers)`` — they need every symbol's row at the same
        # decision time. Both run live: each is split into the right bucket here and the feed
        # assembles a cross-symbol peer view per bar. Tuples are (strategy, id, ver, promoted).
        self._row_strategies: list[tuple] = []
        self._portfolio_strategies: list[tuple] = []
        self._latest_rows: dict[str, dict] = {}  # most-recent feature row per symbol (peers)

        def _add(strat, sid, ver, promoted: bool) -> None:
            if hasattr(strat, "evaluate_portfolio"):
                self._portfolio_strategies.append((strat, sid, ver, promoted))
            else:
                self._row_strategies.append((strat, sid, ver, promoted))

        if strategies:
            for strat, sid, ver in strategies:
                _add(strat, sid, ver, True)
        elif candidate_id:
            strat, sid, ver = lake_candidate_strategy(candidate_id)
            _add(strat, sid, ver, is_strategy_promoted(sid, ver))
        else:
            bt = load_backtest_config()
            strat = make_strategy(bt)
            _add(strat, bt.reference_strategy.name, bt.reference_strategy.strategy_version, False)
        self.feat_cfg = _lake_feature_config(self.timeframe)
        self._toxic_spread = load_regime_config().toxic_spread_bps  # default estimate floor

    def seed(self, rest_source: DataSource | None = None) -> None:
        """Backfill the rolling window (OHLCV + point-in-time) via REST so features are ready."""
        src = rest_source or self._rest()
        iv = timeframe_ms(self.timeframe)
        end = self.seed_end_ms if self.seed_end_ms is not None else int(time.time() * 1000)
        end = (end // iv) * iv
        start = end - self.window_bars * iv
        base_iv = self.data_cfg.base_timeframe
        for sym in self.symbols:
            self._reader.seed_ohlcv(
                sym,
                src.fetch(
                    SeriesKey(self.data_cfg.exchange_id, OHLCV, sym, self.timeframe), start, end
                ),
            )
            for dt in _POINT_IN_TIME:
                tf = (
                    self.data_cfg.oi_grid
                    if dt == OPEN_INTEREST
                    else (self.data_cfg.funding_timeframe if dt == FUNDING else base_iv)
                )
                self._reader.set_series(
                    sym,
                    dt,
                    src.fetch(SeriesKey(self.data_cfg.exchange_id, dt, sym, tf), start, end),
                )

    def _rest(self) -> DataSource:
        if self.rest_source is not None:
            return self.rest_source
        from src.data.source import get_data_source

        self.rest_source = get_data_source(self.data_cfg.exchange_id)
        return self.rest_source

    def _candidates_for(self, sym: str, bar: dict, row: dict) -> list[PaperCandidateInput]:
        """Build candidate inputs for one symbol from every active strategy — per-row strategies
        on ``row`` and cross-asset strategies on ``(sym, row, peers)`` where peers are the other
        symbols' most-recent feature rows."""
        out: list[PaperCandidateInput] = []

        def _emit(sig, sid: str, ver: str, promoted: bool) -> None:
            if sig is None:
                return
            cand = build_candidate(
                sym, row, sig, strat_id=sid, strat_ver=ver,
                entry_price=float(bar["close"]), spread_bps=self._toxic_spread / 5.0,
                promoted=promoted, data_ok=True,
            )
            # Real exits are exchange-side (bracket SL/TP) → no forward move in live mode.
            out.append(PaperCandidateInput(candidate=cand, equity=self.equity, exit_move_frac=0.0))

        for strat, sid, ver, promoted in self._row_strategies:
            _emit(strat.evaluate(row), sid, ver, promoted)
        if self._portfolio_strategies:
            peers = {k: v for k, v in self._latest_rows.items() if k != sym}
            for strat, sid, ver, promoted in self._portfolio_strategies:
                _emit(strat.evaluate_portfolio(sym, row, peers), sid, ver, promoted)
        return out

    def groups(self) -> Iterator[tuple[int, list[PaperCandidateInput]]]:
        if not self._reader.ohlcv(self.symbols[0]):
            self.seed()
        last_ts = dict.fromkeys(self.symbols, -1)
        emitted = 0
        while self.max_groups is None or emitted < self.max_groups:
            if self._should_stop is not None and self._should_stop():
                return  # operator pressed Stop (dashboard) → end the stream cleanly
            now = int(time.time() * 1000)
            if self.data_manager is not None and self.data_manager.poll(now).exchange_halt:
                return
            # Pass 1 — collect every symbol that has a NEW closed bar this cycle and refresh its
            # feature row, so the cross-asset peer view (self._latest_rows) is complete before
            # any portfolio strategy is evaluated.
            advanced: list[tuple[str, dict, dict]] = []  # (sym, bar, row)
            for sym in self.symbols:
                if self.data_manager is not None and not self.data_manager.is_fresh(sym):
                    continue
                got = self.feed_source.latest_bar(sym)
                if got is None:
                    continue
                ts, bar = got
                if ts <= last_ts[sym]:
                    continue
                last_ts[sym] = ts
                self._reader.append_bar(sym, bar)
                frame = compute_features(sym, self._reader, self.feat_cfg)
                if not frame.rows:
                    continue
                row = frame.rows[-1]
                self._latest_rows[sym] = row
                advanced.append((sym, bar, row))

            # Pass 2 — for each advanced symbol, evaluate per-row AND cross-asset strategies; all
            # signals on that symbol compete in one group so ranking + the one-position-per-symbol
            # cap arbitrate (only one trade per symbol across all strategies).
            progressed = False
            for sym, bar, row in advanced:
                cands = self._candidates_for(sym, bar, row)
                if not cands:
                    continue
                emitted += 1
                progressed = True
                yield (int(row["decision_ts"]), cands)
                if self.max_groups is not None and emitted >= self.max_groups:
                    return
            if not progressed:
                if self.poll_sec > 0:
                    # Wait for the next closed bar, but in 1s slices so a Stop is honoured fast.
                    waited = 0.0
                    while waited < self.poll_sec:
                        if self._should_stop is not None and self._should_stop():
                            return
                        time.sleep(min(1.0, self.poll_sec - waited))
                        waited += 1.0
                else:
                    return  # nothing new and not polling → finite stream (tests / one-shot)
