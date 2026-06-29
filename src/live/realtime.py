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

import structlog

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

_log = structlog.get_logger("live.realtime")


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
        on_cycle=None,
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
        # Heartbeat: called once per poll cycle (even when no signal fires) so a healthy-but-QUIET
        # session is visibly alive instead of frozen at "tick 0" (a selective strategy like lead_lag
        # only yields on a signal, so without this it looks dead between setups).
        self._on_cycle = on_cycle
        self._cycles = 0
        # Point-in-time series (funding / OI / spread) are seeded once; without periodic refresh
        # they FREEZE — funding_z then never changes, so funding_carry holds a static basket and
        # the carry charged on held legs goes stale. Re-fetch them on this wall-clock cadence.
        self._pit_refresh_ms = 3_600_000  # 1h (funding posts every ~8h; OI/spread faster)
        self._last_pit_refresh = 0
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
        total_bars = 0
        for sym in self.symbols:
            bars = src.fetch(
                SeriesKey(self.data_cfg.exchange_id, OHLCV, sym, self.timeframe), start, end
            )
            self._reader.seed_ohlcv(sym, bars)
            total_bars += len(bars)
        self._seed_point_in_time(src, start, end)
        self._last_pit_refresh = int(time.time() * 1000)
        # A seed that returns NO OHLCV means the feed can never advance → the session sits at tick 0
        # (the silent-startup symptom). Surface the bar count so it's diagnosable, loud if empty.
        (_log.warning if total_bars == 0 else _log.info)(
            "live_feed_seeded", symbols=len(self.symbols), ohlcv_bars=total_bars,
            timeframe=self.timeframe, exchange_env=self.settings.exchange_env,
        )

    def _seed_point_in_time(self, src: DataSource, start: int, end: int) -> None:
        """Fetch funding / OI / spread for every symbol over ``[start, end]`` — the point-in-time
        series funding_z / premium / oi_change read. Shared by the initial seed and the refresh."""
        base_iv = self.data_cfg.base_timeframe
        for sym in self.symbols:
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

    def _maybe_refresh_point_in_time(self) -> None:
        """Re-fetch funding / OI / spread on the wall-clock cadence so funding_z (and the carry)
        stay CURRENT — without this they freeze at seed time and funding_carry never turns over."""
        now = int(time.time() * 1000)
        if now - self._last_pit_refresh < self._pit_refresh_ms:
            return
        iv = timeframe_ms(self.timeframe)
        end = (now // iv) * iv
        start = end - self.window_bars * iv
        try:
            self._seed_point_in_time(self._rest(), start, end)
            self._last_pit_refresh = now
            _log.info("live_feed_pit_refreshed", funding_tf=self.data_cfg.funding_timeframe)
        except Exception:  # noqa: BLE001 - a refresh fetch error must not kill the stream
            _log.warning("live_feed_pit_refresh_error", exc_info=True)

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

        def _emit(sig, sid: str, ver: str, promoted: bool, risk_scale: float) -> None:
            if sig is None:
                return
            cand = build_candidate(
                sym, row, sig, strat_id=sid, strat_ver=ver,
                entry_price=float(bar["close"]), spread_bps=self._toxic_spread / 5.0,
                promoted=promoted, data_ok=True, risk_scale=risk_scale,
            )
            # Real exits are exchange-side (bracket SL/TP/trailing) → no forward move in live mode.
            out.append(PaperCandidateInput(candidate=cand, equity=self.equity, exit_move_frac=0.0))

        for strat, sid, ver, promoted in self._row_strategies:
            _emit(strat.evaluate(row), sid, ver, promoted, float(getattr(strat, "risk_scale", 1.0)))
        if self._portfolio_strategies:
            peers = {k: v for k, v in self._latest_rows.items() if k != sym}
            for strat, sid, ver, promoted in self._portfolio_strategies:
                _emit(
                    strat.evaluate_portfolio(sym, row, peers), sid, ver, promoted,
                    float(getattr(strat, "risk_scale", 1.0)),
                )
        return out

    def _advance_symbol(
        self, sym: str, last_ts: dict[str, int]
    ) -> tuple[str, dict, dict] | None:
        """Pull the latest closed bar for one symbol and compute its feature row, or None if
        nothing new. Isolated so a transient per-symbol error (a REST blip, a dropped websocket,
        a one-off feature error) is caught by the caller and never kills the whole stream."""
        if self.data_manager is not None and not self.data_manager.is_fresh(sym):
            return None
        got = self.feed_source.latest_bar(sym)
        if got is None:
            return None
        ts, bar = got
        if ts <= last_ts[sym]:
            return None
        last_ts[sym] = ts
        self._reader.append_bar(sym, bar)
        frame = compute_features(sym, self._reader, self.feat_cfg)
        if not frame.rows:
            return None
        row = frame.rows[-1]
        self._latest_rows[sym] = row
        return (sym, bar, row)

    def groups(self) -> Iterator[tuple[int, list[PaperCandidateInput]]]:
        if not self._reader.ohlcv(self.symbols[0]):
            self.seed()
        last_ts = dict.fromkeys(self.symbols, -1)
        emitted = 0
        while self.max_groups is None or emitted < self.max_groups:
            if self._should_stop is not None and self._should_stop():
                return  # operator pressed Stop (dashboard) → end the stream cleanly
            self._maybe_refresh_point_in_time()  # keep funding/OI/spread current (funding_z, carry)
            now = int(time.time() * 1000)
            try:
                halted = (
                    self.data_manager is not None and self.data_manager.poll(now).exchange_halt
                )
            except Exception:  # noqa: BLE001 - a data-manager poll error must not kill the stream
                _log.warning("live_feed_poll_error", exc_info=True)
                if self._sleep_or_stop():
                    return
                continue
            if halted:
                # Section 8: stop trading while exchange-wide data integrity is down. A CONTINUOUS
                # (polling) session does NOT end here — it waits and re-checks so a transient
                # outage (a websocket drop, a REST blip — near-certain over a multi-day run)
                # pauses trading and then resumes when data is healthy again, rather than silently
                # ending the session. A finite/one-shot run ends as before.
                if self.poll_sec > 0:
                    _log.warning("live_feed_exchange_halt_waiting")
                    if self._sleep_or_stop():
                        return
                    continue
                return
            # Pass 1 — collect every symbol that has a NEW closed bar this cycle and refresh its
            # feature row, so the cross-asset peer view (self._latest_rows) is complete before
            # any portfolio strategy is evaluated. A per-symbol error is logged and skipped — the
            # stream survives transient exchange/network faults over a multi-day session.
            advanced: list[tuple[str, dict, dict]] = []  # (sym, bar, row)
            for sym in self.symbols:
                try:
                    got = self._advance_symbol(sym, last_ts)
                except Exception:  # noqa: BLE001 - one bad symbol must not end the session
                    _log.warning("live_feed_symbol_error", symbol=sym, exc_info=True)
                    continue
                if got is not None:
                    advanced.append(got)

            # Pass 2 — for each advanced symbol, evaluate per-row AND cross-asset strategies; all
            # signals on that symbol compete in one group so ranking + the one-position-per-symbol
            # cap arbitrate (only one trade per symbol across all strategies). A build error on
            # one symbol is logged and skipped, never ending the stream.
            progressed = False
            for sym, bar, row in advanced:
                try:
                    cands = self._candidates_for(sym, bar, row)
                except Exception:  # noqa: BLE001 - one bad symbol must not end the session
                    _log.warning("live_candidate_build_error", symbol=sym, exc_info=True)
                    continue
                if not cands:
                    continue
                emitted += 1
                progressed = True
                yield (int(row["decision_ts"]), cands)
                if self.max_groups is not None and emitted >= self.max_groups:
                    return
            # Heartbeat AFTER processing the cycle — fires every cycle, signal or not, so the
            # dashboard can show the session is alive + how many bars it has evaluated.
            self._cycles += 1
            if self._on_cycle is not None:
                self._on_cycle({
                    "cycles": self._cycles, "advanced": len(advanced), "signals": emitted,
                    "last_ts": max((int(r["decision_ts"]) for _, _, r in advanced), default=0),
                })
            if not progressed:
                if self.poll_sec > 0:
                    if self._sleep_or_stop():
                        return  # Stop pressed during the wait
                else:
                    return  # nothing new and not polling → finite stream (tests / one-shot)

    def snapshots(self) -> Iterator[tuple[int, dict, dict]]:
        """Like :meth:`groups` but yields the CROSS-SECTION snapshot ``(decision_ts, {sym: bar},
        {sym: feature_row})`` each cycle — for the basket (cross-sectional) paper loop, which needs
        the whole universe at one bar rather than per-symbol candidate groups. Same poll / halt /
        Stop / transient-fault handling as ``groups``."""
        if not self._reader.ohlcv(self.symbols[0]):
            self.seed()
        last_ts = dict.fromkeys(self.symbols, -1)
        emitted = 0
        while self.max_groups is None or emitted < self.max_groups:
            if self._should_stop is not None and self._should_stop():
                return
            self._maybe_refresh_point_in_time()  # keep funding/OI/spread current (funding_z, carry)
            now = int(time.time() * 1000)
            try:
                halted = self.data_manager is not None and self.data_manager.poll(now).exchange_halt
            except Exception:  # noqa: BLE001 - a poll error must not kill the stream
                _log.warning("live_feed_poll_error", exc_info=True)
                if self._sleep_or_stop():
                    return
                continue
            if halted:
                if self.poll_sec > 0 and not self._sleep_or_stop():
                    continue
                return
            advanced = []
            for sym in self.symbols:
                try:
                    got = self._advance_symbol(sym, last_ts)
                except Exception:  # noqa: BLE001 - one bad symbol must not end the session
                    _log.warning("live_feed_symbol_error", symbol=sym, exc_info=True)
                    continue
                if got is not None:
                    advanced.append(got)
            if advanced:
                ts = max(int(r["decision_ts"]) for _, _, r in advanced)
                bars_at = {
                    s: self._reader.ohlcv(s)[-1] for s in self.symbols if self._reader.ohlcv(s)
                }
                emitted += 1
                yield (ts, bars_at, dict(self._latest_rows))
            elif self.poll_sec > 0:
                if self._sleep_or_stop():
                    return
            else:
                return

    def symbol_inputs(self) -> dict:
        """Per-symbol ``SymbolInput`` (bars + features + funding + spread) from the rolling window —
        the basket loop's ``by_symbol`` (its funding/cost helpers need the funding_events + spread).
        Rebuilt each call from the reader; funding/spread share the engine schema (funding_rate /
        spread_bps); guarded so a missing field degrades to a safe default."""
        from src.backtest.engine import SymbolInput

        out: dict = {}
        for sym in self.symbols:
            bars = self._reader.ohlcv(sym)
            if not bars:
                continue
            funding = [
                {"ts": int(f["ts"]), "funding_rate": float(f.get("funding_rate", 0.0))}
                for f in self._reader.series(sym, FUNDING)
            ]
            spread = [
                {"ts": int(s["ts"]), "spread_bps": float(s.get("spread_bps", 2.0))}
                for s in self._reader.series(sym, SPREAD)
            ]
            out[sym] = SymbolInput(
                symbol=sym,
                bars=list(bars),
                frame=compute_features(sym, self._reader, self.feat_cfg),
                spread_samples=spread,
                funding_events=funding,
            )
        return out

    def latest_price(self, symbol: str) -> float | None:
        """Latest known close for a symbol (to mark open positions) — None if unseen yet."""
        bars = self._reader.ohlcv(symbol)
        return float(bars[-1]["close"]) if bars else None

    def _sleep_or_stop(self) -> bool:
        """Wait ``poll_sec`` (or 1s if not polling) in 1s slices so a dashboard Stop is honoured
        fast. Returns True if the operator pressed Stop during the wait."""
        budget = self.poll_sec if self.poll_sec > 0 else 1.0
        waited = 0.0
        while waited < budget:
            if self._should_stop is not None and self._should_stop():
                return True
            time.sleep(min(1.0, budget - waited))
            waited += 1.0
        return False
