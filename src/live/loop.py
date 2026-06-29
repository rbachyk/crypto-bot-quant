"""Live trading loop (AGENTS.md Section 35).

Drives the SAME decision pipeline (ranking → risk → execution → venue, with
reconciliation and the kill switch) one decision time at a time, against an
injectable venue:

* ``paper``   — offline SimulatedVenue (no network);
* ``testnet`` — the real ccxt venue in sandbox mode (real orders, no real funds);
* ``live``    — real-money mainnet, which the venue refuses unless the M8 activation
  guard authorises it.

A ``MarketFeed`` yields ``(decision_ts, [PaperCandidateInput])`` groups; the loop
processes one group per tick. The replay feed builds groups from a downloaded
DATA_VERSION snapshot (the same candidate builder the lake paper session uses), so a
live run is a faithful tick-by-tick replay through the chosen venue. A real-time feed
(polling the exchange for the latest closed bar) plugs in behind the same Protocol.

This loop never enables live trading by itself: ``mode="live"`` still requires the full
live-safety condition (settings + gates + sign-off) enforced at the venue/guard layer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

from src.config import Settings, get_settings
from src.data.config import DataConfig, load_data_config
from src.exchange.metadata import MetadataConfig, load_metadata_config
from src.execution.live_venue import LiveOrderGuard, get_venue
from src.execution.ownership import OwnershipPolicy
from src.execution.reconciliation import StartupReconResult, reconcile_startup
from src.execution.venue import Venue
from src.killswitch import KillSwitch
from src.monitoring import Alert, AlertSeverity, get_alert_sink
from src.paper.engine import PaperCandidateInput, PaperTradingEngine
from src.paper.session import PaperSession

_log = structlog.get_logger("live.loop")

# A mirror position absent from the exchange book for this many consecutive reconciliations is
# treated as closed and dropped (debounce against fill-latency false drops).
_ABSENT_DROP_TICKS = 2


def _alert_reconcile(
    environment: str,
    recommended_action: str,
    *,
    title: str = "reconciliation: unknown order/position detected",
) -> None:
    get_alert_sink().send(
        Alert(
            title=title,
            severity=AlertSeverity.CRITICAL,
            component="execution",
            environment=environment,
            recommended_action=recommended_action,
        )
    )

_MODES = ("paper", "testnet", "live")


class MarketFeed(Protocol):
    """Yields ``(decision_ts, candidate_inputs)`` groups in time order."""

    def groups(self) -> Iterator[tuple[int, list[PaperCandidateInput]]]: ...


@dataclass(slots=True)
class ReplayFeed:
    """Replays lake-derived candidate inputs, one decision time per tick."""

    inputs: list[PaperCandidateInput]

    def groups(self) -> Iterator[tuple[int, list[PaperCandidateInput]]]:
        by_ts: dict[int, list[PaperCandidateInput]] = {}
        for pin in self.inputs:
            by_ts.setdefault(int(pin.candidate.decision_ts), []).append(pin)
        for ts in sorted(by_ts):
            yield ts, by_ts[ts]


def replay_feed_from_lake(
    data_cfg: DataConfig,
    *,
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    candidate_id: str | None = None,
    settings: Settings | None = None,
) -> ReplayFeed:
    """Build a replay feed from a downloaded snapshot (requires `qbot download` first)."""
    from src.paper.lake import build_lake_paper_inputs

    settings = settings or get_settings()
    tf = timeframe or data_cfg.base_timeframe
    syms = symbols or data_cfg.active_symbols()
    inputs, _, _ = build_lake_paper_inputs(
        data_cfg, timeframe=tf, symbols=syms, candidate_id=candidate_id, settings=settings
    )
    return ReplayFeed(inputs)


@dataclass(slots=True)
class LiveTick:
    decision_ts: int
    candidates: int
    executed: int
    rejected: int


@dataclass(slots=True)
class LiveRunResult:
    session: PaperSession
    mode: str
    ticks: list[LiveTick] = field(default_factory=list)
    halted: bool = False
    startup_recon: StartupReconResult | None = None

    @property
    def executed(self) -> int:
        return self.session.executed_count

    @property
    def rejected(self) -> int:
        return self.session.rejected_count


class LiveLoop:
    """Tick-driven live/replay loop over the real decision pipeline (Section 35)."""

    def __init__(
        self,
        *,
        mode: str = "paper",
        settings: Settings | None = None,
        meta: MetadataConfig | None = None,
        venue: Venue | None = None,
        kill_switch: KillSwitch | None = None,
        guard: LiveOrderGuard | None = None,
        data_manager: Any | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self.settings = settings or get_settings()
        # Paper uses the offline skeleton spec; a real venue (testnet/demo/live) must use the
        # metadata verified for ITS exchange so the venue's pre-trade metadata guard (Section 6)
        # reasons over the right spec — an unverified/placeholder spec blocks order placement.
        if meta is not None:
            self.meta = meta
        elif mode == "paper":
            self.meta = load_metadata_config()
        else:
            from src.exchange.metadata import load_metadata_for

            self.meta = load_metadata_for(self.settings.exchange_id)
        self.kill_switch = kill_switch or KillSwitch(self.settings)
        # Optional Section-8 live data manager; when exchange-wide data integrity fails it
        # halts ALL trading (mirrors the kill switch).
        self.data_manager = data_manager
        live = mode in ("testnet", "live")
        self.venue = (
            venue
            if venue is not None
            else get_venue(self.meta, self.settings, live=live, guard=guard)
        )
        self.engine = PaperTradingEngine(
            meta=self.meta,
            settings=self.settings,
            kill_switch=self.kill_switch,
            venue=self.venue,
        )
        # Bind the bounded-live max_open_positions cap to REAL concurrency: count owned positions
        # in the (per-tick reconciled) venue mirror, so a closed position frees a slot instead of
        # relying on an internal counter that only ever incremented.
        if guard is not None and hasattr(guard, "set_position_source"):
            guard.set_position_source(
                lambda: sum(
                    1 for p in self.venue.positions.values() if getattr(p, "owned", True)
                )
            )
        # Consecutive-absence counter for debounced retirement of exchange-side-closed positions.
        self._absent_ticks: dict[str, int] = {}
        # Active time-stop (hold_bars) bookkeeping: the exchange holds SL/TP/trailing natively but
        # cannot do "close after N bars", so the bot tracks each owned position's entry time + hold
        # horizon and flattens it once aged (parity with the backtest time-stop). Real venues only —
        # in replay paper the engine closes positions itself (exit_move_frac). ``_bar_iv`` is the
        # bar interval, inferred from the spacing of the feed's decision timestamps.
        self._open_age: dict[str, tuple[int, int]] = {}  # symbol -> (entry_decision_ts, hold_bars)
        self._bar_iv: int = 0
        self._prev_decision_ts: int | None = None

    def _reconcile_live(self, session: PaperSession) -> bool:
        """Per-tick reconciliation against the REAL exchange book (real venues only, Section 7).

        Re-pulls live orders + positions, syncs the venue mirror to the owned real state (so a
        position closed exchange-side via its SL/TP drops out and the risk/guard see real
        exposure), HALTS on any foreign/manual order or position, and alerts on an owned position
        missing its exchange-side stop (Section 2.2). A transient fetch error never halts — the
        next tick retries. Returns True if a halt is required."""
        venue = self.venue
        if not (
            hasattr(venue, "fetch_open_orders") and hasattr(venue, "fetch_exchange_positions")
        ):
            return False  # offline paper venue → no real book; engine reconciliation handles it
        try:
            exch_orders = venue.fetch_open_orders()
            exch_positions = venue.fetch_exchange_positions()
        except Exception:  # noqa: BLE001 - a transient fetch error must not halt the loop
            _log.warning("live_reconcile_fetch_error", exc_info=True)
            return False
        own = OwnershipPolicy(self.settings)
        foreign_orders = sorted(o for o, v in exch_orders.items() if not own.is_own(v.client_id))
        foreign_positions = sorted(s for s, p in exch_positions.items() if not p.owned)
        # Refresh/adopt OWNED items from the exchange (real stop/TP protection + positions opened
        # outside this session).
        owned_now = {s for s, p in exch_positions.items() if p.owned}
        for sym, p in exch_positions.items():
            if p.owned:
                venue.positions[sym] = p
                self._absent_ticks.pop(sym, None)
                if not p.has_exchange_side_stop():
                    # Log (not alert) so a multi-day loop can't flood the alert sink every tick.
                    _log.warning("live_owned_position_unprotected", symbol=sym)
        # DEBOUNCED drop of closed positions: a mirror position the exchange no longer lists is
        # retired only after it's been absent for _ABSENT_DROP_TICKS consecutive reconciliations —
        # so a just-placed position lagging in fetch_positions is NOT false-dropped, but a real
        # exchange-side SL/TP close frees its concurrency slot (the bounded-live cap counts the
        # mirror) instead of leaking it forever.
        for sym in list(venue.positions):
            if sym in owned_now:
                continue
            self._absent_ticks[sym] = self._absent_ticks.get(sym, 0) + 1
            if self._absent_ticks[sym] >= _ABSENT_DROP_TICKS:
                venue.positions.pop(sym, None)
                # Also drop it from the ENGINE's risk mirror so the Section-17 concurrency / heat /
                # net-beta caps release the slot — otherwise they over-count forever in a real run
                # (the engine's simulated exit path never fires when exits are exchange-side).
                self.engine._open_positions.pop(sym, None)
                self._absent_ticks.pop(sym, None)
        # Sync owned resting orders to the real book (drop our filled/cancelled orders the exchange
        # no longer lists, so the mirror doesn't grow unbounded over a multi-day run).
        venue.open_orders = {
            oid: v for oid, v in exch_orders.items() if own.is_own(v.client_id)
        }
        if foreign_orders or foreign_positions:
            _alert_reconcile(
                self.env_label,
                f"foreign order(s)={foreign_orders} position(s)={foreign_positions}; halt new "
                "entries and investigate (Section 7).",
            )
            session.foreign_order_halt_triggered = True
            session.reconciliation_events.append(
                {
                    "phase": "tick",
                    "halt_triggered": True,
                    "foreign_orders": foreign_orders,
                    "foreign_positions": foreign_positions,
                }
            )
            return True
        return False

    def reconcile_startup(self) -> StartupReconResult:
        """Reconcile the REAL exchange book against this bot before any tick (Section 7).

        For a real venue (testnet/demo/live) this fetches live open orders + positions,
        adopts the ones carrying our ownership prefix into the venue mirror, and flags any
        foreign/manual item — which must halt new entries. For offline paper it is a no-op
        clean book."""
        return reconcile_startup(
            self.venue,
            OwnershipPolicy(self.settings),
            environment=self.env_label,
        )

    @property
    def env_label(self) -> str:
        """Session-id prefix identifying the trading environment, so statistics separate
        cleanly per environment (demo vs testnet vs live vs offline paper). ``paper`` mode is
        always the offline SimulatedVenue; any real-venue mode is labelled by EXCHANGE_ENV, so a
        Bybit **demo** run is tagged ``demo:`` and never mixed with testnet/live history."""
        return "paper" if self.mode == "paper" else self.settings.exchange_env

    def _record_open_ages(self, decision_ts: int, group: list[PaperCandidateInput]) -> None:
        """Remember the entry time + hold horizon of any position this tick just opened, so the
        time-stop can flatten it once aged. Keyed by symbol (max one owned position per symbol)."""
        if self.mode == "paper":
            return
        for pin in group:
            sym = pin.candidate.symbol
            hold_bars = int(getattr(pin.candidate, "hold_bars", 0) or 0)
            if hold_bars > 0 and sym in self.venue.positions and sym not in self._open_age:
                self._open_age[sym] = (decision_ts, hold_bars)

    def _apply_time_stops(self, decision_ts: int, session: PaperSession) -> None:
        """Flatten owned positions that have reached their hold_bars horizon (bot-side time-stop).

        The exchange-resident stop/TP/trailing keep protecting the position regardless; this only
        adds the time-based exit the exchange can't express. No-op until the bar interval is known
        and only on real venues (replay paper exits via the engine's own model)."""
        if self.mode == "paper" or self._bar_iv <= 0:
            return
        for sym in [s for s in self._open_age if s not in self.venue.positions]:
            self._open_age.pop(sym, None)  # already closed exchange-side
        for sym, (entry_ts, hold_bars) in list(self._open_age.items()):
            if decision_ts - entry_ts >= hold_bars * self._bar_iv:
                closed = self.venue.close_position(sym)
                self._open_age.pop(sym, None)
                if closed:
                    _log.info("live_time_stop", symbol=sym, held_bars=hold_bars, ts=decision_ts)
                    session.reconciliation_events.append(
                        {"phase": "time_stop", "symbol": sym, "decision_ts": decision_ts}
                    )

    def run(
        self,
        feed: MarketFeed,
        *,
        session_name: str = "live",
        max_ticks: int | None = None,
        on_tick: Callable[[LiveTick, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_positions: Callable[[str, list[dict]], None] | None = None,
        on_flush: Callable[[PaperSession], None] | None = None,
        price_of: Callable[[str], float | None] | None = None,
    ) -> LiveRunResult:
        """Process feed groups one tick at a time; halt on kill switch / foreign orders.

        ``on_tick(tick, index)`` is called after each processed tick (for live progress
        reporting); ``should_stop()`` is polled before each tick so an external operator
        (e.g. a dashboard Stop button via the job-cancel flag) can halt the loop cleanly."""
        session = self.engine.new_session(f"{self.env_label}:{session_name}")
        result = LiveRunResult(session=session, mode=self.mode)

        # Section 7: before ANY tick, reconcile the real exchange book. A foreign/manual
        # order or position means we cannot trust exchange state → halt before trading.
        startup = self.reconcile_startup()
        result.startup_recon = startup
        session.reconciliation_events.append({"phase": "startup", **startup.to_dict()})
        if startup.halt_required:
            result.halted = True
            session.foreign_order_halt_triggered = True
            return result

        # The feed interleaves empty groups (a new bar with no signal) so held positions re-mark
        # every bar; only non-empty (signal) groups count toward max_ticks and the on_tick index.
        signal_idx = -1
        for decision_ts, group in feed.groups():
            if should_stop is not None and should_stop():
                result.halted = True
                break
            if self.kill_switch.engaged():
                result.halted = True
                break
            # Section 8: halt ALL trading if exchange-wide live-data integrity fails.
            if self.data_manager is not None and self.data_manager.poll(decision_ts).exchange_halt:
                result.halted = True
                break
            # Infer the bar interval from the decision-time spacing (min positive gap, like the
            # backtest grid), then flatten any position that has reached its hold_bars horizon
            # BEFORE considering new entries this bar (exit-then-enter, matching the backtest).
            if self._prev_decision_ts is not None:
                gap = decision_ts - self._prev_decision_ts
                if gap > 0 and (self._bar_iv <= 0 or gap < self._bar_iv):
                    self._bar_iv = gap
            self._prev_decision_ts = decision_ts
            self._apply_time_stops(decision_ts, session)
            # Paper mode: simulate the exchange-side bracket/time-stop exits the SimulatedVenue does
            # not fill, BEFORE considering new entries (exit-then-enter). Held positions close when
            # the new bar breaches their stop/TP/time-stop — otherwise a paper position (built with
            # exit_move_frac=0) would never close and the session would book no realized P&L.
            if self.mode == "paper" and price_of is not None:
                self.engine.simulate_paper_exits(
                    price_of, decision_ts, session, bar_iv=self._bar_iv
                )
            before_exec = session.executed_count
            before_rej = session.rejected_count
            self.engine.process_candidates(group, session)
            self._record_open_ages(decision_ts, group)
            # Reconcile every tick (Section 7); a foreign order/position halts the loop. A real
            # venue (testnet/demo/live) re-pulls the ACTUAL exchange book (detecting foreign items
            # and closes that happened exchange-side); offline paper uses the engine's own mirror.
            tick_halt = (
                self._reconcile_live(session)
                if self.mode != "paper"
                else self.engine.run_reconciliation(session)
            )
            # Re-mark and publish open positions on EVERY bar (signal or not) so a held position's
            # unrealized P&L tracks the latest close instead of freezing between signals — same
            # panel the basket sessions feed.
            if on_positions is not None:
                on_positions(
                    session.session_id,
                    self.engine.open_positions(price_of or (lambda _s: None)),
                )
            # Flush the session to the DB as it runs (throttled by the callback) so a continuous
            # multi-day session's trades are visible on the dashboard and survive a worker
            # restart — the per-symbol path otherwise only persisted at session end.
            if on_flush is not None:
                on_flush(session)
            if tick_halt:
                result.halted = True
            if group:  # a real signal bar: record a tick and report progress
                signal_idx += 1
                tick = LiveTick(
                    decision_ts,
                    len(group),
                    session.executed_count - before_exec,
                    session.rejected_count - before_rej,
                )
                result.ticks.append(tick)
                if on_tick is not None:
                    on_tick(tick, signal_idx)
            if result.halted or (max_ticks is not None and signal_idx + 1 >= max_ticks):
                break
        return result


def _resolve_live_timeframe(settings: Settings, data_cfg: DataConfig, candidate_id: str | None) -> str:
    """Decide the decision timeframe a live/paper session runs on when none is given explicitly.

    Uses the timeframe the active promoted strategies were VALIDATED on (so a 4h strategy runs on
    4h bars, not the fine base grid that's only a resampling base) — the wrong-timeframe bug where
    every live session silently ran at base_timeframe. Falls back to base_timeframe and logs loudly
    when the promoted timeframe is unrecorded (legacy promotions) or mixed; the operator can pass an
    explicit timeframe or re-validate to pin it."""
    try:
        from src.strategies.promotion import promoted_timeframe

        ids = [candidate_id] if candidate_id else None
        tf = promoted_timeframe(settings.strategy_version, candidate_ids=ids)
    except Exception:  # noqa: BLE001 - a registry/DB hiccup must not block the session start
        tf = None
    if tf:
        _log.info(
            "live_timeframe_from_promotion", timeframe=tf,
            candidate=candidate_id or "ensemble",
        )
        return tf
    _log.warning(
        "live_timeframe_unrecorded", fallback=data_cfg.base_timeframe,
        hint="promoted strategy has no recorded timeframe — re-validate or pass an explicit "
        "timeframe to pin the decision tf (running on the base grid otherwise)",
    )
    return data_cfg.base_timeframe


def run_replay_session(
    data_cfg: DataConfig | None = None,
    *,
    mode: str = "paper",
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    candidate_id: str | None = None,
    multi_strategy: bool = False,
    max_ticks: int | None = None,
    poll_sec: float = 0.0,
    settings: Settings | None = None,
    guard: LiveOrderGuard | None = None,
    transport: str | None = None,
    realtime: bool = False,
    on_tick: Callable[[LiveTick, int], None] | None = None,
    on_heartbeat: Callable[[dict], None] | None = None,
    on_positions: Callable[[str, list[dict]], None] | None = None,
    on_flush: Callable[[PaperSession], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> LiveRunResult:
    """Run the live loop over a snapshot **replay** or the **real-time** live feed in ``mode``.

    ``transport`` ('ws' | 'rest') attaches a live :class:`LiveDataManager` so an exchange-wide
    data-integrity failure halts the loop (Section 8). ``realtime=True`` (requires a transport)
    drives the candidate stream from the live feed — a rolling window → the one feature pipeline
    → the strategy on each newly-closed bar — instead of replaying the snapshot.

    ``multi_strategy=True`` runs the **active promoted strategy ensemble** (top-N by validated
    expectancy_r) concurrently instead of a single ``candidate_id`` — this is how demo/live
    behave: every active promoted strategy emits signals, the engine arbitrates via ranking +
    the one-position-per-symbol cap. With nothing promoted yet, the feed simply has no
    candidates (no trades) — faithful to live."""
    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    tf = timeframe or _resolve_live_timeframe(settings, data_cfg, candidate_id)
    syms = symbols or data_cfg.active_symbols()

    # Section 13: any non-paper run (testnet/demo/live) places orders on a real account and
    # may ONLY run strategies validated on real lake data — never synthetic/reference-only.
    require_real_data = mode != "paper"
    strategies = None
    if multi_strategy:
        from src.paper.lake import resolve_active_strategies

        active, _skipped = resolve_active_strategies(
            settings, require_real_data=require_real_data
        )
        strategies = active
    # Real-money mode is bounded by the activation guard (gates + sign-off + caps).
    if guard is None and mode == "live":
        from src.live.guard import LiveActivationGuard

        guard = LiveActivationGuard(settings)

    source = None
    data_manager = None
    if transport or realtime:
        from src.data.schema import timeframe_ms
        from src.live.data_manager import LiveDataManager
        from src.live.websocket_feed import live_feed_source

        source = live_feed_source(
            syms,
            transport=transport or "rest",
            exchange_id=data_cfg.exchange_id,
            timeframe=tf,
            exchange_env=settings.exchange_env,
        )
        data_manager = LiveDataManager(source, syms, interval_ms=timeframe_ms(tf))

    if realtime:
        from src.live.realtime import LiveCandidateFeed

        feed: MarketFeed = LiveCandidateFeed(
            data_cfg,
            feed_source=source,
            data_manager=data_manager,
            timeframe=tf,
            symbols=syms,
            candidate_id=candidate_id,
            strategies=strategies,
            settings=settings,
            max_groups=max_ticks,
            poll_sec=poll_sec,  # >0 → continuous session (waits for new bars)
            should_stop=should_stop,  # responsive Stop during the wait
            on_cycle=on_heartbeat,  # per-cycle liveness (even when no signal fires)
        )
        # The real-time feed owns the data-manager halt; don't double-poll at the loop level.
        loop = LiveLoop(mode=mode, settings=settings, guard=guard)
    elif multi_strategy:
        from src.paper.lake import build_active_lake_inputs

        inputs, _ids = build_active_lake_inputs(
            data_cfg,
            timeframe=tf,
            symbols=syms,
            settings=settings,
            require_real_data=require_real_data,
        )
        feed = ReplayFeed(inputs)
        loop = LiveLoop(mode=mode, settings=settings, guard=guard, data_manager=data_manager)
    else:
        feed = replay_feed_from_lake(
            data_cfg, timeframe=tf, symbols=syms, candidate_id=candidate_id, settings=settings
        )
        loop = LiveLoop(mode=mode, settings=settings, guard=guard, data_manager=data_manager)
    return loop.run(
        feed,
        session_name=data_cfg.data_version,
        max_ticks=max_ticks,
        on_tick=on_tick,
        should_stop=should_stop,
        on_positions=on_positions,
        on_flush=on_flush,
        price_of=getattr(feed, "latest_price", None),  # only the realtime feed marks live prices
    )


def persist_live_run(
    result: LiveRunResult, settings: Settings | None = None
) -> str:
    """Persist a finished live/demo/testnet loop the same way a paper session is persisted.

    Writes the run summary + per-trade rows + decision logs + trade explainability so the
    dashboard reads demo/testnet/live trades through the exact same tables as paper, while the
    ``env:`` session-id prefix keeps each environment's statistics separated (Section 26/34)."""
    from datetime import UTC, datetime

    from src.paper.report import build_paper_report
    from src.paper.run import persist_paper_session

    session = result.session
    if session.ended_at is None:
        session.ended_at = datetime.now(UTC)
    report = build_paper_report(session)
    return persist_paper_session(session, report, settings)
