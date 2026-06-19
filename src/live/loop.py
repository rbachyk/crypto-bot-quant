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

from src.config import Settings, get_settings
from src.data.config import DataConfig, load_data_config
from src.exchange.metadata import MetadataConfig, load_metadata_config
from src.execution.live_venue import LiveOrderGuard, get_venue
from src.execution.venue import Venue
from src.killswitch import KillSwitch
from src.paper.engine import PaperCandidateInput, PaperTradingEngine
from src.paper.session import PaperSession

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
        self.meta = meta or load_metadata_config()
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

    @property
    def env_label(self) -> str:
        """Session-id prefix identifying the trading environment, so statistics separate
        cleanly per environment (demo vs testnet vs live vs offline paper). ``paper`` mode is
        always the offline SimulatedVenue; any real-venue mode is labelled by EXCHANGE_ENV, so a
        Bybit **demo** run is tagged ``demo:`` and never mixed with testnet/live history."""
        return "paper" if self.mode == "paper" else self.settings.exchange_env

    def run(
        self,
        feed: MarketFeed,
        *,
        session_name: str = "live",
        max_ticks: int | None = None,
        on_tick: Callable[[LiveTick, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> LiveRunResult:
        """Process feed groups one tick at a time; halt on kill switch / foreign orders.

        ``on_tick(tick, index)`` is called after each processed tick (for live progress
        reporting); ``should_stop()`` is polled before each tick so an external operator
        (e.g. a dashboard Stop button via the job-cancel flag) can halt the loop cleanly."""
        session = self.engine.new_session(f"{self.env_label}:{session_name}")
        result = LiveRunResult(session=session, mode=self.mode)
        for i, (decision_ts, group) in enumerate(feed.groups()):
            if max_ticks is not None and i >= max_ticks:
                break
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
            before_exec = session.executed_count
            before_rej = session.rejected_count
            self.engine.process_candidates(group, session)
            # Reconcile the bot's mirror against the venue every tick (Section 7);
            # a foreign order halts the loop.
            if self.engine.run_reconciliation(session):
                result.halted = True
                tick = LiveTick(
                    decision_ts,
                    len(group),
                    session.executed_count - before_exec,
                    session.rejected_count - before_rej,
                )
                result.ticks.append(tick)
                if on_tick is not None:
                    on_tick(tick, i)
                break
            tick = LiveTick(
                decision_ts,
                len(group),
                session.executed_count - before_exec,
                session.rejected_count - before_rej,
            )
            result.ticks.append(tick)
            if on_tick is not None:
                on_tick(tick, i)
        return result


def run_replay_session(
    data_cfg: DataConfig | None = None,
    *,
    mode: str = "paper",
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    candidate_id: str | None = None,
    multi_strategy: bool = False,
    max_ticks: int | None = None,
    settings: Settings | None = None,
    guard: LiveOrderGuard | None = None,
    transport: str | None = None,
    realtime: bool = False,
    on_tick: Callable[[LiveTick, int], None] | None = None,
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
    tf = timeframe or data_cfg.base_timeframe
    syms = symbols or data_cfg.active_symbols()

    strategies = None
    if multi_strategy:
        from src.paper.lake import resolve_active_strategies

        active, _skipped = resolve_active_strategies(settings)
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
        )
        # The real-time feed owns the data-manager halt; don't double-poll at the loop level.
        loop = LiveLoop(mode=mode, settings=settings, guard=guard)
    elif multi_strategy:
        from src.paper.lake import build_active_lake_inputs

        inputs, _ids = build_active_lake_inputs(
            data_cfg, timeframe=tf, symbols=syms, settings=settings
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
