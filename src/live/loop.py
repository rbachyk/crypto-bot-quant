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

from collections.abc import Iterator
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

    def run(
        self, feed: MarketFeed, *, session_name: str = "live", max_ticks: int | None = None
    ) -> LiveRunResult:
        """Process feed groups one tick at a time; halt on kill switch / foreign orders."""
        session = self.engine.new_session(f"{self.mode}:{session_name}")
        result = LiveRunResult(session=session, mode=self.mode)
        for i, (decision_ts, group) in enumerate(feed.groups()):
            if max_ticks is not None and i >= max_ticks:
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
                result.ticks.append(
                    LiveTick(
                        decision_ts,
                        len(group),
                        session.executed_count - before_exec,
                        session.rejected_count - before_rej,
                    )
                )
                break
            result.ticks.append(
                LiveTick(
                    decision_ts,
                    len(group),
                    session.executed_count - before_exec,
                    session.rejected_count - before_rej,
                )
            )
        return result


def run_replay_session(
    data_cfg: DataConfig | None = None,
    *,
    mode: str = "paper",
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    candidate_id: str | None = None,
    max_ticks: int | None = None,
    settings: Settings | None = None,
    guard: LiveOrderGuard | None = None,
) -> LiveRunResult:
    """Convenience: replay a downloaded snapshot through the live loop in ``mode``."""
    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    feed = replay_feed_from_lake(
        data_cfg,
        timeframe=timeframe,
        symbols=symbols,
        candidate_id=candidate_id,
        settings=settings,
    )
    # Real-money mode is bounded by the activation guard (gates + sign-off + caps).
    if guard is None and mode == "live":
        from src.live.guard import LiveActivationGuard

        guard = LiveActivationGuard(settings)
    loop = LiveLoop(mode=mode, settings=settings, guard=guard)
    return loop.run(feed, session_name=data_cfg.data_version, max_ticks=max_ticks)
