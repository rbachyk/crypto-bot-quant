"""Paper Trading Engine (AGENTS.md Section 26 / Phase 8).

The PaperTradingEngine is the system-layer 11 component. It runs the exact same
pipeline as live trading but against a :class:`~src.execution.venue.SimulatedVenue`
(no real orders). It does NOT skip any safety check:

    candidate → ranking (setup quality) → risk manager → revalidate → exec sim

Every step produces a decision log entry. The kill switch, reconciliation, and
simulated stops are exercised exactly as in live (Section 2.2). No strategy may
advance to live from Phase A alone (Section 26).

This module is the single entry-point for both the PAPER-A and PAPER-B gate
checks; the report module formats the session for each gate's criteria.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from src.config import Settings, get_settings
from src.exchange.metadata import MetadataConfig
from src.execution import (
    NO_FIXED_TP_FRAC,
    ExecutionEngine,
    OwnershipPolicy,
    Reconciler,
    SimulatedVenue,
    Venue,
    load_execution_config,
)
from src.killswitch import KillSwitch
from src.paper.session import (
    PaperDecisionLog,
    PaperSession,
    PaperTrade,
    RejectedPaperCandidate,
)
from src.ranking import (
    Candidate,
    CandidateRankingEngine,
    SetupQualityScorer,
    load_ranking_config,
)
from src.risk import (
    AccountState,
    BreakerInputs,
    PortfolioState,
    RiskManager,
    load_risk_config,
)
from src.risk.portfolio import Position

_log = structlog.get_logger("paper.engine")

# Abnormal-slippage breaker input (M7): observed fill slippage over a short rolling window.
# The breaker activates when the MEAN slippage of the last _SLIP_WINDOW fills exceeds the
# execution policy's max_slippage_frac (the same knob the per-order revalidator blocks on —
# the breaker catches sustained observed drift the per-order estimate misses). At least
# _SLIP_MIN_FILLS observations are required so one bad print doesn't halt the book.
_SLIP_WINDOW = 5
_SLIP_MIN_FILLS = 3

# Breaker reasons that require a MANUAL reset (Section 2.2) — persisted via the M9 state
# store so a restart cannot clear the halt. Cooldown-style breakers (consecutive-loss,
# funding, abnormal-slippage) recompute from their inputs and are deliberately not latched.
_MANUAL_RESET_BLOCKERS = ("daily_loss_limit", "max_drawdown_limit", "weekly_loss_limit")


@dataclass(slots=True)
class PaperCandidateInput:
    """A candidate plus the simulated account context at decision time."""

    candidate: Candidate
    equity: float = 10_000.0
    daily_pnl: float = 0.0
    open_risk: float = 0.0
    # Running session peak equity and realized loss streak at decision time. Threaded through
    # to BreakerInputs so the drawdown and consecutive-loss breakers can actually trip in
    # paper (they were previously hardcoded to peak==equity / 0, making them dead). Defaults
    # (0) fall back to "flat / no streak", preserving prior behaviour for callers that omit them.
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    # A foreign (unowned) order injected to exercise reconciliation detection.
    inject_foreign_order: bool = False
    # Price movement fraction after entry (positive = favorable).
    exit_move_frac: float = 0.0
    # Number of bars to hold (0 = close next bar).
    hold_bars: int = 0
    # Replay-path exit already resolved by an intrabar bracket walk (H9): the exit reason and the
    # funding cost fraction (>0 = paid) accrued while held. When ``exit_reason`` is provided the
    # engine books it directly instead of re-classifying from the endpoint move — so the replay
    # path captures intrabar stop/take-profit/trailing hits and funding, matching backtest/live.
    exit_reason: str | None = None
    funding_frac: float = 0.0


class PaperTradingEngine:
    """Runs the full paper trading pipeline on a list of candidates.

    Parameters
    ----------
    meta:
        Exchange metadata config (same object used by risk + execution).
    kill_switch:
        Shared kill-switch instance. If None, a new one is created.
    config_version:
        Versioned config identifier for decision logs (Section 4).
    universe_version:
        Universe version in effect for this session.
    """

    def __init__(
        self,
        meta: MetadataConfig | None = None,
        kill_switch: KillSwitch | None = None,
        config_version: str = "v0.1.0",
        universe_version: str = "u_2026_06_17_001",
        settings: Settings | None = None,
        venue: Venue | None = None,
    ) -> None:
        from src.exchange.metadata import load_metadata_config

        self._settings = settings or get_settings()
        self._meta = meta or load_metadata_config()
        self._kill_switch = kill_switch or KillSwitch()
        self._config_version = config_version
        self._universe_version = universe_version

        # Build the pipeline stack from defaults (same configs as the gate checks).
        risk_cfg = load_risk_config()
        exec_cfg = load_execution_config()
        rank_cfg = load_ranking_config()

        self._risk = RiskManager(risk_cfg, self._meta, self._kill_switch)
        self._scorer = SetupQualityScorer(rank_cfg, self._meta)
        self._ranker = CandidateRankingEngine(rank_cfg, self._meta)
        ownership = OwnershipPolicy(self._settings)
        # The venue is injectable so the live loop can drive the SAME pipeline against a
        # real (testnet) venue; defaults to the offline SimulatedVenue for paper.
        self._venue = venue if venue is not None else SimulatedVenue(self._meta)
        self._exec = ExecutionEngine(
            exec_cfg, self._meta, ownership, self._venue, self._kill_switch
        )
        self._reconciler = Reconciler(ownership)
        # Risk-relevant facts for positions currently open in the venue, keyed by symbol.
        # This is the engine's portfolio mirror — it is what the risk manager reasons over
        # so the concurrency / heat / net-beta caps actually bind against open positions
        # (previously the risk manager always saw an EMPTY portfolio, so the Section-17
        # portfolio caps were dead in paper/demo). Kept in lock-step with the venue book.
        self._open_positions: dict[str, Position] = {}
        # (strategy, entry_ts) per open symbol — the Position drops them, but the dashboard's live
        # open-positions panel wants to label each held position by strategy and entry time.
        self._position_meta: dict[str, tuple[str, int]] = {}
        # (stop_price, tp_price, hold_bars, entry_ts) per open symbol — the bracket levels a PAPER
        # position is held against. Live candidates carry exit_move_frac=0 (real exits are
        # exchange-side), but an offline SimulatedVenue never FILLS the resting stop/TP, so without
        # an explicit simulator a paper position would never close. simulate_paper_exits reads these
        # to flatten a held paper position when a new bar's price breaches its stop/TP/time-stop.
        self._exit_levels: dict[str, tuple[float, float, int, int]] = {}
        # Perpetual funding accrued on each OPEN position — parity with the backtest engine, which
        # charges funding at every funding timestamp a position is held across (src/backtest/engine
        # ._charge_funding). Without this, paper/live directional P&L (e.g. lead_lag held for hours)
        # overstates vs the backtest AND the real exchange, which DOES debit/credit funding. Wired
        # by the live run from the feed's funding series (set_funding_source); with no source
        # (offline unit tests) funding stays 0 — behaviour unchanged. COST convention: >0 = paid.
        from src.backtest.config import load_backtest_config
        from src.backtest.costs import FundingModel

        self._funding = FundingModel(load_backtest_config().costs)
        self._funding_source: Callable[[str], list[dict]] | None = None
        self._position_funding: dict[str, float] = {}  # symbol → accrued funding (cost convention)
        self._funding_watermark: dict[str, int] = {}  # symbol → last settled funding-event ts
        # Trailing-stop simulation (parity with the backtest, which ratchets the stop with the peak
        # favorable price — `src/backtest/engine.py`; and with live, where the trail is an
        # exchange-native order). Without it paper never trails, so momentum winners (lead_lag,
        # whose PRIMARY exit IS the trail) round-trip from profit to loss instead of trailing out.
        self._trail_dist: dict[str, float] = {}  # symbol → trailing distance (price units); 0 = off
        self._peak: dict[str, float] = {}  # symbol → best favorable price (high=long / low=short)
        # Realized PnL accumulated across the session (from CLOSED trades), fed to the loss
        # breakers so they can actually trip (Section 17). Per-symbol for the per-symbol breaker.
        self._realized_pnl: float = 0.0
        self._per_symbol_pnl: dict[str, float] = {}
        # Day/week rollover for the loss breakers: daily_pnl/weekly_pnl are "realized PnL so far
        # today / this week", but _realized_pnl is a LIFETIME accumulator — feeding it raw made a
        # multi-day session charge every prior day's loss against the DAILY limit, tripping it once
        # and never recovering. Snapshot the lifetime total at each day/week boundary so the
        # windowed P&L is (lifetime - boundary snapshot). Keyed off candidate decision_ts (UTC).
        self._sim_peak_equity: float = 0.0  # running peak of simulated equity (drawdown breaker)
        self._day_key: int | None = None
        self._week_key: int | None = None
        self._day_start_realized: float = 0.0
        self._week_start_realized: float = 0.0
        # Real account state (a live/demo venue only): when present, the breakers reason over REAL
        # equity (daily-loss + drawdown) — the engine's simulated exit path is inert when exits
        # happen exchange-side, so this is what makes the loss breakers live in production.
        self._free_margin: float | None = None
        self._account_equity: float | None = None
        self._session_start_equity: float | None = None
        self._peak_equity: float = 0.0
        # Real-venue CALENDAR loss-window anchors (M10): equity at the start of the current UTC
        # day / ISO week (Monday-anchored). bk_daily/bk_weekly = equity − anchor, so a multi-day
        # session rolls its daily window at midnight instead of charging the whole session against
        # today's limit, and (with the M9 store) a mid-day restart keeps the morning anchor
        # instead of forgetting the day's losses.
        self._rv_day_key: int | None = None
        self._rv_week_key: int | None = None
        self._rv_day_anchor: float | None = None
        self._rv_week_anchor: float | None = None
        # Consecutive realized losses (M7): incremented on every realized losing close, reset on
        # a winner — feeds the consecutive-loss cooldown breaker, which was previously dead (its
        # input was hardcoded 0 in every production path).
        self._consecutive_losses: int = 0
        # Funding actually paid (cost convention, M7): realized on closes + reset each ISO week
        # (the funding breaker's "period"); open-position accrual is added at read time.
        self._funding_paid_total: float = 0.0
        self._week_start_funding: float = 0.0
        # Observed fill slippage window (M7) → abnormal_slippage_active breaker input.
        self._recent_slippage: deque[float] = deque(maxlen=_SLIP_WINDOW)
        self._max_slippage_frac: float = exec_cfg.max_slippage_frac
        # Decision-bar interval (L10): exit_ts horizons and time-stops are counted in BARS of the
        # session's actual timeframe — set via set_bar_interval() by the live loop / lake replay;
        # the 1m default only backs offline synthetic tests that never declare a timeframe.
        self._bar_interval_ms: int = 60_000
        # Optional M9 state store (attached for real-venue sessions): persists tripped halts,
        # calendar anchors, peak equity and the loss streak across restarts.
        self._state_store = None

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def new_session(self, session_id: str | None = None) -> PaperSession:
        return PaperSession(session_id=session_id or str(uuid.uuid4()))

    def set_bar_interval(self, interval_ms: int) -> None:
        """Pin the session's decision-bar interval (L10) so hold_bars horizons (exit_ts,
        time-stops) are counted in THIS session's bars, not hardcoded 1-minute bars."""
        if interval_ms > 0:
            self._bar_interval_ms = int(interval_ms)

    def attach_state_store(self, store) -> None:
        """Attach the M9 persistent breaker-state store (real-venue sessions).

        Restores the state that must survive a restart — a tripped manual-reset halt keeps
        refusing entries (see _process_one), the calendar loss-window anchors (M10) keep the
        day's/week's losses, peak equity keeps the drawdown high-water mark, and the loss
        streak keeps the cooldown counter. The ONLY thing that clears a tripped halt is the
        operator's explicit reset (``qbot risk-reset``)."""
        self._state_store = store
        st = store.load()
        self._consecutive_losses = int(st.get("consecutive_losses", 0) or 0)
        self._peak_equity = max(self._peak_equity, float(st.get("peak_equity", 0.0) or 0.0))
        self._rv_day_key = st.get("day_key")
        self._rv_week_key = st.get("week_key")
        self._rv_day_anchor = st.get("day_anchor_equity")
        self._rv_week_anchor = st.get("week_anchor_equity")

    def _persist_state(self) -> None:
        """Write the restart-critical breaker state through to the M9 store (no-op without one)."""
        if self._state_store is None:
            return
        self._state_store.update(
            consecutive_losses=self._consecutive_losses,
            peak_equity=self._peak_equity,
            day_key=self._rv_day_key,
            week_key=self._rv_week_key,
            day_anchor_equity=self._rv_day_anchor,
            week_anchor_equity=self._rv_week_anchor,
        )

    def _register_realized(self, symbol: str, pnl: float, funding: float = 0.0) -> None:
        """Book one realized trade result into every breaker accumulator (M7).

        Single choke-point for the three close paths (simulated exit, immediate synthetic
        exit, exchange-side exit) so the loss breakers, the per-symbol breaker, the
        consecutive-loss cooldown and the funding breaker all see every realized result."""
        self._realized_pnl += pnl
        self._per_symbol_pnl[symbol] = self._per_symbol_pnl.get(symbol, 0.0) + pnl
        self._consecutive_losses = self._consecutive_losses + 1 if pnl < 0 else 0
        self._funding_paid_total += funding
        self._persist_state()

    def _cumulative_funding_paid(self) -> float:
        """Funding paid THIS ISO week (realized since the week boundary + open accrual), the
        funding breaker's input (M7 — previously never passed despite being tracked)."""
        realized = self._funding_paid_total - self._week_start_funding
        return realized + sum(self._position_funding.values())

    def _abnormal_slippage_active(self) -> bool:
        """M7: observed-slippage breaker input — mean fill slippage over the last
        ``_SLIP_WINDOW`` fills exceeds the execution policy's ``max_slippage_frac``
        (requires ``_SLIP_MIN_FILLS`` observations so one bad print can't halt the book)."""
        if len(self._recent_slippage) < _SLIP_MIN_FILLS:
            return False
        return sum(self._recent_slippage) / len(self._recent_slippage) > self._max_slippage_frac

    def _calendar_windows(self, ts: int, equity: float) -> tuple[float, float]:
        """Real-venue daily/weekly loss windows anchored to CALENDAR boundaries (M10).

        Returns ``(daily_pnl, weekly_pnl)`` = equity − the equity anchored at the current UTC
        day / ISO-week (Monday 00:00 UTC) boundary. Anchors roll when the boundary passes and
        persist via the M9 store, so a restart mid-day restores the morning anchor instead of
        re-anchoring at post-loss equity (which would forget the day's losses)."""
        day = ts // 86_400_000
        week = (ts - 4 * 86_400_000) // (7 * 86_400_000)  # epoch Thu → shift 4d to Monday anchor
        changed = False
        if self._rv_day_key != day or self._rv_day_anchor is None:
            self._rv_day_key = day
            self._rv_day_anchor = equity
            changed = True
        if self._rv_week_key != week or self._rv_week_anchor is None:
            self._rv_week_key = week
            self._rv_week_anchor = equity
            changed = True
        if changed:
            self._persist_state()
        return equity - self._rv_day_anchor, equity - self._rv_week_anchor

    def open_positions(self, price_of) -> list[dict]:
        """Snapshot the currently-open positions marked to market — for the dashboard's live
        open-positions panel. ``price_of(symbol)`` returns the latest price (None → mark at entry,
        i.e. 0 unrealized). Matches the basket loop's position-dict shape."""
        out: list[dict] = []
        for sym, pos in self._open_positions.items():
            mark = price_of(sym)
            mark = pos.entry_price if mark is None else float(mark)
            strat, entry_ts = self._position_meta.get(sym, ("", 0))
            funding = self._position_funding.get(sym, 0.0)  # cost convention; net into unrealized
            out.append({
                "symbol": sym, "strategy": strat, "side": pos.side, "qty": pos.qty,
                "entry_price": pos.entry_price, "mark_price": mark, "notional": pos.notional,
                "unrealized_pnl": pos.side * (mark - pos.entry_price) * pos.qty - funding,
                "funding": funding,
                "entry_ts": entry_ts,
            })
        return out

    def _exit_fee(self, symbol: str, notional: float) -> float:
        """Taker fee for a simulated reduce-only market close (mirrors SimulatedVenue._fee)."""
        spec = self._meta.spec(symbol)
        if spec is None:
            return 0.0
        rate = spec.fields.get("taker_fee", 0.0)
        return abs(notional) * float(rate if isinstance(rate, (int, float)) else 0.0)

    def _roll_loss_windows(self, ts: int) -> None:
        """Snapshot the lifetime realized total at each UTC day / week boundary so the daily/weekly
        loss breakers see only THIS day's / week's realized P&L. Called BEFORE any realized P&L is
        booked for ``ts`` (both the bar's exits and its entries) so a loss closed on the first bar
        of a new day is attributed to the new day, not snapshotted away from it. The week is
        aligned to Monday 00:00 UTC (epoch is a Thursday, so a raw //7d would roll on Thursdays).
        """
        if ts <= 0:
            return
        day = ts // 86_400_000
        week = (ts - 4 * 86_400_000) // (7 * 86_400_000)  # epoch Thu → shift 4d to a Monday anchor
        if self._day_key != day:
            self._day_key = day
            self._day_start_realized = self._realized_pnl
        if self._week_key != week:
            self._week_key = week
            self._week_start_realized = self._realized_pnl
            # The funding breaker's period is the ISO week too (M7): snapshot so
            # cumulative_funding_paid measures THIS week's bleed, not the session lifetime.
            self._week_start_funding = self._funding_paid_total

    def set_funding_source(self, source: Callable[[str], list[dict]] | None) -> None:
        """Wire the per-symbol funding-event lookup (``[{"ts", "funding_rate"}, ...]``) so open
        paper positions accrue funding exactly as the backtest does. The live run sources it from
        the feed; a real-venue (demo/testnet/live) run leaves it unset — the exchange charges
        funding itself, reflected in real account equity, so simulating it would double-count."""
        self._funding_source = source

    def _charge_open_funding(self, now_ts: int) -> None:
        """Settle every funding event in ``(watermark, now_ts]`` (and after entry) onto each open
        position — mirrors the basket / backtest funding helper. Iterating the bounded event window
        each call is robust to the live feed's funding series sliding/rebuilding under us."""
        if self._funding_source is None:
            return
        for sym, pos in self._open_positions.items():
            try:
                events = self._funding_source(sym) or []
            except Exception:  # noqa: BLE001 - a feed hiccup must never crash the trading loop
                continue
            last = self._funding_watermark.get(sym, 0)  # last settled funding ts (init = entry ts)
            for ev in events:
                ets = int(ev["ts"])
                if last < ets <= now_ts:
                    self._position_funding[sym] = self._position_funding.get(sym, 0.0) + (
                        self._funding.payment(
                            side=pos.side,
                            notional=pos.notional,
                            funding_rate=float(ev["funding_rate"]),
                        )
                    )
                    last = ets
            self._funding_watermark[sym] = last

    def simulate_paper_exits(
        self, price_of, now_ts: int, session: PaperSession, *, bar_iv: int = 0
    ) -> int:
        """Close held PAPER positions whose bracket (stop / trailing-stop / take-profit) or
        time-stop is breached by the latest bar — the exchange-side exit a real venue would fill but
        the offline SimulatedVenue does not. Without this a per-symbol paper position
        (exit_move_frac=0) opens and is held forever, so the session books no closed trades and no
        realized P&L. The trailing stop ratchets the effective stop with the peak favorable price
        (parity with the backtest / a live exchange-native trailing order), and exits carry a
        distinct ``trailing_stop`` reason so trail-outs are visible vs hard stops.

        ``price_of(symbol)`` returns the latest close; a close-based check is conservative (it can
        miss an intrabar wick) but never invents a fill. Returns the number of positions closed.
        Real venues manage exits themselves, so the loop only calls this in paper mode."""
        # Roll the daily/weekly loss windows for THIS bar before booking any exit, so a loss closed
        # on the first bar of a new day counts toward the new day's daily-loss breaker (B2/R3).
        self._roll_loss_windows(now_ts)
        # Accrue funding on every held position crossing a funding timestamp this bar (parity with
        # the backtest), BEFORE marking exits so a leg that closes here realizes its full funding.
        self._charge_open_funding(now_ts)
        closed = 0
        for sym in list(self._open_positions):
            price = price_of(sym)
            if price is None:
                continue
            price = float(price)
            pos = self._open_positions[sym]
            stop, tp, hold_bars, entry_ts = self._exit_levels.get(sym, (0.0, 0.0, 0, 0))
            trail_dist = self._trail_dist.get(sym, 0.0)
            # Peak = best favorable price BEFORE this close (ratcheted at the end of the tick), so a
            # fresh favorable close can't tighten the very stop it's then tested against — mirrors
            # the backtest's no-look-ahead ordering (src/backtest/engine.py).
            peak = self._peak.get(sym, pos.entry_price)
            reason: str | None = None
            if pos.side > 0:
                # Effective stop = fixed stop ratcheted UP by the trailing stop (peak − trail_dist).
                eff_stop = max(stop, peak - trail_dist) if (trail_dist > 0 and stop > 0) else stop
                if eff_stop > 0 and price <= eff_stop:
                    reason = "trailing_stop" if eff_stop > stop else "stop"
                elif tp > 0 and price >= tp:
                    reason = "take_profit"
            else:  # short: stop is ABOVE entry, take-profit BELOW; the trail ratchets it DOWN
                eff_stop = min(stop, peak + trail_dist) if (trail_dist > 0 and stop > 0) else stop
                if eff_stop > 0 and price >= eff_stop:
                    reason = "trailing_stop" if eff_stop < stop else "stop"
                elif tp > 0 and price <= tp:
                    reason = "take_profit"
            if (
                reason is None and bar_iv > 0 and hold_bars > 0 and entry_ts > 0
                and now_ts - entry_ts >= hold_bars * bar_iv
            ):
                reason = "time_stop"
            if reason is None:
                # No exit → ratchet the peak with this close for the NEXT tick's trailing stop.
                if trail_dist > 0:
                    self._peak[sym] = max(peak, price) if pos.side > 0 else min(peak, price)
                continue
            self._book_paper_exit(sym, price, reason, now_ts, session)
            closed += 1
        return closed

    def _book_paper_exit(
        self, symbol: str, exit_price: float, exit_reason: str, now_ts: int, session: PaperSession
    ) -> None:
        """Flatten one held paper position and CLOSE its existing open trade record in place.

        The entry appended a PaperTrade with exit_reason='open' (its pnl was just the entry fee);
        closing updates that same row (exit price/reason/ts, total fee, realized pnl) rather than
        appending a duplicate, so executed_count stays one-per-position. Releases the risk slot and
        cancels the resting stop/TP legs on the venue (no orphan orders)."""
        pos = self._open_positions.pop(symbol, None)
        if pos is None:
            return
        self._position_meta.pop(symbol, None)
        self._exit_levels.pop(symbol, None)
        # Funding accrued while held (cost convention) — realized into pnl, like the backtest.
        funding = self._position_funding.pop(symbol, 0.0)
        self._funding_watermark.pop(symbol, None)
        self._trail_dist.pop(symbol, None)
        self._peak.pop(symbol, None)
        self._venue.close_position(symbol)  # drop mirror + cancel the resting bracket legs
        exit_fee = self._exit_fee(symbol, exit_price * pos.qty)
        raw_pnl = (exit_price - pos.entry_price) * pos.side * pos.qty
        trade = next(
            (t for t in reversed(session.trades)
             if t.symbol == symbol and t.exit_reason == "open"),
            None,
        )
        if trade is not None:
            total_fee = trade.fee + exit_fee  # entry fee (already on the record) + exit fee
            pnl = raw_pnl - total_fee - funding
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.exit_ts = now_ts
            trade.fee = total_fee
            trade.funding = funding
            trade.pnl = pnl
            trade.pnl_r = pnl / pos.risk_amount if pos.risk_amount > 0 else 0.0
        else:  # no open record (shouldn't happen in the live path) — book a standalone close
            pnl = raw_pnl - exit_fee - funding
            strat, entry_ts = self._position_meta.get(symbol, ("", 0))
            session.trades.append(PaperTrade(
                trade_id=str(uuid.uuid4())[:8], symbol=symbol, strategy=strat, side=pos.side,
                qty=pos.qty, entry_price=pos.entry_price, stop_price=0.0, tp_price=0.0,
                regime=pos.regime, session=0, decision_ts=entry_ts, entry_ts=entry_ts or now_ts,
                exit_ts=now_ts, exit_price=exit_price, exit_reason=exit_reason, fee=exit_fee,
                slippage_cost=0.0, pnl=pnl,
                pnl_r=pnl / pos.risk_amount if pos.risk_amount > 0 else 0.0,
                has_exchange_side_stop=True, execution_route="taker", spread_bps_at_entry=0.0,
                slippage_frac=0.0, funding=funding,
            ))
        self._register_realized(symbol, pnl, funding)

    def book_exchange_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_fee: float,
        now_ts: int,
        session: PaperSession,
        *,
        exit_reason: str = "exchange_exit",
    ) -> bool:
        """Book the realized result of a position that closed ON THE EXCHANGE (H7).

        Real-venue exits happen exchange-side (the resident SL/TP/trailing fired, or the
        bot sent a reduce-only market close for a time-stop) — the engine's simulated exit
        path never fires, so without this the entry's PaperTrade stays ``exit_reason="open"``
        with ``pnl = -entry_fee`` forever and every persisted net_pnl / win_rate /
        expectancy_r is structurally wrong. This CLOSES the existing open trade record in
        place (real exit price + real exit fee from the caller), feeds the realized P&L to
        the loss breakers, and prunes every engine mirror — it never places orders (the
        position is already flat on the exchange) and NEVER fabricates an entry: a position
        with no open trade record (adopted at startup from a prior process) just clears its
        mirrors, logs, and returns False."""
        self._roll_loss_windows(now_ts)
        pos = self._open_positions.pop(symbol, None)
        self._position_meta.pop(symbol, None)
        self._exit_levels.pop(symbol, None)
        funding = self._position_funding.pop(symbol, 0.0)
        self._funding_watermark.pop(symbol, None)
        self._trail_dist.pop(symbol, None)
        self._peak.pop(symbol, None)
        trade = next(
            (t for t in reversed(session.trades)
             if t.symbol == symbol and t.exit_reason == "open"),
            None,
        )
        if trade is None:
            _log.warning(
                "exchange_exit_without_entry_record", symbol=symbol, exit_reason=exit_reason,
                hint="position adopted from a prior process — mirrors cleared, no trade booked",
            )
            return False
        raw_pnl = (exit_price - trade.entry_price) * trade.side * trade.qty
        total_fee = trade.fee + exit_fee  # entry fee (already on the record) + real exit fee
        pnl = raw_pnl - total_fee - funding
        risk_amount = (
            pos.risk_amount
            if pos is not None and pos.risk_amount > 0
            else abs(trade.entry_price - trade.stop_price) * trade.qty
        )
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.exit_ts = now_ts
        trade.fee = total_fee
        trade.funding = funding
        trade.pnl = pnl
        trade.pnl_r = pnl / risk_amount if risk_amount > 0 else 0.0
        # Realized → the loss/drawdown/streak/funding breakers see the booked result (Section 17).
        self._register_realized(symbol, pnl, funding)
        return True

    def process_candidates(
        self,
        inputs: list[PaperCandidateInput],
        session: PaperSession,
    ) -> PaperSession:
        """Process a batch of candidate inputs through the full paper pipeline.

        Each input is processed in sequence (as one decision bar). The kill
        switch and reconciliation are respected on every call.
        """
        # Snapshot real account state ONCE per batch (a real venue only) so the risk manager's
        # pre-trade free-margin blocker AND the loss/drawdown breakers reason over REAL account
        # state; SimulatedVenue has no such methods, so these stay None and paper falls back to
        # the simulated equity / accumulated realized PnL.
        self._free_margin = None
        fetch_fm = getattr(self._venue, "fetch_free_margin", None)
        if callable(fetch_fm):
            try:
                self._free_margin = fetch_fm()
            except Exception:  # noqa: BLE001 - a balance hiccup must not stop processing
                self._free_margin = None
        fetch_eq = getattr(self._venue, "fetch_account_equity", None)
        if callable(fetch_eq):
            try:
                eq = fetch_eq()
            except Exception:  # noqa: BLE001
                eq = None
            if eq is not None:
                self._account_equity = eq
                if self._session_start_equity is None:
                    self._session_start_equity = eq
                self._peak_equity = max(self._peak_equity, eq)
        for inp in inputs:
            self._process_one(inp, session)
        session.ended_at = datetime.now(UTC)
        return session

    def engage_kill_switch(self, session: PaperSession) -> None:
        """Engage the kill switch and record the event in the session."""
        self._kill_switch.engage("paper_test")
        session.kill_switch_exercised = True
        session.kill_switch_events.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "action": "engaged",
                "source": "paper_engine",
            }
        )

    def disengage_kill_switch(self, session: PaperSession) -> None:
        """Disengage the kill switch (for test reset)."""
        self._kill_switch.disengage()
        session.kill_switch_events.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "action": "disengaged",
                "source": "paper_engine",
            }
        )

    def run_reconciliation(
        self,
        session: PaperSession,
        *,
        inject_foreign_order: bool = False,
    ) -> bool:
        """Run reconciliation check and return True if halt is triggered.

        When ``inject_foreign_order=True`` a synthetic unowned order is
        temporarily added to the venue to exercise the halt path (PAPER-A).
        The reconciler detects it (lacks the ownership prefix) and sets
        halt_required=True.
        """
        from src.execution.order import Order, OrderType

        # Snapshot current owned positions/orders.
        bot_positions = list(self._venue.positions.keys())
        # Bot's known order ids (all orders currently in the venue book).
        known_order_ids: set[str] = set(self._venue.open_orders.keys())
        known_position_symbols: set[str] = set(self._venue.positions.keys())

        orders = dict(self._venue.open_orders)
        positions = dict(self._venue.positions)

        if inject_foreign_order:
            # Use a client_id that does NOT carry the bot's prefix — the reconciler
            # detects it as foreign (Section 7 / ORDER-OWN gate).
            foreign_id = "MANUAL_TRADE_UNKNOWN_001"
            foreign_order = Order(
                client_id=foreign_id,
                symbol="BTC/USDT:USDT",
                side="buy",
                qty=0.01,
                order_type=OrderType.LIMIT,
                price=50_000.0,
            )
            orders[foreign_id] = foreign_order
            # Do NOT add to known_order_ids so reconciler sees it as unknown.

        result = self._reconciler.reconcile(
            orders,
            positions,
            known_order_ids=known_order_ids,
            known_position_symbols=known_position_symbols,
        )
        halt_triggered = result.halt_required

        if halt_triggered:
            session.foreign_order_halt_triggered = True

        session.reconciliation_events.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "bot_positions": bot_positions,
                "injected_foreign": inject_foreign_order,
                "halt_triggered": halt_triggered,
                "unknown_orders": list(result.unknown_orders),
            }
        )
        return halt_triggered

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _process_one(self, inp: PaperCandidateInput, session: PaperSession) -> None:
        candidate = inp.candidate
        ks_state = "engaged" if self._kill_switch.engaged() else "clear"

        # M9: a persisted manual-reset halt (daily-loss / max-drawdown / weekly-loss tripped in
        # THIS OR A PRIOR process) refuses every new entry until the operator explicitly resets
        # it (`qbot risk-reset`). Checked before anything else — a restart must not trade.
        halted = self._state_store.tripped_reason() if self._state_store is not None else None
        if halted:
            reject_reason = f"risk_halt_pending_manual_reset({halted})"
            session.rejected.append(
                RejectedPaperCandidate(
                    symbol=candidate.symbol,
                    strategy=candidate.strategy,
                    side=candidate.side,
                    regime=candidate.regime,
                    decision_ts=candidate.decision_ts,
                    reason=reject_reason,
                )
            )
            session.decision_logs.append(
                self._decision_log(candidate, "reject", reject_reason, False, ks_state)
            )
            return

        # Build the account state for risk from the CURRENTLY-OPEN positions in the venue
        # (not an empty tuple): this is what makes the Section-17 portfolio caps —
        # max-concurrent-per-symbol/total/regime, portfolio heat, net-beta — actually
        # enforce against existing exposure.
        # When a real venue reports account equity (live/demo), the breakers reason over REAL
        # equity — daily-loss and max-drawdown then catch a losing session regardless of how the
        # exit happened (exchange-side SL/TP). In paper, fall back to the configured equity and the
        # simulated realized-PnL accumulator.
        if self._account_equity is not None:
            bk_equity = self._account_equity
            bk_peak = max(self._peak_equity, bk_equity, inp.peak_equity)
            self._peak_equity = bk_peak
            # M10: daily/weekly windows anchor to CALENDAR boundaries (UTC day / ISO week), not
            # the session start — multi-day sessions roll the daily anchor at midnight, and the
            # M9-persisted anchors survive a mid-day restart so the day's losses aren't forgotten.
            ts = int(getattr(candidate, "decision_ts", 0) or 0)
            if ts <= 0:
                ts = int(datetime.now(UTC).timestamp() * 1000)
            bk_daily, bk_weekly = self._calendar_windows(ts, bk_equity)
        else:
            # Roll the daily/weekly windows at UTC day/week boundaries (off the decision time) so a
            # multi-day session does not charge old losses against today's limit forever (B2).
            self._roll_loss_windows(int(getattr(candidate, "decision_ts", 0) or 0))
            # Equity reflects realized P&L (B3) so the drawdown breaker actually moves and the
            # daily-loss fraction scales off current equity, not the static seed.
            bk_equity = inp.equity + self._realized_pnl
            self._sim_peak_equity = max(self._sim_peak_equity, bk_equity, inp.peak_equity)
            bk_peak = self._sim_peak_equity
            bk_daily = inp.daily_pnl + (self._realized_pnl - self._day_start_realized)
            bk_weekly = self._realized_pnl - self._week_start_realized
        portfolio = PortfolioState(
            equity=bk_equity, positions=tuple(self._open_positions.values())
        )
        # M7: every breaker input is LIVE now — the loss streak comes from the engine's own
        # realized-close tracker (callers may still inject a higher streak for tests), funding
        # from the existing accrual tracker, and abnormal slippage from observed recent fills.
        breakers = BreakerInputs(
            equity=bk_equity,
            peak_equity=bk_peak,
            daily_pnl=bk_daily,
            consecutive_losses=max(inp.consecutive_losses, self._consecutive_losses),
            abnormal_slippage_active=self._abnormal_slippage_active(),
            reconciled=not inp.inject_foreign_order,
            weekly_pnl=bk_weekly,
            cumulative_funding_paid=self._cumulative_funding_paid(),
            per_symbol_pnl=dict(self._per_symbol_pnl),
        )
        account = AccountState(
            portfolio=portfolio,
            breakers=breakers,
            unknown_order_present=inp.inject_foreign_order,
            free_margin=getattr(self, "_free_margin", None),
            # liquidation_price deliberately omitted (None): no pre-trade producer exists, so
            # the liquidation-distance check is EXPLICITLY not evaluated (M7 — see AccountState
            # and manager step 4a for the documented choice).
        )

        # Setup-quality HARD BLOCKERS (Section 15): a no-trade regime, toxic spread, negative
        # expected-value-after-costs, stale data, halted symbol, etc. can NEVER be bypassed.
        # Previously the score was computed and DISCARDED, so these blockers never ran in the
        # paper/live loop (this engine backs both). The score threshold + ranked selection remain
        # a separate (frequency-changing) policy; only the safety blockers are enforced here.
        score = self._scorer.score(candidate)
        if score.blockers:
            reject_reason = f"setup_blocked({','.join(score.blockers)})"
            session.rejected.append(
                RejectedPaperCandidate(
                    symbol=candidate.symbol,
                    strategy=candidate.strategy,
                    side=candidate.side,
                    regime=candidate.regime,
                    decision_ts=candidate.decision_ts,
                    reason=reject_reason,
                )
            )
            session.decision_logs.append(
                self._decision_log(candidate, "reject", reject_reason, False, ks_state)
            )
            return

        # Risk approval.
        decision = self._risk.evaluate(candidate, account)

        if not decision.approved:
            # M9: a manual-reset breaker (daily-loss / max-drawdown / weekly-loss) just tripped —
            # latch it in the persistent store so the halt survives a restart. Cooldown breakers
            # (consecutive-loss, funding, slippage, per-symbol) recompute and are not latched.
            if (
                self._state_store is not None
                and decision.action == "block"
                and decision.blocker
                and decision.blocker.startswith(_MANUAL_RESET_BLOCKERS)
            ):
                self._state_store.trip(
                    decision.blocker,
                    ts_ms=int(getattr(candidate, "decision_ts", 0) or 0) or None,
                )
            reject_reason = f"risk_{decision.action}"
            session.rejected.append(
                RejectedPaperCandidate(
                    symbol=candidate.symbol,
                    strategy=candidate.strategy,
                    side=candidate.side,
                    regime=candidate.regime,
                    decision_ts=candidate.decision_ts,
                    reason=reject_reason,
                )
            )
            session.decision_logs.append(
                self._decision_log(candidate, "reject", reject_reason, False, ks_state)
            )
            return

        # Execution (revalidate + bracket placement).
        result = self._exec.execute(candidate, decision)

        if not result.placed:
            reject_reason = f"exec_{result.reason}"
            session.rejected.append(
                RejectedPaperCandidate(
                    symbol=candidate.symbol,
                    strategy=candidate.strategy,
                    side=candidate.side,
                    regime=candidate.regime,
                    decision_ts=candidate.decision_ts,
                    reason=reject_reason,
                )
            )
            session.decision_logs.append(
                self._decision_log(candidate, "reject", reject_reason, True, ks_state)
            )
            return

        # Build the paper trade record.
        fill = result.fill
        position = result.position
        assert fill is not None
        assert position is not None

        # M7: feed the OBSERVED fill slippage into the abnormal-slippage breaker window.
        self._recent_slippage.append(float(fill.slippage_frac))

        entry_price = fill.actual_price
        stop_price = entry_price * (1 - candidate.stop_frac * candidate.side)
        # L12: honor the live NO_FIXED_TP sentinel convention — tp_frac ≥ NO_FIXED_TP_FRAC means
        # "no fixed take-profit" (momentum; the trail/time-stop is the exit), so paper arms NO TP
        # level (0.0), exactly like the live order builder. Previously a SHORT with a sentinel in
        # [0.5, 1) got a REACHABLE paper TP at entry×(1−tp_frac) that live never places.
        tp_price = (
            0.0
            if candidate.tp_frac >= NO_FIXED_TP_FRAC
            else entry_price * (1 + candidate.tp_frac * candidate.side)
        )
        exit_move = inp.exit_move_frac
        exit_price = entry_price * (1 + exit_move)
        funding_amount = 0.0
        if inp.exit_reason is not None:
            # Replay path (H9): the exit was already resolved by an intrabar bracket walk that
            # applied the backtest's stop/trailing/take-profit geometry AND accrued funding — trust
            # it directly rather than re-classify from the endpoint move (which misses intrabar hits
            # and books an intrabar stop that recovered by the horizon as a winner).
            exit_reason = inp.exit_reason
            funding_amount = inp.funding_frac * entry_price * fill.qty
        else:
            # Classify the exit by which bracket the forward price reached. ``exit_move`` is the RAW
            # (non-side-adjusted) price move; ``exit_move * side > 0`` means it moved our way → the
            # take-profit side, otherwise the stop side. For a SHORT the levels invert (stop
            # ABOVE entry, TP BELOW), so the comparison flips with side.
            exit_reason = "open"
            if exit_move != 0:
                favorable = exit_move * candidate.side > 0
                # tp_price == 0.0 encodes NO fixed TP (the sentinel, L12): never classify a
                # take-profit exit for it — the position stays open for the trail/time-stop.
                if favorable and tp_price > 0 and (
                    (candidate.side > 0 and exit_price >= tp_price)
                    or (candidate.side < 0 and exit_price <= tp_price)
                ):
                    exit_reason = "take_profit"
                elif not favorable and (
                    (candidate.side > 0 and exit_price <= stop_price)
                    or (candidate.side < 0 and exit_price >= stop_price)
                ):
                    exit_reason = "stop"

        raw_pnl = (exit_price - entry_price) * candidate.side * fill.qty
        fee = fill.fee
        pnl = raw_pnl - fee - funding_amount
        risk_amount = (
            decision.risk_amount
            if decision.risk_amount
            else abs(entry_price - stop_price) * fill.qty
        )
        pnl_r = pnl / risk_amount if risk_amount > 0 else 0.0

        trade_id = str(uuid.uuid4())[:8]
        now_ts = int(datetime.now(UTC).timestamp() * 1000)

        paper_trade = PaperTrade(
            trade_id=trade_id,
            symbol=candidate.symbol,
            strategy=candidate.strategy,
            side=candidate.side,
            qty=fill.qty,
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
            regime=candidate.regime,
            session=candidate.session,
            decision_ts=candidate.decision_ts,
            entry_ts=now_ts,
            # L10: the hold horizon is counted in the SESSION's decision bars (set_bar_interval),
            # not hardcoded 1-minute bars — a 12-bar hold on 1h bars is 12 hours, not 12 minutes.
            exit_ts=now_ts + inp.hold_bars * self._bar_interval_ms,
            exit_price=exit_price,
            exit_reason=exit_reason,
            fee=fee,
            funding=funding_amount,  # replay-path accrued funding (H9); 0.0 for live/realtime
            slippage_cost=fill.slippage_cost,
            pnl=pnl,
            pnl_r=pnl_r,
            has_exchange_side_stop=position.has_exchange_side_stop(),
            execution_route="maker" if fill.maker else "taker",
            spread_bps_at_entry=fill.spread_bps_at_order,
            slippage_frac=fill.slippage_frac,
        )
        session.trades.append(paper_trade)
        # Keep the engine's portfolio mirror in lock-step with the simulated exit so the
        # risk caps bind correctly on subsequent candidates: a position still "open" holds
        # its concurrency/heat/beta slot; a simulated stop/TP exit releases it (and the
        # venue book) so the slot frees. In live/demo the venue book is the real exchange
        # state, so this same mirror reflects genuine open exposure.
        if exit_reason == "open":
            self._open_positions[candidate.symbol] = Position(
                symbol=candidate.symbol,
                side=candidate.side,
                qty=fill.qty,
                entry_price=entry_price,
                risk_amount=risk_amount,
                beta_to_btc=self._risk.cfg.beta_to_btc(candidate.symbol),
                regime=candidate.regime,
            )
            self._position_meta[candidate.symbol] = (
                candidate.strategy, int(getattr(candidate, "decision_ts", 0) or 0)
            )
            self._exit_levels[candidate.symbol] = (
                stop_price, tp_price, int(getattr(candidate, "hold_bars", 0) or 0),
                int(getattr(candidate, "decision_ts", 0) or 0),
            )
            # Start funding accrual at entry: 0 booked, watermark = entry ts so only funding posted
            # strictly after entry is settled (mirrors the backtest's entry-anchored window).
            self._position_funding[candidate.symbol] = 0.0
            self._funding_watermark[candidate.symbol] = now_ts
            # Trailing-stop state: distance in price units (trail_frac × entry); peak seeded at
            # entry (matches the backtest's `peak=entry_price`). trail_frac 0 ⇒ no trailing here.
            self._trail_dist[candidate.symbol] = (
                float(getattr(candidate, "trail_frac", 0.0) or 0.0) * entry_price
            )
            self._peak[candidate.symbol] = entry_price
        else:
            self._open_positions.pop(candidate.symbol, None)
            self._position_meta.pop(candidate.symbol, None)
            self._exit_levels.pop(candidate.symbol, None)
            self._position_funding.pop(candidate.symbol, None)
            self._funding_watermark.pop(candidate.symbol, None)
            self._trail_dist.pop(candidate.symbol, None)
            self._peak.pop(candidate.symbol, None)
            self._venue.positions.pop(candidate.symbol, None)
            # The trade closed → its PnL is REALIZED. Accumulate it for the loss/streak breakers
            # so a losing session halts new entries on the next candidate (Section 17).
            self._register_realized(candidate.symbol, pnl)
        session.decision_logs.append(
            self._decision_log(candidate, "execute", "approved", True, ks_state)
        )
        session.explainability.append(self._explain(candidate, paper_trade, fill))

    def _explain(self, candidate: Candidate, trade, fill):
        """Build the Section-24 TradeExplainability for an executed trade."""
        from src.explainability import TradeExplainability

        edge_after = (
            candidate.expected_edge_frac * trade.entry_price - fill.fee - fill.slippage_cost
        )
        return TradeExplainability(
            trade_id=trade.trade_id,
            symbol=candidate.symbol,
            strategy_id=candidate.strategy,
            setup_type=candidate.strategy,
            regime=candidate.regime,
            signal_features=dict(candidate.features),
            expected_edge_after_costs=edge_after,
            expected_fees=fill.fee,
            expected_slippage=fill.slippage_cost,
            expected_funding_impact=None,
            stop_price=trade.stop_price,
            invalidation_conditions=["stop_hit", "regime_change", "time_stop"],
            execution_route=trade.execution_route,
            risk_approved=True,
            risk_reason="approved",
            model_version=None,
            learner_version=None,
            config_version=self._config_version,
            universe_version=self._universe_version,
            why_selected=f"{candidate.strategy} setup ranked in regime {candidate.regime}",
            why_rejected_others=[],
        )

    def _decision_log(
        self,
        candidate: Candidate,
        action: str,
        reason: str,
        risk_approved: bool,
        ks_state: str,
    ) -> PaperDecisionLog:
        from src.ml.features import candidate_to_row

        entry_price = candidate.entry_price
        # L14: estimate fees from the symbol's VERIFIED metadata (same source the venue charges
        # from), falling back to a generic taker rate only when the spec carries no fee.
        spec = self._meta.spec(candidate.symbol)
        rate = spec.fields.get("taker_fee") if spec is not None else None
        fee_rate = float(rate) if isinstance(rate, (int, float)) and rate > 0 else 0.0006
        fee_est = entry_price * fee_rate * 2
        slip_est = entry_price * candidate.slippage_est
        edge_est = candidate.expected_edge_frac * entry_price - fee_est - slip_est
        return PaperDecisionLog(
            entry_ts=datetime.now(UTC),
            symbol=candidate.symbol,
            strategy=candidate.strategy,
            regime=candidate.regime,
            side=candidate.side,
            action=action,
            reason=reason,
            risk_approved=risk_approved,
            expected_edge=round(edge_est, 6),
            expected_fee=round(fee_est, 6),
            expected_slippage=round(slip_est, 6),
            config_version=self._config_version,
            universe_version=self._universe_version,
            strategy_version=candidate.strategy_version,
            kill_switch_state=ks_state,
            # Persist the canonical ML feature row so the real label pipeline (H18) can rebuild a
            # training feature vector for this decision and join it to the realized outcome.
            features=candidate_to_row(candidate),
        )
