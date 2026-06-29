"""Live/paper execution for cross-sectional (carry/factor) basket strategies.

The per-trade live path (`PaperTradingEngine`) can't run a basket strategy: a carry edge must be
held delta-neutral or the directional variance of unhedged legs buries it (that is exactly why
funding_carry runs through the `CrossSectionalEngine` in the backtest, not the per-trade engine).
This drives the SAME basket math in real time — each rebalance it snapshots the cross-section and
calls the engine's tested rebalance / leg-open-close / funding helpers (so live and backtest agree,
the Parity Rule), then books each closed leg as a `PaperTrade`.

The loop consumes an injectable stream of cross-section snapshots, so it is fully unit-testable
offline; the real-time shell (`run_basket_paper_session`) feeds it from the live REST feed. Paper
mode books SIMULATED fills only — no real orders, no real funds.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestResult, SymbolInput, Trade
from src.backtest.portfolio import CrossSectionalEngine
from src.exchange.metadata import MetadataConfig
from src.paper.session import PAPER_BASE_EQUITY, PaperSession, PaperTrade

# A snapshot: (decision_ts, {symbol: bar}, {symbol: feature_row}) for one bar across the universe.
Snapshot = tuple[int, dict[str, dict], dict[str, dict]]


def _trade_to_paper(t: Trade, *, maker: bool) -> PaperTrade:
    """Map an engine leg-trade to a PaperTrade (a basket leg has no stop/TP — carry exits on
    rebalance / time, not a protective level)."""
    stop_frac = (t.risk_amount / t.notional) if t.notional > 0 else 0.0
    return PaperTrade(
        trade_id=str(uuid.uuid4())[:8],
        symbol=t.symbol,
        strategy=t.strategy,
        side=t.side,
        qty=t.qty,
        entry_price=t.entry_price,
        stop_price=t.entry_price * (1.0 - t.side * stop_frac),  # notional R reference only
        tp_price=0.0,
        regime=t.regime,
        session=t.session,
        decision_ts=t.entry_ts,
        entry_ts=t.entry_ts,
        exit_ts=t.exit_ts,
        exit_price=t.exit_price,
        exit_reason=t.exit_reason,
        fee=t.fee,
        slippage_cost=t.slippage_cost,
        pnl=t.pnl,
        pnl_r=t.pnl_r,
        has_exchange_side_stop=False,  # a basket leg is bot-managed, not an exchange-resident stop
        execution_route="maker" if maker else "taker",
        spread_bps_at_entry=0.0,
        slippage_frac=0.0,
    )


class BasketPaperLoop:
    """Drives the cross-sectional engine's basket math one bar at a time over a snapshot stream,
    booking each closed leg as a PaperTrade. ``by_symbol`` carries each symbol's funding_events +
    spread (so the engine's funding/cost helpers work); it is updated by the caller as data arrives.
    ``bar_interval_ms`` is the bar grid (for the engine's bars-held / rebalance math)."""

    def __init__(
        self,
        cfg: BacktestConfig,
        meta: MetadataConfig,
        strategy: object,
        *,
        bar_interval_ms: int,
        session: PaperSession,
        on_trade: Callable[[PaperTrade], None] | None = None,
        on_event: Callable[[str], None] | None = None,
        on_positions: Callable[[list[dict]], None] | None = None,
        initial_equity: float | None = None,
    ) -> None:
        self.engine = CrossSectionalEngine(cfg, meta, strategy)
        self.engine._grid_iv = max(1, int(bar_interval_ms))
        self.strategy = strategy
        self.session = session
        self.on_trade = on_trade
        self.on_event = on_event
        self.on_positions = on_positions
        self._holdings: dict = {}
        # PAPER sessions seed at the shared paper base so basket $ P&L lines up with the per-symbol
        # paper engine + the dashboard equity curve; the backtest path leaves this None and keeps
        # the config's account.initial_equity (a size-invariant numeraire there).
        eq = cfg.account.initial_equity if initial_equity is None else float(initial_equity)
        self._equity = eq
        self._last_rebal: int | None = None
        self._result = BacktestResult(initial_equity=eq)
        self._booked = 0
        self._rebal_ms = (
            int(self.engine.rebalance_hours * 3_600_000)
            if self.engine.rebalance_hours > 0
            else self.engine.rebalance_bars * self.engine._grid_iv
        )

    def step(
        self, ts: int, bars_at: dict[str, dict], rows_at: dict[str, dict], by_symbol: dict
    ) -> None:
        eng = self.engine
        # 1) Funding on every held leg (the carry).
        for sym, leg in list(self._holdings.items()):
            if sym in by_symbol:
                leg.next_funding_idx = eng._charge_funding(leg, by_symbol[sym], ts)
        # 2) Rebalance on cadence.
        if self._last_rebal is None or ts - self._last_rebal >= self._rebal_ms:
            # Score EXACTLY as the backtest engine does (the Parity Rule): residual modes
            # (residual_momentum) score from the rolling RETURNS series via _residual_score — the
            # per-row strategy.score() is a no-op marker for them, so calling it would yield zero
            # scores and the basket would NEVER form. Rebuild the returns from the live rolling
            # window so _residual_score behaves identically to the backtest.
            if eng._residual:
                eng._prepare_returns([si for si in by_symbol.values() if getattr(si, "bars", None)])
            scores: dict[str, float] = {}
            for sym, row in rows_at.items():
                if sym not in bars_at:
                    continue
                sc = eng._residual_score(sym, ts) if eng._residual else self.strategy.score(row)
                if sc is not None:
                    f = float(sc)
                    if f == f and abs(f) != float("inf"):  # finite
                        scores[sym] = f
            if len(scores) >= eng.min_universe:
                bars_by_ts = {s: {ts: b} for s, b in bars_at.items()}
                rows_by_ts = {s: {ts: r} for s, r in rows_at.items()}
                before = len(self._result.trades)
                self._equity = eng._rebalance(
                    self._holdings, scores, bars_by_ts, rows_by_ts, by_symbol, ts, self._equity,
                    self._result,
                )
                self._last_rebal = ts
                if self.on_event is not None:
                    self.on_event(
                        f"rebalanced: {len(self._holdings)} legs held, "
                        f"{len(self._result.trades) - before} closed, equity={self._equity:.2f}"
                    )
            elif self.on_event is not None:
                self.on_event(
                    f"rebalance skipped: only {len(scores)}/{eng.min_universe} symbols scored "
                    "(need more history for the feature window)"
                )
        self._flush()
        if self.on_positions is not None:
            self.on_positions(self._open_positions(bars_at))

    def _open_positions(self, bars_at: dict[str, dict]) -> list[dict]:
        """Mark every held leg to the latest bar and return its unrealized P&L — what the dashboard
        shows as live open positions until the leg closes (then it becomes a realized trade)."""
        out: list[dict] = []
        for sym, leg in self._holdings.items():
            bar = bars_at.get(sym)
            mark = float(bar["close"]) if bar is not None else leg.entry_price
            unreal = leg.side * (mark - leg.entry_price) * leg.qty - leg.entry_fee - leg.funding
            out.append({
                "symbol": sym, "strategy": self.engine.name, "side": leg.side, "qty": leg.qty,
                "entry_price": leg.entry_price, "mark_price": mark, "notional": leg.notional,
                "unrealized_pnl": unreal, "entry_ts": leg.entry_ts,
            })
        return out

    def close_all(self, ts: int, bars_at: dict[str, dict]) -> None:
        """Flatten every leg at the latest bar (session end / stop)."""
        for sym, leg in list(self._holdings.items()):
            bar = bars_at.get(sym)
            if bar is not None:
                self._equity += self.engine._close_leg(leg, bar, "end_of_data", self._result)
        self._holdings.clear()
        self._flush()
        if self.on_positions is not None:
            self.on_positions([])  # session flat → clear the dashboard's open-positions panel

    def run(
        self, snapshots: Iterable[Snapshot], by_symbol: dict[str, SymbolInput]
    ) -> BacktestResult:
        last: Snapshot | None = None
        for snap in snapshots:
            ts, bars_at, rows_at = snap
            self.step(ts, bars_at, rows_at, by_symbol)
            last = snap
        if last is not None:
            self.close_all(last[0], last[1])
        return self._result

    def _flush(self) -> None:
        """Book any newly-closed legs (engine trades since the last flush) as paper trades."""
        new = self._result.trades[self._booked :]
        self._booked = len(self._result.trades)
        for t in new:
            pt = _trade_to_paper(t, maker=self.engine.maker)
            self.session.trades.append(pt)
            if self.on_trade is not None:
                self.on_trade(pt)


def _halt_check(
    caller_stop: Callable[[], bool] | None, kill_switch: object
) -> Callable[[], bool]:
    """Compose the caller's stop condition with the GLOBAL kill switch so the basket loop halts on
    a kill-switch engage too — not just the dashboard Stop / job-cancel. The per-symbol live loop
    already honours the kill switch (loop.py); the basket path must too, or an emergency halt
    would stop directional trading but leave the carry/factor baskets running. Fail-safe: an error
    reading the kill switch is treated as NOT engaged here (caller_stop / job-cancel still applies),
    matching KillSwitch's own never-raise contract."""

    def _should_stop() -> bool:
        try:
            if kill_switch.engaged():  # type: ignore[attr-defined]
                return True
        except Exception:  # noqa: BLE001 - never let a kill-switch read crash the loop
            pass
        return bool(caller_stop and caller_stop())

    return _should_stop


def _persist_open_positions(session_id: str, positions: list[dict]) -> None:
    """Replace this session's live open-position rows with the current marked-to-market snapshot
    (delete + insert) so the dashboard always shows exactly what's held right now. An empty list
    clears them (the session went flat). Best-effort: a DB blip must not crash the trading loop."""
    from datetime import UTC, datetime

    from src.db.base import session_scope
    from src.db.models import OpenPosition

    try:
        with session_scope() as s:
            s.query(OpenPosition).filter_by(session_id=session_id).delete()
            now = datetime.now(UTC)
            for p in positions:
                s.add(OpenPosition(session_id=session_id, updated_at=now, **p))
    except Exception:  # noqa: BLE001 - position display must never take down the session
        pass


def run_basket_paper_session(
    candidate_id: str,
    *,
    data_cfg=None,
    timeframe: str | None = None,
    poll_sec: float = 0.0,
    max_ticks: int | None = None,
    settings=None,
    should_stop: Callable[[], bool] | None = None,
    on_event: Callable[[str], None] | None = None,
    on_tick: Callable[[int, str], None] | None = None,
) -> int:
    """Continuous PAPER session for ONE cross-sectional (basket) strategy on the live REST feed.

    The shell of the basket path: builds the real-time feed for the universe, drives the proven
    :class:`BasketPaperLoop` from its cross-section snapshots, and persists the booked PaperTrades
    the same way every paper session is (so the dashboard shows them). PAPER only — simulated fills,
    no real orders. ``poll_sec`` > 0 = continuous (waits for new bars). Returns the trade count.
    Halts on the GLOBAL kill switch (flattens + persists) as well as the caller's ``should_stop``.

    NOTE: network-dependent — validated against the live feed on the VPS, not in offline tests (the
    loop math is proven by tests/test_basket_paper.py). Run via `qbot paper-basket`.
    """
    from src.backtest.config import load_backtest_config
    from src.config import get_settings
    from src.data.config import load_data_config
    from src.data.schema import timeframe_ms
    from src.exchange.metadata import load_metadata_for
    from src.killswitch import KillSwitch
    from src.live.data_manager import LiveDataManager
    from src.live.realtime import LiveCandidateFeed
    from src.live.websocket_feed import live_feed_source
    from src.paper.report import build_paper_report
    from src.paper.run import persist_paper_session
    from src.strategies.candidates import build_strategy
    from src.strategies.config import load_strategies_config

    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    tf = timeframe or data_cfg.base_timeframe
    syms = data_cfg.active_symbols()

    sc = load_strategies_config()
    cand = sc.candidate(candidate_id)
    built = build_strategy(cand, sc.strategy_version) if cand is not None else None
    if built is None or not getattr(built, "cross_sectional", False):
        raise ValueError(f"{candidate_id!r} is not a cross-sectional (basket) strategy")
    strategy = build_strategy(cand, sc.strategy_version)

    source = live_feed_source(
        syms, transport="rest", exchange_id=data_cfg.exchange_id,
        timeframe=tf, exchange_env=settings.exchange_env,
    )
    data_manager = LiveDataManager(source, syms, interval_ms=timeframe_ms(tf))
    # Halt on the GLOBAL kill switch (emergency stop) as well as the caller's Stop / job-cancel.
    # When it fires the feed stops yielding, close_all flattens every leg, and the session persists.
    halt = _halt_check(should_stop, KillSwitch(settings))
    feed = LiveCandidateFeed(
        data_cfg, feed_source=source, data_manager=data_manager, timeframe=tf, symbols=syms,
        candidate_id=candidate_id, settings=settings, max_groups=max_ticks, poll_sec=poll_sec,
        should_stop=halt,
    )

    meta = load_metadata_for(data_cfg.exchange_id)
    cfg = load_backtest_config()
    session = PaperSession(session_id=f"paper:basket:{candidate_id}:{data_cfg.data_version}:{tf}")
    loop = BasketPaperLoop(
        cfg, meta, strategy, bar_interval_ms=timeframe_ms(tf), session=session, on_event=on_event,
        initial_equity=PAPER_BASE_EQUITY,  # paper base → equity curve aligns with other sessions
        on_positions=lambda positions: _persist_open_positions(session.session_id, positions),
    )

    last_bars: dict[str, dict] = {}
    persisted = 0
    for ticks, (ts, bars_at, rows_at) in enumerate(feed.snapshots(), start=1):
        last_bars = bars_at
        loop.step(ts, bars_at, rows_at, feed.symbol_inputs())
        # Persist as soon as legs CLOSE (a rebalance), not just at session end — otherwise closed
        # trades stay invisible on the dashboard until the session is stopped.
        if len(session.trades) > persisted:
            persist_paper_session(session, build_paper_report(session), settings)
            persisted = len(session.trades)
        if on_tick is not None:
            on_tick(ticks, f"tick {ticks}: {len(session.trades)} legs, {len(bars_at)} symbols")
    loop.close_all(int(max((b["ts"] for b in last_bars.values()), default=0)), last_bars)

    persist_paper_session(session, build_paper_report(session), settings)
    return len(session.trades)
