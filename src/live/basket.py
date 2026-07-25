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

import time
import uuid
from collections.abc import Callable, Iterable

import structlog

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestResult, SymbolInput, Trade
from src.backtest.portfolio import CrossSectionalEngine
from src.exchange.metadata import MetadataConfig
from src.paper.session import PAPER_BASE_EQUITY, PaperSession, PaperTrade

_log = structlog.get_logger("live.basket")

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
        funding=t.funding,  # carry accrued over the hold (already netted into t.pnl)
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
        # Symbols whose REAL close failed during close_all (demo path only) — the caller escalates
        # to the venue's audited emergency close so no exchange position is left running.
        self.failed_closes: list[str] = []
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
                leg.last_funding_ts = eng._charge_funding(leg, by_symbol[sym], ts)
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
            if eng.can_rebalance(scores):
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
                if len(scores) < eng.min_universe:
                    self.on_event(
                        f"rebalance skipped: only {len(scores)}/{eng.min_universe} symbols scored "
                        "(need more history for the feature window)"
                    )
                else:
                    # Dispersion gate: held legs stay untouched, no turnover paid. Visible so a
                    # quiet market is distinguishable from a broken feed.
                    self.on_event(
                        f"rebalance skipped: score gap {eng.score_gap(scores):.5f} < "
                        f"min_score_gap {eng.min_score_gap:.5f} (too little cross-sectional "
                        "dispersion to pay for the turnover) — holding existing legs"
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
                "unrealized_pnl": unreal, "funding": leg.funding, "entry_ts": leg.entry_ts,
            })
        return out

    def close_all(self, ts: int, bars_at: dict[str, dict], by_symbol: dict) -> None:
        """Flatten every leg at the latest bar (session end / stop). ``by_symbol`` supplies each
        leg's SymbolInput (spread model) for the taker close, same as the rebalance path.

        M16: a leg whose symbol has NO bar this tick (halted / feed gap at session end) is
        closed on its last KNOWN bar at-or-before ``ts`` (never a future price), with a warning
        — and if no priceable bar exists at all it is booked at its entry price so the leg's
        entry fee and accrued funding are still realized. Booked P&L is never silently dropped."""
        for sym, leg in list(self._holdings.items()):
            bar = bars_at.get(sym)
            sym_in = by_symbol.get(sym)
            if bar is None and sym_in is not None:
                bars = list(getattr(sym_in, "bars", None) or [])
                bar = next((b for b in reversed(bars) if int(b["ts"]) <= ts), None)
                if bar is not None:
                    _log.warning(
                        "basket_close_all_last_known_bar", symbol=sym,
                        bar_ts=int(bar["ts"]), close_ts=ts,
                    )
            if bar is not None and sym_in is not None:
                pnl = self.engine._close_leg(leg, bar, "end_of_data", self._result, sym_in)
                if pnl is None:
                    # REAL-venue close failed — or only PARTIALLY filled (the engine books the
                    # closed portion and keeps the remainder). Either way the exchange still holds
                    # this leg: keep it tracked and record it so the caller escalates to the audited
                    # emergency close rather than exiting with a live position adrift.
                    self._equity += self.engine.drain_partial_pnl()  # book what DID close
                    self.failed_closes.append(sym)
                    _log.error(
                        "basket_close_all_venue_close_failed", symbol=sym, close_ts=ts,
                        hint="exchange position still open — escalate to emergency_close_all",
                    )
                    continue
                self._equity += pnl
            else:
                # No priceable bar at all — book the close at the entry price (gross 0) so the
                # entry fee + accrued funding are realized instead of vaporised with the leg.
                pnl = -leg.entry_fee - leg.funding
                self._result.trades.append(Trade(
                    symbol=sym, strategy=self.engine.name, side=leg.side, qty=leg.qty,
                    entry_ts=leg.entry_ts, entry_price=leg.entry_price,
                    exit_ts=ts, exit_price=leg.entry_price, exit_reason="close_no_price",
                    notional=leg.notional, risk_amount=leg.risk_amount,
                    fee=leg.entry_fee, funding=leg.funding, slippage_cost=leg.slippage_cost,
                    pnl=pnl,
                    pnl_r=pnl / leg.risk_amount if leg.risk_amount > 0 else 0.0,
                    regime=leg.regime, session=leg.session,
                    bars_held=max(0, int((ts - leg.entry_ts) // self.engine._grid_iv)),
                ))
                self._equity += pnl
                _log.warning(
                    "basket_close_all_no_price", symbol=sym, close_ts=ts,
                    hint="no bar available — leg booked at entry price (fees+funding realized)",
                )
            del self._holdings[sym]
        self._flush()
        if self.on_positions is not None:
            # Usually flat → clears the dashboard's open-positions panel. If a REAL close failed
            # the leg is still held, so keep showing it: the operator must see what is adrift.
            self.on_positions(self._open_positions(bars_at) if self._holdings else [])

    def book_exchange_closed_leg(
        self, symbol: str, *, exit_price: float, exit_fee: float, exit_ts: int, reason: str
    ) -> bool:
        """Book a leg that the EXCHANGE closed out from under us (demo path).

        A disaster stop firing, a liquidation, or a manual close means the position is gone but
        the engine still holds a leg. Dropping the mirror alone would silently vaporise that
        leg's realized P&L — the trade happened, so it must be booked like any other exit, at the
        observed exit price. Returns False if we weren't holding it."""
        leg = self._holdings.pop(symbol, None)
        if leg is None:
            return False
        gross = leg.side * (exit_price - leg.entry_price) * leg.qty
        total_fee = leg.entry_fee + exit_fee
        pnl = gross - total_fee - leg.funding
        self._result.trades.append(Trade(
            symbol=symbol, strategy=self.engine.name, side=leg.side, qty=leg.qty,
            entry_ts=leg.entry_ts, entry_price=leg.entry_price,
            exit_ts=exit_ts, exit_price=exit_price, exit_reason=reason,
            notional=leg.notional, risk_amount=leg.risk_amount,
            fee=total_fee, funding=leg.funding, slippage_cost=leg.slippage_cost,
            pnl=pnl, pnl_r=pnl / leg.risk_amount if leg.risk_amount > 0 else 0.0,
            regime=leg.regime, session=leg.session,
            bars_held=max(0, int((exit_ts - leg.entry_ts) // self.engine._grid_iv)),
        ))
        self._equity += pnl
        self._flush()
        return True

    def run(
        self, snapshots: Iterable[Snapshot], by_symbol: dict[str, SymbolInput]
    ) -> BacktestResult:
        last: Snapshot | None = None
        for snap in snapshots:
            ts, bars_at, rows_at = snap
            self.step(ts, bars_at, rows_at, by_symbol)
            last = snap
        if last is not None:
            self.close_all(last[0], last[1], by_symbol)
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


def _clear_orphan_open_positions(session_id: str) -> None:
    """At session start, delete leftover open-position rows from PRIOR runs of the SAME stream.

    Each run now has a unique session_id (env:…:<run-stamp>), so a hard-killed run (no graceful
    end-clear) leaves orphan rows that linger on the dashboard forever — the new run uses a new id
    and never overwrites them. We delete only rows whose session_id shares this run's stream prefix
    (everything up to and including the last ':', i.e. the id minus the run stamp) and differ from
    the current id — concurrent OTHER streams (different strategy/env/timeframe) are untouched."""
    from src.db.base import session_scope
    from src.db.models import OpenPosition

    prefix = session_id.rsplit(":", 1)[0] + ":"  # drop the trailing run-stamp segment
    try:
        with session_scope() as s:
            (
                s.query(OpenPosition)
                .filter(OpenPosition.session_id.like(f"{prefix}%"))
                .filter(OpenPosition.session_id != session_id)
                .delete(synchronize_session=False)
            )
    except Exception:  # noqa: BLE001 - cleanup must never take down the session
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
    return _run_basket_session(
        candidate_id, live=False, data_cfg=data_cfg, timeframe=timeframe, poll_sec=poll_sec,
        max_ticks=max_ticks, settings=settings, should_stop=should_stop, on_event=on_event,
        on_tick=on_tick,
    )


def run_basket_demo_session(
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
    venue=None,
) -> int:
    """Continuous DEMO session for ONE basket strategy — REAL orders on virtual funds.

    Same math, same cadence, same sizing as :func:`run_basket_paper_session`; the difference is
    that each leg open/close is placed on the real Bybit **demo/testnet** venue and booked at the
    OBSERVED fill (:mod:`src.live.basket_exec`). That is the whole point: it measures whether the
    backtest's cost model survives contact with a real book — the one thing a simulated-fill paper
    run structurally cannot tell you.

    Refuses to start unless: ``EXCHANGE_ENV`` is demo/testnet (never live — this function has no
    path to a real-money account), the demo readiness gate PASSes for this basket, the exchange
    book reconciles clean, and the account is flat + funded. On stop / kill switch it flattens
    every leg and escalates to the venue's audited emergency close if any close failed.

    To run SEVERAL baskets on the one demo account use
    :func:`run_multi_basket_demo_session` — co-hosting them in one process is what makes their
    positions net correctly (starting a second process against the same account would give each
    an independent mirror of a book they actually share).

    NOTE: network-dependent. The engine/executor seams are offline-tested
    (``tests/test_basket_demo.py`` with a fake venue); the real endpoint is VPS-validated.
    """
    return _run_basket_session(
        candidate_id, live=True, data_cfg=data_cfg, timeframe=timeframe, poll_sec=poll_sec,
        max_ticks=max_ticks, settings=settings, should_stop=should_stop, on_event=on_event,
        on_tick=on_tick, venue=venue,
    )


def run_multi_basket_demo_session(
    candidate_ids: list[str],
    *,
    data_cfg=None,
    timeframe: str | None = None,
    poll_sec: float = 0.0,
    max_ticks: int | None = None,
    settings=None,
    should_stop: Callable[[], bool] | None = None,
    on_event: Callable[[str], None] | None = None,
    on_tick: Callable[[int, str], None] | None = None,
    venue=None,
) -> int:
    """Continuous DEMO session running SEVERAL baskets on ONE demo account.

    An exchange holds one position per symbol, so two baskets that both trade BTC share it
    whether or not they know. Co-hosting them here gives them a single
    :class:`~src.live.basket_exec.NetPositionManager`: each strategy states its own desired
    position, the manager owns the aggregate and sends only the DELTA, and reconciliation is
    against the netted total. Each strategy keeps its own engine, equity slice and PaperSession,
    so per-strategy statistics stay exactly as separate as they are today.

    The account equity is SPLIT across the strategies (rather than each sizing off the whole
    balance), so the aggregate gross respects ``basket_demo.max_gross_pct`` — the shared capital
    allocator Section 17 requires before several strategies trade one account.

    Co-hosted baskets must agree on the timeframe (they share one bar feed); pass ``timeframe``
    explicitly to override the per-strategy resolution.
    """
    return _run_basket_session(
        list(candidate_ids), live=True, data_cfg=data_cfg, timeframe=timeframe,
        poll_sec=poll_sec, max_ticks=max_ticks, settings=settings, should_stop=should_stop,
        on_event=on_event, on_tick=on_tick, venue=venue,
    )


def _strategy_extra(strategy: object) -> dict:
    """The strategy's ``params.extra`` knobs (portfolio_gross / maker_rebalance / …), as the
    engine reads them. Empty dict when the strategy carries none."""
    params = getattr(strategy, "params", None)
    return dict(getattr(params, "extra", {}) or {}) if params is not None else {}


def _preflight_demo_basket(
    candidate_ids: list[str], settings, venue, limits, data_cfg, scope_symbols
) -> float:
    """Every refusal that must happen BEFORE a single real order — returns the account equity.

    Raises ``PermissionError`` on any blocker. Order matters: environment first (a live env must
    be refused before we so much as read a balance), then per-strategy readiness, then the book,
    then funding."""
    from src.execution.ownership import OwnershipPolicy
    from src.execution.reconciliation import reconcile_startup
    from src.live.demo_guard import DemoReadinessGuard

    env = settings.exchange_env
    if env == "live":
        raise PermissionError(
            "refusing to run a basket demo session with EXCHANGE_ENV=live — this path places "
            "unstopped hedged legs and is authorised for virtual funds ONLY (demo/testnet). "
            "Real-money trading goes through the LiveActivationGuard, never here."
        )
    if env not in ("demo", "testnet"):
        raise PermissionError(
            f"EXCHANGE_ENV={env!r} is not a virtual-funds environment — set EXCHANGE_ENV=demo "
            "(Bybit demo account) or testnet before running a basket demo session"
        )

    # EVERY co-hosted strategy must clear the gate — one ineligible basket must not ride along
    # on another's eligibility. Metadata is checked over the SESSION'S SCOPE (the partitioned
    # symbols it will actually trade), not the whole universe — symbols reserved for another
    # strategy are that strategy's problem to verify.
    scope = list(scope_symbols)
    for cid in candidate_ids:
        report = DemoReadinessGuard(
            settings, venue=venue, basket_candidate_id=cid, data_cfg=data_cfg, symbols=scope
        ).evaluate()
        if not report.ok:
            raise PermissionError(
                f"demo readiness is not PASS for {cid!r} — refusing to place real orders:\n"
                + report.report()
            )

    recon = reconcile_startup(
        venue, OwnershipPolicy(settings), environment=env, adopt=True
    )
    if recon.halt_required:
        # A FOREIGN (non-prefix) order/position halts regardless of partitioning — it is not ours.
        raise PermissionError(
            "exchange book is not clean — refusing to start:\n" + recon.report()
        )
    # Under account partitioning we require only OUR SCOPE to be flat. Owned positions in OTHER
    # symbols belong to another of our strategies (basis_reversion on its reserved symbols) and are
    # legitimately present — recognized as ours, left alone. A leftover in our OWN scope, though,
    # still cannot be attributed to a strategy in this session (a crashed prior run) and would
    # corrupt the aggregate, so it still blocks.
    scope_set = set(scope)
    in_scope_pos = [s for s in recon.owned_positions if s in scope_set]
    # reconcile_startup(adopt=True) put the owned Order objects in venue.open_orders — map id→symbol
    # so a resting order (e.g. basis_reversion's stop) only blocks when it is in OUR scope.
    in_scope_ord = [
        oid for oid in recon.owned_orders
        if getattr(getattr(venue, "open_orders", {}).get(oid, None), "symbol", None) in scope_set
    ]
    if limits.require_flat_book and (in_scope_pos or in_scope_ord):
        raise PermissionError(
            "our own scope is NOT flat on the demo account (positions="
            f"{in_scope_pos} orders={in_scope_ord}). These are in symbols THIS session trades but "
            "cannot be attributed to it (a crashed prior run?). Close them first "
            "(`qbot flatten-demo` clears the whole demo account), or co-host all baskets in ONE "
            "session. Positions in OTHER symbols (another strategy's partition) are fine and were "
            "left untouched."
        )

    equity = None
    fetch = getattr(venue, "fetch_account_equity", None)
    if callable(fetch):
        equity = fetch()
    if equity is None:
        raise PermissionError(
            "could not read the demo account equity — refusing to size a basket off a guessed "
            "balance (the paper path's fixed 10k numeraire is meaningless against a real book)"
        )
    if float(equity) < limits.min_account_equity:
        raise PermissionError(
            f"demo account equity {float(equity):.2f} is below the "
            f"{limits.min_account_equity:.2f} floor — legs would round to dust or fail the "
            "venue's min-notional. Fund the demo account first."
        )
    return float(equity)


def _run_basket_session(
    candidate_ids: str | list[str],
    *,
    live: bool,
    data_cfg=None,
    timeframe: str | None = None,
    poll_sec: float = 0.0,
    max_ticks: int | None = None,
    settings=None,
    should_stop: Callable[[], bool] | None = None,
    on_event: Callable[[str], None] | None = None,
    on_tick: Callable[[int, str], None] | None = None,
    venue=None,
) -> int:
    """Shared shell for every basket session. ``live=False`` is the simulated-fill paper path
    (unchanged); ``live=True`` places real orders on the demo/testnet venue.

    Several ``candidate_ids`` run CO-HOSTED in this one process against ONE account: each keeps
    its own engine, equity slice and PaperSession (so per-strategy statistics stay separate), but
    all share a single :class:`NetPositionManager`, which is what makes one demo account correct
    for several baskets — see :mod:`src.live.basket_exec`. Returns the total legs booked."""
    from src.backtest.config import load_backtest_config
    from src.config import get_settings
    from src.data.config import load_data_config
    from src.data.schema import timeframe_ms
    from src.exchange.metadata import load_metadata_for
    from src.killswitch import KillSwitch
    from src.live.data_manager import LiveDataManager
    from src.live.loop import _resolve_live_timeframe, _run_stamp
    from src.live.realtime import LiveCandidateFeed
    from src.live.websocket_feed import live_feed_source
    from src.paper.report import build_paper_report
    from src.paper.run import persist_paper_session
    from src.strategies.candidates import build_strategy
    from src.strategies.config import load_strategies_config

    ids = [candidate_ids] if isinstance(candidate_ids, str) else list(candidate_ids)
    if not ids:
        raise ValueError("a basket session needs at least one cross-sectional strategy")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate strategy in the session: {ids}")

    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    sc = load_strategies_config()

    # Account partitioning: this session trades ONLY its scope — the universe minus every symbol
    # another strategy reserves (basis_reversion et al.). Co-hosted baskets net among themselves
    # inside the shared scope; nothing outside it is fetched, ranked, ordered or reconciled here,
    # so two strategies never touch the same symbol on one account (the netting hazard). An
    # explicitly-restricted basket intersects its own live_symbols. The feed, engine, sizing and
    # the NetPositionManager all run on this scope, not the full universe.
    universe = data_cfg.active_symbols()
    syms = sorted({s for cid in ids for s in sc.live_scope(cid, universe)})
    if not syms:
        raise ValueError(
            f"the account partition leaves {ids} with NO symbols to trade "
            f"(universe={len(universe)}, reserved by others={sorted(sc.reserved_symbols(*ids))}). "
            "Widen live_symbols or the data universe."
        )
    if set(syms) != set(universe) and on_event is not None:
        on_event(
            f"account partition: trading {len(syms)}/{len(universe)} symbols "
            f"(reserved for other strategies: {sorted(set(universe) - set(syms))})"
        )

    strategies: dict[str, object] = {}
    for cid in ids:
        cand = sc.candidate(cid)
        built = build_strategy(cand, sc.strategy_version) if cand is not None else None
        if built is None or not getattr(built, "cross_sectional", False):
            raise ValueError(f"{cid!r} is not a cross-sectional (basket) strategy")
        strategies[cid] = built

    # Run each basket on the timeframe it was validated on (e.g. 1h), not the fine base grid —
    # residual_momentum's signal/beta windows are counted in BARS, so the wrong timeframe silently
    # changes the lookbacks. Co-hosted strategies must AGREE on it: they share one bar feed, and a
    # strategy stepped on someone else's grid is being run at parameters it was never validated at.
    tf = timeframe or _resolve_live_timeframe(settings, data_cfg, ids[0])
    if timeframe is None and len(ids) > 1:
        resolved = {cid: _resolve_live_timeframe(settings, data_cfg, cid) for cid in ids}
        if len(set(resolved.values())) > 1:
            raise ValueError(
                f"co-hosted baskets must share a timeframe, got {resolved}. Pass an explicit "
                "--timeframe to override, or run them as separate sessions on separate accounts."
            )

    source = live_feed_source(
        syms, transport="rest", exchange_id=data_cfg.exchange_id,
        timeframe=tf, exchange_env=settings.exchange_env,
    )
    data_manager = LiveDataManager(source, syms, interval_ms=timeframe_ms(tf))
    # Halt on the GLOBAL kill switch (emergency stop) as well as the caller's Stop / job-cancel.
    # When it fires the feed stops yielding, close_all flattens every leg, and sessions persist.
    halt = _halt_check(should_stop, KillSwitch(settings))
    feed = LiveCandidateFeed(
        data_cfg, feed_source=source, data_manager=data_manager, timeframe=tf, symbols=syms,
        candidate_id=ids[0], settings=settings, max_groups=max_ticks, poll_sec=poll_sec,
        should_stop=halt,
    )

    meta = load_metadata_for(data_cfg.exchange_id)
    cfg = load_backtest_config()

    # -- venue + pre-flight (demo only) ---------------------------------- #
    manager = None
    equity = PAPER_BASE_EQUITY  # paper base → equity curve aligns with the other paper sessions
    env_label = "paper"
    if live:
        from src.execution.live_venue import get_venue
        from src.live.basket_exec import NetPositionManager
        from src.live.guard import load_basket_demo_limits

        limits = load_basket_demo_limits()
        venue = venue if venue is not None else get_venue(meta, settings, live=True)
        equity = _preflight_demo_basket(ids, settings, venue, limits, data_cfg, syms)
        env_label = settings.exchange_env
        manager = NetPositionManager(
            venue, meta, settings,
            disaster_stop_frac=limits.disaster_stop_frac, on_event=on_event,
            scope_symbols=set(syms),  # touch ONLY our partition — never another strategy's symbols
        )

    # -- per-strategy equity slice --------------------------------------- #
    # Co-hosted baskets share ONE balance, so they cannot each size off the whole of it — that is
    # exactly the "N processes ≈ N× the intended exposure" gap (Section 17). Split the account
    # equity equally, then cap each slice so the AGGREGATE gross stays inside max_gross_pct.
    equities = _equity_slices(
        strategies, equity, live=live,
        max_gross_pct=(limits.max_gross_pct if live else 1.0), on_event=on_event,
    )

    stamp = _run_stamp()
    loops: dict[str, BasketPaperLoop] = {}
    sessions: dict[str, PaperSession] = {}
    for cid, strategy in strategies.items():
        # Unique per run so a restart/re-Start does not delete the prior run's persisted legs (the
        # delete-then-insert history-wipe). The env prefix (paper: / demo: / testnet:) keeps env
        # classification + per-strategy grouping intact, so demo stats never mix with paper.
        session = PaperSession(
            session_id=f"{env_label}:basket:{cid}:{data_cfg.data_version}:{tf}:{stamp}"
        )
        _clear_orphan_open_positions(session.session_id)  # drop a prior crashed run's stale legs
        loop = BasketPaperLoop(
            cfg, meta, strategy, bar_interval_ms=timeframe_ms(tf), session=session,
            on_event=on_event, initial_equity=equities[cid],
            on_positions=(
                lambda positions, sid=session.session_id: _persist_open_positions(sid, positions)
            ),
        )
        if manager is not None:
            from src.live.basket_exec import LiveBasketExecutor

            # From here every leg this engine opens/closes is a REAL position delta on the demo
            # account, routed through the shared manager so co-hosted baskets net instead of
            # fighting over (or flattening) each other's positions.
            loop.engine.executor = LiveBasketExecutor(manager, strategy_id=cid)
        loops[cid] = loop
        sessions[cid] = session

    if manager is not None and on_event is not None:
        on_event(
            f"DEMO basket session live on {settings.exchange_env}: strategies={ids} "
            f"equity={equity:.2f} slices={ {k: round(v, 2) for k, v in equities.items()} } "
            f"disaster_stop={manager.disaster_stop_frac:.0%} run={stamp}"
        )

    last_bars: dict[str, dict] = {}
    persisted = dict.fromkeys(ids, 0)
    try:
        for ticks, (ts, bars_at, rows_at) in enumerate(feed.snapshots(), start=1):
            last_bars = bars_at
            inputs = feed.symbol_inputs()
            for cid, loop in loops.items():
                loop.step(ts, bars_at, rows_at, inputs)
                # Persist as soon as legs CLOSE (a rebalance), not just at session end — otherwise
                # closed trades stay invisible on the dashboard until the session is stopped.
                if len(sessions[cid].trades) > persisted[cid]:
                    persist_paper_session(
                        sessions[cid], build_paper_report(sessions[cid]), settings
                    )
                    persisted[cid] = len(sessions[cid].trades)
            if manager is not None:
                _reconcile_tick(manager, loops, on_event)
            if on_tick is not None:
                booked = sum(len(s.trades) for s in sessions.values())
                on_tick(ticks, f"tick {ticks}: {booked} legs, {len(bars_at)} symbols")
    finally:
        # Flatten on ANY exit — normal stop, kill switch, or an exception mid-loop. A demo session
        # that raises must not leave real legs open on the account.
        close_ts = int(max((b["ts"] for b in last_bars.values()), default=0))
        failed: list[str] = []
        for cid, loop in loops.items():
            loop.close_all(close_ts, last_bars, feed.symbol_inputs())
            failed.extend(loop.failed_closes)
            persist_paper_session(sessions[cid], build_paper_report(sessions[cid]), settings)
        if live and venue is not None:
            # Backstop: after flattening every tracked leg, confirm the account is ACTUALLY flat.
            # A mis-observed fill or a failed close can leave a real position no leg tracked, so a
            # clean close_all is not proof of a clean book. The basket demo owns the account
            # exclusively (pre-flight), so anything still on it is ours to remove — emergency-close
            # the remainder. This is the guarantee the SOL orphan violated: a session never EXITS
            # leaving a position open.
            _flatten_book_if_dirty(venue, manager, failed, on_event)
    if manager is not None and on_event is not None:
        on_event(f"execution quality: {manager.execution_summary()}")
    return sum(len(s.trades) for s in sessions.values())


def _flatten_book_if_dirty(venue, manager, failed: list[str], on_event) -> None:
    """Guarantee THIS session's symbols are flat at session end — never the whole account.

    Under account partitioning a basket owns only its scope; another strategy (basis_reversion)
    legitimately holds positions in its own symbols, so the backstop must touch ONLY our scope.
    A residual in-scope position (a mis-observed fill, a failed close) is flattened reduce-only,
    per symbol; anything that survives is surfaced for the operator. emergency_close_all — which
    flattens EVERY position on the account — is used only for a SOLO session that owns the whole
    account (``scope_symbols is None``)."""
    in_scope = getattr(manager, "in_scope", lambda _s: True)
    try:
        residual = {s: (p.side * p.qty) for s, p in venue.fetch_exchange_positions().items()
                    if p.qty > 0 and in_scope(s)}
    except Exception as exc:  # noqa: BLE001 - if we cannot read the book, escalate to be safe
        _log.error("basket_end_book_read_failed", error=str(exc))
        residual = {}
        failed = failed or ["<unreadable book>"]
    for sym, qty in sorted(residual.items()):
        _log.warning("basket_end_residual_position", symbol=sym, qty=qty,
                     hint="in our scope at session end with no tracked leg — flattening")
        if manager is not None:
            manager.flatten_stray(sym, qty, 0.0)
    # Re-read our scope: anything that survived + any failed leg close still needs handling.
    try:
        still = {s for s, p in venue.fetch_exchange_positions().items()
                 if p.qty > 0 and in_scope(s)}
    except Exception:  # noqa: BLE001 - unreadable → escalate
        still = set()
        failed = failed or ["<unreadable book>"]
    remaining = sorted(set(failed) | still)
    if not remaining:
        return
    solo = getattr(manager, "scope_symbols", None) is None
    if solo:
        # SOLO session owns the whole account → the audited whole-account emergency close is safe.
        _emergency_flatten(venue, remaining, on_event)
        return
    # PARTITIONED: reduce-only close ONLY our own symbols; never touch another strategy's.
    _log.error("basket_scoped_flatten", symbols=remaining)
    if on_event is not None:
        on_event(f"EMERGENCY: flattening our own symbols {remaining} (partitioned account)")
    close = getattr(venue, "close_book_position", None)
    unresolved = []
    for sym in remaining:
        if not callable(close) or not close(sym):
            unresolved.append(sym)
    if unresolved and on_event is not None:
        on_event(
            f"EMERGENCY: could not flatten {unresolved} — they may still be OPEN. Close manually "
            "(qbot flatten-demo closes the whole demo account)."
        )


def _equity_slices(
    strategies: dict[str, object],
    equity: float,
    *,
    live: bool,
    max_gross_pct: float,
    on_event: Callable[[str], None] | None,
) -> dict[str, float]:
    """Per-strategy equity for co-hosted baskets sharing one balance.

    Paper sessions each get the full paper numeraire (they are isolated measurements against
    separate simulated books). A DEMO session shares one real balance, so the equity is split
    equally and each slice is scaled so the AGGREGATE gross notional
    (``Σ slice × portfolio_gross``) stays within ``max_gross_pct`` of the account."""
    ids = list(strategies)
    if not live:
        return dict.fromkeys(ids, equity)
    slice_equity = equity / len(ids)
    out: dict[str, float] = {}
    for cid, strategy in strategies.items():
        gross_mult = max(1e-9, float(_strategy_extra(strategy).get("portfolio_gross", 1.0) or 1.0))
        # This strategy may use at most its share of the allowed aggregate gross.
        allowed_gross = equity * max_gross_pct / len(ids)
        out[cid] = min(slice_equity, allowed_gross / gross_mult)
    if on_event is not None and any(out[c] < slice_equity - 1e-9 for c in ids):
        on_event(
            f"equity slices capped to keep aggregate gross ≤ {max_gross_pct:.0%} of the account: "
            f"{ {k: round(v, 2) for k, v in out.items()} }"
        )
    return out


def _reconcile_tick(manager, loops: dict[str, BasketPaperLoop], on_event) -> None:
    """Per-tick check that the AGGREGATE intent still matches the real exchange book (Section 7).

    A drift means a position moved without us — a disaster stop firing, a liquidation, or a manual
    close. Whichever strategies held that symbol have had a real exit, so their legs are BOOKED at
    the observed exit price and dropped: leaving the mirror alone would keep marking a position
    that no longer exists, and dropping it silently would vaporise the leg's realized P&L.
    Best-effort — a reconciliation read must never take the trading loop down."""
    drift = manager.reconcile()
    for symbol, state in sorted(drift.items()):
        expected, actual = state["expected"], state["actual"]
        if abs(expected) < 1e-12 and abs(actual) >= 1e-12:
            # A position on the book we have NO intent for. On this exclusively-owned account
            # (pre-flight required a flat book + forbids a second session) that is a MIS-OBSERVED
            # FILL — the order opened a position the fill poll reported as unfilled, so nothing was
            # recorded. Flatten it reduce-only to restore book==intent instead of logging CRITICAL
            # every tick and leaving it open (the exact SOL orphan that accrued for 12h).
            manager.flatten_stray(symbol, actual, 0.0)
            continue
        if abs(actual) > abs(expected):
            # We DO hold a legit leg here and there is extra we cannot attribute. Blind-closing the
            # whole symbol would take our leg too, so alert rather than act — the operator decides.
            _log.error(
                "basket_unexpected_exchange_position", symbol=symbol,
                expected=expected, actual=actual,
            )
            if on_event is not None:
                on_event(
                    f"CRITICAL: {symbol} exchange position {actual:+g} exceeds our intent "
                    f"{expected:+g} — investigate before trusting this session"
                )
            continue
        dropped = manager.drop_symbol(symbol)
        # A PARTIAL exchange-side close (part of the net was liquidated / manually closed) leaves a
        # residual the intent no longer accounts for once the legs below are booked out. Flatten it
        # in THIS tick rather than waiting for the next reconcile to rediscover it as a stray — the
        # legs are being booked as closed right now, so the residual belongs to nobody from here on.
        if abs(actual) >= 1e-12:
            _log.warning(
                "basket_partial_exchange_close", symbol=symbol, expected=expected, actual=actual,
                hint="only part of the net closed exchange-side — legs booked out, residual "
                "flattened reduce-only",
            )
            manager.flatten_stray(symbol, actual, 0.0)
        for cid, qty in dropped.items():
            loop = loops.get(cid)
            leg = loop._holdings.get(symbol) if loop is not None else None
            if loop is None or leg is None:
                continue
            side = 1 if qty > 0 else -1
            exit_price, exit_fee = manager.observed_exit(
                symbol, side, leg.entry_ts, leg.last_mark or leg.entry_price
            )
            loop.book_exchange_closed_leg(
                symbol, exit_price=exit_price, exit_fee=exit_fee,
                exit_ts=int(time.time() * 1000), reason="exchange_close",
            )
            _log.error(
                "basket_leg_vanished_from_exchange", symbol=symbol, strategy=cid,
                exit_price=exit_price,
                hint="closed exchange-side (disaster stop / liquidation / manual) — booked at "
                "the observed exit and un-mirrored",
            )
            if on_event is not None:
                on_event(
                    f"WARNING: {symbol} ({cid}) was closed on the exchange, not by us — "
                    f"booked at {exit_price:g}"
                )


def _emergency_flatten(venue, failed: list[str], on_event) -> None:
    """Last resort when a leg's reduce-only close failed at session end: the audited
    emergency close (Section 7). Better a logged forced flatten than a live leg left adrift."""
    _log.error("basket_emergency_flatten", symbols=sorted(failed))
    if on_event is not None:
        on_event(f"EMERGENCY: close failed for {sorted(failed)} — flattening the account")
    try:
        n = venue.emergency_close_all(confirm=True)
        if on_event is not None:
            on_event(f"emergency close completed: {n} position/order actions")
    except Exception as exc:  # noqa: BLE001 - surface loudly; the operator must intervene
        _log.critical("basket_emergency_flatten_failed", error=str(exc), symbols=sorted(failed))
        if on_event is not None:
            on_event(
                f"EMERGENCY CLOSE FAILED ({exc}) — {sorted(failed)} may still be OPEN on the "
                "demo account. Close them manually."
            )
