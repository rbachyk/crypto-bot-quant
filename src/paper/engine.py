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
from dataclasses import dataclass
from datetime import UTC, datetime

from src.config import Settings, get_settings
from src.exchange.metadata import MetadataConfig
from src.execution import (
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
        # Realized PnL accumulated across the session (from CLOSED trades), fed to the loss
        # breakers so they can actually trip (Section 17). Per-symbol for the per-symbol breaker.
        self._realized_pnl: float = 0.0
        self._per_symbol_pnl: dict[str, float] = {}
        # Real account state (a live/demo venue only): when present, the breakers reason over REAL
        # equity (daily-loss + drawdown) — the engine's simulated exit path is inert when exits
        # happen exchange-side, so this is what makes the loss breakers live in production.
        self._free_margin: float | None = None
        self._account_equity: float | None = None
        self._session_start_equity: float | None = None
        self._peak_equity: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def new_session(self, session_id: str | None = None) -> PaperSession:
        return PaperSession(session_id=session_id or str(uuid.uuid4()))

    def open_positions(self, price_of) -> list[dict]:
        """Snapshot the currently-open positions marked to market — for the dashboard's live
        open-positions panel. ``price_of(symbol)`` returns the latest price (None → mark at entry,
        i.e. 0 unrealized). Matches the basket loop's position-dict shape."""
        out: list[dict] = []
        for sym, pos in self._open_positions.items():
            mark = price_of(sym)
            mark = pos.entry_price if mark is None else float(mark)
            strat, entry_ts = self._position_meta.get(sym, ("", 0))
            out.append({
                "symbol": sym, "strategy": strat, "side": pos.side, "qty": pos.qty,
                "entry_price": pos.entry_price, "mark_price": mark, "notional": pos.notional,
                "unrealized_pnl": pos.side * (mark - pos.entry_price) * pos.qty,
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

    def simulate_paper_exits(
        self, price_of, now_ts: int, session: PaperSession, *, bar_iv: int = 0
    ) -> int:
        """Close held PAPER positions whose bracket (stop / take-profit) or time-stop is breached
        by the latest bar — the exchange-side exit a real venue would fill but the offline
        SimulatedVenue does not. Without this a per-symbol paper position (exit_move_frac=0) opens
        and is held forever, so the session books no closed trades and shows no realized P&L.

        ``price_of(symbol)`` returns the latest close; a close-based check is conservative (it can
        miss an intrabar wick) but never invents a fill. Returns the number of positions closed.
        Real venues manage exits themselves, so the loop only calls this in paper mode."""
        closed = 0
        for sym in list(self._open_positions):
            price = price_of(sym)
            if price is None:
                continue
            price = float(price)
            pos = self._open_positions[sym]
            stop, tp, hold_bars, entry_ts = self._exit_levels.get(sym, (0.0, 0.0, 0, 0))
            reason: str | None = None
            if pos.side > 0:
                if stop > 0 and price <= stop:
                    reason = "stop"
                elif tp > 0 and price >= tp:
                    reason = "take_profit"
            else:  # short: stop is ABOVE entry, take-profit BELOW
                if stop > 0 and price >= stop:
                    reason = "stop"
                elif tp > 0 and price <= tp:
                    reason = "take_profit"
            if (
                reason is None and bar_iv > 0 and hold_bars > 0 and entry_ts > 0
                and now_ts - entry_ts >= hold_bars * bar_iv
            ):
                reason = "time_stop"
            if reason is None:
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
            pnl = raw_pnl - total_fee
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.exit_ts = now_ts
            trade.fee = total_fee
            trade.pnl = pnl
            trade.pnl_r = pnl / pos.risk_amount if pos.risk_amount > 0 else 0.0
        else:  # no open record (shouldn't happen in the live path) — book a standalone close
            pnl = raw_pnl - exit_fee
            strat, entry_ts = self._position_meta.get(symbol, ("", 0))
            session.trades.append(PaperTrade(
                trade_id=str(uuid.uuid4())[:8], symbol=symbol, strategy=strat, side=pos.side,
                qty=pos.qty, entry_price=pos.entry_price, stop_price=0.0, tp_price=0.0,
                regime=pos.regime, session=0, decision_ts=entry_ts, entry_ts=entry_ts or now_ts,
                exit_ts=now_ts, exit_price=exit_price, exit_reason=exit_reason, fee=exit_fee,
                slippage_cost=0.0, pnl=pnl, pnl_r=pnl / pos.risk_amount if pos.risk_amount > 0 else 0.0,
                has_exchange_side_stop=True, execution_route="taker", spread_bps_at_entry=0.0,
                slippage_frac=0.0,
            ))
        self._realized_pnl += pnl
        self._per_symbol_pnl[symbol] = self._per_symbol_pnl.get(symbol, 0.0) + pnl

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
            start = (
                self._session_start_equity
                if self._session_start_equity is not None
                else bk_equity
            )
            bk_daily = bk_equity - start
            bk_weekly = bk_daily
        else:
            bk_equity = inp.equity
            bk_peak = max(inp.peak_equity, inp.equity)
            bk_daily = inp.daily_pnl + self._realized_pnl
            bk_weekly = self._realized_pnl
        portfolio = PortfolioState(
            equity=bk_equity, positions=tuple(self._open_positions.values())
        )
        breakers = BreakerInputs(
            equity=bk_equity,
            peak_equity=bk_peak,
            daily_pnl=bk_daily,
            consecutive_losses=inp.consecutive_losses,
            abnormal_slippage_active=False,
            reconciled=not inp.inject_foreign_order,
            weekly_pnl=bk_weekly,
            per_symbol_pnl=dict(self._per_symbol_pnl),
        )
        account = AccountState(
            portfolio=portfolio,
            breakers=breakers,
            unknown_order_present=inp.inject_foreign_order,
            free_margin=getattr(self, "_free_margin", None),
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

        entry_price = fill.actual_price
        stop_price = entry_price * (1 - candidate.stop_frac * candidate.side)
        tp_price = entry_price * (1 + candidate.tp_frac * candidate.side)
        exit_move = inp.exit_move_frac
        exit_price = entry_price * (1 + exit_move)
        exit_reason = "open"
        if exit_move > 0 and candidate.side > 0 and exit_price >= tp_price:
            exit_reason = "take_profit"
        elif exit_move < 0 and candidate.side > 0 and exit_price <= stop_price:
            exit_reason = "stop"
        elif exit_move > 0 and candidate.side < 0 and exit_price <= tp_price:
            exit_reason = "take_profit"
        elif exit_move < 0 and candidate.side < 0 and exit_price >= stop_price:
            exit_reason = "stop"

        raw_pnl = (exit_price - entry_price) * candidate.side * fill.qty
        fee = fill.fee
        pnl = raw_pnl - fee
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
            exit_ts=now_ts + inp.hold_bars * 60_000,
            exit_price=exit_price,
            exit_reason=exit_reason,
            fee=fee,
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
        else:
            self._open_positions.pop(candidate.symbol, None)
            self._position_meta.pop(candidate.symbol, None)
            self._exit_levels.pop(candidate.symbol, None)
            self._venue.positions.pop(candidate.symbol, None)
            # The trade closed → its PnL is REALIZED. Accumulate it for the loss breakers so a
            # losing session halts new entries on the next candidate (Section 17).
            self._realized_pnl += pnl
            self._per_symbol_pnl[candidate.symbol] = (
                self._per_symbol_pnl.get(candidate.symbol, 0.0) + pnl
            )
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
        entry_price = candidate.entry_price
        fee_rate = 0.0006  # default taker estimate
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
        )
