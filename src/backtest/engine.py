"""Event-based backtest engine (AGENTS.md Section 19).

Strictly event-based (vectorized backtests are exploration-only, Section 19). The
engine walks a single chronological bar grid shared by all symbols. At each bar
it, in order:

1. **charges funding** on every open position crossing a funding timestamp;
2. **manages open positions** against the bar's intrabar high/low (exchange-style
   stop / take-profit) and the time-stop;
3. **opens new positions** from signals decided on the PREVIOUS bar's close,
   filling at THIS bar's open with realistic fees + slippage, after risk sizing
   and the execution hard-blockers — anything rejected is logged.

The look-ahead guard is *structural*: a signal for bar ``k`` (feature row with
``decision_ts = (k+1)·iv``) can only fill at bar ``k+1``'s open, so no decision
ever reads its own bar's close or any future bar. Survivorship / future-universe
leakage is prevented by ``activation_ts``: a symbol is tradable only at decision
times at or after it entered the universe (Section 19).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import cast

import structlog

from src.backtest.config import BacktestConfig
from src.backtest.costs import BUY, SELL, FeeModel, FundingModel, SlippageModel
from src.backtest.risk import RiskSimulator
from src.backtest.strategy import (
    ExitDecision,
    PortfolioStrategy,
    PositionView,
    Signal,
    Strategy,
)
from src.exchange.metadata import MetadataConfig
from src.features.pipeline import FeatureFrame

_log = structlog.get_logger("backtest.engine")


# --------------------------------------------------------------------------- #
# Inputs & outputs                                                             #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SymbolInput:
    """Everything the engine needs for one symbol over the test window."""

    symbol: str
    bars: list[dict]  # OHLCV rows on the grid, sorted by ts
    frame: FeatureFrame  # decision-time feature rows (same grid)
    spread_samples: list[dict]  # {ts, spread_bps}, sorted by ts
    funding_events: list[dict]  # {ts, funding_rate}
    activation_ts: int = 0  # earliest decision_ts at which the symbol is in-universe
    _spread_ts: list[int] = field(default_factory=list, repr=False)
    _spread_bps: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._spread_ts = [int(s["ts"]) for s in self.spread_samples]
        self._spread_bps = [float(s["spread_bps"]) for s in self.spread_samples]

    def spread_bps_at(self, decision_ts: int) -> float:
        """Last spread sample with ts <= decision_ts (modelled spread at decision)."""
        i = bisect.bisect_right(self._spread_ts, decision_ts) - 1
        return self._spread_bps[i] if i >= 0 else 2.0


@dataclass(slots=True)
class Trade:
    symbol: str
    strategy: str
    side: int
    qty: float
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    exit_reason: str
    notional: float
    risk_amount: float
    fee: float
    funding: float
    slippage_cost: float
    pnl: float  # net of fees + funding (slippage is embedded in fill prices)
    pnl_r: float
    regime: str
    session: int
    bars_held: int
    planned_rr: float = 0.0  # |tp-entry|/|entry-stop| at entry (∞-ish when TP unreachable)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "side": "long" if self.side > 0 else "short",
            "qty": self.qty,
            "entry_ts": self.entry_ts,
            "entry_price": self.entry_price,
            "exit_ts": self.exit_ts,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "notional": self.notional,
            "risk_amount": self.risk_amount,
            "fee": self.fee,
            "funding": self.funding,
            "slippage_cost": self.slippage_cost,
            "pnl": self.pnl,
            "pnl_r": self.pnl_r,
            "regime": self.regime,
            "session": self.session,
            "bars_held": self.bars_held,
            "planned_rr": self.planned_rr,
        }


@dataclass(slots=True)
class RejectedCandidate:
    symbol: str
    decision_ts: int
    side: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "decision_ts": self.decision_ts,
            "side": "long" if self.side > 0 else "short",
            "reason": self.reason,
        }


@dataclass(slots=True)
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    rejected: list[RejectedCandidate] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    equity_ts: list[int] = field(default_factory=list)
    initial_equity: float = 0.0
    symbols: list[str] = field(default_factory=list)
    rejected_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.initial_equity


# --------------------------------------------------------------------------- #
# Internal open-position record                                                #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Open:
    symbol: str
    strategy: str
    side: int
    qty: float
    entry_ts: int
    entry_price: float
    notional: float
    risk_amount: float
    stop_price: float
    tp_price: float
    hold_until_ts: int
    entry_fee: float
    funding: float
    slippage_cost: float
    regime: str
    session: int
    next_funding_idx: int
    trail_dist: float = 0.0  # price distance for the trailing stop (0 = no trailing)
    peak: float = 0.0  # best favorable price since entry (high for longs, low for shorts)
    maker: bool = False  # entered as a passive limit → take-profit exits as a maker limit too


def _regime_of(row: dict, spread_bps: float = 0.0) -> str:
    """Deterministic Section-11 regime label (R-code) from decision-time features."""
    from src.regime import detect_regime

    return detect_regime(row, spread_bps=spread_bps, data_ok=True)


class BacktestEngine:
    def __init__(
        self,
        cfg: BacktestConfig,
        meta: MetadataConfig,
        strategy: Strategy | PortfolioStrategy,
    ) -> None:
        self.cfg = cfg
        self.meta = meta
        self.strategy = strategy
        # Cross-asset strategies decide from peer rows at the same decision time.
        self.is_portfolio = hasattr(strategy, "evaluate_portfolio")
        # Optional per-bar management hook (duck-typed): a strategy that exposes ``manage``
        # (per-symbol) or ``manage_portfolio`` (cross-asset) gets an early thesis-driven exit
        # consulted each bar after stop/take-profit. Absent ⇒ engine-only exits (unchanged).
        self._has_manage = hasattr(strategy, "manage_portfolio" if self.is_portfolio else "manage")
        self.fees = FeeModel(meta, cfg.costs)
        self.slippage = SlippageModel(cfg.costs)
        self.funding = FundingModel(cfg.costs)
        self.risk = RiskSimulator(cfg.account, meta)
        # Shared-timeline state, (re)built per run() call (see the epoch-time note there).
        self._grid_iv: int = 1  # bar interval (ms), only for hold/bars-held duration math
        self._bars_by_ts: dict[str, dict[int, dict]] = {}  # symbol -> {bar ts -> bar}
        # Decision-time feature rows keyed by their decision_ts, for the management hook to look
        # up "this symbol's row at the bar being managed" (and peers, for cross-asset). Built only
        # when a strategy exposes manage()/manage_portfolio() (see run()).
        self._rows_by_ts: dict[str, dict[int, dict]] = {}

    def run(self, inputs: list[SymbolInput]) -> BacktestResult:
        result = BacktestResult(
            initial_equity=self.cfg.account.initial_equity,
            symbols=[s.symbol for s in inputs],
        )
        if not inputs:
            return result

        # Walk the shared timeline by EPOCH TIME, not by bar index. Time is the one coordinate
        # every symbol shares: at each timestamp a symbol either has a bar or it does not, looked
        # up directly by ``ts``. This is robust as the universe grows and symbols list/delist on
        # different dates — there is no per-symbol index offset to compute (the array-index model
        # silently produced zero trades for any contract listed after the window start, whose
        # first bar is not at index 0). ``iv`` is kept only to express bar-count durations
        # (hold period, bars-held), never as a position offset.
        self._grid_iv = self._grid_interval(inputs)
        self._bars_by_ts = {s.symbol: {b["ts"]: b for b in s.bars} for s in inputs}
        # Feature rows keyed by decision_ts so the management hook can fetch the causally-available
        # row for the bar it is managing (decision_ts == the managed bar's ts). Only built when the
        # strategy manages positions per bar (avoids the dict churn for engine-only-exit runs).
        self._rows_by_ts = (
            {s.symbol: {int(r["decision_ts"]): r for r in s.frame.rows} for s in inputs}
            if self._has_manage
            else {}
        )
        # The timeline is the ascending union of every symbol's real bar timestamps, so we visit
        # only timestamps where some symbol actually trades (no empty pre-listing slots).
        timeline = sorted({b["ts"] for s in inputs for b in s.bars})
        # Per-symbol: signals keyed by entry timestamp (decision made on the prior bar's close).
        # Each entry carries the originating feature row so entry never re-scans.
        if self.is_portfolio:
            signals_by_ts = self._portfolio_signals(inputs)
        else:
            signals_by_ts = {s.symbol: self._signals(s) for s in inputs}
        inputs_by_symbol = {s.symbol: s for s in inputs}

        equity = self.cfg.account.initial_equity
        open_positions: list[_Open] = []
        last_ts = timeline[-1] if timeline else 0

        for ts in timeline:
            # 1) Funding on open positions crossing a funding timestamp.
            for pos in open_positions:
                sym_in = inputs_by_symbol[pos.symbol]
                pos.next_funding_idx = self._charge_funding(pos, sym_in, ts)

            # 2) Manage existing open positions against this timestamp's bar.
            still_open: list[_Open] = []
            for pos in open_positions:
                bar = self._bars_by_ts[pos.symbol].get(ts)
                if bar is None:
                    still_open.append(pos)
                    continue
                trade = self._maybe_exit(pos, bar, ts, final=(ts == last_ts))
                if trade is not None:
                    equity += trade.pnl
                    result.trades.append(trade)
                else:
                    still_open.append(pos)
            open_positions = still_open

            # 3) New entries: signals whose entry timestamp == ts fill at this bar's open.
            for sym, by_ts in signals_by_ts.items():
                entry = by_ts.get(ts)
                if entry is None:
                    continue
                sig, row = entry
                sym_in = inputs_by_symbol[sym]
                bar = self._bars_by_ts[sym].get(ts)
                if bar is None:
                    continue
                new_pos = self._maybe_enter(
                    sym_in, sig, row, bar, ts, equity, open_positions, result
                )
                if new_pos is not None:
                    open_positions.append(new_pos)
                    # Intrabar stop/tp can trigger on the entry bar itself.
                    trade = self._maybe_exit(new_pos, bar, ts, final=(ts == last_ts))
                    if trade is not None:
                        equity += trade.pnl
                        result.trades.append(trade)
                        open_positions.remove(new_pos)

            # Mark-to-market equity for an honest drawdown curve.
            mtm = equity + self._unrealized(open_positions, inputs_by_symbol, ts)
            result.equity_curve.append(mtm)
            result.equity_ts.append(ts)

        # Force-close anything still open at the last timestamp (handled above when
        # ts == last_ts via final=True), but guard against symbols with shorter
        # series by closing at their own last bar.
        for pos in open_positions:
            sym_in = inputs_by_symbol[pos.symbol]
            last_bar = sym_in.bars[-1]
            trade = self._close(pos, last_bar, "end_of_data")
            equity += trade.pnl
            result.trades.append(trade)
        if open_positions:
            result.equity_curve[-1] = equity
        open_positions = []

        result.rejected_by_reason = self._reject_summary(result.rejected)
        return result

    # -- signal precomputation ------------------------------------------- #
    def _signals(self, sym_in: SymbolInput) -> dict[int, tuple[Signal, dict]]:
        """Map entry timestamp -> (Signal, originating row). decision_ts == entry ts.

        Keyed by the entry timestamp (``decision_ts`` == the entry bar's ts), so a signal only
        survives if a real bar exists at that timestamp (skipped before listing / inside a gap).
        Keying by epoch time — not array position — is what lets a symbol listed mid-window
        still trade."""
        bars = self._bars_by_ts[sym_in.symbol]
        out: dict[int, tuple[Signal, dict]] = {}
        for row in sym_in.frame.rows:
            if row["decision_ts"] < sym_in.activation_ts:
                continue  # symbol not yet in-universe (future-universe guard)
            entry_ts = int(row["decision_ts"])
            if entry_ts not in bars:
                continue  # no tradable bar at this timestamp (pre-listing or interior gap)
            # Per-symbol path runs only when is_portfolio is False (see run()), so the
            # strategy implements the plain Strategy protocol.
            sig = cast(Strategy, self.strategy).evaluate(row)
            if sig is not None:
                # Deterministic on a collision: if two feature rows share an entry timestamp
                # (mixed-grid data where decision_ts spacing != the OHLCV interval), KEEP THE
                # FIRST (earliest decision_ts) and log the drop — never a silent last-writer-wins.
                if entry_ts in out:
                    _log.warning(
                        "backtest_signal_bar_collision", symbol=sym_in.symbol, ts=entry_ts
                    )
                else:
                    out[entry_ts] = (sig, row)
        return out

    def _portfolio_signals(
        self, inputs: list[SymbolInput]
    ) -> dict[str, dict[int, tuple[Signal, dict]]]:
        """Cross-asset signals: each symbol evaluated with peer rows at the same ts.

        For every decision time present across the universe, build the peer
        snapshot (each symbol's feature row for that exact ``decision_ts`` — the
        close of the previous bar, so strictly causal) and let the portfolio
        strategy decide per symbol. Timing matches the per-symbol path: a signal
        for ``decision_ts`` fills at the bar whose ts == ``decision_ts``.
        """
        out: dict[str, dict[int, tuple[Signal, dict]]] = {s.symbol: {} for s in inputs}
        rows_by_dts: dict[str, dict[int, dict]] = {
            s.symbol: {int(r["decision_ts"]): r for r in s.frame.rows} for s in inputs
        }
        bars_by_symbol = self._bars_by_ts
        activation_by_symbol = {s.symbol: s.activation_ts for s in inputs}

        all_dts = sorted({dts for m in rows_by_dts.values() for dts in m})
        for dts in all_dts:
            peers = {sym: m[dts] for sym, m in rows_by_dts.items() if dts in m}
            for sym, row in peers.items():
                if dts < activation_by_symbol[sym]:
                    continue  # symbol not yet in-universe (future-universe guard)
                if dts not in bars_by_symbol[sym]:
                    continue  # no tradable bar at this timestamp (pre-listing or interior gap)
                others = {k: v for k, v in peers.items() if k != sym}
                # Reached only via the is_portfolio branch in run(), so the strategy
                # implements the PortfolioStrategy protocol.
                sig = cast(PortfolioStrategy, self.strategy).evaluate_portfolio(sym, row, others)
                if sig is not None:
                    if dts in out[sym]:  # collision → keep first (earliest dts), log the drop
                        _log.warning("backtest_signal_bar_collision", symbol=sym, ts=dts)
                    else:
                        out[sym][dts] = (sig, row)
        return out

    # -- entry ----------------------------------------------------------- #
    def _maybe_enter(
        self,
        sym_in: SymbolInput,
        sig: Signal,
        row: dict,
        bar: dict,
        ts: int,
        equity: float,
        open_positions: list[_Open],
        result: BacktestResult,
    ) -> _Open | None:
        decision_ts = bar["ts"]

        def reject(reason: str) -> None:
            result.rejected.append(RejectedCandidate(sym_in.symbol, decision_ts, sig.side, reason))

        # Concurrency caps (Section 17: max positions total / per symbol).
        per_symbol = sum(1 for p in open_positions if p.symbol == sym_in.symbol)
        if per_symbol >= self.cfg.account.max_concurrent_per_symbol:
            reject("open_position_conflict")
            return None
        if len(open_positions) >= self.cfg.account.max_concurrent_total:
            reject("max_concurrent_total")
            return None

        # Execution hard-blockers (Section 15/18): toxic spread, slippage cap.
        spread_bps = float(sym_in.spread_bps_at(decision_ts))
        if spread_bps > self.cfg.execution.max_spread_bps:
            reject(f"toxic_spread({spread_bps:.1f}bps)")
            return None

        # Risk sizing (Section 17). Entry reference is THIS bar's open.
        ref_price = float(bar["open"])
        sizing = self.risk.size(
            sym_in.symbol, equity=equity, entry_price=ref_price, stop_frac=sig.stop_frac
        )
        if not sizing.approved:
            reject(sizing.reason)
            return None

        if sig.maker:
            # Passive limit entry: post ``limit_offset_frac`` inside the fill-bar open and fill
            # ONLY if the bar trades through the limit (a buy fills when the low reaches it; a sell
            # when the high reaches it). No fill ⇒ the order is cancelled and no position opens
            # (the accepted "fewer trades" cost of maker execution). A maker fill is exact at the
            # limit price with zero slippage and pays the maker fee. The toxic-spread blocker above
            # still applies; the taker slippage cap does not (a resting limit has no slippage).
            offset = max(0.0, sig.limit_offset_frac)
            limit_price = ref_price * (1.0 - sig.side * offset)
            if sig.side > 0:
                if float(bar["low"]) > limit_price:
                    reject("maker_no_fill")
                    return None
            elif float(bar["high"]) < limit_price:
                reject("maker_no_fill")
                return None
            entry_price = limit_price
            notional = sizing.qty * entry_price
            entry_fee = self.fees.fee(sym_in.symbol, notional, maker=True)
            entry_slip_cost = 0.0
        else:
            bar_notional = float(bar["volume"]) * ref_price
            slip = self.slippage.slippage_frac(
                spread_bps=spread_bps, notional=sizing.notional, bar_notional=bar_notional
            )
            if slip > self.cfg.execution.max_slippage_frac:
                reject(f"slippage_estimate_exceeds_cap({slip:.4f})")
                return None

            # Fill at the next-bar open with adverse slippage; entry is taker.
            entry_side = BUY if sig.side > 0 else SELL
            entry_price = self.slippage.fill_price(ref_price, entry_side, slip)
            notional = sizing.qty * entry_price
            entry_fee = self.fees.fee(sym_in.symbol, notional, maker=False)
            entry_slip_cost = abs(entry_price - ref_price) * sizing.qty

        stop_price = entry_price * (1.0 - sig.side * sig.stop_frac)
        tp_price = entry_price * (1.0 + sig.side * sig.tp_frac)

        return _Open(
            symbol=sym_in.symbol,
            strategy=self.strategy.name,
            side=sig.side,
            qty=sizing.qty,
            entry_ts=decision_ts,
            entry_price=entry_price,
            notional=notional,
            risk_amount=sizing.qty * entry_price * sig.stop_frac,
            stop_price=stop_price,
            tp_price=tp_price,
            # Hold duration is expressed in BARS; convert to a time horizon (entry ts + N·iv) so
            # the exit fires at the right wall-clock moment regardless of array position.
            hold_until_ts=ts
            + (
                sig.hold_bars
                if sig.hold_bars is not None
                else self.cfg.reference_strategy.hold_bars
            )
            * self._grid_iv,
            entry_fee=entry_fee,
            funding=0.0,
            slippage_cost=entry_slip_cost,
            regime=_regime_of(row, sym_in.spread_bps_at(decision_ts)),
            session=int(row.get("session_code", 0)),
            next_funding_idx=self._first_funding_after(sym_in, decision_ts),
            trail_dist=sig.trail_frac * entry_price,
            peak=entry_price,
            maker=sig.maker,
        )

    # -- exit ------------------------------------------------------------ #
    def _maybe_exit(self, pos: _Open, bar: dict, ts: int, *, final: bool) -> Trade | None:
        high, low = float(bar["high"]), float(bar["low"])
        # Effective stop: the initial stop, ratcheted by a trailing stop set from the best
        # favorable excursion BEFORE this bar (so a fresh high on this bar can't raise the stop
        # that this same bar's low then hits — conservative, no intrabar look-ahead). When
        # trail_dist is 0 the effective stop is just the fixed initial stop.
        if pos.side > 0:
            stop_level = pos.stop_price
            if pos.trail_dist > 0:
                stop_level = max(stop_level, pos.peak - pos.trail_dist)
            if low <= stop_level:
                reason = "trailing_stop" if stop_level > pos.stop_price else "stop"
                return self._close(pos, bar, reason, price=stop_level)
            if high >= pos.tp_price:
                # A take-profit is a resting limit: for a maker position it fills as a maker
                # (no slippage, maker fee); risk exits below stay taker (you cross to get out).
                return self._close(pos, bar, "take_profit", price=pos.tp_price, maker=pos.maker)
        else:
            stop_level = pos.stop_price
            if pos.trail_dist > 0:
                stop_level = min(stop_level, pos.peak + pos.trail_dist)
            if high >= stop_level:
                reason = "trailing_stop" if stop_level < pos.stop_price else "stop"
                return self._close(pos, bar, reason, price=stop_level)
            if low <= pos.tp_price:
                return self._close(pos, bar, "take_profit", price=pos.tp_price, maker=pos.maker)
        # Early thesis-driven exit (manage hook) — consulted AFTER stop/take-profit (so a
        # protective stop always wins) and BEFORE the time-stop backstop (so a position whose
        # edge has played out exits on its own signal, with its own reason, rather than bleeding
        # to the time-stop). No-op when the strategy exposes no manage hook.
        if self._has_manage:
            exit_dec = self._manage_decision(pos, ts)
            if exit_dec is not None:
                return self._close_via_manage(pos, bar, exit_dec)
        if ts >= pos.hold_until_ts:
            return self._close(pos, bar, "time_stop", price=float(bar["close"]))
        if final:
            return self._close(pos, bar, "end_of_data", price=float(bar["close"]))
        # Survived this bar — ratchet the favorable-excursion peak for the next bar's trail.
        if pos.trail_dist > 0:
            pos.peak = max(pos.peak, high) if pos.side > 0 else min(pos.peak, low)
        return None

    def _manage_decision(self, pos: _Open, ts: int) -> ExitDecision | None:
        """Ask the strategy whether to exit ``pos`` early at ``ts``, or None.

        Uses the decision-time feature row for the managed bar (``decision_ts == ts`` — the
        prior bar's close, so strictly causal, same as entries). For cross-asset families the
        peer snapshot is every OTHER symbol's row at the same ``ts``. Returns None when no row
        exists at this ts (gap / pre-listing) or the strategy declines."""
        row = self._rows_by_ts.get(pos.symbol, {}).get(ts)
        if row is None:
            return None
        view = PositionView(
            side=pos.side,
            entry_price=pos.entry_price,
            bars_held=int((ts - pos.entry_ts) // self._grid_iv),
            regime=pos.regime,
        )
        if self.is_portfolio:
            peers = {
                sym: m[ts]
                for sym, m in self._rows_by_ts.items()
                if sym != pos.symbol and ts in m
            }
            return cast(PortfolioStrategy, self.strategy).manage_portfolio(  # type: ignore[attr-defined]
                pos.symbol, row, peers, view
            )
        return cast(Strategy, self.strategy).manage(row, view)  # type: ignore[attr-defined]

    def _close_via_manage(self, pos: _Open, bar: dict, exit_dec: ExitDecision) -> Trade:
        """Execute a manage-hook exit on this bar: maker passive limit with taker fallback.

        For a maker position, post the exit as a passive limit ``limit_offset_frac`` FAVORABLE to
        the closing side (a long sells above the open, a short buys below) and fill maker (no
        slippage, maker fee) if the bar trades through it. If the bar never reaches the limit — or
        the position entered taker — cross the spread as a TAKER at the bar close, so the exit is
        guaranteed on the signal bar, paying taker cost only when the passive fill missed."""
        offset = max(0.0, exit_dec.limit_offset_frac)
        if pos.maker and offset > 0.0:
            ref = float(bar["open"])
            # Favorable passive close: long → sell at ref·(1+offset) above; short → buy below.
            limit_price = ref * (1.0 + pos.side * offset)
            if pos.side > 0 and float(bar["high"]) >= limit_price:
                return self._close(pos, bar, exit_dec.reason, price=limit_price, maker=True)
            if pos.side < 0 and float(bar["low"]) <= limit_price:
                return self._close(pos, bar, exit_dec.reason, price=limit_price, maker=True)
        # Taker fallback — guarantee the exit at the managed bar's close.
        return self._close(pos, bar, exit_dec.reason, price=float(bar["close"]), maker=False)

    def _close(
        self, pos: _Open, bar: dict, reason: str, price: float | None = None, *, maker: bool = False
    ) -> Trade:
        ref_price = float(bar["close"]) if price is None else price
        exit_ts = int(bar["ts"])
        if maker:
            # Maker exit (take-profit limit that price came into): fill exact at the limit, no
            # slippage, maker fee. Only the take-profit of a maker position takes this path.
            exit_price = ref_price
            exit_fee = self.fees.fee(pos.symbol, pos.qty * exit_price, maker=True)
            exit_slip_cost = 0.0
        else:
            # Closing is taker on the opposite side (adverse slippage).
            exit_side = SELL if pos.side > 0 else BUY
            spread_bps = 2.0  # modelled exit spread floor; refined via min_half_spread_frac
            # Use the REAL exit-bar notional for the impact term (mirrors the entry side). A
            # hardcoded bar_notional=1.0 made the impact term ~`notional`× too large on exits, so
            # any non-zero impact_coeff would blow up exit fills and corrupt every expectancy.
            bar_notional = float(bar.get("volume", 0.0) or 0.0) * ref_price
            slip = self.slippage.slippage_frac(
                spread_bps=spread_bps,
                notional=pos.notional,
                bar_notional=bar_notional or pos.notional,
            )
            exit_price = self.slippage.fill_price(ref_price, exit_side, slip)
            exit_fee = self.fees.fee(pos.symbol, pos.qty * exit_price, maker=False)
            exit_slip_cost = abs(exit_price - ref_price) * pos.qty

        gross = pos.side * (exit_price - pos.entry_price) * pos.qty
        total_fee = pos.entry_fee + exit_fee
        pnl = gross - total_fee - pos.funding
        pnl_r = pnl / pos.risk_amount if pos.risk_amount > 0 else 0.0
        # Planned reward:risk at entry (target distance / stop distance). For momentum candidates
        # the TP is intentionally unreachable, so this is large — which is itself the signal that
        # the edge depends on the time-stop, not a target (compare against realized RR).
        stop_dist = abs(pos.entry_price - pos.stop_price)
        planned_rr = abs(pos.tp_price - pos.entry_price) / stop_dist if stop_dist > 0 else 0.0
        return Trade(
            symbol=pos.symbol,
            strategy=pos.strategy,
            side=pos.side,
            qty=pos.qty,
            entry_ts=pos.entry_ts,
            entry_price=pos.entry_price,
            exit_ts=exit_ts,
            exit_price=exit_price,
            exit_reason=reason,
            notional=pos.notional,
            risk_amount=pos.risk_amount,
            fee=total_fee,
            funding=pos.funding,
            slippage_cost=pos.slippage_cost + exit_slip_cost,
            pnl=pnl,
            pnl_r=pnl_r,
            regime=pos.regime,
            session=pos.session,
            # Bars held in grid terms: elapsed time / bar interval (was an index delta).
            bars_held=int((exit_ts - pos.entry_ts) // self._grid_iv),
            planned_rr=round(planned_rr, 6),
        )

    # -- funding --------------------------------------------------------- #
    def _charge_funding(self, pos: _Open, sym_in: SymbolInput, ts_j: int) -> int:
        events = sym_in.funding_events
        idx = pos.next_funding_idx
        while idx < len(events) and events[idx]["ts"] <= ts_j:
            ev = events[idx]
            if ev["ts"] > pos.entry_ts:  # only funding while the position is open
                pos.funding += self.funding.payment(
                    side=pos.side, notional=pos.notional, funding_rate=float(ev["funding_rate"])
                )
            idx += 1
        return idx

    def _first_funding_after(self, sym_in: SymbolInput, ts: int) -> int:
        for i, ev in enumerate(sym_in.funding_events):
            if ev["ts"] > ts:
                return i
        return len(sym_in.funding_events)

    # -- helpers --------------------------------------------------------- #
    def _grid_interval(self, inputs: list[SymbolInput]) -> int:
        """The bar interval (ms) defining the shared grid: the SMALLEST positive gap between
        consecutive bars across all symbols. Taking the minimum recovers the true timeframe even
        when a series has interior holes (a single missing candle would inflate a naive
        ``bars[1]-bars[0]``)."""
        best: int | None = None
        for s in inputs:
            for a, b in zip(s.bars, s.bars[1:], strict=False):
                d = int(b["ts"] - a["ts"])
                if d > 0 and (best is None or d < best):
                    best = d
        return best or 1

    def _unrealized(
        self, open_positions: list[_Open], inputs_by_symbol: dict[str, SymbolInput], ts: int
    ) -> float:
        total = 0.0
        for pos in open_positions:
            bar = self._bars_by_ts[pos.symbol].get(ts)
            if bar is None:
                continue
            mark = float(bar["close"])
            total += pos.side * (mark - pos.entry_price) * pos.qty - pos.entry_fee - pos.funding
        return total

    def _reject_summary(self, rejected: list[RejectedCandidate]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rejected:
            key = r.reason.split("(")[0]
            out[key] = out.get(key, 0) + 1
        return out
