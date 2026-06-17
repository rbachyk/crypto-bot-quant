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

from src.backtest.config import BacktestConfig
from src.backtest.costs import BUY, SELL, FeeModel, FundingModel, SlippageModel
from src.backtest.risk import RiskSimulator
from src.backtest.strategy import PortfolioStrategy, Signal, Strategy
from src.exchange.metadata import MetadataConfig
from src.features.pipeline import FeatureFrame


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
    entry_bar: int
    entry_price: float
    notional: float
    risk_amount: float
    stop_price: float
    tp_price: float
    hold_until_bar: int
    entry_fee: float
    funding: float
    slippage_cost: float
    regime: str
    session: int
    next_funding_idx: int


def _regime_of(row: dict) -> str:
    """Coarse volatility/trend regime label from decision-time features."""
    rank = float(row.get("atr_pct_rank", 0.5))
    vol = "high_vol" if rank >= 0.66 else ("low_vol" if rank <= 0.33 else "mid_vol")
    trend = "up" if float(row.get("trend_slope", 0.0)) >= 0 else "down"
    return f"{vol}_{trend}"


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
        self.fees = FeeModel(meta, cfg.costs)
        self.slippage = SlippageModel(cfg.costs)
        self.funding = FundingModel(cfg.costs)
        self.risk = RiskSimulator(cfg.account, meta)

    def run(self, inputs: list[SymbolInput]) -> BacktestResult:
        result = BacktestResult(
            initial_equity=self.cfg.account.initial_equity,
            symbols=[s.symbol for s in inputs],
        )
        if not inputs:
            return result

        n_bars = max(len(s.bars) for s in inputs)
        # Per-symbol: signals indexed by entry bar (decision made on bar-1's close).
        # Each entry carries the originating feature row so entry never re-scans.
        if self.is_portfolio:
            signals_by_bar = self._portfolio_signals(inputs)
        else:
            signals_by_bar = {s.symbol: self._signals(s) for s in inputs}
        inputs_by_symbol = {s.symbol: s for s in inputs}

        equity = self.cfg.account.initial_equity
        open_positions: list[_Open] = []

        for j in range(n_bars):
            ts_j = self._bar_ts(inputs, j)

            # 1) Funding on open positions crossing a funding timestamp.
            for pos in open_positions:
                sym_in = inputs_by_symbol[pos.symbol]
                pos.next_funding_idx = self._charge_funding(pos, sym_in, ts_j)

            # 2) Manage existing open positions against bar j's range.
            still_open: list[_Open] = []
            for pos in open_positions:
                bar = self._bar_at(inputs_by_symbol[pos.symbol], j)
                if bar is None:
                    still_open.append(pos)
                    continue
                trade = self._maybe_exit(pos, bar, j, final=(j == n_bars - 1))
                if trade is not None:
                    equity += trade.pnl
                    result.trades.append(trade)
                else:
                    still_open.append(pos)
            open_positions = still_open

            # 3) New entries: signals whose entry bar == j fill at bar j's open.
            for sym, by_bar in signals_by_bar.items():
                entry = by_bar.get(j)
                if entry is None:
                    continue
                sig, row = entry
                sym_in = inputs_by_symbol[sym]
                bar = self._bar_at(sym_in, j)
                if bar is None:
                    continue
                new_pos = self._maybe_enter(
                    sym_in, sig, row, bar, j, equity, open_positions, result
                )
                if new_pos is not None:
                    open_positions.append(new_pos)
                    # Intrabar stop/tp can trigger on the entry bar itself.
                    trade = self._maybe_exit(new_pos, bar, j, final=(j == n_bars - 1))
                    if trade is not None:
                        equity += trade.pnl
                        result.trades.append(trade)
                        open_positions.remove(new_pos)

            # Mark-to-market equity for an honest drawdown curve.
            mtm = equity + self._unrealized(open_positions, inputs_by_symbol, j)
            result.equity_curve.append(mtm)
            result.equity_ts.append(ts_j)

        # Force-close anything still open at the last bar (handled above when
        # j == n_bars-1 via final=True), but guard against symbols with shorter
        # series by closing at their own last bar.
        for pos in open_positions:
            sym_in = inputs_by_symbol[pos.symbol]
            last_bar = sym_in.bars[-1]
            trade = self._close(pos, last_bar, len(sym_in.bars) - 1, "end_of_data")
            equity += trade.pnl
            result.trades.append(trade)
        if open_positions:
            result.equity_curve[-1] = equity
        open_positions = []

        result.rejected_by_reason = self._reject_summary(result.rejected)
        return result

    # -- signal precomputation ------------------------------------------- #
    def _signals(self, sym_in: SymbolInput) -> dict[int, tuple[Signal, dict]]:
        """Map entry-bar index -> (Signal, originating row). decision_ts == entry ts."""
        iv = self._iv(sym_in)
        out: dict[int, tuple[Signal, dict]] = {}
        for row in sym_in.frame.rows:
            if row["decision_ts"] < sym_in.activation_ts:
                continue  # symbol not yet in-universe (future-universe guard)
            entry_bar = row["decision_ts"] // iv
            if entry_bar >= len(sym_in.bars):
                continue
            # Per-symbol path runs only when is_portfolio is False (see run()), so the
            # strategy implements the plain Strategy protocol.
            sig = cast(Strategy, self.strategy).evaluate(row)
            if sig is not None:
                out[int(entry_bar)] = (sig, row)
        return out

    def _portfolio_signals(
        self, inputs: list[SymbolInput]
    ) -> dict[str, dict[int, tuple[Signal, dict]]]:
        """Cross-asset signals: each symbol evaluated with peer rows at the same ts.

        For every decision time present across the universe, build the peer
        snapshot (each symbol's feature row for that exact ``decision_ts`` — the
        close of the previous bar, so strictly causal) and let the portfolio
        strategy decide per symbol. Timing matches the per-symbol path: a signal
        for ``decision_ts`` fills at entry bar ``decision_ts // iv``.
        """
        out: dict[str, dict[int, tuple[Signal, dict]]] = {s.symbol: {} for s in inputs}
        rows_by_dts: dict[str, dict[int, dict]] = {
            s.symbol: {int(r["decision_ts"]): r for r in s.frame.rows} for s in inputs
        }
        n_bars_by_symbol = {s.symbol: len(s.bars) for s in inputs}
        iv_by_symbol = {s.symbol: self._iv(s) for s in inputs}
        activation_by_symbol = {s.symbol: s.activation_ts for s in inputs}

        all_dts = sorted({dts for m in rows_by_dts.values() for dts in m})
        for dts in all_dts:
            peers = {sym: m[dts] for sym, m in rows_by_dts.items() if dts in m}
            for sym, row in peers.items():
                if dts < activation_by_symbol[sym]:
                    continue  # symbol not yet in-universe (future-universe guard)
                entry_bar = dts // iv_by_symbol[sym]
                if entry_bar >= n_bars_by_symbol[sym]:
                    continue
                others = {k: v for k, v in peers.items() if k != sym}
                # Reached only via the is_portfolio branch in run(), so the strategy
                # implements the PortfolioStrategy protocol.
                sig = cast(PortfolioStrategy, self.strategy).evaluate_portfolio(sym, row, others)
                if sig is not None:
                    out[sym][int(entry_bar)] = (sig, row)
        return out

    # -- entry ----------------------------------------------------------- #
    def _maybe_enter(
        self,
        sym_in: SymbolInput,
        sig: Signal,
        row: dict,
        bar: dict,
        bar_idx: int,
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
            entry_bar=bar_idx,
            entry_price=entry_price,
            notional=notional,
            risk_amount=sizing.qty * entry_price * sig.stop_frac,
            stop_price=stop_price,
            tp_price=tp_price,
            hold_until_bar=bar_idx
            + (
                sig.hold_bars
                if sig.hold_bars is not None
                else self.cfg.reference_strategy.hold_bars
            ),
            entry_fee=entry_fee,
            funding=0.0,
            slippage_cost=entry_slip_cost,
            regime=_regime_of(row),
            session=int(row.get("session_code", 0)),
            next_funding_idx=self._first_funding_after(sym_in, decision_ts),
        )

    # -- exit ------------------------------------------------------------ #
    def _maybe_exit(self, pos: _Open, bar: dict, bar_idx: int, *, final: bool) -> Trade | None:
        high, low = float(bar["high"]), float(bar["low"])
        # Stop checked before TP (conservative: assume the adverse level first).
        if pos.side > 0:
            if low <= pos.stop_price:
                return self._close(pos, bar, bar_idx, "stop", price=pos.stop_price)
            if high >= pos.tp_price:
                return self._close(pos, bar, bar_idx, "take_profit", price=pos.tp_price)
        else:
            if high >= pos.stop_price:
                return self._close(pos, bar, bar_idx, "stop", price=pos.stop_price)
            if low <= pos.tp_price:
                return self._close(pos, bar, bar_idx, "take_profit", price=pos.tp_price)
        if bar_idx >= pos.hold_until_bar:
            return self._close(pos, bar, bar_idx, "time_stop", price=float(bar["close"]))
        if final:
            return self._close(pos, bar, bar_idx, "end_of_data", price=float(bar["close"]))
        return None

    def _close(
        self, pos: _Open, bar: dict, bar_idx: int, reason: str, price: float | None = None
    ) -> Trade:
        ref_price = float(bar["close"]) if price is None else price
        exit_ts = int(bar["ts"])
        # Closing is taker on the opposite side (adverse slippage).
        exit_side = SELL if pos.side > 0 else BUY
        spread_bps = 2.0  # modelled exit spread floor; refined via min_half_spread_frac
        slip = self.slippage.slippage_frac(
            spread_bps=spread_bps, notional=pos.notional, bar_notional=1.0
        )
        exit_price = self.slippage.fill_price(ref_price, exit_side, slip)
        exit_notional = pos.qty * exit_price
        exit_fee = self.fees.fee(pos.symbol, exit_notional, maker=False)
        exit_slip_cost = abs(exit_price - ref_price) * pos.qty

        gross = pos.side * (exit_price - pos.entry_price) * pos.qty
        total_fee = pos.entry_fee + exit_fee
        pnl = gross - total_fee - pos.funding
        pnl_r = pnl / pos.risk_amount if pos.risk_amount > 0 else 0.0
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
            bars_held=bar_idx - pos.entry_bar,
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
    def _iv(self, sym_in: SymbolInput) -> int:
        if len(sym_in.bars) >= 2:
            return int(sym_in.bars[1]["ts"] - sym_in.bars[0]["ts"])
        return 1

    def _bar_ts(self, inputs: list[SymbolInput], j: int) -> int:
        for s in inputs:
            if j < len(s.bars):
                return int(s.bars[j]["ts"])
        return j

    def _bar_at(self, sym_in: SymbolInput, j: int) -> dict | None:
        if 0 <= j < len(sym_in.bars):
            return sym_in.bars[j]
        return None

    def _unrealized(
        self, open_positions: list[_Open], inputs_by_symbol: dict[str, SymbolInput], j: int
    ) -> float:
        total = 0.0
        for pos in open_positions:
            bar = self._bar_at(inputs_by_symbol[pos.symbol], j)
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
