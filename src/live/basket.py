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
from src.paper.session import PaperSession, PaperTrade

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
    ) -> None:
        self.engine = CrossSectionalEngine(cfg, meta, strategy)
        self.engine._grid_iv = max(1, int(bar_interval_ms))
        self.strategy = strategy
        self.session = session
        self.on_trade = on_trade
        self._holdings: dict = {}
        self._equity = cfg.account.initial_equity
        self._last_rebal: int | None = None
        self._result = BacktestResult(initial_equity=cfg.account.initial_equity)
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
            scores: dict[str, float] = {}
            for sym, row in rows_at.items():
                if sym not in bars_at:
                    continue
                sc = self.strategy.score(row)  # type: ignore[attr-defined]
                if sc is not None:
                    f = float(sc)
                    if f == f and abs(f) != float("inf"):  # finite
                        scores[sym] = f
            if len(scores) >= eng.min_universe:
                bars_by_ts = {s: {ts: b} for s, b in bars_at.items()}
                rows_by_ts = {s: {ts: r} for s, r in rows_at.items()}
                self._equity = eng._rebalance(
                    self._holdings, scores, bars_by_ts, rows_by_ts, by_symbol, ts, self._equity,
                    self._result,
                )
                self._last_rebal = ts
        self._flush()

    def close_all(self, ts: int, bars_at: dict[str, dict]) -> None:
        """Flatten every leg at the latest bar (session end / stop)."""
        for sym, leg in list(self._holdings.items()):
            bar = bars_at.get(sym)
            if bar is not None:
                self._equity += self.engine._close_leg(leg, bar, "end_of_data", self._result)
        self._holdings.clear()
        self._flush()

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
