"""Cross-sectional (basket) backtest engine — the proper vehicle for carry / factor strategies.

The per-trade :class:`~src.backtest.engine.BacktestEngine` sizes each signal as an independent,
risk-bounded position with a stop/TP — right for directional edges, wrong for a CARRY edge whose
signal (e.g. funding dispersion) is a few bps of cash flow that gets buried under the directional
variance of unhedged legs. This engine instead holds a **dollar-neutral basket**, rebalanced
periodically: at each rebalance it ranks the universe by the strategy's ``score(row)``, goes LONG
the top fraction and SHORT the bottom fraction in equal dollar amounts (net ≈ 0, so the common
market factor cancels and the cross-sectional signal — funding carry — is what's left), holds
across funding settlements (the funding model books the carry), and only trades the DELTA on each
rebalance (stable basket members are not churned). It emits the same :class:`BacktestResult`
(equity curve + per-leg trades) so the unchanged walk-forward / hold-out / deflated-Sharpe gate
judges it on the identical bar as every other strategy.

Event-based (walks the shared bar grid, explicit fills/costs/funding per the cost models), NOT a
vectorized shortcut — so it is validation-grade (Section 19).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from src.backtest.config import BacktestConfig
from src.backtest.costs import BUY, SELL, FeeModel, FundingModel, SlippageModel
from src.backtest.engine import BacktestResult, SymbolInput, Trade, _regime_of
from src.exchange.metadata import MetadataConfig


@dataclass(slots=True)
class _Leg:
    """One open basket leg (held between rebalances)."""

    symbol: str
    side: int
    qty: float
    entry_ts: int
    entry_price: float
    notional: float
    risk_amount: float
    entry_fee: float
    funding: float
    slippage_cost: float
    regime: str
    session: int
    next_funding_idx: int


class CrossSectionalEngine:
    """Dollar-neutral, periodically-rebalanced basket engine. Same ``run(inputs) -> BacktestResult``
    interface as :class:`BacktestEngine`, so ``run_engine`` routes to it for ``cross_sectional``
    strategies with no other change. Portfolio knobs come from the strategy's ``params.extra``:
    ``basket_frac`` (top/bottom fraction per side), ``rebalance_bars`` (cadence),
    ``portfolio_gross`` (gross exposure ÷ equity), ``min_universe`` (min names to form a basket)."""

    def __init__(
        self, cfg: BacktestConfig, meta: MetadataConfig, strategy: object
    ) -> None:
        self.cfg = cfg
        self.meta = meta
        self.strategy = strategy
        self.fees = FeeModel(meta, cfg.costs)
        self.slippage = SlippageModel(cfg.costs)
        self.funding = FundingModel(cfg.costs)
        params = getattr(strategy, "params", None)
        ex = dict(getattr(params, "extra", {}) or {})
        self.basket_frac = float(ex.get("basket_frac", 0.2))
        # Rebalance cadence: prefer rebalance_hours (timeframe-INDEPENDENT — the funding rank is a
        # slow signal, so the right cadence is a wall-clock period, not a bar count). Falls back to
        # rebalance_bars × the bar interval. Turnover is the carry edge's main cost.
        self.rebalance_hours = float(ex.get("rebalance_hours", 0.0))
        self.rebalance_bars = max(1, int(ex.get("rebalance_bars", 8)))
        self.portfolio_gross = float(ex.get("portfolio_gross", 1.0))
        self.min_universe = int(ex.get("min_universe", 4))
        # Neutralization: "dollar" (equal long-$/short-$) or "beta" (weight legs so NET beta-to-the-
        # market-factor ≈ 0 — strips the residual market exposure a dollar-neutral basket leaves
        # when the long/short legs differ in beta, which adds variance/noise). Beta is
        # computed INSIDE the engine from the bars (rolling cov/var vs the equal-weight universe
        # return) — no feature column, so it runs on existing cached inputs with no rebuild.
        self.neutralization = str(ex.get("neutralization", "dollar"))
        self.beta_window = max(10, int(ex.get("beta_window", 120)))
        # Maker rebalancing: a basket rebalance is low-urgency, so post passive limits (maker fee,
        # no slippage) rather than cross the spread — the realistic basket execution, and turnover
        # cost is the carry edge's tightest margin (the fee-stress blocker). Final force-close at
        # end-of-data stays taker (you can't be patient at the boundary).
        self.maker = float(ex.get("maker_rebalance", 0.0)) > 0
        self.stop_frac = float(getattr(params, "stop_frac", 0.02)) or 0.02
        self.risk_scale = min(1.0, max(0.0, float(getattr(strategy, "risk_scale", 1.0))))
        self.name = str(getattr(strategy, "name", "cross_sectional"))
        self._grid_iv = 1
        self._sym_rets: dict[str, tuple[list[int], list[float]]] = {}
        self._mkt_ret: dict[int, float] = {}

    # -- public ---------------------------------------------------------- #
    def run(self, inputs: list[SymbolInput]) -> BacktestResult:
        result = BacktestResult(
            initial_equity=self.cfg.account.initial_equity,
            symbols=[s.symbol for s in inputs],
        )
        if not inputs:
            return result
        by_symbol = {s.symbol: s for s in inputs}
        bars_by_ts = {s.symbol: {b["ts"]: b for b in s.bars} for s in inputs}
        rows_by_ts = {
            s.symbol: {int(r["decision_ts"]): r for r in s.frame.rows} for s in inputs
        }
        self._grid_iv = self._grid_interval(inputs)
        if self.neutralization == "beta":
            self._prepare_returns(inputs)
        timeline = sorted({b["ts"] for s in inputs for b in s.bars})
        last_ts = timeline[-1]
        rebal_ms = (
            int(self.rebalance_hours * 3_600_000)
            if self.rebalance_hours > 0
            else self.rebalance_bars * self._grid_iv
        )

        equity = self.cfg.account.initial_equity
        holdings: dict[str, _Leg] = {}
        last_rebal: int | None = None

        for ts in timeline:
            # 1) Funding on every held leg crossing a funding timestamp (the carry).
            for sym, leg in holdings.items():
                leg.next_funding_idx = self._charge_funding(leg, by_symbol[sym], ts)
            # 2) Rebalance on cadence.
            if last_rebal is None or ts - last_rebal >= rebal_ms:
                scores: dict[str, float] = {}
                for s in inputs:
                    if ts < s.activation_ts:
                        continue
                    row = rows_by_ts[s.symbol].get(ts)
                    if row is None or bars_by_ts[s.symbol].get(ts) is None:
                        continue
                    sc = self.strategy.score(row)  # type: ignore[attr-defined]
                    if sc is not None and math.isfinite(float(sc)):
                        scores[s.symbol] = float(sc)
                if len(scores) >= self.min_universe:
                    equity = self._rebalance(
                        holdings, scores, bars_by_ts, rows_by_ts, by_symbol, ts, equity, result
                    )
                    last_rebal = ts
            # 3) Mark to market for an honest drawdown curve.
            result.equity_curve.append(equity + self._unrealized(holdings, bars_by_ts, ts))
            result.equity_ts.append(ts)

        # Force-close everything at the last bar.
        for sym, leg in list(holdings.items()):
            bar = bars_by_ts[sym].get(last_ts) or by_symbol[sym].bars[-1]
            equity += self._close_leg(leg, bar, "end_of_data", result)
        holdings.clear()
        if result.equity_curve:
            result.equity_curve[-1] = equity
        return result

    # -- rebalance ------------------------------------------------------- #
    def _rebalance(self, holdings, scores, bars_by_ts, rows_by_ts, by_symbol, ts, equity, result):
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)  # high score first
        k = max(1, int(len(ranked) * self.basket_frac))
        longs = {s for s, _ in ranked[:k]}
        shorts = {s for s, _ in ranked[-k:]} - longs  # disjoint if the universe is small
        target = {**dict.fromkeys(longs, 1), **dict.fromkeys(shorts, -1)}

        # Close legs that leave the basket or flip side (realize pnl into cash equity first).
        for sym, leg in list(holdings.items()):
            if target.get(sym) != leg.side:
                bar = bars_by_ts[sym].get(ts)
                if bar is not None:
                    equity += self._close_leg(leg, bar, "rebalance", result)
                    del holdings[sym]
        # Open new legs (stable same-side members are kept — no churn). Per-symbol target notionals
        # are dollar- or beta-neutral depending on config.
        gross = equity * self.portfolio_gross * self.risk_scale
        notionals = self._target_notionals(longs, shorts, ts, gross)
        for sym, side in target.items():
            if sym in holdings:
                continue
            bar = bars_by_ts[sym].get(ts)
            row = rows_by_ts[sym].get(ts)
            if bar is None or row is None:
                continue
            leg = self._open_leg(sym, side, notionals.get(sym, 0.0), bar, row, ts, by_symbol[sym])
            if leg is not None:
                holdings[sym] = leg
        return equity

    def _target_notionals(self, longs, shorts, ts, gross) -> dict[str, float]:
        """Per-leg notional. Dollar-neutral = equal weight. Beta-neutral scales the two legs so the
        NET dollar-beta is zero: a·Σβ_long = b·Σβ_short, k_l·a + k_s·b = gross (legs ≥ 0)."""
        kl, ks = len(longs), len(shorts)
        equal = {s: gross / max(1, kl + ks) for s in (*longs, *shorts)}
        if self.neutralization != "beta" or not kl or not ks:
            return equal
        betas = {s: self._beta(s, ts) for s in (*longs, *shorts)}
        sum_lb = sum(betas[s] for s in longs)
        sum_sb = sum(betas[s] for s in shorts)
        if sum_lb <= 1e-9 or sum_sb <= 1e-9:  # degenerate (mixed-sign / ~0 betas) → dollar-neutral
            return equal
        a = gross / (kl + ks * sum_lb / sum_sb)
        b = a * sum_lb / sum_sb
        return {**dict.fromkeys(longs, a), **dict.fromkeys(shorts, b)}

    def _open_leg(self, sym, side, notional, bar, row, ts, sym_in) -> _Leg | None:
        ref_price = float(bar["open"])
        if ref_price <= 0 or notional <= 0:
            return None
        spread_bps = float(sym_in.spread_bps_at(ts))
        if spread_bps > self.cfg.execution.max_spread_bps:
            return None  # toxic spread — skip this leg
        qty = notional / ref_price
        if self.maker:
            # A basket rebalance is low-urgency → post a passive limit: fill at the reference, no
            # slippage, maker fee. Realistic basket execution; cuts the turnover cost.
            entry_price = ref_price
            entry_fee = self.fees.fee(sym, qty * ref_price, maker=True)
            slip_cost = 0.0
        else:
            bar_notional = float(bar.get("volume", 0.0) or 0.0) * ref_price
            slip = self.slippage.slippage_frac(
                spread_bps=spread_bps, notional=notional, bar_notional=bar_notional or notional
            )
            entry_price = self.slippage.fill_price(ref_price, BUY if side > 0 else SELL, slip)
            entry_fee = self.fees.fee(sym, qty * entry_price, maker=False)
            slip_cost = abs(entry_price - ref_price) * qty
        real_notional = qty * entry_price
        return _Leg(
            symbol=sym,
            side=side,
            qty=qty,
            entry_ts=ts,
            entry_price=entry_price,
            notional=real_notional,
            risk_amount=real_notional * self.stop_frac,
            entry_fee=entry_fee,
            funding=0.0,
            slippage_cost=slip_cost,
            regime=_regime_of(row, spread_bps),
            session=int(row.get("session_code", 0)),
            next_funding_idx=self._first_funding_after(sym_in, ts),
        )

    def _close_leg(self, leg: _Leg, bar: dict, reason: str, result: BacktestResult) -> float:
        ref_price = float(bar["close"])
        exit_ts = int(bar["ts"])
        if self.maker and reason != "end_of_data":
            exit_price = ref_price  # passive maker exit on a planned rebalance
            exit_fee = self.fees.fee(leg.symbol, leg.qty * ref_price, maker=True)
        else:
            bar_notional = float(bar.get("volume", 0.0) or 0.0) * ref_price
            slip = self.slippage.slippage_frac(
                spread_bps=2.0, notional=leg.notional, bar_notional=bar_notional or leg.notional
            )
            exit_price = self.slippage.fill_price(ref_price, SELL if leg.side > 0 else BUY, slip)
            exit_fee = self.fees.fee(leg.symbol, leg.qty * exit_price, maker=False)
        gross = leg.side * (exit_price - leg.entry_price) * leg.qty
        total_fee = leg.entry_fee + exit_fee
        pnl = gross - total_fee - leg.funding  # funding>0 = paid by the leg; carry is its negative
        pnl_r = pnl / leg.risk_amount if leg.risk_amount > 0 else 0.0
        result.trades.append(
            Trade(
                symbol=leg.symbol,
                strategy=self.name,
                side=leg.side,
                qty=leg.qty,
                entry_ts=leg.entry_ts,
                entry_price=leg.entry_price,
                exit_ts=exit_ts,
                exit_price=exit_price,
                exit_reason=reason,
                notional=leg.notional,
                risk_amount=leg.risk_amount,
                fee=total_fee,
                funding=leg.funding,
                slippage_cost=leg.slippage_cost + abs(exit_price - ref_price) * leg.qty,
                pnl=pnl,
                pnl_r=pnl_r,
                regime=leg.regime,
                session=leg.session,
                bars_held=int((exit_ts - leg.entry_ts) // self._grid_iv),
            )
        )
        return pnl

    # -- helpers (mirror the per-trade engine's funding/grid math) ------- #
    def _charge_funding(self, leg: _Leg, sym_in: SymbolInput, ts: int) -> int:
        events = sym_in.funding_events
        idx = leg.next_funding_idx
        while idx < len(events) and events[idx]["ts"] <= ts:
            ev = events[idx]
            if ev["ts"] > leg.entry_ts:
                leg.funding += self.funding.payment(
                    side=leg.side, notional=leg.notional, funding_rate=float(ev["funding_rate"])
                )
            idx += 1
        return idx

    def _first_funding_after(self, sym_in: SymbolInput, ts: int) -> int:
        for i, ev in enumerate(sym_in.funding_events):
            if ev["ts"] > ts:
                return i
        return len(sym_in.funding_events)

    def _unrealized(self, holdings: dict[str, _Leg], bars_by_ts, ts: int) -> float:
        total = 0.0
        for leg in holdings.values():
            bar = bars_by_ts[leg.symbol].get(ts)
            if bar is None:
                continue
            mark = float(bar["close"])
            total += leg.side * (mark - leg.entry_price) * leg.qty - leg.entry_fee - leg.funding
        return total

    def _grid_interval(self, inputs: list[SymbolInput]) -> int:
        best: int | None = None
        for s in inputs:
            for a, b in zip(s.bars, s.bars[1:], strict=False):
                d = int(b["ts"] - a["ts"])
                if d > 0 and (best is None or d < best):
                    best = d
        return best or 1

    # -- beta (computed internally from bars; no feature column / no rebuild) -- #
    def _prepare_returns(self, inputs: list[SymbolInput]) -> None:
        """Per-symbol ordered (ts, bar-return) series + the equal-weight universe return per ts (the
        market factor for beta). Built once per run from the bars already in hand."""
        self._sym_rets: dict[str, tuple[list[int], list[float]]] = {}
        ret_at_ts: dict[int, list[float]] = {}
        for s in inputs:
            ts_list: list[int] = []
            ret_list: list[float] = []
            prev: float | None = None
            for b in s.bars:
                c = float(b["close"])
                if prev is not None and prev > 0:
                    r = c / prev - 1.0
                    ts_list.append(int(b["ts"]))
                    ret_list.append(r)
                    ret_at_ts.setdefault(int(b["ts"]), []).append(r)
                prev = c
            self._sym_rets[s.symbol] = (ts_list, ret_list)
        self._mkt_ret = {t: sum(v) / len(v) for t, v in ret_at_ts.items()}

    def _beta(self, symbol: str, ts: int) -> float:
        """Rolling beta of ``symbol`` to the universe factor over the last ``beta_window`` returns
        STRICTLY before ``ts`` (causal). 1.0 on too little history / degenerate variance."""
        ts_list, ret_list = self._sym_rets.get(symbol, ([], []))
        hi = bisect.bisect_left(ts_list, ts)  # first index at/after ts → exclusive upper bound
        lo = max(0, hi - self.beta_window)
        if hi - lo < 10:
            return 1.0
        rs = ret_list[lo:hi]
        ms = [self._mkt_ret.get(ts_list[i], 0.0) for i in range(lo, hi)]
        n = len(rs)
        mr = sum(rs) / n
        mm = sum(ms) / n
        cov = sum((rs[i] - mr) * (ms[i] - mm) for i in range(n))
        var = sum((ms[i] - mm) ** 2 for i in range(n))
        return cov / var if var > 1e-12 else 1.0
