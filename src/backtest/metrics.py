"""Backtest report generator — all required outputs (AGENTS.md Section 19).

Turns a :class:`~src.backtest.engine.BacktestResult` into the full metric set the
spec mandates (Section 19 "Backtest output must include"): total return, net PnL,
expectancy, profit factor, max drawdown, trade count; symbol / strategy / regime /
session / long-short breakdowns; cost / slippage / funding breakdowns; rejected-
candidate breakdown; worst trades; and stability metrics. Every number is a pure
function of the recorded trades + equity curve, so a report is reproducible.

Returns are reported capital-agnostically (fractions and R-multiples); the engine
never claims or implies future profitability (Section 0 / output rules).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from src.backtest.engine import BacktestResult, Trade


def _pf(trades: Iterable[Trade]) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _expectancy_r(trades: list[Trade]) -> float:
    return sum(t.pnl_r for t in trades) / len(trades) if trades else 0.0


def max_drawdown(equity_curve: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction of the peak."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst


def _breakdown(trades: list[Trade], key) -> dict[str, dict]:  # type: ignore[no-untyped-def]
    groups: dict[str, list[Trade]] = {}
    for t in trades:
        groups.setdefault(str(key(t)), []).append(t)
    out: dict[str, dict] = {}
    for name, ts in sorted(groups.items()):
        out[name] = {
            "trades": len(ts),
            "net_pnl": round(sum(t.pnl for t in ts), 6),
            "expectancy_r": round(_expectancy_r(ts), 6),
            "profit_factor": round(_pf(ts), 6),
            "win_rate": round(sum(1 for t in ts if t.pnl > 0) / len(ts), 6),
        }
    return out


def _max_consecutive_losses(trades: list[Trade]) -> int:
    worst = run = 0
    for t in sorted(trades, key=lambda x: x.exit_ts):
        if t.pnl < 0:
            run += 1
            worst = max(worst, run)
        else:
            run = 0
    return worst


def _stability(trades: list[Trade], segments: int = 5) -> dict:
    """Edge stability across equal time-ordered trade segments (Section 16)."""
    if not trades:
        return {"segments": 0, "positive_segments": 0, "segment_expectancy_r": [], "pnl_std": 0.0}
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    n = len(ordered)
    seg = max(1, n // segments)
    seg_exp: list[float] = []
    positive = 0
    for i in range(0, n, seg):
        chunk = ordered[i : i + seg]
        e = _expectancy_r(chunk)
        seg_exp.append(round(e, 6))
        if sum(t.pnl for t in chunk) > 0:
            positive += 1
    pnls = [t.pnl for t in ordered]
    mean = sum(pnls) / n
    pnl_std = math.sqrt(sum((p - mean) ** 2 for p in pnls) / n) if n > 1 else 0.0
    return {
        "segments": len(seg_exp),
        "positive_segments": positive,
        "segment_expectancy_r": seg_exp,
        "pnl_std": round(pnl_std, 6),
        "max_consecutive_losses": _max_consecutive_losses(ordered),
    }


@dataclass(slots=True)
class BacktestReport:
    payload: dict

    def to_dict(self) -> dict:
        return self.payload

    # Convenience accessors used by gates / walk-forward / stress.
    @property
    def trade_count(self) -> int:
        return int(self.payload["trade_count"])

    @property
    def expectancy_r(self) -> float:
        return float(self.payload["expectancy_r"])

    @property
    def net_pnl(self) -> float:
        return float(self.payload["net_pnl"])

    @property
    def profit_factor(self) -> float:
        return float(self.payload["profit_factor"])

    @property
    def total_return(self) -> float:
        return float(self.payload["total_return"])

    @property
    def max_drawdown(self) -> float:
        return float(self.payload["max_drawdown"])


def build_report(result: BacktestResult, *, label: str = "") -> BacktestReport:
    trades = result.trades
    initial = result.initial_equity
    net_pnl = sum(t.pnl for t in trades)
    total_return = net_pnl / initial if initial > 0 else 0.0
    longs = [t for t in trades if t.side > 0]
    shorts = [t for t in trades if t.side < 0]

    payload = {
        "label": label,
        "initial_equity": round(initial, 6),
        "final_equity": round(result.final_equity, 6),
        "total_return": round(total_return, 8),
        "net_pnl": round(net_pnl, 6),
        "expectancy_r": round(_expectancy_r(trades), 6),
        "profit_factor": round(_pf(trades), 6),
        "max_drawdown": round(max_drawdown(result.equity_curve), 6),
        "trade_count": len(trades),
        "win_rate": round(sum(1 for t in trades if t.pnl > 0) / len(trades), 6) if trades else 0.0,
        "symbols": list(result.symbols),
        # Required breakdowns (Section 19).
        "symbol_breakdown": _breakdown(trades, lambda t: t.symbol),
        "strategy_breakdown": _breakdown(trades, lambda t: t.strategy),
        "regime_breakdown": _breakdown(trades, lambda t: t.regime),
        "session_breakdown": _breakdown(trades, lambda t: t.session),
        "side_breakdown": {
            "long": {
                "trades": len(longs),
                "expectancy_r": round(_expectancy_r(longs), 6),
                "net_pnl": round(sum(t.pnl for t in longs), 6),
                "profit_factor": round(_pf(longs), 6),
            },
            "short": {
                "trades": len(shorts),
                "expectancy_r": round(_expectancy_r(shorts), 6),
                "net_pnl": round(sum(t.pnl for t in shorts), 6),
                "profit_factor": round(_pf(shorts), 6),
            },
        },
        "cost_breakdown": {
            "total_fees": round(sum(t.fee for t in trades), 6),
            "total_funding": round(sum(t.funding for t in trades), 6),
            "total_slippage": round(sum(t.slippage_cost for t in trades), 6),
            "gross_pnl": round(net_pnl + sum(t.fee + t.funding for t in trades), 6),
        },
        "slippage_breakdown": _breakdown(trades, lambda t: t.symbol) if trades else {},
        "funding_breakdown": {
            "total": round(sum(t.funding for t in trades), 6),
            "by_symbol": {
                sym: round(sum(t.funding for t in trades if t.symbol == sym), 6)
                for sym in result.symbols
            },
        },
        "rejected_candidates": {
            "total": len(result.rejected),
            "by_reason": dict(sorted(result.rejected_by_reason.items())),
        },
        "exit_reason_breakdown": _count_by(trades, lambda t: t.exit_reason),
        "worst_trades": [t.to_dict() for t in sorted(trades, key=lambda x: x.pnl)[:10]],
        "stability": _stability(trades),
    }
    return BacktestReport(payload)


def _count_by(trades: list[Trade], key) -> dict[str, int]:  # type: ignore[no-untyped-def]
    out: dict[str, int] = {}
    for t in trades:
        k = str(key(t))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))
