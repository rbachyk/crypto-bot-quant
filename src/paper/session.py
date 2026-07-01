"""Paper trading session records (AGENTS.md Section 26).

A :class:`PaperSession` accumulates all candidates evaluated, executed trades,
rejected candidates, and per-fill execution quality in one simulated paper run.
It is the single source of truth that both PAPER-A (technical check) and
PAPER-B (strategy check) gate checks read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

# Notional equity every PAPER account is seeded at (per-symbol engine + basket loop) and that the
# dashboard reconstructs its equity curve on. ONE source so all paper P&L / % returns share a base
# (the backtest's larger account.initial_equity is a size-invariant numeraire, used only there).
PAPER_BASE_EQUITY = 10_000.0


@dataclass(slots=True)
class PaperTrade:
    """One executed paper trade with all Section 18 execution-quality fields."""

    trade_id: str
    symbol: str
    strategy: str
    side: int  # +1 long / -1 short
    qty: float
    entry_price: float
    stop_price: float
    tp_price: float
    regime: str
    session: int
    decision_ts: int
    entry_ts: int
    exit_ts: int
    exit_price: float
    exit_reason: str  # "stop"|"trailing_stop"|"take_profit"|"time_stop"|"kill_switch"|"open"
    fee: float
    slippage_cost: float
    pnl: float
    pnl_r: float
    has_exchange_side_stop: bool
    execution_route: str  # "maker" | "taker"
    spread_bps_at_entry: float
    slippage_frac: float
    # Funding booked into pnl, COST convention (>0 = paid, <0 = carry received). Already in pnl;
    # carried separately so it survives persistence (basket legs accrue it; per-trade engine: 0).
    funding: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "tp_price": self.tp_price,
            "regime": self.regime,
            "session": self.session,
            "decision_ts": self.decision_ts,
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "fee": self.fee,
            "slippage_cost": self.slippage_cost,
            "pnl": self.pnl,
            "pnl_r": self.pnl_r,
            "has_exchange_side_stop": self.has_exchange_side_stop,
            "execution_route": self.execution_route,
            "spread_bps_at_entry": self.spread_bps_at_entry,
            "slippage_frac": self.slippage_frac,
            "funding": self.funding,
        }


@dataclass(slots=True)
class RejectedPaperCandidate:
    """A candidate evaluated but not executed, with the rejection reason."""

    symbol: str
    strategy: str
    side: int
    regime: str
    decision_ts: int
    reason: str  # "risk_rejected" | "ranking_blocked" | "kill_switch" | "exec_revalidate" | ...

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "side": self.side,
            "regime": self.regime,
            "decision_ts": self.decision_ts,
            "reason": self.reason,
        }


@dataclass(slots=True)
class PaperDecisionLog:
    """Per-signal decision record (AGENTS.md Section 24 decision_log).

    Written asynchronously (never blocks execution), stores all fields required
    by the Trade Explainability Schema (Section 24).
    """

    entry_ts: datetime
    symbol: str
    strategy: str
    regime: str
    side: int
    action: str  # "execute" | "reject" | "block"
    reason: str
    risk_approved: bool
    expected_edge: float
    expected_fee: float
    expected_slippage: float
    config_version: str
    universe_version: str
    strategy_version: str
    kill_switch_state: str  # "engaged" | "clear"

    def to_dict(self) -> dict:
        return {
            "entry_ts": self.entry_ts.isoformat(),
            "symbol": self.symbol,
            "strategy": self.strategy,
            "regime": self.regime,
            "side": self.side,
            "action": self.action,
            "reason": self.reason,
            "risk_approved": self.risk_approved,
            "expected_edge": self.expected_edge,
            "expected_fee": self.expected_fee,
            "expected_slippage": self.expected_slippage,
            "config_version": self.config_version,
            "universe_version": self.universe_version,
            "strategy_version": self.strategy_version,
            "kill_switch_state": self.kill_switch_state,
        }


@dataclass
class PaperSession:
    """Accumulates the full paper trading session state (AGENTS.md Section 26).

    Tracks candidates evaluated, executed trades, rejected candidates, and
    kill-switch events. Both PAPER-A and PAPER-B gate checks consume this.
    """

    session_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None

    trades: list[PaperTrade] = field(default_factory=list)
    rejected: list[RejectedPaperCandidate] = field(default_factory=list)
    decision_logs: list[PaperDecisionLog] = field(default_factory=list)
    # Per-trade TradeExplainability records (Section 24); typed loosely to avoid an import cycle.
    explainability: list = field(default_factory=list)

    # Kill-switch events (ts, reason).
    kill_switch_events: list[dict] = field(default_factory=list)

    # Reconciliation events (ts, result, halt_triggered).
    reconciliation_events: list[dict] = field(default_factory=list)

    # True if the kill switch was exercised at least once during this session.
    kill_switch_exercised: bool = False
    # True if reconciliation detected a foreign order and triggered halt.
    foreign_order_halt_triggered: bool = False

    @property
    def total_candidates(self) -> int:
        return len(self.trades) + len(self.rejected)

    @property
    def decision_log_count(self) -> int:
        return len(self.decision_logs)

    @property
    def executed_count(self) -> int:
        return len(self.trades)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def symbols(self) -> list[str]:
        return sorted({t.symbol for t in self.trades})

    @property
    def regimes(self) -> list[str]:
        return sorted({t.regime for t in self.trades})

    def symbol_breakdown(self) -> dict[str, dict]:
        """Per-symbol trade summary (PAPER-B requirement)."""
        groups: dict[str, list[PaperTrade]] = {}
        for t in self.trades:
            groups.setdefault(t.symbol, []).append(t)
        out: dict[str, dict] = {}
        for sym, ts in sorted(groups.items()):
            wins = [t for t in ts if t.pnl > 0]
            out[sym] = {
                "trades": len(ts),
                "wins": len(wins),
                "win_rate": round(len(wins) / len(ts), 4) if ts else 0.0,
                "net_pnl": round(sum(t.pnl for t in ts), 6),
                "expectancy_r": round(sum(t.pnl_r for t in ts) / len(ts), 4) if ts else 0.0,
                "total_fee": round(sum(t.fee for t in ts), 6),
            }
        return out

    def regime_breakdown(self) -> dict[str, dict]:
        """Per-regime trade summary (PAPER-B requirement)."""
        groups: dict[str, list[PaperTrade]] = {}
        for t in self.trades:
            groups.setdefault(t.regime, []).append(t)
        out: dict[str, dict] = {}
        for reg, ts in sorted(groups.items()):
            wins = [t for t in ts if t.pnl > 0]
            out[reg] = {
                "trades": len(ts),
                "wins": len(wins),
                "win_rate": round(len(wins) / len(ts), 4) if ts else 0.0,
                "net_pnl": round(sum(t.pnl for t in ts), 6),
                "expectancy_r": round(sum(t.pnl_r for t in ts) / len(ts), 4) if ts else 0.0,
            }
        return out

    def strategy_breakdown(self) -> dict[str, dict]:
        """Per-strategy trade summary."""
        groups: dict[str, list[PaperTrade]] = {}
        for t in self.trades:
            groups.setdefault(t.strategy, []).append(t)
        out: dict[str, dict] = {}
        for strat, ts in sorted(groups.items()):
            wins = [t for t in ts if t.pnl > 0]
            out[strat] = {
                "trades": len(ts),
                "wins": len(wins),
                "win_rate": round(len(wins) / len(ts), 4) if ts else 0.0,
                "net_pnl": round(sum(t.pnl for t in ts), 6),
                "expectancy_r": round(sum(t.pnl_r for t in ts) / len(ts), 4) if ts else 0.0,
            }
        return out

    def rejection_breakdown(self) -> dict[str, int]:
        """Count rejections by reason (PAPER-B rejected-candidate analysis)."""
        counts: dict[str, int] = {}
        for r in self.rejected:
            counts[r.reason] = counts.get(r.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_candidates": self.total_candidates,
            "executed_count": self.executed_count,
            "rejected_count": self.rejected_count,
            "kill_switch_exercised": self.kill_switch_exercised,
            "foreign_order_halt_triggered": self.foreign_order_halt_triggered,
            "symbols": self.symbols,
            "regimes": self.regimes,
            "symbol_breakdown": self.symbol_breakdown(),
            "regime_breakdown": self.regime_breakdown(),
            "strategy_breakdown": self.strategy_breakdown(),
            "rejection_breakdown": self.rejection_breakdown(),
            "decision_logs": [d.to_dict() for d in self.decision_logs],
            "trades": [t.to_dict() for t in self.trades],
            "rejected": [r.to_dict() for r in self.rejected],
            "kill_switch_events": self.kill_switch_events,
            "reconciliation_events": self.reconciliation_events,
        }
