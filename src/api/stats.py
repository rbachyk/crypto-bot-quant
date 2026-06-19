"""Dashboard statistics with time-period filtering (Phase 7, Section 25).

Provides aggregate and per-symbol statistics for the dashboard. Trading-level
metrics (PnL, win rate, expectancy, profit factor, drawdown, fees, slippage and
breakdowns by strategy / regime / session / symbol) are computed from the
real ``paper_trades`` produced by paper sessions — the data that powers the
performance ("TradeZella-style") dashboard.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from src.db.base import session_scope
from src.db.models import (
    GateResult,
    GateStatus,
    Job,
    JobStatus,
    PaperTradeRecord,
    RemediationAction,
    RemediationStatus,
    UniverseMember,
    UniverseVersion,
)

# Paper sessions seed each account at this notional equity (src/paper/run.py),
# so drawdown is expressed as a fraction of this base.
_PAPER_BASE_EQUITY = 10_000.0


def _session_bucket(hour: int) -> str:
    """Coarse trading session for the hour-of-day (UTC) — matches the feature pipeline."""
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 16:
        return "europe"
    return "us"


# Trading environments, separated by the ``env:`` prefix on a session_id (see
# src.live.loop.LiveLoop.env_label). "paper" = everything that is NOT a real-venue run.
ENVIRONMENTS = ("paper", "demo", "testnet", "live")
_REAL_ENV_PREFIXES = ("demo:", "testnet:", "live:")


def _apply_env(query, env: str | None):
    """Scope a paper_trades query to one trading environment by session_id prefix."""
    if not env or env == "all":
        return query
    if env == "paper":
        # Paper = NOT a demo/testnet/live session.
        for pfx in _REAL_ENV_PREFIXES:
            query = query.where(~PaperTradeRecord.session_id.like(f"{pfx}%"))
        return query
    return query.where(PaperTradeRecord.session_id.like(f"{env}:%"))


class TimePeriod(str, enum.Enum):
    TODAY = "today"
    YESTERDAY = "yesterday"
    LAST_7D = "last_7d"
    LAST_30D = "last_30d"
    CURRENT_MONTH = "current_month"
    PREV_MONTH = "prev_month"
    CUSTOM = "custom"
    ALL = "all"


@dataclass(slots=True)
class TimeWindow:
    start: datetime | None
    end: datetime | None


def resolve_window(period: str, from_ts: str | None, to_ts: str | None) -> TimeWindow:
    """Convert a period string + optional custom bounds to a UTC time window."""
    now = datetime.now(UTC)

    if period == TimePeriod.TODAY:
        start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        return TimeWindow(start, now)

    if period == TimePeriod.YESTERDAY:
        today_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        yesterday_start = today_start - timedelta(days=1)
        return TimeWindow(yesterday_start, today_start)

    if period == TimePeriod.LAST_7D:
        return TimeWindow(now - timedelta(days=7), now)

    if period == TimePeriod.LAST_30D:
        return TimeWindow(now - timedelta(days=30), now)

    if period == TimePeriod.CURRENT_MONTH:
        start = datetime(now.year, now.month, 1, tzinfo=UTC)
        return TimeWindow(start, now)

    if period == TimePeriod.PREV_MONTH:
        first_of_this_month = date(now.year, now.month, 1)
        last_of_prev = first_of_this_month - timedelta(days=1)
        start = datetime(last_of_prev.year, last_of_prev.month, 1, tzinfo=UTC)
        end = datetime(first_of_this_month.year, first_of_this_month.month, 1, tzinfo=UTC)
        return TimeWindow(start, end)

    if period == TimePeriod.CUSTOM:
        # URL-decode: '+' may arrive as ' ' when passed as a query parameter.
        def _parse_ts(ts: str | None) -> datetime | None:
            if not ts:
                return None
            return datetime.fromisoformat(ts.replace(" ", "+"))

        return TimeWindow(_parse_ts(from_ts), _parse_ts(to_ts))

    # ALL — no filter
    return TimeWindow(None, None)


@dataclass
class GateStats:
    passed: int = 0
    failed: int = 0
    blocked: int = 0
    not_run: int = 0
    total: int = 0
    live_readiness_score: float = 0.0
    critical_gates_passed: int = 0
    total_critical_gates: int = 0
    next_critical_action: str = ""


@dataclass
class JobStats:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    running: int = 0
    queued: int = 0
    cancelled: int = 0


@dataclass
class UniverseStats:
    active_symbols: int = 0
    total_symbols: int = 0
    universe_version: str = ""


@dataclass
class TradingStats:
    """Realized trading performance computed from ``paper_trades`` over a window."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees_paid: float = 0.0
    total_slippage: float = 0.0
    total_funding_paid: float = 0.0
    max_drawdown_pct: float = 0.0
    current_equity: float = 0.0
    symbols_traded: list[str] = field(default_factory=list)
    # Richer performance fields for the TradeZella-style views.
    avg_win: float = 0.0
    avg_loss: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    # Per-trade series for interactive charts: (epoch_ms, pnl). The client computes the running
    # equity + day/week/month/year buckets from this, so the same data powers every grouping.
    trade_series: list[tuple[int, float]] = field(default_factory=list)
    by_strategy: list[dict[str, Any]] = field(default_factory=list)
    by_regime: list[dict[str, Any]] = field(default_factory=list)
    by_session: list[dict[str, Any]] = field(default_factory=list)
    by_symbol: list[dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """The richer fields, for the performance API/page."""
        return {
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "gross_win": self.gross_win,
            "gross_loss": self.gross_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "equity_curve": self.equity_curve,
            "trade_series": self.trade_series,
            "by_strategy": self.by_strategy,
            "by_regime": self.by_regime,
            "by_session": self.by_session,
            "by_symbol": self.by_symbol,
        }


def _breakdown(rows: list[Any], key) -> list[dict[str, Any]]:
    """Group trade rows by ``key(row)`` → per-group performance, sorted by pnl desc."""
    groups: dict[str, list[Any]] = {}
    for r in rows:
        groups.setdefault(str(key(r)) or "—", []).append(r)
    out: list[dict[str, Any]] = []
    for name, items in groups.items():
        pnl = sum(t.pnl for t in items)
        wins = sum(1 for t in items if t.pnl > 0)
        out.append(
            {
                "group": name,
                "trades": len(items),
                "pnl": round(pnl, 2),
                "win_rate": round(wins / len(items), 4) if items else 0.0,
                "expectancy_r": round(sum(t.pnl_r for t in items) / len(items), 4)
                if items
                else 0.0,
            }
        )
    return sorted(out, key=lambda d: d["pnl"], reverse=True)


def compute_trading_stats(
    window: TimeWindow,
    *,
    env: str | None = None,
    symbol: str | None = None,
    strategy: str | None = None,
    session_id: str | None = None,
) -> TradingStats:
    """Realized performance from ``paper_trades`` in ``window`` (optionally entity-scoped).

    Scopes (Section 25 entity filters): ``env`` (paper / demo / testnet / live — by session_id
    prefix, so each environment's statistics stay SEPARATED), ``symbol``, ``strategy``, and
    ``session_id`` (one specific run — covers "by paper/live session")."""
    st = TradingStats()
    with session_scope() as session:
        q = select(PaperTradeRecord)
        if window.start:
            q = q.where(PaperTradeRecord.created_at >= window.start)
        if window.end:
            q = q.where(PaperTradeRecord.created_at <= window.end)
        q = _apply_env(q, env)
        if symbol:
            q = q.where(PaperTradeRecord.symbol == symbol)
        if strategy:
            q = q.where(PaperTradeRecord.strategy == strategy)
        if session_id:
            q = q.where(PaperTradeRecord.session_id == session_id)
        rows = list(session.execute(q.order_by(PaperTradeRecord.created_at)).scalars().all())

    st.total_trades = len(rows)
    if not rows:
        return st
    wins = [t for t in rows if t.pnl > 0]
    losses = [t for t in rows if t.pnl < 0]
    st.winning_trades = len(wins)
    st.losing_trades = len(losses)
    st.win_rate = round(len(wins) / len(rows), 4)
    st.expectancy_r = round(sum(t.pnl_r for t in rows) / len(rows), 4)
    st.realized_pnl = round(sum(t.pnl for t in rows), 2)
    st.total_fees_paid = round(sum(t.fee for t in rows), 2)
    st.total_slippage = round(sum(t.slippage_cost for t in rows), 2)
    st.gross_win = round(sum(t.pnl for t in wins), 2)
    st.gross_loss = round(sum(t.pnl for t in losses), 2)  # negative
    st.profit_factor = round(st.gross_win / abs(st.gross_loss), 4) if st.gross_loss else 0.0
    st.avg_win = round(st.gross_win / len(wins), 2) if wins else 0.0
    st.avg_loss = round(st.gross_loss / len(losses), 2) if losses else 0.0
    st.largest_win = round(max((t.pnl for t in rows), default=0.0), 2)
    st.largest_loss = round(min((t.pnl for t in rows), default=0.0), 2)
    st.symbols_traded = sorted({t.symbol for t in rows})

    # Equity curve + max drawdown (fraction of running peak).
    equity = _PAPER_BASE_EQUITY
    peak = equity
    max_dd = 0.0
    curve = [round(equity, 2)]
    for t in rows:
        equity += t.pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        curve.append(round(equity, 2))
    st.current_equity = round(equity, 2)
    st.max_drawdown_pct = round(max_dd, 4)
    st.equity_curve = curve
    st.trade_series = [
        (int(t.created_at.timestamp() * 1000), round(t.pnl, 4)) for t in rows
    ]

    st.by_strategy = _breakdown(rows, lambda r: r.strategy)
    st.by_regime = _breakdown(rows, lambda r: r.regime)
    st.by_symbol = _breakdown(rows, lambda r: r.symbol)
    st.by_session = _breakdown(rows, lambda r: _session_bucket(r.created_at.hour))
    return st


@dataclass
class AggregateStats:
    period: str
    window_start: str | None
    window_end: str | None
    gates: GateStats = field(default_factory=GateStats)
    jobs: JobStats = field(default_factory=JobStats)
    universe: UniverseStats = field(default_factory=UniverseStats)
    trading: TradingStats = field(default_factory=TradingStats)
    open_remediation_items: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "gates": {
                "passed": self.gates.passed,
                "failed": self.gates.failed,
                "blocked": self.gates.blocked,
                "not_run": self.gates.not_run,
                "total": self.gates.total,
                "live_readiness_score": round(self.gates.live_readiness_score, 1),
                "critical_gates_passed": self.gates.critical_gates_passed,
                "total_critical_gates": self.gates.total_critical_gates,
                "next_critical_action": self.gates.next_critical_action,
            },
            "jobs": {
                "total": self.jobs.total,
                "succeeded": self.jobs.succeeded,
                "failed": self.jobs.failed,
                "running": self.jobs.running,
                "queued": self.jobs.queued,
                "cancelled": self.jobs.cancelled,
            },
            "universe": {
                "active_symbols": self.universe.active_symbols,
                "total_symbols": self.universe.total_symbols,
                "universe_version": self.universe.universe_version,
            },
            "trading": {
                "total_trades": self.trading.total_trades,
                "winning_trades": self.trading.winning_trades,
                "losing_trades": self.trading.losing_trades,
                "win_rate": self.trading.win_rate,
                "expectancy_r": self.trading.expectancy_r,
                "profit_factor": self.trading.profit_factor,
                "realized_pnl": self.trading.realized_pnl,
                "unrealized_pnl": self.trading.unrealized_pnl,
                "total_fees_paid": self.trading.total_fees_paid,
                "total_slippage": self.trading.total_slippage,
                "total_funding_paid": self.trading.total_funding_paid,
                "max_drawdown_pct": self.trading.max_drawdown_pct,
                "current_equity": self.trading.current_equity,
                "symbols_traded": self.trading.symbols_traded,
            },
            "open_remediation_items": self.open_remediation_items,
        }


def compute_gate_stats(window: TimeWindow) -> GateStats:
    """Compute gate pass/fail/blocked/not_run counts from latest results."""
    stats = GateStats()
    from src.gates.catalog import load_catalog

    catalog = load_catalog()

    with session_scope() as session:
        # For each gate, find its latest result.
        latest_by_gate: dict[str, GateStatus] = {}
        q = select(GateResult).order_by(GateResult.gate_id, GateResult.id.desc())
        if window.start:
            q = q.where(GateResult.started_at >= window.start)
        if window.end:
            q = q.where(GateResult.started_at <= window.end)
        rows = session.execute(q).scalars().all()
        for row in rows:
            if row.gate_id not in latest_by_gate:
                latest_by_gate[row.gate_id] = row.status

        for gate_id in catalog:
            status = latest_by_gate.get(gate_id, GateStatus.NOT_RUN)
            if status is GateStatus.PASSED:
                stats.passed += 1
            elif status is GateStatus.FAILED:
                stats.failed += 1
            elif status is GateStatus.BLOCKED:
                stats.blocked += 1
            else:
                stats.not_run += 1
            stats.total += 1

        # Live readiness: critical gates that are PASSED.
        critical_ids = [g for g, spec in catalog.items() if spec.blocks_live == "true"]
        stats.total_critical_gates = len(critical_ids)
        critical_passed = sum(
            1
            for gid in critical_ids
            if latest_by_gate.get(gid, GateStatus.NOT_RUN) is GateStatus.PASSED
        )
        stats.critical_gates_passed = critical_passed
        if stats.total_critical_gates > 0:
            stats.live_readiness_score = (critical_passed / stats.total_critical_gates) * 100.0

        # Next critical action: first non-PASS critical gate's remediation step 0.
        for gid in critical_ids:
            if latest_by_gate.get(gid, GateStatus.NOT_RUN) is not GateStatus.PASSED:
                spec = catalog[gid]
                if spec.remediation_steps:
                    stats.next_critical_action = f"[{gid}] {spec.remediation_steps[0]}"
                else:
                    stats.next_critical_action = f"Re-run gate {gid}"
                break

    return stats


def compute_job_stats(window: TimeWindow) -> JobStats:
    stats = JobStats()
    with session_scope() as session:
        q = select(Job)
        if window.start:
            q = q.where(Job.created_at >= window.start)
        if window.end:
            q = q.where(Job.created_at <= window.end)
        jobs = session.execute(q).scalars().all()
        stats.total = len(jobs)
        for j in jobs:
            if j.status is JobStatus.SUCCEEDED:
                stats.succeeded += 1
            elif j.status is JobStatus.FAILED:
                stats.failed += 1
            elif j.status is JobStatus.RUNNING:
                stats.running += 1
            elif j.status is JobStatus.QUEUED:
                stats.queued += 1
            elif j.status is JobStatus.CANCELLED:
                stats.cancelled += 1
    return stats


def compute_universe_stats() -> UniverseStats:
    stats = UniverseStats()
    with session_scope() as session:
        latest_version = session.execute(
            select(UniverseVersion).order_by(UniverseVersion.created_at.desc()).limit(1)
        ).scalar_one_or_none()
        if latest_version:
            stats.universe_version = latest_version.version
            members = (
                session.execute(
                    select(UniverseMember).where(
                        UniverseMember.universe_version == latest_version.version
                    )
                )
                .scalars()
                .all()
            )
            stats.total_symbols = len(members)
            stats.active_symbols = sum(1 for m in members if m.status.value == "active")
    return stats


def compute_open_remediation_count() -> int:
    with session_scope() as session:
        result = session.execute(
            select(func.count())
            .select_from(RemediationAction)
            .where(RemediationAction.status == RemediationStatus.OPEN)
        ).scalar_one()
    return int(result)


def get_aggregate_stats(
    period: str = "all",
    from_ts: str | None = None,
    to_ts: str | None = None,
    *,
    env: str | None = None,
    symbol: str | None = None,
    strategy: str | None = None,
    session_id: str | None = None,
) -> AggregateStats:
    window = resolve_window(period, from_ts, to_ts)
    return AggregateStats(
        period=period,
        window_start=window.start.isoformat() if window.start else None,
        window_end=window.end.isoformat() if window.end else None,
        gates=compute_gate_stats(window),
        jobs=compute_job_stats(window),
        universe=compute_universe_stats(),
        trading=compute_trading_stats(
            window, env=env, symbol=symbol, strategy=strategy, session_id=session_id
        ),
        open_remediation_items=compute_open_remediation_count(),
    )


def get_environment_summary() -> list[dict[str, Any]]:
    """Per-environment trade counts + net P&L, so the dashboard shows the SEPARATION at a glance
    (paper / demo / testnet / live never mixed unless 'All' is chosen)."""
    out: list[dict[str, Any]] = []
    full = TimeWindow(None, None)
    for env in ENVIRONMENTS:
        t = compute_trading_stats(full, env=env)
        out.append(
            {"env": env, "trades": t.total_trades, "net_pnl": t.realized_pnl,
             "win_rate": t.win_rate}
        )
    return out


def get_traded_symbols(env: str | None = None) -> list[str]:
    """Symbols that actually have trades (env-scoped) — the real per-symbol drill-down list,
    independent of whether a universe version has been built yet."""
    with session_scope() as session:
        q = _apply_env(select(PaperTradeRecord.symbol).distinct(), env)
        return sorted({s for (s,) in session.execute(q) if s})


def get_trade_scopes() -> dict[str, list[str]]:
    """Distinct entity scopes for the dashboard selectors: strategies + sessions (Section 25)."""
    from src.db.models import PaperTradeRecord

    with session_scope() as session:
        strategies = sorted(
            {s for (s,) in session.execute(select(PaperTradeRecord.strategy).distinct()) if s}
        )
        sessions = [
            sid
            for (sid,) in session.execute(
                select(PaperTradeRecord.session_id)
                .distinct()
                .order_by(PaperTradeRecord.session_id.desc())
            )
            if sid
        ][:50]
    return {"strategies": strategies, "sessions": sessions}


def get_per_symbol_stats(
    symbol: str,
    period: str = "all",
    from_ts: str | None = None,
    to_ts: str | None = None,
    *,
    env: str | None = None,
) -> dict[str, Any]:
    """Per-symbol realized trading stats from ``paper_trades`` (optionally env-scoped)."""
    window = resolve_window(period, from_ts, to_ts)
    t = compute_trading_stats(window, env=env, symbol=symbol)
    return {
        "symbol": symbol,
        "period": period,
        "window_start": window.start.isoformat() if window.start else None,
        "window_end": window.end.isoformat() if window.end else None,
        "trading": {
            "total_trades": t.total_trades,
            "winning_trades": t.winning_trades,
            "losing_trades": t.losing_trades,
            "win_rate": t.win_rate,
            "expectancy_r": t.expectancy_r,
            "profit_factor": t.profit_factor,
            "realized_pnl": t.realized_pnl,
            "total_fees_paid": t.total_fees_paid,
            "total_slippage": t.total_slippage,
            "total_funding_paid": t.total_funding_paid,
            "max_drawdown_pct": t.max_drawdown_pct,
            "current_equity": t.current_equity,
        },
        "summary": t.to_summary(),
    }


def get_symbols_list() -> list[str]:
    """Return all symbols known from the latest universe version."""
    with session_scope() as session:
        latest_version = session.execute(
            select(UniverseVersion).order_by(UniverseVersion.created_at.desc()).limit(1)
        ).scalar_one_or_none()
        if not latest_version:
            return []
        members = (
            session.execute(
                select(UniverseMember).where(
                    UniverseMember.universe_version == latest_version.version
                )
            )
            .scalars()
            .all()
        )
        return [m.symbol for m in members]
