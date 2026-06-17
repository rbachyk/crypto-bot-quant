"""Dashboard statistics with time-period filtering (Phase 7, Section 25).

Provides aggregate and per-symbol statistics for the dashboard.
Trading-level metrics (PnL, drawdown) are scaffolded with zero-safe defaults
until Phase 8 paper/live data populates them.
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
    RemediationAction,
    RemediationStatus,
    UniverseMember,
    UniverseVersion,
)


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
    """Phase 7 scaffold — zeroed until paper/live data in Phase 8."""

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
) -> AggregateStats:
    window = resolve_window(period, from_ts, to_ts)
    return AggregateStats(
        period=period,
        window_start=window.start.isoformat() if window.start else None,
        window_end=window.end.isoformat() if window.end else None,
        gates=compute_gate_stats(window),
        jobs=compute_job_stats(window),
        universe=compute_universe_stats(),
        trading=TradingStats(),  # Phase 8
        open_remediation_items=compute_open_remediation_count(),
    )


def get_per_symbol_stats(
    symbol: str,
    period: str = "all",
    from_ts: str | None = None,
    to_ts: str | None = None,
) -> dict[str, Any]:
    """Per-symbol stats scaffold — trading metrics populated in Phase 8."""
    window = resolve_window(period, from_ts, to_ts)
    return {
        "symbol": symbol,
        "period": period,
        "window_start": window.start.isoformat() if window.start else None,
        "window_end": window.end.isoformat() if window.end else None,
        "trading": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "realized_pnl": 0.0,
            "total_fees_paid": 0.0,
            "total_slippage": 0.0,
            "total_funding_paid": 0.0,
            "max_drawdown_pct": 0.0,
        },
        "note": "Phase 7 scaffold: trading data populated in Phase 8.",
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
