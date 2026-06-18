"""Backtest iteration leaderboard (M3 — research iteration comparison).

Ranks persisted ``backtest_runs`` so real-data iterations can be compared while
hunting for an edge, with NO prior run lost (every distinct snapshot/strategy is
its own immutable row). Ordering mirrors the authoritative profitability bar
(the walk-forward ``KillCriteria``): runs that clear the bar rank first, then by
expectancy (R), profit factor, drawdown and trade count. ``meets_bar`` is a
quick display flag — the BT/WF/FEE/SLIP gates remain the binding judgement before
anything is promoted toward live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, select

from src.backtest.config import BacktestConfig, KillCriteria, load_backtest_config
from src.db.base import session_scope
from src.db.models import BacktestRun


@dataclass(slots=True)
class LeaderboardEntry:
    rank: int
    run_id: str
    kind: str
    created_at: str
    strategy_id: str
    strategy_version: str
    dataset_version: str | None
    timeframe: str
    symbols: list[str]
    trade_count: int
    expectancy_r: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    passed: bool
    meets_bar: bool
    report_path: str | None

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "run_id": self.run_id,
            "kind": self.kind,
            "created_at": self.created_at,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "dataset_version": self.dataset_version,
            "timeframe": self.timeframe,
            "symbols": self.symbols,
            "trade_count": self.trade_count,
            "expectancy_r": self.expectancy_r,
            "profit_factor": self.profit_factor,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "passed": self.passed,
            "meets_bar": self.meets_bar,
            "report_path": self.report_path,
        }


def meets_bar(row: BacktestRun, kc: KillCriteria) -> bool:
    """Whether a run clears the profitability bar (walk-forward kill criteria)."""
    return (
        row.expectancy_r >= kc.min_oos_expectancy_r
        and row.profit_factor >= kc.min_oos_profit_factor
        and row.max_drawdown <= kc.max_oos_drawdown
        and row.trade_count >= kc.min_trades_per_fold
    )


def _sort_key(row: BacktestRun, kc: KillCriteria) -> tuple:
    # Clears-the-bar first, then strongest expectancy / PF, smallest DD, most trades.
    return (
        0 if meets_bar(row, kc) else 1,
        -row.expectancy_r,
        -min(row.profit_factor, 1e9),
        row.max_drawdown,
        -row.trade_count,
    )


def _timeframe_of(row: BacktestRun) -> str:
    return str((row.summary or {}).get("timeframe", ""))


def _iteration_key(row: BacktestRun) -> tuple:
    """Identity of a tuning iteration: same strategy, snapshot, timeframe, symbols."""
    return (
        row.strategy_id,
        row.strategy_version,
        row.dataset_version,
        _timeframe_of(row),
        tuple(row.symbols or []),
    )


def _collapse_best(rows: list[BacktestRun], kc: KillCriteria) -> list[BacktestRun]:
    """Keep only the best run per iteration key (latest re-runs don't flood the board)."""
    best: dict[tuple, BacktestRun] = {}
    for row in rows:
        key = _iteration_key(row)
        cur = best.get(key)
        if cur is None or _sort_key(row, kc) < _sort_key(cur, kc):
            best[key] = row
    return list(best.values())


def _entry(rank: int, row: BacktestRun, kc: KillCriteria) -> LeaderboardEntry:
    created = row.created_at
    return LeaderboardEntry(
        rank=rank,
        run_id=row.run_id,
        kind=row.kind,
        created_at=created.isoformat() if created is not None else "",
        strategy_id=row.strategy_id,
        strategy_version=row.strategy_version,
        dataset_version=row.dataset_version,
        timeframe=_timeframe_of(row),
        symbols=list(row.symbols or []),
        trade_count=row.trade_count,
        expectancy_r=row.expectancy_r,
        profit_factor=min(row.profit_factor, 1e9),
        total_return=row.total_return,
        max_drawdown=row.max_drawdown,
        passed=row.passed,
        meets_bar=meets_bar(row, kc),
        report_path=row.report_path,
    )


def build_leaderboard(
    *,
    kind: str | None = "backtest",
    dataset_version: str | None = None,
    strategy_id: str | None = None,
    limit: int = 50,
    best_per_iteration: bool = True,
    session: Any | None = None,
    cfg: BacktestConfig | None = None,
) -> list[LeaderboardEntry]:
    """Return ranked leaderboard entries for persisted backtest runs.

    ``best_per_iteration`` collapses re-runs of the same (strategy, snapshot,
    timeframe, symbols) to their best result so the board shows distinct
    iterations, not duplicates. Pass ``kind=None`` to rank across all run kinds.
    """
    cfg = cfg or load_backtest_config()
    kc = cfg.walk_forward.kill_criteria

    def _run(s: Any) -> list[LeaderboardEntry]:
        q = select(BacktestRun)
        if kind:
            q = q.where(BacktestRun.kind == kind)
        if dataset_version:
            q = q.where(BacktestRun.dataset_version == dataset_version)
        if strategy_id:
            q = q.where(BacktestRun.strategy_id == strategy_id)
        rows = list(s.execute(q.order_by(desc(BacktestRun.created_at))).scalars())
        if best_per_iteration:
            rows = _collapse_best(rows, kc)
        rows.sort(key=lambda r: _sort_key(r, kc))
        return [_entry(i + 1, row, kc) for i, row in enumerate(rows[:limit])]

    if session is not None:
        return _run(session)
    with session_scope() as scoped:
        return _run(scoped)
