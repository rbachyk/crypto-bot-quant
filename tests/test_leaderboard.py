"""M3: backtest iteration leaderboard + dataset_version persistence.

Inserts ``backtest_runs`` rows directly to exercise ranking/collapse/filter logic
deterministically, and round-trips ``persist_backtest_run`` to prove a run records
the DATA_VERSION snapshot it ran over (so iterations are grouped and never lost).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from src.api import create_app
from src.backtest.config import load_backtest_config
from src.backtest.leaderboard import build_leaderboard, meets_bar
from src.backtest.service import persist_backtest_run
from src.config import Settings
from src.db.base import session_scope
from src.db.models import BacktestRun

KC = load_backtest_config().walk_forward.kill_criteria

_SETTINGS = Settings(
    _env_file=None,
    app_env="paper",
    dashboard_auth_mode="basic",
    dashboard_username="admin",
    dashboard_password="secret",
)
_AUTH = ("admin", "secret")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(_SETTINGS))


@pytest.fixture(scope="module", autouse=True)
def _cleanup_backtest_runs():
    """Delete exactly the backtest_runs rows this module adds (shared dev DB hygiene)."""
    with session_scope() as s:
        before = {rid for (rid,) in s.query(BacktestRun.run_id).all()}
    yield
    with session_scope() as s:
        after = {rid for (rid,) in s.query(BacktestRun.run_id).all()}
        new = after - before
        if new:
            s.query(BacktestRun).filter(BacktestRun.run_id.in_(new)).delete(
                synchronize_session=False
            )


def _insert(
    run_id: str,
    *,
    strategy_id: str,
    dataset_version: str,
    expectancy_r: float,
    profit_factor: float,
    max_drawdown: float,
    trade_count: int,
    timeframe: str = "1h",
    kind: str = "backtest",
) -> None:
    with session_scope() as s:
        # Idempotent across re-runs: the dev DB persists between runs and run_id is unique.
        s.query(BacktestRun).filter_by(run_id=run_id).delete()
        s.add(
            BacktestRun(
                run_id=run_id,
                kind=kind,
                strategy_id=strategy_id,
                strategy_version="v1",
                dataset_version=dataset_version,
                symbols=["BTC/USDT:USDT"],
                passed=expectancy_r > 0,
                trade_count=trade_count,
                expectancy_r=expectancy_r,
                profit_factor=profit_factor,
                total_return=expectancy_r * trade_count * 0.01,
                max_drawdown=max_drawdown,
                summary={"timeframe": timeframe},
            )
        )


def test_meets_bar_thresholds() -> None:
    ok = SimpleNamespace(
        expectancy_r=KC.min_oos_expectancy_r,
        profit_factor=KC.min_oos_profit_factor,
        max_drawdown=KC.max_oos_drawdown,
        trade_count=KC.min_trades_per_fold,
    )
    assert meets_bar(ok, KC)  # exactly on the bar passes
    # each dimension independently disqualifies
    assert not meets_bar(SimpleNamespace(**{**ok.__dict__, "expectancy_r": 0.0}), KC)
    assert not meets_bar(SimpleNamespace(**{**ok.__dict__, "profit_factor": 1.0}), KC)
    assert not meets_bar(SimpleNamespace(**{**ok.__dict__, "max_drawdown": 0.99}), KC)
    assert not meets_bar(SimpleNamespace(**{**ok.__dict__, "trade_count": 0}), KC)


def test_leaderboard_ranks_passers_first_then_expectancy() -> None:
    sid = "lb_rank"
    _insert(
        "lb_a",
        strategy_id=sid,
        dataset_version="ds_A",
        expectancy_r=0.05,
        profit_factor=1.3,
        max_drawdown=0.10,
        trade_count=50,
    )
    _insert(
        "lb_b",
        strategy_id=sid,
        dataset_version="ds_B",
        expectancy_r=0.01,
        profit_factor=1.05,
        max_drawdown=0.30,
        trade_count=50,
    )  # misses bar
    _insert(
        "lb_c",
        strategy_id=sid,
        dataset_version="ds_C",
        expectancy_r=0.08,
        profit_factor=1.5,
        max_drawdown=0.08,
        trade_count=60,
    )

    board = build_leaderboard(strategy_id=sid, limit=10)
    assert [e.run_id for e in board] == ["lb_c", "lb_a", "lb_b"]
    assert [e.rank for e in board] == [1, 2, 3]
    assert [e.meets_bar for e in board] == [True, True, False]


def test_best_per_iteration_collapses_reruns() -> None:
    sid = "lb_collapse"
    # Same (strategy, dataset, timeframe, symbols) iteration run twice.
    _insert(
        "lb_w",
        strategy_id=sid,
        dataset_version="ds_X",
        expectancy_r=0.02,
        profit_factor=1.2,
        max_drawdown=0.15,
        trade_count=40,
    )
    _insert(
        "lb_better",
        strategy_id=sid,
        dataset_version="ds_X",
        expectancy_r=0.06,
        profit_factor=1.4,
        max_drawdown=0.12,
        trade_count=45,
    )

    collapsed = build_leaderboard(strategy_id=sid, best_per_iteration=True)
    assert [e.run_id for e in collapsed] == ["lb_better"]  # only the best survives

    every = build_leaderboard(strategy_id=sid, best_per_iteration=False)
    assert {e.run_id for e in every} == {"lb_w", "lb_better"}


def test_zero_trade_runs_rank_below_runs_that_traded() -> None:
    sid = "lb_zero"
    # A no-trade run has expectancy 0; a losing run has negative expectancy. The losing
    # run is more informative for finding an edge, so it must rank ABOVE the no-trade one.
    _insert("lb_notrade", strategy_id=sid, dataset_version="ds_nt", expectancy_r=0.0,
            profit_factor=0.0, max_drawdown=0.0, trade_count=0)
    _insert("lb_losing", strategy_id=sid, dataset_version="ds_loss", expectancy_r=-0.30,
            profit_factor=0.5, max_drawdown=0.20, trade_count=12)
    board = build_leaderboard(strategy_id=sid)
    assert [e.run_id for e in board] == ["lb_losing", "lb_notrade"]


def test_leaderboard_filters_by_dataset_version() -> None:
    sid = "lb_filter"
    _insert(
        "lb_d1",
        strategy_id=sid,
        dataset_version="ds_only",
        expectancy_r=0.04,
        profit_factor=1.25,
        max_drawdown=0.10,
        trade_count=30,
    )
    _insert(
        "lb_d2",
        strategy_id=sid,
        dataset_version="ds_other",
        expectancy_r=0.07,
        profit_factor=1.5,
        max_drawdown=0.09,
        trade_count=30,
    )
    board = build_leaderboard(strategy_id=sid, dataset_version="ds_only")
    assert [e.run_id for e in board] == ["lb_d1"]


def test_persist_records_dataset_version_and_keys_run_id() -> None:
    cfg = load_backtest_config()
    report = SimpleNamespace(
        payload={"label": "persist_test", "win_rate": 0.5},
        trade_count=33,
        expectancy_r=0.05,
        profit_factor=1.4,
        total_return=0.1,
        max_drawdown=0.1,
    )
    rid_a = persist_backtest_run(
        cfg,
        report,
        kind="backtest",
        report_path="/tmp/a.json",
        dataset_version="snap_A",
        symbols=["BTC/USDT:USDT"],
        summary_extra={"timeframe": "1h"},
    )
    rid_b = persist_backtest_run(
        cfg,
        report,
        kind="backtest",
        report_path="/tmp/b.json",
        dataset_version="snap_B",
        symbols=["BTC/USDT:USDT"],
        summary_extra={"timeframe": "1h"},
    )
    # Same report + config but different snapshots ⇒ distinct rows.
    assert rid_a != rid_b
    with session_scope() as s:
        row = s.query(BacktestRun).filter_by(run_id=rid_a).one()
        assert row.dataset_version == "snap_A"
        assert row.symbols == ["BTC/USDT:USDT"]
        assert row.summary.get("timeframe") == "1h"


def test_persist_keys_run_id_by_timeframe() -> None:
    """Two timeframes over the SAME snapshot/strategy/label must not collide."""
    cfg = load_backtest_config()
    report = SimpleNamespace(
        payload={"label": "lake"},
        trade_count=0,
        expectancy_r=0.0,
        profit_factor=0.0,
        total_return=0.0,
        max_drawdown=0.0,
    )
    common = {
        "kind": "backtest",
        "report_path": "/tmp/x.json",
        "dataset_version": "snap_same",
        "symbols": ["BTC/USDT:USDT"],
    }
    rid_1h = persist_backtest_run(
        cfg, report, summary_extra={"label": "lake", "timeframe": "1h"}, **common
    )
    rid_4h = persist_backtest_run(
        cfg, report, summary_extra={"label": "lake", "timeframe": "4h"}, **common
    )
    assert rid_1h != rid_4h


# --------------------------------------------------------------------------- #
# Dashboard surface                                                           #
# --------------------------------------------------------------------------- #
def test_leaderboard_api_returns_ranked_entries(client: TestClient) -> None:
    _insert(
        "lb_api",
        strategy_id="lb_api_sid",
        dataset_version="ds_api",
        expectancy_r=0.06,
        profit_factor=1.4,
        max_drawdown=0.10,
        trade_count=40,
    )
    resp = client.get("/api/backtests/leaderboard?strategy=lb_api_sid", auth=_AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data and data[0]["run_id"] == "lb_api" and data[0]["rank"] == 1
    assert data[0]["meets_bar"] is True


def test_leaderboard_api_requires_auth(client: TestClient) -> None:
    assert client.get("/api/backtests/leaderboard").status_code == 401


def test_leaderboard_dashboard_page_renders(client: TestClient) -> None:
    resp = client.get("/dashboard/leaderboard", auth=_AUTH)
    assert resp.status_code == 200
    assert "Iteration Leaderboard" in resp.text
