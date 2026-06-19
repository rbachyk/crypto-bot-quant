"""Section 25: entity-scoped stats filters (strategy / paper-or-live session)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api import create_app
from src.api.stats import compute_trading_stats, get_trade_scopes, resolve_window
from src.config import Settings
from src.db.base import session_scope
from src.db.models import PaperTradeRecord

_SETTINGS = Settings(
    _env_file=None,
    app_env="paper",
    dashboard_auth_mode="basic",
    dashboard_username="admin",
    dashboard_password="secret",
)
_AUTH = ("admin", "secret")
_A = "ef_session_A"
_B = "ef_session_B"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(_SETTINGS))


@pytest.fixture(autouse=True)
def _seed():
    with session_scope() as s:
        s.query(PaperTradeRecord).filter(PaperTradeRecord.session_id.in_([_A, _B])).delete(
            synchronize_session=False
        )
        for i in range(3):
            s.add(
                PaperTradeRecord(
                    session_id=_A,
                    trade_id=f"a{i}",
                    symbol="BTC/USDT:USDT",
                    strategy="ef_strat",
                    side=1,
                    pnl=10.0,
                    pnl_r=1.0,
                )
            )
        s.add(
            PaperTradeRecord(
                session_id=_B,
                trade_id="b0",
                symbol="ETH/USDT:USDT",
                strategy="ef_strat",
                side=1,
                pnl=-5.0,
                pnl_r=-0.5,
            )
        )
    yield
    with session_scope() as s:
        s.query(PaperTradeRecord).filter(PaperTradeRecord.session_id.in_([_A, _B])).delete(
            synchronize_session=False
        )


def test_session_scope_filters_trades() -> None:
    w = resolve_window("all", None, None)
    assert compute_trading_stats(w, session_id=_A).total_trades == 3
    assert compute_trading_stats(w, session_id=_B).total_trades == 1
    assert compute_trading_stats(w, session_id=_A).realized_pnl == 30.0


def test_trade_scopes_lists_strategies_and_sessions() -> None:
    scopes = get_trade_scopes()
    assert "ef_strat" in scopes["strategies"]
    assert _A in scopes["sessions"] and _B in scopes["sessions"]


def test_stats_api_accepts_entity_filters(client: TestClient) -> None:
    resp = client.get(f"/api/stats?session={_A}", auth=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["trading"]["total_trades"] == 3
    scopes = client.get("/api/stats/scopes", auth=_AUTH)
    assert scopes.status_code == 200 and "ef_strat" in scopes.json()["strategies"]


def test_overview_renders_scope_selector(client: TestClient) -> None:
    text = client.get("/", auth=_AUTH).text
    assert "All strategies" in text and "All sessions" in text
    assert 'name="session"' in text and 'name="strategy"' in text
