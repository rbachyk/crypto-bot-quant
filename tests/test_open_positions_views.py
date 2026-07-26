"""Open positions must be visible for REAL environments, not only paper.

A demo or live session persists its held legs exactly as a paper one does (src/live/basket.py
marks them to market every tick), but the only panel that rendered them was hardcoded to paper
sessions. An operator running a demo basket could see realized trades and nothing about the
exposure the account was actually carrying.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from src.api.stats import compute_trading_stats, resolve_window
from src.config import Settings
from src.db.base import session_scope
from src.db.models import OpenPosition

from tests.conftest import requires_db

_SYM = "XLM/USDT:USDT"


@pytest.fixture
def demo_leg():
    """One held leg on a DEMO session, cleaned up afterwards."""
    sid = f"demo:basket:funding_carry:d:1h:{uuid.uuid4().hex[:6]}"
    with session_scope() as s:
        s.add(OpenPosition(
            session_id=sid, strategy="funding_carry", symbol=_SYM, side=1, qty=88.0,
            entry_price=0.31, mark_price=0.33, notional=27.3, unrealized_pnl=1.76,
            funding=-0.12, entry_ts=1, updated_at=datetime.now(UTC),
        ))
    yield sid
    with session_scope() as s:
        s.query(OpenPosition).filter_by(session_id=sid).delete()


@pytest.fixture
def client():
    from src.api.app import create_app

    return TestClient(create_app(Settings(_env_file=None, dashboard_auth_mode="none")))


@requires_db
def test_statistics_page_shows_open_positions_for_the_selected_environment(client, demo_leg):
    for env in ("demo", "real", "all"):
        assert _SYM in client.get(f"/dashboard/stats?env={env}").text, env


@requires_db
def test_a_demo_leg_never_leaks_into_the_paper_view(client, demo_leg):
    """Environment separation is the whole point of the env selector — a real-venue position must
    not appear in, or contribute to, the paper numbers."""
    assert _SYM not in client.get("/dashboard/stats?env=paper").text
    assert _SYM not in client.get("/dashboard/paper").text
    assert compute_trading_stats(resolve_window("all", None, None), env="paper").unrealized_pnl == 0


@requires_db
def test_live_page_shows_held_legs_across_every_real_venue(client, demo_leg):
    """The Live page lists demo, testnet AND live runs, so its positions panel is scoped the same
    way. Keying it to settings.exchange_env would hide a demo session's legs from a deployment
    configured for testnet — on the page built to watch them."""
    assert _SYM in client.get("/dashboard/live").text


@requires_db
def test_unrealized_pnl_is_computed_not_a_hardcoded_zero(demo_leg):
    """The field sat in the stats payload since the dataclass was written, always 0.00."""
    w = resolve_window("all", None, None)
    assert compute_trading_stats(w, env="demo").unrealized_pnl == pytest.approx(1.76)
    assert compute_trading_stats(w, env="real").unrealized_pnl == pytest.approx(1.76)
    assert compute_trading_stats(w, env="paper").unrealized_pnl == pytest.approx(0.0)


@requires_db
def test_the_refresh_fragment_honours_the_environment(client, demo_leg):
    """The panel polls this endpoint every 15s; it must scope the same way the page does, or the
    first refresh would silently swap the contents for another environment's."""
    assert _SYM in client.get("/api/open-positions?env=demo").text
    assert _SYM not in client.get("/api/open-positions?env=paper").text
    assert _SYM not in client.get("/api/open-positions").text  # defaults to paper, as before
