"""Dashboard completeness + chrome (AGENTS.md §25 / Appendix B.8 / B.9).

Guards two things that are easy to regress silently:
  * every one of the 23 required dashboard pages is reachable (HTTP 200), and
  * the redesigned left-sidebar shell renders on every page with an active-link highlight.
"""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from src.config import Settings  # noqa: E402

from tests.conftest import requires_db  # noqa: E402

# (required-page label, route) — the 23 pages enumerated in §25 and Appendix B.8.
REQUIRED_PAGES: list[tuple[str, str]] = [
    ("Overview", "/"),
    ("Data Coverage", "/dashboard/data-coverage"),
    ("Universe", "/dashboard/universe"),
    ("Jobs", "/dashboard/jobs"),
    ("Gates", "/dashboard/gates"),
    ("Remediation Actions", "/dashboard/remediation"),
    ("Backtests", "/dashboard/backtests"),
    ("Paper Trading", "/dashboard/paper"),
    ("Live Trading", "/dashboard/live"),
    ("General Statistics", "/dashboard/stats"),
    ("Per-Symbol Statistics", "/dashboard/stats/BTCUSDT"),
    ("Strategy Analytics", "/dashboard/strategy"),
    ("Regime Analytics", "/dashboard/regime"),
    ("Session Analytics", "/dashboard/session-analytics"),
    ("Execution Quality", "/dashboard/execution"),
    ("Risk", "/dashboard/risk"),
    ("ML Shadow", "/dashboard/shadow"),
    ("Online Learning", "/dashboard/learning"),
    ("RL", "/dashboard/rl"),
    ("Reports", "/dashboard/reports"),
    ("Approvals", "/dashboard/approvals"),
    ("System Health", "/dashboard/health"),
    ("Settings", "/dashboard/settings"),
]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    return TestClient(create_app(Settings(_env_file=None, dashboard_auth_mode="none")))


@requires_db
def test_strategy_detail_surfaces_full_validation_evidence(client) -> None:
    """The per-strategy detail view renders every metric captured during validation from the
    persisted summary — per-fold + locked hold-out breakdown, deflated Sharpe, cost stress, and
    the backtest report — so a shelved verdict is inspectable without a re-run."""
    from datetime import UTC, datetime

    from src.db.base import session_scope
    from src.db.models import StrategyPromotion
    from src.strategies.config import load_strategies_config

    ver = load_strategies_config().strategy_version
    cid = "lead_lag_xasset"
    summary = {
        "data_source": "lake",
        "timeframe": "4h",
        "walk_forward": {
            "passed": False,
            "folds_passed": 5,
            "n_folds": 5,
            "overfitting": {"deflated_sharpe": 0.73},
            "folds": [
                {"index": 0, "lo_ts": 1735689600000, "hi_ts": 1738368000000, "passed": True,
                 "failures": [], "trade_count": 120, "expectancy_r": 0.031,
                 "profit_factor": 1.2, "max_drawdown": 0.08, "total_return": 0.05},
            ],
            "holdout": {"lo_ts": 1743465600000, "hi_ts": 1751328000000, "passed": False,
                        "trade_count": 90, "expectancy_r": -0.012, "profit_factor": 0.95,
                        "net_pnl": -14.2, "max_drawdown": 0.06},
        },
        "fee_stress": {
            "baseline_expectancy_r": 0.03, "stressed_expectancy_r": 0.011, "passed": True,
        },
        "slippage_stress": {"baseline_expectancy_r": 0.03, "stressed_expectancy_r": 0.008},
        "report": {
            "trade_count": 1803, "expectancy_r": 0.024, "profit_factor": 1.15,
            "net_pnl": 512.0, "total_return": 0.42, "max_drawdown": 0.11,
            "by_side": {"long": {"expectancy_r": 0.03, "net_pnl": 300.0, "profit_factor": 1.2}},
        },
    }
    with session_scope() as s:
        s.query(StrategyPromotion).filter_by(candidate_id=cid, strategy_version=ver).delete()
        s.add(StrategyPromotion(
            candidate_id=cid, strategy_version=ver, family="A", promoted=False, status="shelved",
            expectancy_r=0.024, allow_long=True, allow_short=False,
            shelved_reasons=["locked hold-out not positive net of costs"],
            summary=summary, validated_at=datetime.now(UTC), engine_version="eng_0002",
        ))

    r = client.get(f"/dashboard/strategies/{cid}")
    assert r.status_code == 200
    body = r.text
    assert "HOLD-OUT" in body  # the locked hold-out row
    assert "0.730" in body or "0.73" in body  # deflated Sharpe
    assert "-0.012" in body  # the hold-out's negative expectancy (the reason it shelved)
    assert "locked hold-out not positive net of costs" in body  # shelved reason surfaced
    assert "Cost stress" in body and "Backtest report" in body
    assert "SHELVED" in body


@requires_db
@pytest.mark.parametrize("label,route", REQUIRED_PAGES, ids=[p[0] for p in REQUIRED_PAGES])
def test_required_page_reachable(client, label: str, route: str) -> None:
    r = client.get(route)
    assert r.status_code == 200, f"{label} ({route}) -> {r.status_code}"
    # The left-sidebar shell renders on every page.
    assert 'class="sidebar"' in r.text
    assert "navlink" in r.text


@requires_db
def test_sidebar_lists_every_required_page(client) -> None:
    """Every required page has a sidebar link (so it is reachable by navigation, not just URL)."""
    from src.api.app import _NAV_GROUPS

    hrefs = {href for _, items in _NAV_GROUPS for _, href, _, _ in items}
    for _label, route in REQUIRED_PAGES:
        # per-symbol is reached from the Statistics page's symbol links, not a top-level item
        if route.startswith("/dashboard/stats/"):
            continue
        assert route in hrefs, f"{route} missing from sidebar nav"


@requires_db
def test_active_link_highlighted(client) -> None:
    assert "navlink active" in client.get("/dashboard/gates").text


@requires_db
def test_period_selector_is_custom_segmented_control(client) -> None:
    # Not a native <select> — a styled segmented pill control (the spec's "non-standard control").
    html = client.get("/dashboard/execution").text
    assert 'class="segment"' in html


@requires_db
def test_the_control_centre_asks_every_crawler_to_stay_out(client) -> None:
    """In production Caddy publishes this dashboard on a real hostname. It is authenticated, but a
    discoverable, indexed control centre for a trading bot is not something to leave to the auth
    wall alone — every response carries the header, pages carry the meta tag, and robots.txt
    disallows everything (unauthenticated, or a crawler would never read it)."""
    page = client.get("/dashboard/gates")
    assert "noindex" in page.headers.get("X-Robots-Tag", "")
    assert "<meta name='robots' content='noindex,nofollow,noarchive'>" in page.text

    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert robots.text.strip() == "User-agent: *\nDisallow: /"
