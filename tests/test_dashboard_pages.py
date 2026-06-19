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
