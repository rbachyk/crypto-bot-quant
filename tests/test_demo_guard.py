"""DemoReadinessGuard (Section 6/7/13/17/35): the single pre-flight PASS/FAIL/BLOCKED gate
for safe Bybit demo execution. These tests prove each control is wired and the verdict
precedence (FAIL > BLOCKED > PASS) holds."""

from __future__ import annotations

import uuid

from src.config import Settings
from src.live.demo_guard import BLOCKED, FAIL, PASS, DemoReadinessGuard

from tests.conftest import requires_db

_PREFIX = "QBOT_LOCAL_v1_"


def _settings(**over) -> Settings:
    base = {"_env_file": None, "exchange_env": "demo", "order_client_id_prefix": _PREFIX}
    base.update(over)
    return Settings(**base)


class _CleanVenue:
    def fetch_open_orders(self):
        return {}

    def fetch_exchange_positions(self):
        return {}


class _DisengagedKill:
    def engaged(self):
        return False


class _EngagedKill:
    def engaged(self):
        return True


def test_demo_guard_blocks_on_unverified_bybit_metadata() -> None:
    """A Bybit demo run is BLOCKED because the shipped Bybit metadata is unverified
    (operator review pending) — no orders until the spec is verified (Section 6)."""
    report = DemoReadinessGuard(
        _settings(exchange_id="bybit"), kill_switch=_DisengagedKill()
    ).evaluate()
    assert report.verdict == BLOCKED
    meta = next(c for c in report.checks if c.name == "exchange_metadata")
    assert meta.status == BLOCKED and "UNVERIFIED" in meta.detail.upper()


def test_demo_guard_fails_on_engaged_kill_switch() -> None:
    report = DemoReadinessGuard(
        _settings(exchange_id="bybit"), kill_switch=_EngagedKill()
    ).evaluate()
    assert report.verdict == FAIL  # FAIL outranks the metadata BLOCKED
    ks = next(c for c in report.checks if c.name == "kill_switch")
    assert ks.status == FAIL


def test_demo_guard_fails_on_live_environment() -> None:
    """The demo guard must never green-light real-money live trading."""
    report = DemoReadinessGuard(
        _settings(exchange_env="live", exchange_id="bybit"), kill_switch=_DisengagedKill()
    ).evaluate()
    assert report.verdict == FAIL
    env = next(c for c in report.checks if c.name == "environment")
    assert env.status == FAIL


def test_demo_guard_report_lists_all_checks() -> None:
    report = DemoReadinessGuard(
        _settings(exchange_id="bybit"), kill_switch=_DisengagedKill()
    ).evaluate()
    text = report.report()
    for name in (
        "environment",
        "kill_switch",
        "order_ownership",
        "risk_caps",
        "exchange_metadata",
        "tp_sl_capability",
        "strategy_eligibility",
        "reconciliation",
    ):
        assert name in text
    assert "BLOCKED" in text


def test_demo_guard_reconciliation_without_venue_is_blocked() -> None:
    """With no connected venue the clean-book check cannot be confirmed up front → BLOCKED."""
    report = DemoReadinessGuard(
        _settings(exchange_id="skeleton"), kill_switch=_DisengagedKill()
    ).evaluate()
    recon = next(c for c in report.checks if c.name == "reconciliation")
    assert recon.status == BLOCKED


@requires_db
def test_demo_guard_passes_when_all_controls_green() -> None:
    """End-to-end PASS: skeleton (verified, matched) metadata, a clean venue, a disengaged
    kill switch, configured ownership, and a strategy validated on real lake data."""
    from src.strategies.promotion import persist_validations
    from src.strategies.research import CandidateValidation, SideDecision

    ver = f"strat_test_{uuid.uuid4().hex[:6]}"
    sd = SideDecision(
        allow_long=True, allow_short=False, long_expectancy_r=0.2, short_expectancy_r=-0.1,
        long_trades=30, short_trades=5, disabled=["short"],
    )
    cand = CandidateValidation(
        candidate_id="basis_reversion", family="B", strategy_version=ver, promoted=True,
        status="promoted", shelved_reasons=[], side_decision=sd, hypothesis={},
        report={"expectancy_r": 0.2}, walk_forward={}, fee_stress={}, slippage_stress={},
        noise_control={},
    )
    # Validated on REAL lake data → eligible for demo.
    assert persist_validations([cand], data_source="lake") == 1

    settings = _settings(exchange_id="skeleton", strategy_version=ver)
    report = DemoReadinessGuard(
        settings, kill_switch=_DisengagedKill(), venue=_CleanVenue()
    ).evaluate()
    assert report.verdict == PASS, report.report()
    assert all(c.status == PASS for c in report.checks)
