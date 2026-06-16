"""Config & environment-safety tests (AGENTS.md Section 4, Appendix B.1).

These are the environment-safety tests required by Section 31. They assert that
unsafe configurations FAIL startup (Priority Stack: capital protection).
"""

from __future__ import annotations

import pytest
from src.config import AppEnv, DashboardAuthMode, Settings, TradingMode


def _s(**kwargs) -> Settings:
    # _env_file=None keeps these unit tests independent of any local .env.
    return Settings(_env_file=None, **kwargs)


def test_safe_defaults_no_live() -> None:
    s = _s()
    assert s.trading_mode is TradingMode.PAPER
    assert s.enable_live_trading is False
    assert s.live_trading_allowed is False
    assert s.app_env is AppEnv.LOCAL


def test_live_mode_rejected_outside_production() -> None:
    with pytest.raises(ValueError, match="APP_ENV=production"):
        _s(trading_mode=TradingMode.LIVE, app_env=AppEnv.LOCAL)


def test_live_mode_requires_enable_flag() -> None:
    with pytest.raises(ValueError, match="ENABLE_LIVE_TRADING"):
        _s(trading_mode=TradingMode.LIVE, app_env=AppEnv.PRODUCTION, enable_live_trading=False)


def test_enable_live_trading_only_in_production() -> None:
    with pytest.raises(ValueError, match="only allowed when APP_ENV=production"):
        _s(enable_live_trading=True, app_env=AppEnv.PAPER)


def test_fully_live_config_is_allowed_and_flagged() -> None:
    s = _s(
        trading_mode=TradingMode.LIVE,
        app_env=AppEnv.PRODUCTION,
        enable_live_trading=True,
        dashboard_password="a-real-secret",
    )
    assert s.live_trading_allowed is True


def test_research_env_rejects_api_keys() -> None:
    with pytest.raises(ValueError, match="research must not carry"):
        _s(app_env=AppEnv.RESEARCH, exchange_api_key="leaked")


def test_dashboard_auth_required_outside_local() -> None:
    with pytest.raises(ValueError, match="dashboard authentication is mandatory"):
        _s(app_env=AppEnv.PAPER, dashboard_auth_mode=DashboardAuthMode.NONE)


def test_production_rejects_placeholder_password() -> None:
    with pytest.raises(ValueError, match="DASHBOARD_PASSWORD"):
        _s(app_env=AppEnv.PRODUCTION, dashboard_password="change-me-in-env")


def test_sync_database_url_uses_psycopg_driver() -> None:
    s = _s(database_url="postgresql://u:p@h:5432/db")
    assert s.sync_database_url.startswith("postgresql+psycopg://")


def test_versions_payload_has_all_identifiers() -> None:
    v = _s().versions()
    for key in (
        "CONFIG_VERSION",
        "UNIVERSE_VERSION",
        "DATA_VERSION",
        "STRATEGY_VERSION",
        "FEATURE_SET_VERSION",
        "RISK_POLICY_VERSION",
        "EXECUTION_POLICY_VERSION",
    ):
        assert key in v
