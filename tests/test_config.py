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
        # conftest exports ALLOW_DEFAULT_DASHBOARD_CREDENTIALS=true for the suite;
        # production refuses that override, so disable it explicitly here.
        allow_default_dashboard_credentials=False,
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


def test_placeholder_password_rejected_in_every_environment() -> None:
    # Default credentials must not survive ANY deployment (a staging/paper VPS publishes
    # the dashboard behind Caddy on 443), not just production.
    for env in (AppEnv.LOCAL, AppEnv.RESEARCH, AppEnv.PAPER, AppEnv.STAGING):
        with pytest.raises(ValueError, match="DASHBOARD_PASSWORD"):
            _s(
                app_env=env,
                dashboard_password="change-me-in-env",
                allow_default_dashboard_credentials=False,
            )
        with pytest.raises(ValueError, match="DASHBOARD_PASSWORD"):
            _s(app_env=env, dashboard_password="", allow_default_dashboard_credentials=False)


def test_placeholder_password_opt_out_allowed_outside_production() -> None:
    # Local-development ergonomics: the explicit opt-out permits the placeholder.
    s = _s(
        app_env=AppEnv.LOCAL,
        dashboard_password="change-me-in-env",
        allow_default_dashboard_credentials=True,
    )
    assert s.allow_default_dashboard_credentials is True


def test_placeholder_password_opt_out_refused_in_production() -> None:
    # The escape hatch itself must never take effect in production.
    with pytest.raises(ValueError, match="ALLOW_DEFAULT_DASHBOARD_CREDENTIALS"):
        _s(
            app_env=AppEnv.PRODUCTION,
            dashboard_password="change-me-in-env",
            allow_default_dashboard_credentials=True,
        )
    with pytest.raises(ValueError, match="ALLOW_DEFAULT_DASHBOARD_CREDENTIALS"):
        _s(
            app_env=AppEnv.PRODUCTION,
            dashboard_password="a-real-secret",
            allow_default_dashboard_credentials=True,
        )


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


def test_heartbeat_ttl_must_exceed_heartbeat_interval() -> None:
    # L36: a TTL at or below the heartbeat interval makes the reaper declare LIVE workers
    # dead and duplicate their in-flight jobs — refuse the config at startup.
    with pytest.raises(ValueError, match="WORKER_HEARTBEAT_TTL_SEC"):
        _s(worker_heartbeat_sec=30, worker_heartbeat_ttl_sec=30)
    with pytest.raises(ValueError, match="WORKER_HEARTBEAT_TTL_SEC"):
        _s(worker_heartbeat_sec=30, worker_heartbeat_ttl_sec=10)
    assert _s(worker_heartbeat_sec=10, worker_heartbeat_ttl_sec=30).worker_heartbeat_ttl_sec == 30


def test_dead_settings_removed() -> None:
    # L37: settings with zero consumers were removed — silently-ignored knobs are worse than
    # none (an operator setting OBJECT_STORAGE_URL/ENABLE_RL_SHADOW would believe they work).
    for dead in ("object_storage_url", "enable_online_learning_shadow", "enable_rl_shadow"):
        assert dead not in Settings.model_fields
