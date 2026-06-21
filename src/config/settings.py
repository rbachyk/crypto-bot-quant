"""Versioned, environment-validated configuration.

Implements AGENTS.md Section 4 (Configuration & Versioning) and Appendix B.1
(Required Runtime Environments). Safety rules enforced here (Priority Stack
1 Capital protection, 2 Exchange safety):

* ``TRADING_MODE=LIVE`` is **never** the default and cannot be selected unless
  ``APP_ENV=production`` *and* ``ENABLE_LIVE_TRADING=true`` (Section 2.1).
* Environment mismatch fails startup (Appendix B.1).
* The ``research`` environment may not carry live trading keys (Appendix B.1).

All runtime behaviour is config-driven and every artifact carries a version
identifier (Section 4).
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, enum.Enum):
    """Runtime environment (Appendix B.1)."""

    LOCAL = "local"
    RESEARCH = "research"
    PAPER = "paper"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingMode(str, enum.Enum):
    """Trading mode. ``LIVE`` is never the default (Section 2.1)."""

    BACKTEST = "BACKTEST"
    RESEARCH = "RESEARCH"
    PAPER = "PAPER"
    LIVE = "LIVE"


class DashboardAuthMode(str, enum.Enum):
    """Dashboard authentication mode. Auth is mandatory outside ``local``."""

    BASIC = "basic"
    NONE = "none"


# Repo root = two levels up from this file (src/config/settings.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Process configuration, loaded from environment / ``.env``.

    Defaults are deliberately safe (no live trading, no real keys). Any
    material change must create a new ``CONFIG_VERSION`` (Section 4).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Environment & mode -------------------------------------------------
    app_env: AppEnv = AppEnv.LOCAL
    trading_mode: TradingMode = TradingMode.PAPER

    # --- Identity / versioning (Section 4) ---------------------------------
    exchange_id: str = "bybit"
    exchange_env: str = "testnet"
    exchange_account_type: str = "swap"
    bot_instance_id: str = "QBOT_LOCAL"
    order_client_id_prefix: str = "QBOT_LOCAL_v1_"
    config_version: str = "cfg_0001"
    universe_version: str = "univ_0001"
    data_version: str = "data_0001"
    metadata_version: str = "meta_0001"
    strategy_version: str = "strat_0001"
    feature_set_version: str = "feat_0001"
    risk_policy_version: str = "risk_0001"
    execution_policy_version: str = "exec_0001"
    online_learner_version: str = "learner_0001"

    # --- Infrastructure endpoints (Appendix B.1) ---------------------------
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/trading_bot"
    redis_url: str = "redis://localhost:6379/0"
    # --- Worker reliability + queue routing (Appendix B.13) ---
    # A worker refreshes a liveness beacon every `worker_heartbeat_sec`; the reaper treats a
    # beacon gone for `worker_heartbeat_ttl_sec` as a dead worker and re-queues its in-flight
    # jobs. TTL must exceed the heartbeat interval. `worker_queues` is a comma-separated list of
    # queue classes this worker serves ('' = serve all: ml, rl, backtest, data, gates, default).
    worker_heartbeat_sec: int = 10
    worker_heartbeat_ttl_sec: int = 30
    worker_reaper_interval_sec: int = 30
    worker_queues: str = ""
    object_storage_url: str = ""  # empty => use local data lake path below
    data_lake_path: Path = REPO_ROOT / "var" / "datalake"
    artifact_path: Path = REPO_ROOT / "var" / "artifacts"
    reports_path: Path = REPO_ROOT / "reports"
    backup_path: Path = REPO_ROOT / "var" / "backups"

    # --- Dashboard auth -----------------------------------------------------
    dashboard_auth_mode: DashboardAuthMode = DashboardAuthMode.BASIC
    dashboard_username: str = "admin"
    dashboard_password: str = "change-me-in-env"

    # --- Feature toggles (safe defaults; Appendix B.3) ---------------------
    enable_live_trading: bool = False
    # Gate the SCHEDULER (src/scheduler.py): research/paper jobs run only when
    # enable_background_research_jobs is set; ML shadow passes only when enable_ml_shadow is set.
    # OFF by default so nothing auto-enqueues recurring paper sessions / validations that add
    # dashboard trades (matches compose + .env.example); opt in explicitly to run research.
    enable_background_research_jobs: bool = False
    enable_ml_shadow: bool = False
    # Reserved: the online-learner and RL shadow layers currently run via their gates (LEARN-PROMO,
    # RL-SIM/RL-SHADOW), not scheduled jobs, so these toggles have no scheduled job to gate yet.
    enable_online_learning_shadow: bool = False
    enable_rl_shadow: bool = False

    # --- Scheduler (periodic recurring jobs; runs as the `scheduler` service) ---
    # Off by default so tests/host tooling never auto-enqueue; the compose scheduler service
    # sets SCHEDULER_ENABLED=true. Per-job cadence is in src/scheduler.py.
    scheduler_enabled: bool = False
    scheduler_tick_sec: int = 60

    # --- Exchange API credentials (absent by default) ----------------------
    exchange_api_key: str = ""
    exchange_api_secret: str = ""

    # --- Monitoring / alert transports --------------------------------------
    # Push channels are OPTIONAL: a transport activates only when its fields are set, else the
    # alert sink is log-only (so tests/gates are unaffected). Both are fail-safe at runtime.
    alert_telegram_bot_token: str = ""
    alert_telegram_chat_id: str = ""
    alert_email_host: str = ""
    alert_email_port: int = 587
    alert_email_from: str = ""
    alert_email_to: str = ""  # comma-separated recipients
    alert_email_username: str = ""
    alert_email_password: str = ""
    alert_email_use_tls: bool = True

    # --- Service identity (set per container by docker-compose) ------------
    service_name: str = "backend"

    # ------------------------------------------------------------------ #
    # Validation — environment mismatch fails startup (Appendix B.1)      #
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _enforce_safety(self) -> Settings:
        errors: list[str] = []

        # LIVE mode requires production env + explicit live enable flag.
        if self.trading_mode is TradingMode.LIVE:
            if self.app_env is not AppEnv.PRODUCTION:
                errors.append(
                    "TRADING_MODE=LIVE is only allowed when APP_ENV=production "
                    f"(got APP_ENV={self.app_env.value})."
                )
            if not self.enable_live_trading:
                errors.append(
                    "TRADING_MODE=LIVE requires ENABLE_LIVE_TRADING=true "
                    "(live trading must be explicitly enabled)."
                )

        # ENABLE_LIVE_TRADING may only be true in production.
        if self.enable_live_trading and self.app_env is not AppEnv.PRODUCTION:
            errors.append(
                "ENABLE_LIVE_TRADING=true is only allowed when APP_ENV=production "
                f"(got APP_ENV={self.app_env.value})."
            )

        # Exchange environment must be one of the three distinct Bybit envs (Section 6).
        if self.exchange_env not in ("live", "testnet", "demo"):
            errors.append(
                f"EXCHANGE_ENV={self.exchange_env!r} is invalid; must be 'live', 'testnet', or "
                "'demo' (testnet and demo are different environments with different endpoints)."
            )

        # Research environment must not carry live trading keys (Appendix B.1).
        if self.app_env is AppEnv.RESEARCH and (self.exchange_api_key or self.exchange_api_secret):
            errors.append(
                "APP_ENV=research must not carry exchange API credentials "
                "(no withdrawal/live keys in research)."
            )

        # Dashboard auth mandatory outside local (Appendix C, B.17).
        if self.app_env is not AppEnv.LOCAL and self.dashboard_auth_mode is DashboardAuthMode.NONE:
            errors.append(
                f"DASHBOARD_AUTH_MODE=none is not allowed in APP_ENV={self.app_env.value}; "
                "dashboard authentication is mandatory outside local."
            )

        # A live-capable deployment must not ship the placeholder password.
        if (
            self.app_env is AppEnv.PRODUCTION
            and self.dashboard_auth_mode is DashboardAuthMode.BASIC
            and self.dashboard_password in ("", "change-me-in-env")
        ):
            errors.append("DASHBOARD_PASSWORD must be set to a real secret in production.")

        if errors:
            raise ValueError("Unsafe / inconsistent configuration:\n  - " + "\n  - ".join(errors))
        return self

    @property
    def live_trading_allowed(self) -> bool:
        """True only when the full live-safety condition holds.

        Phase 1 never enables live trading; this is the single predicate the
        rest of the system must consult before any live action.
        """
        return (
            self.trading_mode is TradingMode.LIVE
            and self.app_env is AppEnv.PRODUCTION
            and self.enable_live_trading
        )

    @property
    def sync_database_url(self) -> str:
        """Database URL usable by SQLAlchemy/Alembic (psycopg driver)."""
        url = self.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    def versions(self) -> dict[str, str]:
        """All active version identifiers (for reports / decision logs)."""
        return {
            "CONFIG_VERSION": self.config_version,
            "UNIVERSE_VERSION": self.universe_version,
            "DATA_VERSION": self.data_version,
            "METADATA_VERSION": self.metadata_version,
            "STRATEGY_VERSION": self.strategy_version,
            "FEATURE_SET_VERSION": self.feature_set_version,
            "RISK_POLICY_VERSION": self.risk_policy_version,
            "EXECUTION_POLICY_VERSION": self.execution_policy_version,
            "ONLINE_LEARNER_VERSION": self.online_learner_version,
        }


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (validated on first call)."""
    return Settings()
