"""Phase 13 — Controlled Live Readiness tests (AGENTS.md Section 32, Appendix D).

Tests cover:
  - LEARN-PROMO-L gate (learner recommend → bounded live)
  - SEC gate (security)
  - DEPLOY gate (deployment)
  - BACKUP gate (Phase 13 full version)
  - MON gate (Phase 13 full version)
  - CONFIG-FREEZE gate (config freeze)
  - LIVE gate (final live readiness)

Each test verifies a specific gate criterion. Gate integration tests at the
bottom verify full gate runs via the CHECKS registry.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.config import get_settings

# ======================================================================== #
# LEARN-PROMO-L                                                             #
# ======================================================================== #


class TestLearnPromoL:
    def test_recommendations_track_record_created(self):
        """RECOMMEND-mode learner log entries can be written to DB."""
        from src.adaptation.action_space import BoundedAction
        from src.adaptation.store import reset_memory_sink, write_learner_log

        reset_memory_sink()
        settings = get_settings()
        action = BoundedAction(
            size_bucket=0.5,
            take=True,
            exec_style="maker",
            param_nudges={},
            learner_id="online_shadow_v1",
            learner_version="learner_0001",
            mode="RECOMMEND",
            rationale="test recommendation",
        )
        entry = write_learner_log(
            learner_id="online_shadow_v1",
            learner_version="learner_0001",
            mode="RECOMMEND",
            symbol="BTCUSDT",
            context_features={"signal_strength": 0.75},
            proposed_action=action,
            projected_outcome=0.005,
            realized_outcome=0.004,
            applied=False,
            clamped_fields=[],
            config_version=settings.config_version,
            write_to_db=True,
        )
        assert entry.mode == "RECOMMEND"
        assert entry.applied is False

    def test_rollback_triggers_on_envelope_breaker(self):
        """RollbackGuard freezes controller on envelope breaker (LIVE_BOUNDED trigger)."""
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.rollback import RollbackGuard

        ctrl = LearnerController(
            policy=OnlineLogRegPolicy(),
            bounds=ActionBounds(),
            mode=LearnerMode.RECOMMEND,
        )
        guard = RollbackGuard()
        guard.set_envelope_breaker(True)
        event = guard.check(ctrl)

        assert event is not None
        assert event.trigger == "envelope_breaker"
        assert ctrl.is_frozen()
        assert ctrl.mode == LearnerMode.FROZEN

    def test_rollback_triggers_on_unsafe_regime(self):
        """RollbackGuard freezes on R7/R8 regime."""
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.rollback import RollbackGuard

        ctrl = LearnerController(
            policy=OnlineLogRegPolicy(),
            bounds=ActionBounds(),
            mode=LearnerMode.RECOMMEND,
        )
        guard = RollbackGuard()
        guard.set_regime_unsafe("R8_DATA_UNSAFE")
        event = guard.check(ctrl)

        assert event is not None
        assert event.trigger == "unsafe_regime"
        assert ctrl.is_frozen()

    def test_learner_kill_independent_of_trading_kill_switch(self):
        """Freezing learner does not engage the trading kill switch."""
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.killswitch import KillSwitch

        settings = get_settings()
        ks = KillSwitch(settings)
        trading_state_before = ks.engaged()

        ctrl = LearnerController(
            policy=OnlineLogRegPolicy(),
            bounds=ActionBounds(),
            mode=LearnerMode.RECOMMEND,
        )
        ctrl.freeze(reason="test learner kill")

        trading_state_after = ks.engaged()

        assert ctrl.is_frozen()
        # Trading kill switch must be unchanged.
        assert trading_state_before == trading_state_after

    def test_auto_freeze_on_breaker_is_immutable(self):
        """auto_freeze_on_breaker is always True; config loader enforces it."""
        from src.adaptation.config import load_adaptation_config
        from src.adaptation.rollback import RollbackGuard

        guard = RollbackGuard()
        assert guard.auto_freeze_on_breaker is True

        cfg = load_adaptation_config()
        assert cfg.rollback.auto_freeze_on_breaker is True

    def test_bounded_update_config_present(self):
        """Adaptation config has max_change_per_update and max_change_rate."""
        from src.adaptation.config import load_adaptation_config

        cfg = load_adaptation_config()
        assert cfg.bounds.max_change_per_update > 0
        assert cfg.bounds.max_change_rate > 0

    def test_frozen_fallback_revert_restores_policy(self):
        """Frozen fallback save/load/restore round-trip works."""
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.versioning import (
            load_frozen_fallback,
            make_frozen_fallback,
            save_snapshot,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir)
            policy = OnlineLogRegPolicy(learner_id="revert_test", learner_version="v1")
            blob = policy.snapshot()
            save_snapshot(blob, "revert_test", "v1", "RECOMMEND", snap_dir)
            make_frozen_fallback(blob, snap_dir)
            ff_blob = load_frozen_fallback(snap_dir)
            restored = OnlineLogRegPolicy()
            restored.load(ff_blob)

        assert ff_blob == blob
        assert restored.learner_id == "revert_test"

    def test_learn_promo_l_gate_passes(self):
        """Full LEARN-PROMO-L gate check returns all criteria PASS."""
        from src.gates.phase13 import check_learn_promo_l

        settings = get_settings()
        criteria = check_learn_promo_l(settings)

        assert len(criteria) > 0
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"LEARN-PROMO-L gate failed criteria: {failed}"


# ======================================================================== #
# SEC                                                                       #
# ======================================================================== #


class TestSec:
    def test_env_example_exists(self):
        """The .env.example file is present."""
        repo_root = Path(__file__).resolve().parents[1]
        assert (repo_root / ".env.example").exists()

    def test_no_live_keys_in_env_example(self):
        """EXCHANGE_API_KEY is empty in .env.example."""
        repo_root = Path(__file__).resolve().parents[1]
        env_example = (repo_root / ".env.example").read_text()

        for line in env_example.splitlines():
            if line.strip().startswith("EXCHANGE_API_KEY="):
                val = line.split("=", 1)[1].strip()
                assert len(val) == 0, f"EXCHANGE_API_KEY has a real value in .env.example: {val}"
                break

    def test_enable_live_trading_false_default(self):
        """ENABLE_LIVE_TRADING=false in .env.example."""
        repo_root = Path(__file__).resolve().parents[1]
        env_example = (repo_root / ".env.example").read_text()
        assert "ENABLE_LIVE_TRADING=false" in env_example

    def test_trading_mode_paper_default(self):
        """TRADING_MODE=PAPER in .env.example."""
        repo_root = Path(__file__).resolve().parents[1]
        env_example = (repo_root / ".env.example").read_text()
        assert "TRADING_MODE=PAPER" in env_example

    def test_dashboard_auth_not_none(self):
        """DASHBOARD_AUTH_MODE is set and not 'none' in .env.example."""
        repo_root = Path(__file__).resolve().parents[1]
        env_example = (repo_root / ".env.example").read_text()
        has_auth = "DASHBOARD_AUTH_MODE=" in env_example
        not_none = "DASHBOARD_AUTH_MODE=none" not in env_example.lower()
        assert has_auth and not_none

    def test_live_engine_requires_compose_profile(self):
        """trading-engine-live requires the 'live' compose profile."""
        repo_root = Path(__file__).resolve().parents[1]
        compose = (repo_root / "docker-compose.yml").read_text()
        assert "trading-engine-live:" in compose
        assert 'profiles: ["live"]' in compose or "profiles:\n      - live" in compose

    def test_audit_log_model_importable(self):
        """AuditLog DB model is importable."""
        from src.db.models import AuditLog  # noqa: F401

    def test_sec_gate_passes(self):
        """Full SEC gate check returns all criteria PASS."""
        from src.gates.phase13 import check_sec

        settings = get_settings()
        criteria = check_sec(settings)

        assert len(criteria) > 0
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"SEC gate failed criteria: {failed}"


# ======================================================================== #
# DEPLOY                                                                    #
# ======================================================================== #


class TestDeploy:
    def test_compose_file_present(self):
        """docker-compose.yml exists."""
        repo_root = Path(__file__).resolve().parents[1]
        assert (repo_root / "docker-compose.yml").exists()

    def test_compose_parses_as_yaml(self):
        """docker-compose.yml is valid YAML with a 'services' key."""
        import yaml

        repo_root = Path(__file__).resolve().parents[1]
        parsed = yaml.safe_load((repo_root / "docker-compose.yml").read_text())
        assert isinstance(parsed, dict)
        assert "services" in parsed

    def test_live_engine_service_defined(self):
        """trading-engine-live service is defined in docker-compose.yml."""
        repo_root = Path(__file__).resolve().parents[1]
        content = (repo_root / "docker-compose.yml").read_text()
        assert "trading-engine-live:" in content

    def test_live_engine_behind_profile(self):
        """trading-engine-live requires the 'live' compose profile."""
        repo_root = Path(__file__).resolve().parents[1]
        content = (repo_root / "docker-compose.yml").read_text()
        assert 'profiles: ["live"]' in content or "profiles:\n      - live" in content

    def test_health_module_importable(self):
        """check_health is importable from src.monitoring.health."""
        from src.monitoring.health import check_health  # noqa: F401

    def test_rollback_plan_documented(self):
        """docs/decisions/phase13_rollback_plan.md exists."""
        repo_root = Path(__file__).resolve().parents[1]
        assert (repo_root / "docs" / "decisions" / "phase13_rollback_plan.md").exists()

    def test_deploy_gate_passes(self):
        """Full DEPLOY gate check returns all criteria PASS."""
        from src.gates.phase13 import check_deploy

        settings = get_settings()
        criteria = check_deploy(settings)

        assert len(criteria) > 0
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"DEPLOY gate failed criteria: {failed}"


# ======================================================================== #
# BACKUP                                                                    #
# ======================================================================== #


class TestBackup:
    def test_backup_script_present(self):
        """scripts/backup_db.sh exists."""
        repo_root = Path(__file__).resolve().parents[1]
        assert (repo_root / "scripts" / "backup_db.sh").exists()

    def test_restore_script_present(self):
        """scripts/restore_test.sh exists."""
        repo_root = Path(__file__).resolve().parents[1]
        assert (repo_root / "scripts" / "restore_test.sh").exists()

    def test_backup_script_syntax_valid(self):
        """scripts/backup_db.sh passes bash -n syntax check."""
        import subprocess

        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["bash", "-n", str(repo_root / "scripts" / "backup_db.sh")],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"backup_db.sh syntax error: {result.stderr}"

    def test_restore_script_syntax_valid(self):
        """scripts/restore_test.sh passes bash -n syntax check."""
        import subprocess

        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["bash", "-n", str(repo_root / "scripts" / "restore_test.sh")],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"restore_test.sh syntax error: {result.stderr}"

    def test_restore_job_registered(self):
        """run_restore_test_check is registered in the job registry."""
        from src.jobs.handlers import ensure_handlers_registered

        ensure_handlers_registered()
        from src.jobs.registry import registry

        assert registry.has("run_restore_test_check")

    def test_backup_path_writable(self):
        """Backup path is writable."""
        settings = get_settings()
        settings.backup_path.mkdir(parents=True, exist_ok=True)
        probe = settings.backup_path / ".probe_test_phase13"
        probe.write_text("ok")
        probe.unlink()

    def test_backup_gate_passes(self):
        """Full BACKUP gate (Phase 13) check returns all criteria PASS."""
        from src.gates.phase13 import check_backup_phase13

        settings = get_settings()
        criteria = check_backup_phase13(settings)

        assert len(criteria) > 0
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"BACKUP gate failed criteria: {failed}"


# ======================================================================== #
# MON                                                                       #
# ======================================================================== #


class TestMon:
    def test_health_module_active(self):
        """Health module probes ≥ 3 components."""
        from src.monitoring.health import check_health

        settings = get_settings()
        report = check_health(settings=settings)
        assert len(report.components) >= 3

    def test_alert_model_has_required_fields(self):
        """Alert model has severity, escalation_path, ts, component."""
        from src.monitoring.alerts import Alert, AlertSeverity

        alert = Alert(
            title="test",
            severity=AlertSeverity.CRITICAL,
            component="test",
            environment="local",
            recommended_action="test",
            escalation_path="if unacknowledged in 15 min -> escalate",
        )
        assert alert.severity == AlertSeverity.CRITICAL
        assert "15 min" in alert.escalation_path
        assert alert.ts is not None

    def test_alert_delivered_to_sink(self):
        """Sending an alert to the sink is received."""
        from src.monitoring.alerts import Alert, AlertSeverity, get_alert_sink

        sink = get_alert_sink()
        before = len(sink.recent(limit=1000))
        sink.send(
            Alert(
                title="test_mon_gate",
                severity=AlertSeverity.INFO,
                component="test",
                environment="local",
            )
        )
        after = len(sink.recent(limit=1000))
        assert after > before

    def test_critical_alert_types_definable(self):
        """All critical alert types from Appendix B.14 can be defined."""
        from src.monitoring.alerts import Alert, AlertSeverity, get_alert_sink

        sink = get_alert_sink()
        critical_types = [
            "kill_switch_triggered",
            "position_mismatch",
            "drawdown_limit_reached",
            "learner_rollback",
            "backup_failed",
        ]
        for alert_type in critical_types:
            alert = Alert(
                title=alert_type,
                severity=AlertSeverity.CRITICAL,
                component="test",
                environment="local",
            )
            result = sink.send(alert)
            assert result is True

    def test_mon_gate_passes(self):
        """Full MON gate (Phase 13) check returns all criteria PASS."""
        from src.gates.phase13 import check_mon_phase13

        settings = get_settings()
        criteria = check_mon_phase13(settings)

        assert len(criteria) > 0
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"MON gate failed criteria: {failed}"


# ======================================================================== #
# CONFIG-FREEZE                                                             #
# ======================================================================== #


class TestConfigFreeze:
    def test_config_version_set(self):
        """CONFIG_VERSION is non-empty."""
        settings = get_settings()
        assert settings.config_version.strip()

    def test_strategy_version_set(self):
        """STRATEGY_VERSION is non-empty."""
        settings = get_settings()
        assert settings.strategy_version.strip()

    def test_risk_policy_version_set(self):
        """RISK_POLICY_VERSION is non-empty."""
        settings = get_settings()
        assert settings.risk_policy_version.strip()

    def test_execution_policy_version_set(self):
        """EXECUTION_POLICY_VERSION is non-empty."""
        settings = get_settings()
        assert settings.execution_policy_version.strip()

    def test_data_version_set(self):
        """DATA_VERSION is non-empty."""
        settings = get_settings()
        assert settings.data_version.strip()

    def test_universe_version_set(self):
        """UNIVERSE_VERSION is non-empty."""
        settings = get_settings()
        assert settings.universe_version.strip()

    def test_versions_dict_complete(self):
        """settings.versions() returns ≥ 8 non-empty version strings."""
        settings = get_settings()
        versions = settings.versions()
        assert len(versions) >= 8
        empty = [k for k, v in versions.items() if not v.strip()]
        assert not empty, f"empty version values: {empty}"

    def test_live_trading_disabled(self):
        """ENABLE_LIVE_TRADING is False in default settings."""
        settings = get_settings()
        assert not settings.enable_live_trading

    def test_config_freeze_gate_passes(self):
        """Full CONFIG-FREEZE gate check returns all criteria PASS."""
        from src.gates.phase13 import check_config_freeze

        settings = get_settings()
        criteria = check_config_freeze(settings)

        assert len(criteria) > 0
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"CONFIG-FREEZE gate failed criteria: {failed}"


# ======================================================================== #
# LIVE                                                                      #
# ======================================================================== #


class TestLive:
    def test_soak_infrastructure_ready(self):
        """Infrastructure for a 72h soak test is importable."""
        from src.killswitch import KillSwitch
        from src.monitoring.health import check_health

        settings = get_settings()
        report = check_health(settings=settings)
        ks = KillSwitch(settings)
        assert len(report.components) >= 3
        assert ks is not None

    def test_circuit_breaker_trips_controller(self):
        """Envelope breaker freezes the controller (LIVE-2 circuit breaker test)."""
        from src.adaptation.action_space import ActionBounds
        from src.adaptation.controller import LearnerController, LearnerMode
        from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
        from src.adaptation.rollback import RollbackGuard

        ctrl = LearnerController(
            policy=OnlineLogRegPolicy(),
            bounds=ActionBounds(),
            mode=LearnerMode.SHADOW,
        )
        guard = RollbackGuard()
        guard.set_envelope_breaker(True)
        event = guard.check(ctrl)

        assert event is not None
        assert ctrl.is_frozen()

    def test_reconciliation_importable(self):
        """Reconciliation module is importable (LIVE-3)."""
        try:
            from src.execution.reconciliation import Reconciler  # noqa: F401
        except ImportError:
            from src.execution import reconciliation  # noqa: F401

    def test_risk_manager_has_portfolio_limits(self):
        """RiskManager source has heat cap, beta cap, position checks (LIVE-4)."""
        import inspect

        from src.risk.manager import RiskManager

        src_text = inspect.getsource(RiskManager)
        assert "heat" in src_text.lower() or "portfolio_heat" in src_text.lower()
        assert "beta" in src_text.lower()

    def test_critical_alerts_e2e(self):
        """Critical alerts deliver end-to-end (LIVE-5)."""
        from src.monitoring.alerts import Alert, AlertSeverity, get_alert_sink

        sink = get_alert_sink()
        before = len(sink.recent(limit=1000))
        for title, comp in [
            ("kill_switch_triggered", "safety"),
            ("position_mismatch", "reconciliation"),
            ("drawdown_limit_reached", "risk"),
        ]:
            sink.send(
                Alert(
                    title=title,
                    severity=AlertSeverity.CRITICAL,
                    component=comp,
                    environment="local",
                )
            )
        after = len(sink.recent(limit=1000))
        assert after - before >= 3

    def test_all_versions_frozen(self):
        """All version strings are non-empty (LIVE-9)."""
        settings = get_settings()
        versions = settings.versions()
        empty = [k for k, v in versions.items() if not v.strip()]
        assert not empty, f"unfrozen versions: {empty}"

    def test_operator_signoff_exists(self):
        """Operator sign-off file exists and is acknowledged (LIVE-10)."""
        settings = get_settings()
        signoff_path = settings.reports_path / "phase_13" / "operator_signoff.json"
        assert signoff_path.exists(), f"sign-off file missing: {signoff_path}"
        signoff = json.loads(signoff_path.read_text())
        assert signoff.get("acknowledged") is True
        assert signoff.get("capital_is_loseable_confirmed") is True
        assert signoff.get("risk_params_reviewed") is True

    def test_live_trading_disabled_final(self):
        """ENABLE_LIVE_TRADING remains False (final safety check)."""
        settings = get_settings()
        assert not settings.enable_live_trading

    def test_live_gate_passes(self):
        """Full LIVE gate check returns all criteria PASS.

        NOTE: When run via the gate runner (make run-gate GATE=LIVE), all upstream
        dependencies are checked first. Here we call check_live() directly.
        """
        from src.gates.phase13 import check_live

        settings = get_settings()
        criteria = check_live(settings)

        assert len(criteria) > 0
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"LIVE gate failed criteria: {failed}"


# ======================================================================== #
# Gate runner integration                                                   #
# ======================================================================== #


class TestPhase13GateIntegration:
    """Verify Phase 13 gates are registered in CHECKS and can run."""

    def test_all_phase13_gates_registered(self):
        """All 7 Phase 13 gates are in the CHECKS registry."""
        from src.gates.checks import CHECKS

        phase13_gates = ["LEARN-PROMO-L", "SEC", "DEPLOY", "BACKUP", "MON", "CONFIG-FREEZE", "LIVE"]
        for gate_id in phase13_gates:
            assert gate_id in CHECKS, f"gate {gate_id!r} not in CHECKS registry"

    def test_learn_promo_l_registered(self):
        from src.gates.checks import CHECKS, has_check

        assert has_check("LEARN-PROMO-L")
        assert CHECKS["LEARN-PROMO-L"].__module__ == "src.gates.phase13"

    def test_sec_registered(self):
        from src.gates.checks import CHECKS, has_check

        assert has_check("SEC")
        assert CHECKS["SEC"].__module__ == "src.gates.phase13"

    def test_deploy_registered(self):
        from src.gates.checks import CHECKS, has_check

        assert has_check("DEPLOY")
        assert CHECKS["DEPLOY"].__module__ == "src.gates.phase13"

    def test_backup_registered_phase13(self):
        """BACKUP now points to check_backup_phase13 (Phase 13 full version)."""
        from src.gates.checks import CHECKS, has_check

        assert has_check("BACKUP")
        assert "phase13" in CHECKS["BACKUP"].__module__

    def test_mon_registered_phase13(self):
        """MON now points to check_mon_phase13 (Phase 13 full version)."""
        from src.gates.checks import CHECKS, has_check

        assert has_check("MON")
        assert "phase13" in CHECKS["MON"].__module__

    def test_config_freeze_registered(self):
        from src.gates.checks import CHECKS, has_check

        assert has_check("CONFIG-FREEZE")
        assert CHECKS["CONFIG-FREEZE"].__module__ == "src.gates.phase13"

    def test_live_registered(self):
        from src.gates.checks import CHECKS, has_check

        assert has_check("LIVE")
        assert CHECKS["LIVE"].__module__ == "src.gates.phase13"

    def test_learn_promo_l_gate_full(self):
        """LEARN-PROMO-L gate runs and passes via CHECKS registry."""
        from src.gates.checks import run_check

        criteria = run_check("LEARN-PROMO-L")
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"LEARN-PROMO-L via CHECKS failed: {failed}"

    def test_sec_gate_full(self):
        """SEC gate runs and passes via CHECKS registry."""
        from src.gates.checks import run_check

        criteria = run_check("SEC")
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"SEC via CHECKS failed: {failed}"

    def test_deploy_gate_full(self):
        """DEPLOY gate runs and passes via CHECKS registry."""
        from src.gates.checks import run_check

        criteria = run_check("DEPLOY")
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"DEPLOY via CHECKS failed: {failed}"

    def test_backup_gate_full(self):
        """BACKUP gate (Phase 13) runs and passes via CHECKS registry."""
        from src.gates.checks import run_check

        criteria = run_check("BACKUP")
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"BACKUP via CHECKS failed: {failed}"

    def test_mon_gate_full(self):
        """MON gate (Phase 13) runs and passes via CHECKS registry."""
        from src.gates.checks import run_check

        criteria = run_check("MON")
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"MON via CHECKS failed: {failed}"

    def test_config_freeze_gate_full(self):
        """CONFIG-FREEZE gate runs and passes via CHECKS registry."""
        from src.gates.checks import run_check

        criteria = run_check("CONFIG-FREEZE")
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"CONFIG-FREEZE via CHECKS failed: {failed}"

    def test_live_gate_full(self):
        """LIVE gate runs and passes via CHECKS registry."""
        from src.gates.checks import run_check

        criteria = run_check("LIVE")
        failed = [c.id for c in criteria if not c.passed]
        assert not failed, f"LIVE via CHECKS failed: {failed}"
