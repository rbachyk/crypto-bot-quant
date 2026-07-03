"""Named report generators (AGENTS.md Section 34).

The spec enumerates the reports the system must generate and store; several (live, online
learning, RL simulation/shadow, live readiness, daily review) had no dedicated generator. This
module produces them from the live database state, each wrapped in the standard Section-34
envelope (versions / period / methodology / results / limitations / recommendations) and written
under ``reports/<name>/``. ``generate_report`` dispatches by name; ``generate_standard_reports``
produces the full set. Surfaced via ``qbot reports``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import desc, select

from src.config import Settings, get_settings
from src.db.base import session_scope
from src.reporting import validate_report_envelope, wrap_report

# name -> (subdir, methodology) for the generators below.
REPORT_NAMES = (
    "live",
    "online_learning",
    "rl_simulation",
    "rl_shadow",
    "live_readiness",
    "daily_review",
)

# Rationale marker of the synthetic learner-log rows the pre-fix LEARN-PROMO-L gate fabricated
# (see src/gates/phase13.py) — those rows are test residue, never real learner decisions, so
# every learner report must exclude them (M26).
_SYNTHETIC_RATIONALE_PREFIX = "approved recommendation #"


def _online_learner_id() -> str:
    """The online learner's id (configs/adaptation.yaml); falls back to the well-known default
    so report generation never dies on a missing/broken adaptation config."""
    try:
        from src.adaptation.config import load_adaptation_config

        return load_adaptation_config().learner_id
    except Exception:  # noqa: BLE001
        return "online_shadow_v1"


def _rl_learner_id() -> str:
    """The RL shadow policy's learner id (single source: src.adaptation.policies.rl_policy)."""
    try:
        from src.adaptation.policies.rl_policy import LEARNER_ID

        return str(LEARNER_ID)
    except Exception:  # noqa: BLE001
        return "rl_policy_v1"


def _write(settings: Settings, name: str, payload: dict) -> str:
    reports_dir = settings.reports_path / name
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"{name}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _live_results(settings: Settings) -> dict:
    from src.db.models import PaperRun

    with session_scope() as s:
        runs = list(s.execute(select(PaperRun).order_by(desc(PaperRun.created_at))).scalars())
        # All real-venue environments (M27): demo is the DEFAULT real-venue env (a Bybit demo
        # soak must show up here, not report session_count=0). Same prefix convention as
        # src.api.stats._REAL_ENV_PREFIXES / src.live.loop.LiveLoop.env_label.
        live = [r for r in runs if str(r.session_id).startswith(("demo:", "testnet:", "live:"))]
        return {
            "trading_mode": settings.trading_mode.value,
            "live_trading_allowed": settings.live_trading_allowed,
            "exchange_env": settings.exchange_env,
            "sessions": [
                {
                    "session_id": r.session_id,
                    "executed": r.executed_count,
                    "net_pnl": r.net_pnl,
                    "win_rate": r.win_rate,
                }
                for r in live[:50]
            ],
            "session_count": len(live),
        }


def _learner_results(settings: Settings, *, label: str, learner_ids: tuple[str, ...]) -> dict:
    from src.db.models import LearnerLog

    with session_scope() as s:
        rows = list(
            s.execute(
                select(LearnerLog)
                .where(LearnerLog.learner_id.in_(learner_ids))  # ONLY this layer's learner (M26)
                .order_by(desc(LearnerLog.ts))
                .limit(500)
            ).scalars()
        )
        # Drop known-synthetic rows (pre-fix LEARN-PROMO-L fabrications) so they can never
        # inflate a shadow report's entry/applied counts.
        rows = [
            r
            for r in rows
            if not str((r.proposed_action or {}).get("rationale", "")).startswith(
                _SYNTHETIC_RATIONALE_PREFIX
            )
        ]
        applied = sum(1 for r in rows if r.applied)
        rollbacks = [r.rollback_event for r in rows if r.rollback_event]
        by_mode: dict[str, int] = {}
        for r in rows:
            by_mode[r.mode] = by_mode.get(r.mode, 0) + 1
        return {
            "layer": label,
            "learner_ids": list(learner_ids),
            "log_entries": len(rows),
            "applied_count": applied,  # MUST be 0 in shadow (Section 20/21)
            "by_mode": by_mode,
            "rollback_events": rollbacks[:20],
        }


def _live_readiness_results() -> dict:
    from src.api.stats import compute_gate_stats, resolve_window

    g = compute_gate_stats(resolve_window("all", None, None))
    return {
        "live_readiness_score": round(g.live_readiness_score, 1),
        "critical_gates_passed": g.critical_gates_passed,
        "total_critical_gates": g.total_critical_gates,
        "gates_passed": g.passed,
        "gates_failed": g.failed,
        "gates_blocked": g.blocked,
        "gates_not_run": g.not_run,
        "next_critical_action": g.next_critical_action,
        "ready": g.total_critical_gates > 0 and g.critical_gates_passed == g.total_critical_gates,
    }


def _daily_review_results() -> dict:
    from src.api.stats import compute_open_remediation_count, get_aggregate_stats

    agg = get_aggregate_stats("today")
    t = agg.trading
    return {
        "trading": {
            "trades": t.total_trades,
            "win_rate": t.win_rate,
            "expectancy_r": t.expectancy_r,
            "profit_factor": t.profit_factor,
            "realized_pnl": t.realized_pnl,
            "max_drawdown_pct": t.max_drawdown_pct,
        },
        "gates": {
            "live_readiness_score": round(agg.gates.live_readiness_score, 1),
            "next_action": agg.gates.next_critical_action,
        },
        "jobs": {"failed": agg.jobs.failed, "running": agg.jobs.running},
        "open_remediation_items": compute_open_remediation_count(),
    }


def generate_report(name: str, settings: Settings | None = None) -> str:
    """Generate one named report (enveloped) and write it; returns the path."""
    settings = settings or get_settings()
    if name == "live":
        payload = wrap_report(
            _live_results(settings),
            report_type="live",
            methodology="Summary of real-venue sessions (paper_runs with a demo:/testnet:/live: "
            "id) and the live-safety configuration predicate.",
            limitations="Live trading is gated off by default; testnet sessions use no real funds.",
            recommendations="Progress to live only after Road to Live = 100% + operator sign-off.",
            period={"scope": "all"},
        )
    elif name == "online_learning":
        payload = wrap_report(
            _learner_results(
                settings, label="online_learning", learner_ids=(_online_learner_id(),)
            ),
            report_type="online_learning",
            methodology="Online-learner decision log (this learner's rows only, synthetic gate "
            "residue excluded); applied_count must be 0 in shadow.",
            limitations="Shadow-only — never influences live trading until promoted.",
            recommendations="Promote only via LEARN-PROMO gates + manual review.",
            period={"scope": "last 500 entries"},
        )
    elif name in ("rl_simulation", "rl_shadow"):
        payload = wrap_report(
            _learner_results(settings, label=name, learner_ids=(_rl_learner_id(),)),
            report_type=name,
            methodology="RL would-be decisions and bounds (the RL policy's learner rows only, "
            "synthetic gate residue excluded); shadow-only, no live influence.",
            limitations="Shadow-only; RL is gated behind RL-SIM/RL-SHADOW.",
            recommendations="Evaluate against the RL kill-criteria before any promotion.",
            period={"scope": "last 500 entries"},
        )
    elif name == "live_readiness":
        payload = wrap_report(
            _live_readiness_results(),
            report_type="live_readiness",
            methodology="Latest result of every blocks_live gate → live-readiness score.",
            limitations="Operator-attested gate criteria PASS deterministically in this env.",
            recommendations="Clear the next_critical_action; reach 100% before activation.",
            period={"scope": "all"},
        )
    elif name == "daily_review":
        payload = wrap_report(
            _daily_review_results(),
            report_type="daily_review",
            methodology="Today's realized trading performance + gate/job/remediation status.",
            limitations="Reflects paper_trades (shadow-only) until live is enabled.",
            recommendations="Triage failed jobs and open remediation items.",
            period={"scope": "today"},
        )
    else:
        raise ValueError(f"unknown report name {name!r}; known: {', '.join(REPORT_NAMES)}")
    # Real enforcement (not an assert — asserts vanish under `python -O`, L23): an incomplete
    # envelope must never be written to reports/.
    missing = validate_report_envelope(payload)
    if missing:
        raise ValueError(f"report {name!r} envelope incomplete: missing/empty {missing}")
    return _write(settings, name, payload)


def generate_standard_reports(settings: Settings | None = None) -> dict[str, str]:
    """Generate every Section-34 report that previously had no generator. Returns name → path."""
    settings = settings or get_settings()
    return {name: generate_report(name, settings) for name in REPORT_NAMES}
