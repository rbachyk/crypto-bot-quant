"""LiveActivationRequest — the typed live-activation record (AGENTS.md Section 27).

When every ``blocks_live`` gate passes, the operator's "Request live activation" creates a
typed :class:`LiveActivationRequest` pinning the exact gate results + every version the
request was granted against. A second operator approves/rejects it; the
:class:`~src.live.guard.LiveActivationGuard` only authorises orders once an APPROVED record
exists. Building the request is refused unless the gates are actually green, so the typed
record can never claim readiness it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.config import Settings, get_settings


@dataclass(slots=True)
class LiveActivationRequest:
    request_id: str
    requested_by: str
    requested_at: datetime
    gate_results: list[dict[str, Any]]  # [{gate_id, status}] for every blocks_live gate
    config_version: str
    strategy_versions: list[str]
    risk_policy_version: str
    execution_policy_version: str
    model_version: str | None = None
    learner_version: str | None = None
    status: str = "pending"  # pending | approved | rejected
    approved_by: str | None = None
    approved_at: datetime | None = None
    rejection_reason: str | None = None
    live_readiness_score: float = 0.0
    versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "gate_results": self.gate_results,
            "config_version": self.config_version,
            "strategy_versions": self.strategy_versions,
            "risk_policy_version": self.risk_policy_version,
            "execution_policy_version": self.execution_policy_version,
            "model_version": self.model_version,
            "learner_version": self.learner_version,
            "status": self.status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "live_readiness_score": self.live_readiness_score,
            "versions": self.versions,
        }


class LiveActivationError(RuntimeError):
    """Raised when a live-activation request is built while the gates are not green."""


def build_live_activation_request(
    *, requested_by: str, settings: Settings | None = None
) -> LiveActivationRequest:
    """Gather gate results + versions into a typed request — only when gates are 100%.

    Raises :class:`LiveActivationError` if any ``blocks_live`` gate is not PASS, so a request
    can never be created (and therefore never approved) before the bot is gate-ready."""
    from sqlalchemy import select

    from src.api.stats import compute_gate_stats, resolve_window
    from src.db.base import session_scope
    from src.db.models import GateResult, GateStatus
    from src.gates.catalog import load_catalog

    settings = settings or get_settings()
    stats = compute_gate_stats(resolve_window("all", None, None))
    if not (
        stats.total_critical_gates > 0 and stats.critical_gates_passed == stats.total_critical_gates
    ):
        raise LiveActivationError(
            f"live activation refused: {stats.critical_gates_passed}/{stats.total_critical_gates} "
            "blocks_live gates pass (Road to Live < 100%)"
        )

    # The gate chain can go green on synthetic/reference fixtures; live trading additionally
    # REQUIRES at least one active strategy validated on REAL downloaded lake data (Section 13),
    # so a fixture-only promotion can never reach a real account.
    from src.strategies.promotion import active_strategy_ids

    if not active_strategy_ids(settings.strategy_version, require_real_data=True):
        raise LiveActivationError(
            "live activation refused: no active strategy is validated on REAL lake data "
            "(reference/synthetic-only promotions may not trade live — download data and "
            "re-validate the strategy before requesting live activation)"
        )

    catalog = load_catalog()
    blocks_live = [g for g, spec in catalog.items() if spec.blocks_live == "true"]
    with session_scope() as s:
        latest: dict[str, str] = {}
        for row in s.execute(
            select(GateResult).order_by(GateResult.gate_id, GateResult.id.desc())
        ).scalars():
            latest.setdefault(row.gate_id, row.status.value)
        gate_results = [
            {"gate_id": gid, "status": latest.get(gid, GateStatus.NOT_RUN.value)}
            for gid in blocks_live
        ]

    versions = settings.versions()
    now = datetime.now(UTC)
    return LiveActivationRequest(
        request_id=f"live_activation_{now.strftime('%Y%m%dT%H%M%S')}",
        requested_by=requested_by,
        requested_at=now,
        gate_results=gate_results,
        config_version=versions.get("CONFIG_VERSION", ""),
        strategy_versions=[versions.get("STRATEGY_VERSION", "")],
        risk_policy_version=versions.get("RISK_POLICY_VERSION", ""),
        execution_policy_version=versions.get("EXECUTION_POLICY_VERSION", ""),
        learner_version=versions.get("ONLINE_LEARNER_VERSION"),
        live_readiness_score=stats.live_readiness_score,
        versions=versions,
    )
