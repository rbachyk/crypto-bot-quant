"""ORM models for the Phase 1 base schema.

Covers the operational / auditable state required by AGENTS.md Appendix B.4
and the Phase 1 acceptance criteria (Appendix D): jobs, job_logs, gates,
gate_results, remediation_actions, approvals, audit_logs, plus skeleton
tables for exchange metadata and the symbol universe.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #
class JobStatus(str, enum.Enum):
    """Background-job lifecycle (Appendix B.6)."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    EXPIRED = "expired"


class GateStatus(str, enum.Enum):
    """Gate state machine (Appendix B.10). Stored upper-cased in reports as
    PASS/FAIL/BLOCKED/NOT_RUN to match the Gate Runner contract."""

    NOT_RUN = "not_run"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    EXPIRED = "expired"
    NEEDS_MANUAL_APPROVAL = "needs_manual_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class RemediationStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class VerificationStatus(str, enum.Enum):
    """Exchange-metadata verification (Section 6, [VERIFIED]/[UNVERIFIED])."""

    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"


class SymbolStatus(str, enum.Enum):
    """Universe membership status (Section 9)."""

    ACTIVE = "active"
    DISABLED = "disabled"
    QUARANTINED = "quarantined"
    RESEARCH_ONLY = "research_only"


# --------------------------------------------------------------------------- #
# Jobs                                                                         #
# --------------------------------------------------------------------------- #
class Job(Base):
    """Background-job record (Appendix B.6)."""

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=16), default=JobStatus.QUEUED, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    requested_by: Mapped[str] = mapped_column(String(64), default="system")
    environment: Mapped[str] = mapped_column(String(16), default="local")
    input_params: Mapped[dict] = mapped_column(JSON, default=dict)

    related_gate_id: Mapped[str | None] = mapped_column(String(32))
    related_dataset_version: Mapped[str | None] = mapped_column(String(32))
    related_universe_version: Mapped[str | None] = mapped_column(String(32))
    related_strategy_version: Mapped[str | None] = mapped_column(String(32))
    related_model_version: Mapped[str | None] = mapped_column(String(32))

    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    progress_message: Mapped[str] = mapped_column(Text, default="")

    logs_uri: Mapped[str | None] = mapped_column(Text)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    next_action_hint: Mapped[str | None] = mapped_column(Text)

    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=1)

    logs: Mapped[list[JobLog]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobLog.id"
    )

    __table_args__ = (Index("ix_jobs_type_status", "job_type", "status"),)


class JobLog(Base):
    """Streamed log line for a job (Appendix B.6: job logs linked)."""

    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    level: Mapped[str] = mapped_column(String(8), default="INFO")
    message: Mapped[str] = mapped_column(Text)

    job: Mapped[Job] = relationship(back_populates="logs")


# --------------------------------------------------------------------------- #
# Gates                                                                        #
# --------------------------------------------------------------------------- #
class Gate(Base):
    """Catalogued gate definition, seeded from ``configs/gates.yaml``."""

    __tablename__ = "gates"

    gate_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    phase: Mapped[str] = mapped_column(String(16), default="")
    depends_on: Mapped[list] = mapped_column(JSON, default=list)
    blocks_live: Mapped[str] = mapped_column(String(32), default="true")
    pass_condition: Mapped[str] = mapped_column(Text, default="")
    remediation_steps: Mapped[list] = mapped_column(JSON, default=list)

    results: Mapped[list[GateResult]] = relationship(
        back_populates="gate", cascade="all, delete-orphan", order_by="GateResult.id"
    )


class GateResult(Base):
    """A single execution of a gate (Appendix B.10)."""

    __tablename__ = "gate_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gate_id: Mapped[str] = mapped_column(
        ForeignKey("gates.gate_id", ondelete="CASCADE"), index=True
    )
    status: Mapped[GateStatus] = mapped_column(
        Enum(GateStatus, native_enum=False, length=24), default=GateStatus.NOT_RUN, index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    run_by: Mapped[str] = mapped_column(String(64), default="system")
    environment: Mapped[str] = mapped_column(String(16), default="local")
    criteria: Mapped[list] = mapped_column(JSON, default=list)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    report_path: Mapped[str | None] = mapped_column(Text)
    related_versions: Mapped[dict] = mapped_column(JSON, default=dict)

    gate: Mapped[Gate] = relationship(back_populates="results")
    remediation_actions: Mapped[list[RemediationAction]] = relationship(
        back_populates="gate_result", cascade="all, delete-orphan"
    )


class RemediationAction(Base):
    """An ordered, actionable remediation item produced by a failed gate.

    A failed gate is never a dead end (AGENTS.md Section 0, B.9): every
    non-PASS criterion produces ordered action items here.
    """

    __tablename__ = "remediation_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gate_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("gate_results.id", ondelete="CASCADE"), index=True
    )
    gate_id: Mapped[str] = mapped_column(String(32), index=True)
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[RemediationStatus] = mapped_column(
        Enum(RemediationStatus, native_enum=False, length=16),
        default=RemediationStatus.OPEN,
    )
    owner_role: Mapped[str] = mapped_column(String(64), default="operator")
    recommended_job: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    gate_result: Mapped[GateResult | None] = relationship(back_populates="remediation_actions")


# --------------------------------------------------------------------------- #
# Approvals & audit                                                            #
# --------------------------------------------------------------------------- #
class Approval(Base):
    """Manual approval record (e.g. live-activation requests, Section 27)."""

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_type: Mapped[str] = mapped_column(String(48), index=True)
    subject_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[ApprovalStatus] = mapped_column(
        Enum(ApprovalStatus, native_enum=False, length=16),
        default=ApprovalStatus.PENDING,
    )
    requested_by: Mapped[str] = mapped_column(String(64), default="system")
    approver: Mapped[str | None] = mapped_column(String(64))
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(Base):
    """Append-only audit log. Every dashboard action is audited (B.17)."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str | None] = mapped_column(String(128))
    environment: Mapped[str] = mapped_column(String(16), default="local")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


# --------------------------------------------------------------------------- #
# Exchange metadata & universe (skeletons for Phase 1)                         #
# --------------------------------------------------------------------------- #
class ExchangeMetadata(Base):
    """Versioned, timestamped exchange-metadata snapshot (Section 6)."""

    __tablename__ = "exchange_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange_id: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(48), index=True)
    metadata_version: Mapped[str] = mapped_column(String(32), default="meta_0000")
    verification_status: Mapped[VerificationStatus] = mapped_column(
        Enum(VerificationStatus, native_enum=False, length=16),
        default=VerificationStatus.UNVERIFIED,
    )
    source: Mapped[str] = mapped_column(String(32), default="skeleton")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint(
            "exchange_id", "symbol", "metadata_version", name="uq_exchange_meta_version"
        ),
    )


class UniverseVersion(Base):
    """A versioned snapshot of the tradable symbol universe (Section 9)."""

    __tablename__ = "universe_versions"

    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    exchange_id: Mapped[str] = mapped_column(String(32), default="")
    criteria: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[str] = mapped_column(Text, default="")

    members: Mapped[list[UniverseMember]] = relationship(
        back_populates="universe", cascade="all, delete-orphan"
    )


class UniverseMember(Base):
    """Per-symbol membership within a universe version (Section 9)."""

    __tablename__ = "universe_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    universe_version: Mapped[str] = mapped_column(
        ForeignKey("universe_versions.version", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(48), index=True)
    status: Mapped[SymbolStatus] = mapped_column(
        Enum(SymbolStatus, native_enum=False, length=16), default=SymbolStatus.RESEARCH_ONLY
    )
    reason: Mapped[str] = mapped_column(Text, default="")

    universe: Mapped[UniverseVersion] = relationship(back_populates="members")

    __table_args__ = (UniqueConstraint("universe_version", "symbol", name="uq_universe_member"),)


class UniverseChange(Base):
    """Membership-history log: symbols entering/leaving or changing status across
    universe versions (Section 9 "store universe membership history"). Every
    universe rebuild records its diff against the previous version here so the
    dashboard and audit trail can answer "what changed and why"."""

    __tablename__ = "universe_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    universe_version: Mapped[str] = mapped_column(String(64), index=True)
    prev_version: Mapped[str | None] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(48), index=True)
    change_type: Mapped[str] = mapped_column(String(24))  # entered | left | status_changed
    from_status: Mapped[str | None] = mapped_column(String(16))
    to_status: Mapped[str | None] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# --------------------------------------------------------------------------- #
# Feature pipeline: feature-set versions (Phase 3)                             #
# --------------------------------------------------------------------------- #
class FeatureSetVersion(Base):
    """Immutable, versioned feature-store build (Section 10 Parity Rule).

    Features are built from exactly one dataset snapshot through a single code
    path; this row is the relational index/manifest pointer (the feature matrices
    live in the data lake as Parquet). The id is content-addressed so an
    identical rebuild reuses it (reproducibility — the FEAT gate)."""

    __tablename__ = "feature_set_versions"

    version: Mapped[str] = mapped_column(String(80), primary_key=True)  # = feature snapshot id
    feature_set_version: Mapped[str] = mapped_column(String(32), default="feat_0001", index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), default="", index=True)
    universe_version: Mapped[str | None] = mapped_column(String(64), index=True)
    exchange_id: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    timeframe: Mapped[str] = mapped_column(String(8), default="")
    feature_names: Mapped[list] = mapped_column(JSON, default=list)
    label_horizon: Mapped[int] = mapped_column(Integer, default=0)
    row_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    checksum: Mapped[str] = mapped_column(String(64), default="")
    manifest_path: Mapped[str | None] = mapped_column(Text)
    source_jobs: Mapped[list] = mapped_column(JSON, default=list)


# --------------------------------------------------------------------------- #
# Data platform: dataset versions & data-quality reports (Phase 2)             #
# --------------------------------------------------------------------------- #
class DatasetVersion(Base):
    """An immutable, versioned dataset snapshot (Appendix B.4 'dataset versions',
    B.5 manifest). Large history lives in the data lake as Parquet; this row is
    the relational index/manifest pointer (Appendix B.4: don't store large
    candle history in relational tables)."""

    __tablename__ = "dataset_versions"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)  # = snapshot id
    data_version: Mapped[str] = mapped_column(String(32), default="data_0001", index=True)
    exchange_id: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    data_types: Mapped[list] = mapped_column(JSON, default=list)
    timeframes: Mapped[list] = mapped_column(JSON, default=list)
    time_range: Mapped[dict] = mapped_column(JSON, default=dict)
    row_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    missing_ranges: Mapped[list] = mapped_column(JSON, default=list)
    checksum: Mapped[str] = mapped_column(String(64), default="")
    validation_status: Mapped[str] = mapped_column(String(24), default="unvalidated")
    manifest_path: Mapped[str | None] = mapped_column(Text)
    source_jobs: Mapped[list] = mapped_column(JSON, default=list)


class BacktestRun(Base):
    """An immutable record of one backtest / walk-forward / stress run (Phase 4).

    The full report (all Section 19 metrics + breakdowns) is written to the
    reports lake as JSON; this row is the relational index/summary the dashboard
    Backtests page and the BT/WF/FEE/SLIP gates read. ``kind`` distinguishes a
    plain backtest from a walk-forward or a fee/slippage stress run."""

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(24), default="backtest", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    backtest_version: Mapped[str] = mapped_column(String(32), default="bt_0001", index=True)
    strategy_id: Mapped[str] = mapped_column(String(64), default="")
    strategy_version: Mapped[str] = mapped_column(String(32), default="")
    dataset_version: Mapped[str | None] = mapped_column(String(64))
    feature_set_version: Mapped[str | None] = mapped_column(String(80))
    universe_version: Mapped[str | None] = mapped_column(String(64))
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    expectancy_r: Mapped[float] = mapped_column(default=0.0)
    profit_factor: Mapped[float] = mapped_column(default=0.0)
    total_return: Mapped[float] = mapped_column(default=0.0)
    max_drawdown: Mapped[float] = mapped_column(default=0.0)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    report_path: Mapped[str | None] = mapped_column(Text)
    related_versions: Mapped[dict] = mapped_column(JSON, default=dict)


class DataQualityReportRow(Base):
    """Persisted data-validation report (Section 8/23/34). A report is generated
    before every research/paper/live run; the DQ gate reads the latest."""

    __tablename__ = "data_quality_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    dataset_version: Mapped[str | None] = mapped_column(String(64), index=True)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    critical_count: Mapped[int] = mapped_column(Integer, default=0)
    violation_count: Mapped[int] = mapped_column(Integer, default=0)
    series_validated: Mapped[int] = mapped_column(Integer, default=0)
    window: Mapped[dict] = mapped_column(JSON, default=dict)
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    report_path: Mapped[str | None] = mapped_column(Text)
