"""Gate result value objects and the PASS/FAIL/BLOCKED/NOT_RUN vocabulary.

The Gate Runner contract emits upper-cased verdicts (PASS/FAIL/BLOCKED/NOT_RUN)
on stdout for the orchestrator; the DB stores the richer state machine of
Appendix B.10 via :class:`src.db.models.GateStatus`.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field


class GateVerdict(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


@dataclass(slots=True)
class Criterion:
    """One checked condition within a gate.

    ``id`` is the stable criterion identifier the orchestrator parses
    (``configs/gate_result.schema.json`` requires ``id``, not ``name``).
    """

    id: str
    status: str  # "PASS" | "FAIL"
    detail: str = ""

    @classmethod
    def ok(cls, id: str, detail: str = "") -> Criterion:  # noqa: A002 - schema field name
        return cls(id=id, status="PASS", detail=detail)

    @classmethod
    def fail(cls, id: str, detail: str = "") -> Criterion:  # noqa: A002 - schema field name
        return cls(id=id, status="FAIL", detail=detail)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, str]:
        """Serialize matching ``gate_result.schema.json`` (criterion uses ``id``)."""
        out: dict[str, str] = {"id": self.id, "status": self.status, "detail": self.detail}
        if not self.passed and self.detail:
            out["failure_reason"] = self.detail
        return out


@dataclass(slots=True)
class GateRunResult:
    """Result of running one gate (matches the orchestrator JSON shape)."""

    gate_id: str
    overall: str  # GateVerdict value
    criteria: list[dict] = field(default_factory=list)
    report_path: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
