"""Standard report envelope (AGENTS.md Section 34).

Every report the system stores must include the same envelope: the versions used (config,
universe, data, strategy, model), the time period, the methodology, the results, the
limitations, and the recommendations. :func:`wrap_report` attaches it consistently and
:func:`validate_report_envelope` is the single check the tests + writers use so no report
can be persisted without the required context.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.config import get_settings

# The keys every stored report must carry (Section 34).
REQUIRED_ENVELOPE_KEYS = (
    "report_type",
    "generated_at",
    "versions",
    "period",
    "methodology",
    "results",
    "limitations",
    "recommendations",
)


def wrap_report(
    results: dict[str, Any],
    *,
    report_type: str,
    methodology: str,
    limitations: str | list[str] = "",
    recommendations: str | list[str] = "",
    period: dict[str, Any] | None = None,
    versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Wrap a report's ``results`` in the standard Section-34 envelope."""
    return {
        "report_type": report_type,
        "generated_at": datetime.now(UTC).isoformat(),
        "versions": versions if versions is not None else get_settings().versions(),
        "period": period or {"scope": "all"},
        "methodology": methodology,
        "results": results,
        "limitations": limitations,
        "recommendations": recommendations,
    }


def validate_report_envelope(payload: dict[str, Any]) -> list[str]:
    """Return the list of missing/empty required envelope fields (empty list = valid)."""
    missing: list[str] = []
    for key in REQUIRED_ENVELOPE_KEYS:
        if key not in payload:
            missing.append(key)
            continue
        val = payload[key]
        # methodology must be non-empty; the rest just need to be present.
        if key == "methodology" and not str(val or "").strip():
            missing.append(key)
    return missing
