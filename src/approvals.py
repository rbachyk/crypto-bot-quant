"""Manual approval requests (AGENTS.md Section 27).

A guarded action (most importantly **live activation**) must not proceed on automation alone:
an operator raises a PENDING :class:`~src.db.models.Approval`, and a second operator approves or
rejects it from the dashboard. This module is the *create* side (the dashboard endpoints handle
the decide side); it gives the rest of the system a single way to request a sign-off so the
approvals surface is actually populated, not read-only.
"""

from __future__ import annotations

import subprocess

from sqlalchemy import select

from src.db.base import session_scope
from src.db.models import Approval, ApprovalStatus


def current_git_commit() -> str | None:
    """Best-effort current commit, recorded on an approval so a sign-off is pinned to the
    exact code/config it was granted against. Returns None outside a git checkout."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def request_approval(
    subject_type: str,
    subject_id: str,
    *,
    requested_by: str = "system",
    evidence: dict | None = None,
) -> int:
    """Create a PENDING approval for ``(subject_type, subject_id)`` and return its id.

    Idempotent per pending subject: if a pending approval for the same subject already exists,
    its id is returned instead of creating a duplicate."""
    with session_scope() as session:
        existing = (
            session.execute(
                select(Approval).where(
                    Approval.subject_type == subject_type,
                    Approval.subject_id == subject_id,
                    Approval.status == ApprovalStatus.PENDING,
                )
            )
            .scalars()
            .first()
        )
        if existing is not None:
            return existing.id
        approval = Approval(
            subject_type=subject_type,
            subject_id=subject_id,
            requested_by=requested_by,
            evidence=evidence or {},
            git_commit=current_git_commit(),
        )
        session.add(approval)
        session.flush()
        return approval.id
