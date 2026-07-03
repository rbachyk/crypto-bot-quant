"""Section 34: named report generators + report-envelope coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.config import Settings
from src.reporting import validate_report_envelope
from src.reports import REPORT_NAMES, generate_report, generate_standard_reports


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, reports_path=tmp_path)


@pytest.mark.parametrize("name", REPORT_NAMES)
def test_each_named_report_is_enveloped(tmp_path, name: str) -> None:
    path = generate_report(name, _settings(tmp_path))
    payload = json.loads(Path(path).read_text())
    assert validate_report_envelope(payload) == []
    assert payload["report_type"] == name
    assert payload["methodology"] and payload["results"]


def test_generate_standard_reports_produces_all(tmp_path) -> None:
    paths = generate_standard_reports(_settings(tmp_path))
    assert set(paths) == set(REPORT_NAMES)
    for p in paths.values():
        assert Path(p).exists()


def test_online_learning_report_tracks_applied_count(tmp_path) -> None:
    path = generate_report("online_learning", _settings(tmp_path))
    results = json.loads(Path(path).read_text())["results"]
    assert "applied_count" in results  # shadow-only: must be 0 (no live influence)


def test_live_readiness_report_has_score(tmp_path) -> None:
    results = json.loads(Path(generate_report("live_readiness", _settings(tmp_path))).read_text())[
        "results"
    ]
    assert "live_readiness_score" in results and "ready" in results


def test_unknown_report_name_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown report name"):
        generate_report("nope", _settings(tmp_path))


def test_paper_report_writer_emits_envelope(tmp_path) -> None:
    from src.paper.report import build_paper_report, write_report
    from src.paper.session import PaperSession

    session = PaperSession(session_id="env_test")
    report = build_paper_report(session)
    out = tmp_path / "paper.json"
    write_report(report, out)
    payload = json.loads(out.read_text())
    assert validate_report_envelope(payload) == []
    assert payload["report_type"] == "paper"


# --- M26/M27/L23 regressions -------------------------------------------------


def test_incomplete_envelope_raises_a_real_error(tmp_path, monkeypatch) -> None:
    """L23: envelope enforcement must be a real raise, not an ``assert`` (stripped by -O)."""
    import src.reports as reports_mod

    monkeypatch.setattr(reports_mod, "validate_report_envelope", lambda payload: ["methodology"])
    with pytest.raises(ValueError, match="envelope incomplete"):
        reports_mod.generate_report("live_readiness", _settings(tmp_path))
    assert list(tmp_path.rglob("*.json")) == []  # nothing written on failure


def test_live_report_includes_demo_sessions(tmp_path) -> None:
    """M27: demo is the DEFAULT real-venue env — a demo soak must appear in the live report."""
    import uuid

    from src.db.base import session_scope
    from src.db.models import PaperRun

    sid = f"demo:m27_{uuid.uuid4().hex[:8]}"
    with session_scope() as s:
        s.add(PaperRun(session_id=sid, executed_count=1, net_pnl=1.0, win_rate=1.0))
    try:
        payload = json.loads(Path(generate_report("live", _settings(tmp_path))).read_text())
        sessions = payload["results"]["sessions"]
        assert any(r["session_id"] == sid for r in sessions), "demo session missing (M27)"
    finally:
        with session_scope() as s:
            s.query(PaperRun).filter_by(session_id=sid).delete()


def test_learner_reports_scope_by_learner_id_and_exclude_synthetic(tmp_path) -> None:
    """M26: rl_simulation/rl_shadow summarize ONLY the RL policy's learner rows, and the
    online_learning report ONLY the online learner's; pre-fix synthetic gate rows (rationale
    'approved recommendation #N') are excluded everywhere."""
    import uuid

    from src.db.base import session_scope
    from src.db.models import LearnerLog

    mark = f"m26_{uuid.uuid4().hex[:8]}"
    rows = [
        # Genuine RL row → must be visible in the RL reports only.
        LearnerLog(learner_id="rl_policy_v1", mode="SHADOW",
                   proposed_action={"rationale": "bounded rl action"},
                   rollback_event=f"{mark}:rl_genuine"),
        # Genuine online-learner row → online_learning report only.
        LearnerLog(learner_id="online_shadow_v1", mode="SHADOW",
                   proposed_action={"rationale": "weight nudge"},
                   rollback_event=f"{mark}:ol_genuine"),
        # Foreign learner (e.g. a gate self-test id) → in NO report.
        LearnerLog(learner_id=f"gate_test_{mark[:8]}", mode="SHADOW",
                   rollback_event=f"{mark}:foreign"),
        # Synthetic pre-fix LEARN-PROMO-L fabrication on the RL id → excluded despite the id.
        LearnerLog(learner_id="rl_policy_v1", mode="RECOMMEND",
                   proposed_action={"rationale": "approved recommendation #1"},
                   rollback_event=f"{mark}:synthetic"),
    ]
    with session_scope() as s:
        for r in rows:
            s.add(r)
    try:
        settings = _settings(tmp_path)
        rl = json.loads(Path(generate_report("rl_shadow", settings)).read_text())["results"]
        ol = json.loads(Path(generate_report("online_learning", settings)).read_text())["results"]

        assert rl["learner_ids"] == ["rl_policy_v1"]
        rl_events = set(rl["rollback_events"])
        assert f"{mark}:rl_genuine" in rl_events
        assert f"{mark}:foreign" not in rl_events  # other learners' logs no longer leak in
        assert f"{mark}:ol_genuine" not in rl_events
        assert f"{mark}:synthetic" not in rl_events  # synthetic gate residue excluded

        ol_events = set(ol["rollback_events"])
        assert f"{mark}:ol_genuine" in ol_events
        assert f"{mark}:rl_genuine" not in ol_events
        assert f"{mark}:foreign" not in ol_events
        assert f"{mark}:synthetic" not in ol_events
    finally:
        with session_scope() as s:
            s.query(LearnerLog).filter(LearnerLog.rollback_event.like(f"{mark}%")).delete(
                synchronize_session=False
            )
