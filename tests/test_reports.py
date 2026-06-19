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
