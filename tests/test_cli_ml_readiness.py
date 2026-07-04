"""CLI test for `qbot ml-readiness` — the operator's view of how much REAL data has accumulated
toward training the shadow models and passing ML-PROMO.

Monkeypatches the two data sources the command reuses (the same functions the trainer and gate
call) so the threshold/label-balance logic is asserted deterministically without a DB.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _samples(wins: int, losses: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(label=1)] * wins + [SimpleNamespace(label=0)] * losses


def test_ml_readiness_reports_below_threshold_state(monkeypatch) -> None:
    # 37 real samples (< 50 → synthetic) but ≥30 with both classes → per-model fit OK.
    # 22 linked outcomes (< 30) → ML-PROMO fails closed.
    monkeypatch.setattr(
        "src.ml.labels.build_labels_from_paper_outcomes", lambda: _samples(22, 15)
    )
    monkeypatch.setattr(
        "src.gates.phase9._load_real_shadow_outcomes", lambda: (40, [(1, 0.5)] * 22)
    )
    res = runner.invoke(app, ["ml-readiness", "--json"])
    assert res.exit_code == 0
    out = json.loads(res.stdout)
    assert out["min_real"] == 50 and out["min_fit"] == 30
    assert out["training"] == {
        "real_samples": 37, "wins": 22, "losses": 15,
        "source": "synthetic", "per_model_fit_ok": True,
    }
    assert out["ml_promo"] == {
        "real_shadow_rows": 40, "linked_outcomes": 22, "scoreable": False,
    }


def test_ml_readiness_reports_ready_state(monkeypatch) -> None:
    # 60 real samples (≥50 → real training) and 35 linked (≥30 → gate can score).
    monkeypatch.setattr(
        "src.ml.labels.build_labels_from_paper_outcomes", lambda: _samples(35, 25)
    )
    monkeypatch.setattr(
        "src.gates.phase9._load_real_shadow_outcomes", lambda: (80, [(1, 0.5)] * 35)
    )
    res = runner.invoke(app, ["ml-readiness", "--json"])
    assert res.exit_code == 0
    out = json.loads(res.stdout)
    assert out["training"]["source"] == "real"
    assert out["training"]["per_model_fit_ok"] is True
    assert out["ml_promo"]["scoreable"] is True


def test_ml_readiness_single_class_blocks_per_model_fit(monkeypatch) -> None:
    # 40 samples but ALL wins (single class) → per-model fit not possible despite ≥30 samples.
    monkeypatch.setattr(
        "src.ml.labels.build_labels_from_paper_outcomes", lambda: _samples(40, 0)
    )
    monkeypatch.setattr(
        "src.gates.phase9._load_real_shadow_outcomes", lambda: (0, [])
    )
    res = runner.invoke(app, ["ml-readiness", "--json"])
    assert res.exit_code == 0
    out = json.loads(res.stdout)
    assert out["training"]["per_model_fit_ok"] is False  # single-class guard


def test_ml_readiness_degrades_gracefully_when_data_unavailable(monkeypatch) -> None:
    # DB down / schema drift on either source → report an error field, never crash (exit 0).
    def _boom() -> object:
        raise RuntimeError("db down")

    monkeypatch.setattr("src.ml.labels.build_labels_from_paper_outcomes", _boom)
    monkeypatch.setattr("src.gates.phase9._load_real_shadow_outcomes", _boom)
    res = runner.invoke(app, ["ml-readiness", "--json"])
    assert res.exit_code == 0
    out = json.loads(res.stdout)
    assert "error" in out["training"] and "error" in out["ml_promo"]
