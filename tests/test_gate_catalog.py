"""Gate catalog ``blocks_live`` normalisation + conditional resolution (audit H13/M25).

Covers:
  - YAML-boolean ``blocks_live: true`` normalises to "true" (M25).
  - Case-insensitive string normalisation; unknown values rejected loudly.
  - The ``defaults:`` block supplies the fallback for gates omitting the key.
  - Conditional values resolve against the owning subsystem's config flags (H13).
  - Resolution fails closed when a subsystem config is unreadable.
"""

from __future__ import annotations

import pytest
from src.adaptation.config import AdaptationConfig
from src.gates.catalog import GateSpec, _normalize_blocks_live, load_catalog
from src.ml.config import FilterCfg, MLConfig, RecommendationCfg

# --------------------------------------------------------------------------- #
# normalisation (M25)                                                          #
# --------------------------------------------------------------------------- #


def test_normalize_yaml_booleans():
    assert _normalize_blocks_live(True) == "true"
    assert _normalize_blocks_live(False) == "false"


def test_normalize_strings_case_insensitive():
    assert _normalize_blocks_live("True") == "true"
    assert _normalize_blocks_live("FALSE") == "false"
    assert _normalize_blocks_live("  if_ml_enabled ") == "if_ml_enabled"
    assert _normalize_blocks_live("IF_RL_ENABLED") == "if_rl_enabled"


def test_normalize_rejects_unknown_value():
    with pytest.raises(ValueError, match="invalid blocks_live"):
        _normalize_blocks_live("if_quantum_enabled")


def _write_catalog(tmp_path, body: str):
    p = tmp_path / "gates.yaml"
    p.write_text(body, encoding="utf-8")
    load_catalog.cache_clear()
    return str(p)


def test_yaml_boolean_true_loads_as_lowercase_string(tmp_path):
    """A bare YAML ``blocks_live: true`` must not become the string "True" (M25)."""
    path = _write_catalog(
        tmp_path,
        """
gates:
  - id: G1
    name: Gate One
    phase: 1
    blocks_live: true
  - id: G2
    name: Gate Two
    phase: 1
    blocks_live: false
""",
    )
    catalog = load_catalog(path)
    assert catalog["G1"].blocks_live == "true"
    assert catalog["G1"].blocks_live_resolved() is True
    assert catalog["G2"].blocks_live == "false"
    assert catalog["G2"].blocks_live_resolved() is False
    load_catalog.cache_clear()


def test_defaults_block_supplies_fallback(tmp_path):
    """The ``defaults:`` block is honoured for gates that omit blocks_live (M25)."""
    path = _write_catalog(
        tmp_path,
        """
defaults:
  blocks_live: false
gates:
  - id: G1
    name: Gate One
    phase: 1
  - id: G2
    name: Gate Two
    phase: 1
    blocks_live: true
""",
    )
    catalog = load_catalog(path)
    assert catalog["G1"].blocks_live == "false"
    assert catalog["G2"].blocks_live == "true"
    load_catalog.cache_clear()


def test_missing_defaults_falls_back_to_true(tmp_path):
    path = _write_catalog(
        tmp_path,
        """
gates:
  - id: G1
    name: Gate One
    phase: 1
""",
    )
    catalog = load_catalog(path)
    assert catalog["G1"].blocks_live == "true"
    load_catalog.cache_clear()


# --------------------------------------------------------------------------- #
# conditional resolution (H13)                                                 #
# --------------------------------------------------------------------------- #


def _spec(blocks_live: str) -> GateSpec:
    return GateSpec(gate_id="X", name="X", phase="9", blocks_live=blocks_live)


def _ml_cfg(stage: int, rec: bool = False, filt: bool = False) -> MLConfig:
    return MLConfig(
        ml_stage=stage,
        recommendation=RecommendationCfg(enabled=rec),
        filter=FilterCfg(enabled=filt),
    )


def test_if_ml_enabled_false_when_pure_shadow(monkeypatch):
    monkeypatch.setattr("src.ml.config.load_ml_config", lambda *a, **k: _ml_cfg(2))
    assert _spec("if_ml_enabled").blocks_live_resolved() is False


def test_if_ml_enabled_true_at_stage_3_plus(monkeypatch):
    monkeypatch.setattr("src.ml.config.load_ml_config", lambda *a, **k: _ml_cfg(3))
    assert _spec("if_ml_enabled").blocks_live_resolved() is True


def test_if_ml_enabled_true_when_filter_enabled_despite_low_stage(monkeypatch):
    """Inconsistent config (stage 2 but live filter on) still fails closed."""
    monkeypatch.setattr("src.ml.config.load_ml_config", lambda *a, **k: _ml_cfg(2, filt=True))
    assert _spec("if_ml_enabled").blocks_live_resolved() is True


def test_if_ml_enabled_true_when_recommendation_enabled(monkeypatch):
    monkeypatch.setattr("src.ml.config.load_ml_config", lambda *a, **k: _ml_cfg(2, rec=True))
    assert _spec("if_ml_enabled").blocks_live_resolved() is True


def test_if_learner_enabled_follows_adaptation_master_switch(monkeypatch):
    monkeypatch.setattr(
        "src.adaptation.config.load_adaptation_config",
        lambda *a, **k: AdaptationConfig(enabled=True),
    )
    assert _spec("if_learner_enabled").blocks_live_resolved() is True
    monkeypatch.setattr(
        "src.adaptation.config.load_adaptation_config",
        lambda *a, **k: AdaptationConfig(enabled=False),
    )
    assert _spec("if_learner_enabled").blocks_live_resolved() is False


def test_if_rl_enabled_follows_adaptation_master_switch(monkeypatch):
    """RL has no independent switch; RL policies only run inside the adaptation
    controller, so RL gates block live exactly when adaptation does."""
    monkeypatch.setattr(
        "src.adaptation.config.load_adaptation_config",
        lambda *a, **k: AdaptationConfig(enabled=True),
    )
    assert _spec("if_rl_enabled").blocks_live_resolved() is True
    monkeypatch.setattr(
        "src.adaptation.config.load_adaptation_config",
        lambda *a, **k: AdaptationConfig(enabled=False),
    )
    assert _spec("if_rl_enabled").blocks_live_resolved() is False


def test_resolution_fails_closed_on_config_error(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("unreadable config")

    monkeypatch.setattr("src.ml.config.load_ml_config", _boom)
    monkeypatch.setattr("src.adaptation.config.load_adaptation_config", _boom)
    assert _spec("if_ml_enabled").blocks_live_resolved() is True
    assert _spec("if_learner_enabled").blocks_live_resolved() is True
    assert _spec("if_rl_enabled").blocks_live_resolved() is True


def test_real_catalog_conditional_gates_resolve_critical():
    """With the repo's current configs (ml_stage=4, adaptation.enabled=true) the
    conditional gates MUST count as live-critical (the audit H13 regression)."""
    load_catalog.cache_clear()
    catalog = load_catalog()
    for gate_id in ("ML-PROMO", "LEARN-PROMO-S", "LEARN-PROMO-L", "RL-SIM", "RL-SHADOW"):
        assert catalog[gate_id].blocks_live != "true"  # still conditional in yaml
        assert catalog[gate_id].blocks_live_resolved() is True, (
            f"{gate_id} must block live while its subsystem is enabled"
        )
