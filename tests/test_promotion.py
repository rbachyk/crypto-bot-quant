"""Strategy-promotion registry tests (Section 12/13): research verdicts are persisted and the
paper/live pipeline can query which strategies are promoted (previously a dangling output)."""

from __future__ import annotations

import json
import math
import uuid

from src.strategies.promotion import (
    _json_safe,
    active_strategy_ids,
    is_strategy_promoted,
    persist_validations,
    promoted_strategies,
    promoted_timeframe,
    reference_only_active_ids,
)
from src.strategies.research import CandidateValidation, SideDecision

from tests.conftest import requires_db


def test_json_safe_sanitizes_non_finite_so_persist_never_breaks() -> None:
    """A nested inf/nan (e.g. profit_factor of a group with zero losers) must not be able to sink
    a completed multi-hour validation at the Postgres-JSON persist step. inf → ±1e9, nan → None,
    and the result must be strict-JSON serializable (allow_nan=False)."""
    raw = {
        "report": {
            "profit_factor": math.inf,
            "side_breakdown": {"long": {"profit_factor": -math.inf, "expectancy_r": 0.1}},
            "curve": [1.0, math.nan, 2.0],
        },
        "ok": "string",
        "n": 5,
    }
    safe = _json_safe(raw)
    assert safe["report"]["profit_factor"] == 1e9
    assert safe["report"]["side_breakdown"]["long"]["profit_factor"] == -1e9
    assert safe["report"]["curve"][1] is None
    json.dumps(safe, allow_nan=False)  # must not raise → safe for the Postgres JSON column


def _validation(candidate_id: str, version: str, *, promoted: bool) -> CandidateValidation:
    sd = SideDecision(
        allow_long=True,
        allow_short=False,
        long_expectancy_r=0.2,
        short_expectancy_r=-0.1,
        long_trades=30,
        short_trades=5,
        disabled=["short"],
    )
    return CandidateValidation(
        candidate_id=candidate_id,
        family="B",
        strategy_version=version,
        promoted=promoted,
        status="promoted" if promoted else "shelved",
        shelved_reasons=[] if promoted else ["both sides non-positive"],
        side_decision=sd,
        hypothesis={},
        report={"expectancy_r": 0.2 if promoted else -0.5},
        walk_forward={},
        fee_stress={},
        slippage_stress={},
        noise_control={},
    )


@requires_db
def test_persist_and_query_promotions() -> None:
    ver = f"strat_test_{uuid.uuid4().hex[:6]}"
    good = _validation("good_one", ver, promoted=True)
    bad = _validation("bad_one", ver, promoted=False)

    assert persist_validations([good, bad]) == 2

    promoted = promoted_strategies(ver)
    assert "good_one" in promoted and "bad_one" not in promoted
    assert is_strategy_promoted("good_one", ver) is True
    assert is_strategy_promoted("bad_one", ver) is False

    # Upsert is idempotent per (candidate_id, version) — no duplicate row.
    assert persist_validations([good]) == 1
    assert promoted_strategies(ver).count("good_one") == 1


@requires_db
def test_promoted_timeframe_round_trips_for_live_resolution() -> None:
    """The decision timeframe a candidate was validated on is persisted and read back, so the
    live/paper loop runs each promoted strategy on its OWN timeframe (e.g. 4h) instead of silently
    defaulting to the base grid. Mixed/unrecorded timeframes resolve to None (caller falls back)."""
    ver = f"strat_tf_{uuid.uuid4().hex[:6]}"
    persist_validations([_validation("lead_lag_xasset", ver, promoted=True)],
                        data_source="lake", timeframe="4h")
    assert promoted_timeframe(ver, candidate_ids=["lead_lag_xasset"]) == "4h"
    assert promoted_timeframe(ver) == "4h"  # single promoted strategy → its tf

    # A second promotion on a DIFFERENT timeframe → mixed → None (no single live tf to assume).
    persist_validations([_validation("residual_momentum", ver, promoted=True)],
                        data_source="lake", timeframe="1h")
    assert promoted_timeframe(ver, candidate_ids=["residual_momentum"]) == "1h"
    assert promoted_timeframe(ver) is None  # 4h + 1h → mixed

    # A legacy promotion with no recorded timeframe resolves to None (caller falls back + logs).
    legacy = f"strat_legacy_{uuid.uuid4().hex[:6]}"
    persist_validations([_validation("old_one", legacy, promoted=True)])  # no timeframe=
    assert promoted_timeframe(legacy) is None


@requires_db
def test_demo_live_requires_real_lake_data() -> None:
    """A strategy promoted on synthetic/reference data only must be BLOCKED from demo/live;
    re-validating it on real lake data makes it eligible (Section 13)."""
    ver = f"strat_test_{uuid.uuid4().hex[:6]}"
    # ``basis_reversion`` is a known candidate id in configs/strategies.yaml.
    cand = _validation("basis_reversion", ver, promoted=True)

    # Promoted on reference (synthetic) data → eligible for paper, BLOCKED for demo/live.
    assert persist_validations([cand], data_source="reference") == 1
    assert "basis_reversion" in active_strategy_ids(ver, require_real_data=False)
    assert "basis_reversion" not in active_strategy_ids(ver, require_real_data=True)
    assert "basis_reversion" in reference_only_active_ids(ver)

    # Re-validated on real lake data → now eligible for demo/live too.
    assert persist_validations([cand], data_source="lake") == 1
    assert "basis_reversion" in active_strategy_ids(ver, require_real_data=True)
    assert "basis_reversion" not in reference_only_active_ids(ver)
