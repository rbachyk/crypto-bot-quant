"""Research / validation harness for the Phase 5 candidates (Section 13/16).

For every candidate it: builds the deterministic edge fixture and a no-structure
noise control through the ONE feature pipeline; runs the event-based engine to get
the full backtest report; computes long/short expectancy SEPARATELY and disables
any structurally-losing side (Appendix D Phase 5); then validates the surviving
configuration with walk-forward + fee ×2 + slippage +50% stress against the
up-front kill-criteria. A candidate is PROMOTED only when WF + FEE + SLIP all pass
on the surviving side(s); otherwise it is SHELVED with the reason (Section 16:
"validation exists to reject"). Each candidate yields a Strategy Report
(Section 13) with per-symbol / regime / side breakdowns and the stress evidence.

The data is a labelled deterministic fixture (no live data exists offline), so
these are research candidates (lifecycle Stage 2-4), NOT proven live edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from src.backtest.config import BacktestConfig, load_backtest_config
from src.backtest.metrics import BacktestReport
from src.backtest.service import run_engine
from src.backtest.stress import fee_stress, slippage_stress
from src.backtest.walkforward import run_walk_forward
from src.config import Settings, get_settings
from src.exchange.metadata import MetadataConfig, load_metadata_config
from src.strategies.candidates import build_strategy
from src.strategies.config import (
    CandidateConfig,
    StrategiesConfig,
    StrategyParams,
    load_strategies_config,
)
from src.strategies.fixtures import build_candidate_inputs


@dataclass(slots=True)
class SideDecision:
    allow_long: bool
    allow_short: bool
    long_expectancy_r: float
    short_expectancy_r: float
    long_trades: int
    short_trades: int
    disabled: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CandidateValidation:
    candidate_id: str
    family: str
    strategy_version: str
    promoted: bool
    status: str  # "promoted" | "shelved"
    shelved_reasons: list[str]
    side_decision: SideDecision
    hypothesis: dict
    report: dict  # full backtest report (promoted sides, edge fixture)
    walk_forward: dict
    fee_stress: dict
    slippage_stress: dict
    noise_control: dict

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "strategy_version": self.strategy_version,
            "promoted": self.promoted,
            "status": self.status,
            "shelved_reasons": self.shelved_reasons,
            "side_decision": self.side_decision.to_dict(),
            "hypothesis": self.hypothesis,
            "report": self.report,
            "walk_forward": self.walk_forward,
            "fee_stress": self.fee_stress,
            "slippage_stress": self.slippage_stress,
            "noise_control": self.noise_control,
        }


def _decide_sides(report: BacktestReport, min_side_expectancy_r: float) -> SideDecision:
    """Keep a side only if it traded and its expectancy_r clears the floor."""
    sb = report.payload["side_breakdown"]
    long_e = float(sb["long"]["expectancy_r"])
    short_e = float(sb["short"]["expectancy_r"])
    long_n = int(sb["long"]["trades"])
    short_n = int(sb["short"]["trades"])
    allow_long = long_n > 0 and long_e > min_side_expectancy_r
    allow_short = short_n > 0 and short_e > min_side_expectancy_r
    disabled = []
    if not allow_long:
        disabled.append("long")
    if not allow_short:
        disabled.append("short")
    return SideDecision(
        allow_long=allow_long,
        allow_short=allow_short,
        long_expectancy_r=long_e,
        short_expectancy_r=short_e,
        long_trades=long_n,
        short_trades=short_n,
        disabled=disabled,
    )


def validate_candidate(
    cand: CandidateConfig,
    strat_cfg: StrategiesConfig,
    cfg: BacktestConfig,
    meta: MetadataConfig,
) -> CandidateValidation:
    inputs = build_candidate_inputs(cand, edge=True)

    # 1) Full backtest with BOTH sides to measure each side independently.
    both = build_strategy(cand, strat_cfg.strategy_version, cand.params)
    full_both = run_engine(cfg, meta, inputs, strategy=both, label=f"{cand.id}_both").report
    side_decision = _decide_sides(full_both, strat_cfg.min_side_expectancy_r)

    shelved: list[str] = []
    if not (side_decision.allow_long or side_decision.allow_short):
        shelved.append("both sides have non-positive expectancy")

    # 2) Promoted configuration: surviving sides only.
    promoted_params: StrategyParams = cand.params.with_sides(
        allow_long=side_decision.allow_long, allow_short=side_decision.allow_short
    )
    strategy = build_strategy(cand, strat_cfg.strategy_version, promoted_params)

    # 3) Full report on the promoted configuration (the Strategy Report body).
    promoted_report = run_engine(
        cfg, meta, inputs, strategy=strategy, label=f"{cand.id}_promoted"
    ).report

    # 4) Walk-forward + fee/slippage stress on the promoted configuration.
    wf = run_walk_forward(cfg, meta, inputs, strategy=strategy)
    base_e = promoted_report.expectancy_r
    fee = fee_stress(cfg, meta, inputs, baseline_expectancy_r=base_e, strategy=strategy)
    slip = slippage_stress(cfg, meta, inputs, baseline_expectancy_r=base_e, strategy=strategy)

    # 5) Noise control: same strategy on the no-structure fixture must be ~flat.
    noise_inputs = build_candidate_inputs(cand, edge=False)
    noise_report = run_engine(
        cfg, meta, noise_inputs, strategy=strategy, label=f"{cand.id}_noise"
    ).report
    noise_control = {
        "expectancy_r": noise_report.expectancy_r,
        "net_pnl": noise_report.net_pnl,
        "trade_count": noise_report.trade_count,
        # A causal strategy must NOT be meaningfully profitable on structureless data.
        "passed": noise_report.expectancy_r <= 0.05,
    }

    if not wf.passed:
        shelved.append(f"walk-forward failed: {wf.reasons}")
    if not fee.survives:
        shelved.append(f"fee stress failed (expectancy_r={fee.stressed_expectancy_r})")
    if not slip.survives:
        shelved.append(f"slippage stress failed (expectancy_r={slip.stressed_expectancy_r})")
    if not noise_control["passed"]:
        shelved.append("noise control not flat (possible look-ahead)")

    promoted = not shelved
    return CandidateValidation(
        candidate_id=cand.id,
        family=cand.family,
        strategy_version=strat_cfg.strategy_version,
        promoted=promoted,
        status="promoted" if promoted else "shelved",
        shelved_reasons=shelved,
        side_decision=side_decision,
        hypothesis=strategy.hypothesis.to_dict(),
        report=promoted_report.payload,
        walk_forward=wf.to_dict(),
        fee_stress=fee.to_dict(),
        slippage_stress=slip.to_dict(),
        noise_control=noise_control,
    )


def validate_all(
    strat_cfg: StrategiesConfig | None = None,
    cfg: BacktestConfig | None = None,
    meta: MetadataConfig | None = None,
) -> list[CandidateValidation]:
    strat_cfg = strat_cfg or load_strategies_config()
    cfg = cfg or load_backtest_config()
    meta = meta or load_metadata_config()
    # lake_only candidates have no synthetic fixture — they are validated on REAL lake data
    # (validate_all_on_lake), so the synthetic-fixture path skips them.
    return [
        validate_candidate(c, strat_cfg, cfg, meta)
        for c in strat_cfg.enabled_candidates()
        if not c.lake_only
    ]


_VALIDATIONS_CACHE: dict[tuple[str, str, str], list[CandidateValidation]] = {}


def get_validations() -> list[CandidateValidation]:
    """Validate every enabled candidate once, memoized by the active versions.

    The whole pipeline is deterministic, so the WF / FEE / SLIP gates (which each
    only need their own slice of the result) share a single validation pass rather
    than re-running the heavy backtests three times.
    """
    strat_cfg = load_strategies_config()
    cfg = load_backtest_config()
    meta = load_metadata_config()
    key = (strat_cfg.strategy_version, cfg.backtest_version, meta.metadata_version)
    cached = _VALIDATIONS_CACHE.get(key)
    if cached is None:
        cached = validate_all(strat_cfg, cfg, meta)
        _VALIDATIONS_CACHE[key] = cached
    return cached


_TARGET_FAMILIES = ("A", "B", "G")  # Appendix D Phase 5 minimum deliverable.


def families_promoted(validations: list[CandidateValidation]) -> dict[str, bool]:
    """Map each required family to whether it has a promoted candidate."""
    promoted = {v.family for v in validations if v.promoted}
    return {fam: fam in promoted for fam in _TARGET_FAMILIES}


# --------------------------------------------------------------------------- #
# Strategy Report persistence (Section 13 / Section 24)                        #
# --------------------------------------------------------------------------- #
def strategy_report_payload(v: CandidateValidation) -> dict:
    """The Section 13 Strategy Report for one candidate (hypothesis + evidence)."""
    p = v.report
    return {
        "candidate_id": v.candidate_id,
        "family": v.family,
        "strategy_version": v.strategy_version,
        "status": v.status,
        "promoted": v.promoted,
        "shelved_reasons": v.shelved_reasons,
        "hypothesis": v.hypothesis,
        "side_decision": v.side_decision.to_dict(),
        "headline": {
            "trade_count": p["trade_count"],
            "expectancy_r": p["expectancy_r"],
            "profit_factor": p["profit_factor"],
            "win_rate": p["win_rate"],
            "max_drawdown": p["max_drawdown"],
            "total_return": p["total_return"],
        },
        "symbol_breakdown": p["symbol_breakdown"],
        "regime_breakdown": p["regime_breakdown"],
        "session_breakdown": p["session_breakdown"],
        "side_breakdown": p["side_breakdown"],
        "cost_breakdown": p["cost_breakdown"],
        "slippage_breakdown": p["slippage_breakdown"],
        "funding_breakdown": p["funding_breakdown"],
        "rejected_candidates": p["rejected_candidates"],
        "exit_reason_breakdown": p["exit_reason_breakdown"],
        "worst_trades": p["worst_trades"],
        "stability": p["stability"],
        "walk_forward": v.walk_forward,
        "fee_stress": v.fee_stress,
        "slippage_stress": v.slippage_stress,
        "noise_control": v.noise_control,
    }


def write_strategy_reports(
    validations: list[CandidateValidation], settings: Settings | None = None
) -> dict[str, str]:
    """Persist one Strategy Report per candidate under ``reports/strategies/``."""
    import json

    settings = settings or get_settings()
    out_dir = settings.reports_path / "strategies"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    paths: dict[str, str] = {}
    for v in validations:
        payload = {"versions": settings.versions(), **strategy_report_payload(v)}
        path = out_dir / f"{v.candidate_id}_{stamp}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        paths[v.candidate_id] = str(path)
    return paths
