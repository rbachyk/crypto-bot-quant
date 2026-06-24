"""Real-data strategy validation (AGENTS.md Section 13/16 — the Parity-Rule twin of
``src.strategies.research`` on REAL downloaded market data).

``src.strategies.research.validate_candidate`` validates on synthetic deterministic *fixtures*
(it plants a known causal structure to prove the harness). This module runs the SAME gates —
full backtest, side decision, walk-forward, fee/slippage stress — over a downloaded
``DATA_VERSION`` snapshot via the one feature pipeline (``build_lake_inputs``), so a promotion
here is established on REAL prices, not fixtures. The only fixture-specific step that is dropped
is the synthetic "noise control" (there is no structureless control series for real data — the
real market IS the test of whether the edge survives).

Requires a snapshot to be downloaded first (dashboard Data page → Download, or
``qbot download --config configs/data.bybit.yaml``). Verdicts are persisted with
``data_source="lake"`` so the dashboard distinguishes real-data promotions from fixture ones.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

from src.backtest.config import BacktestConfig, load_backtest_config
from src.backtest.service import build_lake_inputs, run_engine
from src.backtest.stress import fee_stress, slippage_stress
from src.backtest.walkforward import run_walk_forward
from src.config import Settings, get_settings
from src.data.config import DataConfig
from src.data.store import SeriesStore
from src.exchange.metadata import MetadataConfig, load_metadata_for
from src.strategies.candidates import build_strategy
from src.strategies.config import CandidateConfig, StrategiesConfig, load_strategies_config
from src.strategies.research import CandidateValidation, SideDecision, _decide_sides

_log = structlog.get_logger("strategies.lake_research")

# A real-data promotion needs enough executed trades to be more than noise (Section 13/16).
_MIN_REAL_TRADES = 20

# A progress sink: ``(message)`` for human log lines. Optional so library callers can ignore it,
# while the long-running job wires it to ``ctx.log`` so the operator sees per-stage progress
# instead of an 11-hour black box.
Emit = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


def _shelved(cand: CandidateConfig, version: str, reason: str) -> CandidateValidation:
    return CandidateValidation(
        candidate_id=cand.id, family=cand.family, strategy_version=version, promoted=False,
        status="shelved", shelved_reasons=[reason],
        side_decision=SideDecision(False, False, 0.0, 0.0, 0, 0, ["long", "short"]),
        hypothesis={}, report={"expectancy_r": 0.0}, walk_forward={}, fee_stress={},
        slippage_stress={}, noise_control={"skipped": "real-data validation"},
    )


def validate_candidate_on_lake(
    cand: CandidateConfig,
    strat_cfg: StrategiesConfig,
    cfg: BacktestConfig,
    meta: MetadataConfig,
    lake_inputs: list,
    *,
    emit: Emit = _noop,
) -> CandidateValidation:
    """Validate ONE candidate over real lake inputs (same gates as the fixture harness,
    minus the synthetic noise control).

    ``emit`` receives a human line at each expensive stage (both-sides backtest, promoted
    backtest, walk-forward, fee/slippage stress) so a long run is observable rather than opaque.
    """
    emit(f"{cand.id}: both-sides backtest")
    _log.info("lake_validate_stage", candidate=cand.id, stage="backtest_both")
    both = build_strategy(cand, strat_cfg.strategy_version, cand.params)
    full_both = run_engine(
        cfg, meta, lake_inputs, strategy=both, label=f"{cand.id}_lake_both"
    ).report

    side_decision = _decide_sides(full_both, strat_cfg.min_side_expectancy_r)

    # No side cleared the expectancy floor → the strategy has no edge here. Short-circuit with the
    # ACTUAL per-side expectancy (the meaningful reason) and DO NOT run the promoted backtest /
    # walk-forward / stress: with both sides disabled the "promoted" strategy trades nothing, so
    # those gates would only pile on misleading "insufficient trades (0 < 20) / 0 folds / stress
    # 0.0" noise that buries the real cause (and wastes minutes of compute per candidate). Keep the
    # BOTH-sides report — it carries the real trade count + metrics the operator wants to see.
    if not (side_decision.allow_long or side_decision.allow_short):
        reason = (
            "no side has positive expectancy net of costs on real data "
            f"(long={side_decision.long_expectancy_r:+.3f}R over {side_decision.long_trades} "
            f"trades, short={side_decision.short_expectancy_r:+.3f}R over "
            f"{side_decision.short_trades} trades)"
        )
        emit(f"{cand.id}: SHELVED — {reason}")
        _log.info(
            "lake_validate_no_edge", candidate=cand.id,
            long_expectancy_r=side_decision.long_expectancy_r,
            short_expectancy_r=side_decision.short_expectancy_r,
            both_trades=full_both.trade_count,
        )
        return CandidateValidation(
            candidate_id=cand.id,
            family=cand.family,
            strategy_version=strat_cfg.strategy_version,
            promoted=False,
            status="shelved",
            shelved_reasons=[reason],
            side_decision=side_decision,
            hypothesis=both.hypothesis.to_dict(),
            report=full_both.payload,  # the both-sides report has the real trades + breakdown
            walk_forward={"skipped": "no side enabled (no positive expectancy)"},
            fee_stress={"skipped": "no side enabled (no positive expectancy)"},
            slippage_stress={"skipped": "no side enabled (no positive expectancy)"},
            noise_control={"skipped": "real-data validation (the live market is the control)"},
        )

    shelved: list[str] = []
    promoted_params = cand.params.with_sides(
        allow_long=side_decision.allow_long, allow_short=side_decision.allow_short
    )
    strategy = build_strategy(cand, strat_cfg.strategy_version, promoted_params)
    emit(f"{cand.id}: promoted-side backtest ({full_both.trade_count} both-side trades)")
    _log.info(
        "lake_validate_stage", candidate=cand.id, stage="backtest_promoted",
        both_trades=full_both.trade_count,
    )
    promoted_report = run_engine(
        cfg, meta, lake_inputs, strategy=strategy, label=f"{cand.id}_lake"
    ).report

    if promoted_report.trade_count < _MIN_REAL_TRADES:
        shelved.append(
            f"insufficient trades on real data ({promoted_report.trade_count} < {_MIN_REAL_TRADES})"
        )

    emit(f"{cand.id}: walk-forward ({promoted_report.trade_count} trades)")
    _log.info(
        "lake_validate_stage", candidate=cand.id, stage="walk_forward",
        trades=promoted_report.trade_count,
    )
    wf = run_walk_forward(cfg, meta, lake_inputs, strategy=strategy)
    base_e = promoted_report.expectancy_r
    emit(f"{cand.id}: fee/slippage stress")
    _log.info("lake_validate_stage", candidate=cand.id, stage="stress")
    fee = fee_stress(cfg, meta, lake_inputs, baseline_expectancy_r=base_e, strategy=strategy)
    slip = slippage_stress(cfg, meta, lake_inputs, baseline_expectancy_r=base_e, strategy=strategy)
    if not wf.passed:
        shelved.append(f"walk-forward failed on real data: {wf.reasons}")
    if not fee.survives:
        shelved.append(f"fee stress failed (expectancy_r={fee.stressed_expectancy_r})")
    if not slip.survives:
        shelved.append(f"slippage stress failed (expectancy_r={slip.stressed_expectancy_r})")

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
        noise_control={"skipped": "real-data validation (the live market is the control)"},
    )


def validate_all_on_lake(
    data_cfg: DataConfig,
    *,
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    store: SeriesStore | None = None,
    settings: Settings | None = None,
    emit: Emit = _noop,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[CandidateValidation]:
    """Validate every enabled candidate over a downloaded snapshot. One candidate failing
    (e.g. a cross-asset family needing a shape this path can't build) shelves only that one.

    ``emit`` gets a human line at each stage and ``progress(done, total, msg)`` is called as each
    candidate completes, so the long-running job is observable instead of an opaque CPU spin.
    """
    settings = settings or get_settings()
    strat_cfg = load_strategies_config()
    cfg = load_backtest_config()
    # Validate on the SAME metadata the live venue executes with (per the data config's exchange),
    # so a strategy is sized/promoted on the exact tick/lot/min-notional it will trade on — not the
    # offline skeleton spec. Falls back to skeleton for the skeleton exchange.
    meta = load_metadata_for(data_cfg.exchange_id)
    tf = timeframe or data_cfg.base_timeframe
    syms = symbols or data_cfg.active_symbols()
    store = store if store is not None else SeriesStore(settings.data_lake_path)

    emit(f"building inputs for {len(syms)} symbol(s) on {tf} (this can take a few minutes)")
    lake_inputs = build_lake_inputs(
        store,
        exchange_id=data_cfg.exchange_id,
        symbols=syms,
        timeframe=tf,
        base_timeframe=data_cfg.base_timeframe,
        funding_timeframe=data_cfg.funding_timeframe,
        start_ms=data_cfg.window_start_ms,
        end_ms=data_cfg.window_end_ms,
        oi_timeframe=data_cfg.oi_grid,
    )
    if not lake_inputs or all(not getattr(s, "bars", None) for s in lake_inputs):
        raise ValueError(
            "no real data in the lake for this window — download a snapshot first "
            "(Data page → Download real history)"
        )
    # Report the actual loaded shape — a thin window or a symbol with few bars is the usual
    # reason a candidate later shelves on "insufficient trades".
    shape = ", ".join(f"{s.symbol}={len(s.bars)}b" for s in lake_inputs)
    emit(f"inputs ready: {len(lake_inputs)} symbol(s) [{shape}]")
    _log.info(
        "lake_validate_inputs",
        symbols=len(lake_inputs),
        bars={s.symbol: len(s.bars) for s in lake_inputs},
        timeframe=tf,
    )

    candidates = list(strat_cfg.enabled_candidates())
    total = len(candidates)
    out: list[CandidateValidation] = []
    for i, cand in enumerate(candidates):
        if progress is not None:
            progress(i, total, f"validating {cand.id} ({i + 1}/{total})")
        emit(f"[{i + 1}/{total}] validating {cand.id} ({cand.family})")
        try:
            out.append(
                validate_candidate_on_lake(cand, strat_cfg, cfg, meta, lake_inputs, emit=emit)
            )
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not abort the batch
            emit(f"[{i + 1}/{total}] {cand.id} ERRORED: {exc}")
            out.append(
                _shelved(cand, strat_cfg.strategy_version, f"real-data validation error: {exc}")
            )
    if progress is not None:
        progress(total, total, "all candidates validated")
    return out
