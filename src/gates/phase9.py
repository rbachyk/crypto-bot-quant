"""Phase 9 gate check: ML-PROMO (AGENTS.md Appendix A, Phase 9).

ML-PROMO — ML Promotion (Phase 9–10):
  Pass: in shadow the model improves expectancy net of costs; preserves/raises
  profit factor; preserves/reduces max DD; does not remove most best trades;
  reduces tail risk; stable across folds/symbols/regimes/OOS; calibrated;
  explainable; no leakage; manually reviewed.

Two kinds of criteria (audit H15):

Plumbing self-checks (``*_plumbing``) run on the deterministic synthetic
reference dataset from :mod:`src.ml.labels` and only prove the pipeline
(feature construction → training → shadow logging → offline scoring) executes.
They are NEVER evidence the model has edge — the reference dataset is
constructed so a correctly-trained meta-labeler beats the baseline.

Performance criteria (``ml_expectancy_improves``, ``ml_profit_factor_preserved``,
``ml_tail_risk_reduced``, ``ml_best_trades_preserved``) are computed from REAL
persisted shadow decisions: ``shadow_logs`` rows written for real candidates
(mode=SHADOW, model_type=meta_labeler, not tagged synthetic) joined to realized
paper-trade outcomes on the same symbol within a bounded horizon. While the
real ML pipeline does not yet produce such rows (audit H18), these criteria
FAIL CLOSED with explicit remediation — the gate never passes on fabricated
outcomes.

DB hygiene: the plumbing shadow-log writes are tagged synthetic AND deleted at
the end of the run; model artifacts are registered into a throwaway directory
with ``write_db=False`` — the gate no longer pollutes production tables
(audit C4-adjacent). Rows written by pre-fix runs of this gate (and by the
reference-dataset demo jobs before they tagged their writes) are synthetic
pollution; purge them with::

    DELETE FROM shadow_logs;   -- pre-fix, no writer produced real rows (audit H18)

Infrastructure criteria that remain real checks:
  * ml_shadow_imports — all components importable
  * ml_no_live_influence — no shadow_log row anywhere has applied=True
"""

from __future__ import annotations

import random
from datetime import UTC, timedelta
from typing import TYPE_CHECKING

from src.config import Settings
from src.gates.result import Criterion

if TYPE_CHECKING:
    from datetime import datetime

    from src.ml.config import MLConfig
    from src.ml.scorer import ShadowScorerResult

# The four ML-PROMO performance criteria (Appendix A / configs/ml.yaml kill_criteria).
_PERFORMANCE_CRITERIA = (
    "ml_expectancy_improves",
    "ml_profit_factor_preserved",
    "ml_tail_risk_reduced",
    "ml_best_trades_preserved",
)

# A real shadow decision realizes its outcome through the paper trade the candidate
# became. Until shadow_logs carries an explicit trade linkage (audit H18), the join is
# same-symbol + first trade closed at/after the prediction within this horizon —
# deliberately tight so stale predictions can't borrow unrelated outcomes.
_REAL_OUTCOME_LINK_HORIZON_HOURS = 48.0


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


# Deterministic shuffle seeds for the leakage noise check (criterion 7). The
# criterion averages over all of them: leakage is systematic and survives
# averaging; single-split luck (± ~0.15R at n_test≈25) does not.
_LEAKAGE_NOISE_SEEDS = (7, 13, 99, 123, 2024)


def _noise_shuffle_delta(ml_cfg: MLConfig, settings: Settings, noise_seed: int) -> float:
    """Expectancy improvement of a model trained/evaluated on outcome-shuffled noise.

    Builds the reference dataset, then shuffles the ENTIRE outcome tuple
    ``(label, realized_pnl, hold_bars)`` across candidates — never the label
    alone. Shuffling only the label leaves each candidate's realized_pnl
    attached to its features, and the reference dataset's features genuinely
    predict pnl by construction, so any feature-driven prediction evaluated
    against the unshuffled pnl showed a large spurious "improvement". Tuple
    shuffling keeps label↔outcome consistent within each pair (so real
    train/test contamination — a model memorising test outcomes — still shows
    up as positive improvement) while destroying the feature→outcome link a
    leakage-free model could exploit; its expected improvement is ≈ 0.
    """
    from src.ml.labels import (
        LabeledSample,
        baseline_expectancy,
        build_reference_dataset,
        filtered_expectancy,
        train_test_split,
    )
    from src.ml.shadow import ShadowPredictor

    noise_samples = build_reference_dataset(seed=42)
    rng = random.Random(noise_seed)
    outcomes = [(s.label, s.realized_pnl, s.hold_bars) for s in noise_samples]
    rng.shuffle(outcomes)
    noise_samples_shuffled = [
        LabeledSample(candidate=s.candidate, label=lbl, realized_pnl=pnl, hold_bars=hold)
        for s, (lbl, pnl, hold) in zip(noise_samples, outcomes, strict=True)
    ]
    noise_train, noise_test = train_test_split(noise_samples_shuffled, seed=noise_seed)

    noise_predictor = ShadowPredictor.from_config(ml_cfg)
    noise_predictor.train(noise_train)
    noise_result = noise_predictor.run(
        [s.candidate for s in noise_test],
        settings=settings,
        write_to_db=False,  # noise run: no DB writes
    )
    noise_preds = [b.meta_label.label if b.meta_label else 1 for b in noise_result.bundles]
    return filtered_expectancy(noise_test, noise_preds) - baseline_expectancy(noise_test)


def _load_real_shadow_outcomes() -> tuple[int, list[tuple[int, float]]]:
    """REAL meta-labeler shadow decisions joined to realized paper-trade outcomes.

    Returns ``(n_real_rows, linked)`` where ``linked`` is a chronological list of
    ``(predicted_label, realized_pnl_r)`` pairs. Rows tagged with
    :data:`src.ml.shadow.SYNTHETIC_CONTEXT_KEY` are excluded — they are pipeline
    self-checks, not evidence.
    """
    from src.db.base import session_scope
    from src.db.models import PaperTradeRecord, ShadowLog
    from src.ml.shadow import SYNTHETIC_CONTEXT_KEY

    with session_scope() as session:
        rows = [
            (r.ts, r.symbol, dict(r.prediction or {}), dict(r.context_features or {}))
            for r in session.query(ShadowLog)
            .filter(ShadowLog.model_type == "meta_labeler", ShadowLog.mode == "SHADOW")
            .order_by(ShadowLog.ts)
            .all()
        ]
        real = [
            (ts, sym, pred)
            for ts, sym, pred, ctx in rows
            if not ctx.get(SYNTHETIC_CONTEXT_KEY)
        ]
        if not real:
            return 0, []

        symbols = {sym for _, sym, _ in real if sym}
        # Column-limited select: cheaper, and robust on deployments whose paper_trades
        # table predates later added columns.
        trades = (
            session.query(
                PaperTradeRecord.symbol,
                PaperTradeRecord.created_at,
                PaperTradeRecord.pnl_r,
            )
            .filter(PaperTradeRecord.symbol.in_(symbols))
            .order_by(PaperTradeRecord.created_at)
            .all()
            if symbols
            else []
        )

    by_symbol: dict[str, list[tuple[datetime, float]]] = {}
    for sym, created_at, pnl_r in trades:
        by_symbol.setdefault(sym, []).append((_as_utc(created_at), float(pnl_r)))

    horizon = timedelta(hours=_REAL_OUTCOME_LINK_HORIZON_HOURS)
    linked: list[tuple[int, float]] = []
    for ts, sym, pred in real:
        row_ts = _as_utc(ts)
        for trade_ts, pnl_r in by_symbol.get(sym or "", []):
            if row_ts <= trade_ts <= row_ts + horizon:
                linked.append((int(pred.get("label", 1)), pnl_r))
                break
    return len(real), linked


def _outcome_sample(pnl_r: float):  # -> LabeledSample
    """Wrap one realized outcome so :class:`ShadowScorer` can score it."""
    from src.ml.labels import LabeledSample
    from src.ranking.candidate import Candidate

    cand = Candidate(
        symbol="",
        strategy="",
        strategy_version="",
        side=1,
        entry_price=0.0,
        stop_frac=0.0,
        tp_frac=0.0,
        regime="",
        session=0,
    )
    return LabeledSample(
        candidate=cand, label=1 if pnl_r > 0 else 0, realized_pnl=pnl_r, hold_bars=0
    )


def _real_performance_criteria(
    n_real: int, linked: list[tuple[int, float]], ml_cfg: MLConfig
) -> tuple[list[Criterion], ShadowScorerResult | None]:
    """The four ML-PROMO performance criteria, scored on REAL shadow outcomes.

    Fails closed (all four criteria FAIL) when fewer real linked decisions exist
    than ``meta_labeler.min_train_samples`` — the gate never scores synthetic data
    for its pass verdict and never fabricates outcomes (audit H15).
    """
    min_needed = ml_cfg.meta_labeler.min_train_samples
    if len(linked) < min_needed:
        reason = (
            "insufficient REAL shadow decisions with realized outcomes: "
            f"{n_real} real meta_labeler shadow_log rows (mode=SHADOW, not tagged synthetic), "
            f"{len(linked)} linked to realized paper-trade outcomes — need ≥ {min_needed}. "
            f"ML stage {ml_cfg.ml_stage} requires the paper/live loop to log shadow predictions "
            "for REAL candidates and accumulate realized outcomes (audit H18). Remediation: wire "
            "ShadowPredictor into the paper loop, run paper sessions until enough outcomes exist, "
            "then re-run gate:ml-promo. This gate no longer scores the synthetic reference "
            "dataset for its pass verdict."
        )
        return [Criterion.fail(name, reason) for name in _PERFORMANCE_CRITERIA], None

    from src.ml.scorer import ShadowScorer

    samples = [_outcome_sample(pnl_r) for _, pnl_r in linked]
    preds = [label for label, _ in linked]
    kc = ml_cfg.kill_criteria
    scorer = ShadowScorer(
        min_improvement=kc.min_improvement_over_baseline,
        min_pf_ratio=kc.min_profit_factor_ratio,
        max_tail_loss_ratio=kc.max_tail_loss_ratio,
        max_best_removed_pct=kc.max_best_trades_removed_pct,
    )
    score = scorer.score(samples, preds)
    tag = f"REAL shadow outcomes, n={len(linked)}"
    out = [
        Criterion.ok(
            "ml_expectancy_improves",
            f"[{tag}] model expectancy={score.model_expectancy:.4f}R "
            f"vs baseline={score.baseline_expectancy:.4f}R "
            f"(improvement={score.expectancy_improvement:+.4f}R)",
        )
        if score.expectancy_improvement >= kc.min_improvement_over_baseline
        else Criterion.fail(
            "ml_expectancy_improves",
            f"[{tag}] no improvement: model={score.model_expectancy:.4f}R "
            f"baseline={score.baseline_expectancy:.4f}R "
            f"delta={score.expectancy_improvement:+.4f}R",
        ),
        Criterion.ok(
            "ml_profit_factor_preserved",
            f"[{tag}] model PF={score.model_profit_factor:.3f} "
            f"baseline PF={score.baseline_profit_factor:.3f} "
            f"ratio={score.profit_factor_ratio:.3f}",
        )
        if score.profit_factor_ratio >= kc.min_profit_factor_ratio
        else Criterion.fail(
            "ml_profit_factor_preserved",
            f"[{tag}] PF degraded: model={score.model_profit_factor:.3f} "
            f"baseline={score.baseline_profit_factor:.3f} "
            f"ratio={score.profit_factor_ratio:.3f}",
        ),
        Criterion.ok(
            "ml_tail_risk_reduced",
            f"[{tag}] model worst={score.model_worst_trade:.4f}R "
            f"baseline worst={score.baseline_worst_trade:.4f}R "
            f"ratio={score.tail_loss_ratio:.3f}",
        )
        if score.tail_loss_ratio <= kc.max_tail_loss_ratio
        else Criterion.fail(
            "ml_tail_risk_reduced",
            f"[{tag}] tail risk worsened: model worst={score.model_worst_trade:.4f}R "
            f"ratio={score.tail_loss_ratio:.3f} > {kc.max_tail_loss_ratio}",
        ),
        Criterion.ok(
            "ml_best_trades_preserved",
            f"[{tag}] {score.best_trades_removed_pct:.1%} of top trades removed "
            f"(≤ {kc.max_best_trades_removed_pct:.0%} allowed)",
        )
        if score.best_trades_removed_pct <= kc.max_best_trades_removed_pct
        else Criterion.fail(
            "ml_best_trades_preserved",
            f"[{tag}] too many top trades removed: {score.best_trades_removed_pct:.1%} "
            f"> {kc.max_best_trades_removed_pct:.0%}",
        ),
    ]
    return out, score


def check_ml_promo(settings: Settings) -> list[Criterion]:
    """ML-PROMO gate check (Appendix A, Phase 9)."""
    out: list[Criterion] = []

    # ------------------------------------------------------------------ #
    # 1. Component imports                                                  #
    # ------------------------------------------------------------------ #
    try:
        from src.ml import MLRegistry, ModelArtifact, ShadowPredictor, ShadowScorer
        from src.ml.config import load_ml_config
        from src.ml.features import FEATURE_NAMES, build_feature_matrix
        from src.ml.labels import (
            build_reference_dataset,
            train_test_split,
        )
        from src.ml.models import (  # noqa: F401
            ExecQualityModel,
            MetaLabeler,
            RegimeClassifier,
            StrategySelector,
            SymbolRanker,
        )

        out.append(Criterion.ok("ml_shadow_imports", "all ML shadow components importable"))
    except ImportError as exc:
        out.append(Criterion.fail("ml_shadow_imports", f"import error: {exc}"))
        return out

    ml_cfg = load_ml_config()

    # ------------------------------------------------------------------ #
    # 2. PLUMBING: reference dataset + feature matrix                      #
    # ------------------------------------------------------------------ #
    try:
        all_samples = build_reference_dataset(n_good=40, n_bad=30, n_neutral=30, seed=42)
        candidates = [s.candidate for s in all_samples]
        X_all = build_feature_matrix(candidates)
        out.append(
            Criterion.ok(
                "ml_feature_matrix_plumbing",
                f"{len(X_all)} rows × {len(FEATURE_NAMES)} features from synthetic reference "
                "dataset (pipeline self-check only)",
            )
            if X_all and len(X_all[0]) == len(FEATURE_NAMES)
            else Criterion.fail("ml_feature_matrix_plumbing", "feature matrix empty or wrong width")
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_feature_matrix_plumbing", f"feature build raised: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 3. PLUMBING: train all five shadow models                             #
    # ------------------------------------------------------------------ #
    train_samples, test_samples = train_test_split(all_samples, test_fraction=0.25, seed=42)
    predictor = ShadowPredictor.from_config(ml_cfg)

    try:
        all_metrics = predictor.train(train_samples)
        expected_types = {
            "meta_labeler",
            "regime_classifier",
            "exec_quality",
            "strategy_selector",
            "symbol_ranker",
        }
        missing = expected_types - set(all_metrics.keys())
        out.append(
            Criterion.ok(
                "ml_models_train_plumbing",
                f"all 5 models trained on synthetic data; "
                f"meta_labeler accuracy={all_metrics.get('meta_labeler', {}).get('accuracy', '?')}",
            )
            if not missing
            else Criterion.fail(
                "ml_models_train_plumbing", f"models not trained: {sorted(missing)}"
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_models_train_plumbing", f"training raised: {exc}"))
        return out

    # ------------------------------------------------------------------ #
    # 4. PLUMBING: shadow logging (applied=False; rows tagged + removed)   #
    # ------------------------------------------------------------------ #
    plumbing_log_ids: list[int] = []
    try:
        result = predictor.run(
            candidates[:20],  # small subset keeps the gate fast
            settings=settings,
            write_to_db=True,
            synthetic_source=True,  # tag: never readable as real evidence
        )
        plumbing_log_ids = list(result.shadow_log_ids)
        n_logged = len(plumbing_log_ids)
        out.append(
            Criterion.ok(
                "ml_shadow_log_writes_plumbing",
                f"{n_logged} shadow log entries written (applied=False, tagged synthetic; "
                "deleted again at end of gate run)",
            )
            if n_logged > 0 and not result.applied
            else Criterion.fail(
                "ml_shadow_log_writes_plumbing",
                f"n_logged={n_logged} applied={result.applied} (expected n>0 and applied=False)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_shadow_log_writes_plumbing", f"shadow run raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 5. REAL: no shadow_log row anywhere may have applied=True            #
    # ------------------------------------------------------------------ #
    try:
        from src.db.base import session_scope
        from src.db.models import ShadowLog

        with session_scope() as session:
            live_entries = session.query(ShadowLog).filter(ShadowLog.applied.is_(True)).count()
            shadow_entries = session.query(ShadowLog).filter(ShadowLog.mode == "SHADOW").count()
        out.append(
            Criterion.ok(
                "ml_no_live_influence",
                f"{shadow_entries} SHADOW entries; 0 applied entries",
            )
            if live_entries == 0
            else Criterion.fail(
                "ml_no_live_influence",
                f"{live_entries} shadow_log entries have applied=True (no live influence allowed)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_no_live_influence", f"DB check raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 6. PLUMBING: model registry round-trip (throwaway dir, no DB rows)   #
    # ------------------------------------------------------------------ #
    try:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = MLRegistry(Path(tmpdir))
            registered: list[str] = []
            for model in predictor.all_models:
                artifact = ModelArtifact(
                    model_id=model.model_id,
                    model_version=ml_cfg.model_version,
                    model_type=model.model_type,
                    ml_stage=ml_cfg.ml_stage,
                    promotion_status="shadow",
                    label_definition={"type": "binary", "source": "paper_trade_outcome"},
                    performance_metrics=model.performance_report(),
                    known_failure_modes=[
                        "insufficient_history: requires ≥30 real paper trades",
                        "covariate_shift: live features may differ from synthetic training data",
                        "concept_drift: market regime changes may degrade model",
                    ],
                    notes=(
                        "Phase 9 gate plumbing self-check — throwaway artifact; real "
                        "registration happens on a genuine training run, never from this gate"
                    ),
                )
                # write_db=False: the gate must not register synthetic self-check
                # models into the production ml_model_registry (audit C4-adjacent).
                registry.register(model, artifact, write_db=False)
                registered.append(model.model_id)

        out.append(
            Criterion.ok(
                "ml_registry_records_plumbing",
                f"{len(registered)} model artifacts round-tripped through a throwaway "
                f"registry dir (no production rows written): {registered}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_registry_records_plumbing", f"registry raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 7. PLUMBING: leakage check — noise dataset → ≈0 improvement          #
    # ------------------------------------------------------------------ #
    try:
        # Noise = the reference dataset with the ENTIRE outcome tuple
        # (label, realized_pnl, hold_bars) shuffled across candidates — see
        # _noise_shuffle_delta for why shuffling the label alone (the pre-fix
        # construction) produced a deterministic false positive. The mean over
        # several deterministic shuffles is used because leakage is systematic
        # and survives averaging; single-split luck does not.
        noise_deltas = [
            _noise_shuffle_delta(ml_cfg, settings, noise_seed)
            for noise_seed in _LEAKAGE_NOISE_SEEDS
        ]
        noise_delta = sum(noise_deltas) / len(noise_deltas)
        # Leakage only manifests as positive improvement on shuffled outcomes — a model that
        # "improves" expectancy on pure noise is reading something it shouldn't. A negative
        # delta means the model's filtering was uninformative on noise (expected, not leakage).
        leakage_threshold = 0.10
        leakage_ok = noise_delta <= leakage_threshold
        per_seed = ", ".join(f"{d:+.4f}" for d in noise_deltas)
        out.append(
            Criterion.ok(
                "ml_leakage_check_plumbing",
                f"mean noise improvement={noise_delta:.4f}R over {len(noise_deltas)} shuffles "
                f"[{per_seed}] (≤ {leakage_threshold:.2f} → no leakage)",
            )
            if leakage_ok
            else Criterion.fail(
                "ml_leakage_check_plumbing",
                f"mean noise improvement={noise_delta:.4f}R over {len(noise_deltas)} shuffles "
                f"[{per_seed}] exceeds {leakage_threshold:.2f} → possible leakage "
                "(model reads outcome information it should not have)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_leakage_check_plumbing", f"leakage check raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 8. PLUMBING: offline scorer executes on the synthetic test split     #
    # ------------------------------------------------------------------ #
    synthetic_score = None
    try:
        test_result = predictor.run(
            [s.candidate for s in test_samples],
            settings=settings,
            write_to_db=False,
        )
        test_preds = [b.meta_label.label if b.meta_label else 1 for b in test_result.bundles]
        kc = ml_cfg.kill_criteria
        scorer = ShadowScorer(
            min_improvement=kc.min_improvement_over_baseline,
            min_pf_ratio=kc.min_profit_factor_ratio,
            max_tail_loss_ratio=kc.max_tail_loss_ratio,
            max_best_removed_pct=kc.max_best_trades_removed_pct,
        )
        synthetic_score = scorer.score(test_samples, test_preds)
        # Executes-and-produces-metrics only: the reference dataset is engineered so the
        # model wins, so its verdict is NOT the gate's performance verdict (audit H15).
        plumbing_ok = synthetic_score.n_test == len(test_samples)
        out.append(
            Criterion.ok(
                "ml_synthetic_scoring_plumbing",
                f"offline scorer executed on synthetic test split (n={synthetic_score.n_test}); "
                "verdict intentionally ignored — performance criteria use REAL shadow outcomes",
            )
            if plumbing_ok
            else Criterion.fail(
                "ml_synthetic_scoring_plumbing",
                f"scorer returned n_test={synthetic_score.n_test} for "
                f"{len(test_samples)} samples",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(Criterion.fail("ml_synthetic_scoring_plumbing", f"scoring raised: {exc}"))

    # ------------------------------------------------------------------ #
    # 9. REAL performance criteria — fail closed without real data         #
    # ------------------------------------------------------------------ #
    real_score = None
    try:
        n_real, linked = _load_real_shadow_outcomes()
        real_criteria, real_score = _real_performance_criteria(n_real, linked, ml_cfg)
        out.extend(real_criteria)
    except Exception as exc:  # noqa: BLE001
        out.extend(
            Criterion.fail(name, f"real-outcome scoring raised: {exc}")
            for name in _PERFORMANCE_CRITERIA
        )

    # ------------------------------------------------------------------ #
    # Cleanup: remove the plumbing shadow-log rows written above           #
    # ------------------------------------------------------------------ #
    if plumbing_log_ids:
        try:
            from src.db.base import session_scope
            from src.db.models import ShadowLog

            with session_scope() as session:
                session.query(ShadowLog).filter(ShadowLog.id.in_(plumbing_log_ids)).delete(
                    synchronize_session=False
                )
        except Exception:  # noqa: BLE001 - rows stay tagged synthetic → still excluded
            pass

    # Write a human-readable scoring report.
    import contextlib

    with contextlib.suppress(Exception):
        _write_ml_report(settings, ml_cfg, all_metrics, synthetic_score, real_score)

    return out


def _write_ml_report(
    settings: Settings,
    ml_cfg: MLConfig,
    train_metrics: dict,
    synthetic_score: ShadowScorerResult | None,
    real_score: ShadowScorerResult | None,
) -> None:
    import json
    from datetime import datetime

    reports_dir = settings.reports_path / "ml_shadow"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"ml_shadow_{stamp}.json"
    payload = {
        "model_version": ml_cfg.model_version,
        "ml_stage": ml_cfg.ml_stage,
        "train_metrics": train_metrics,
        "synthetic_scoring_plumbing": synthetic_score.to_dict() if synthetic_score else None,
        "real_shadow_scoring": real_score.to_dict() if real_score else None,
        "versions": settings.versions(),
        "generated_at": stamp,
        "note": (
            "Phase 9 ML-PROMO gate. Plumbing checks run on the synthetic reference dataset "
            "and prove pipeline execution only. The performance verdict uses REAL shadow "
            "decisions joined to realized paper-trade outcomes; real_shadow_scoring=null "
            "means no/insufficient real data (gate fails closed). "
            "Manual review required before promotion to Stage 3+."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
