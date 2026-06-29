"""Built-in job handlers (AGENTS.md Appendix B.7).

Phase 1 ships the gate-runner jobs, the data/universe skeleton jobs
(``sync_exchange_metadata``, ``build_symbol_universe``), backup/restore jobs,
and a few ``selftest_*`` handlers the QUEUE gate uses to prove the queue works.
Heavy research/ML/RL jobs are added in their phases.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from src.config.settings import REPO_ROOT
from src.db.base import session_scope
from src.db.models import ExchangeMetadata, VerificationStatus
from src.exchange import get_adapter
from src.jobs.context import JobContext
from src.jobs.registry import job_handler

_registered = False


def ensure_handlers_registered() -> None:
    """Importing this module registers all handlers; call to be explicit."""
    global _registered
    _registered = True


# --------------------------------------------------------------------------- #
# Self-test handlers (used by the QUEUE gate)                                  #
# --------------------------------------------------------------------------- #
@job_handler("selftest_echo")
def _selftest_echo(ctx: JobContext, params: dict) -> dict:
    steps = int(params.get("steps", 1))
    ctx.log("selftest_echo starting")
    for i in range(steps):
        ctx.check_cancelled()
        ctx.progress(i + 1, steps, f"step {i + 1}/{steps}")
    ctx.log("selftest_echo done")
    return {"message": f"echoed {steps} steps", "steps": steps}


@job_handler("selftest_fail")
def _selftest_fail(ctx: JobContext, params: dict) -> dict:
    ctx.log("selftest_fail will raise", level="WARNING")
    raise RuntimeError("intentional failure for retry/failure-visibility test")


# --------------------------------------------------------------------------- #
# Data / universe skeleton jobs                                               #
# --------------------------------------------------------------------------- #
@job_handler("sync_exchange_metadata")
def _sync_exchange_metadata(ctx: JobContext, params: dict) -> dict:
    """Skeleton metadata sync: fetch symbols from the adapter and persist
    placeholder, ``[UNVERIFIED]`` metadata snapshots (Section 6)."""
    adapter = get_adapter(params.get("exchange_id"))
    symbols = adapter.fetch_symbols()
    ctx.log(f"syncing metadata for {len(symbols)} symbols from {adapter.exchange_id}")
    version = params.get("metadata_version", "meta_0001")
    with session_scope() as session:
        for i, symbol in enumerate(symbols):
            ctx.check_cancelled()
            meta = adapter.fetch_metadata(symbol)
            session.add(
                ExchangeMetadata(
                    exchange_id=adapter.exchange_id,
                    symbol=symbol,
                    metadata_version=version,
                    verification_status=VerificationStatus.UNVERIFIED,
                    source="skeleton",
                    fetched_at=datetime.now(UTC),
                    raw=meta.raw,
                )
            )
            ctx.progress(i + 1, len(symbols), f"synced {symbol}")
    return {"message": f"synced {len(symbols)} symbols (UNVERIFIED)", "symbols": symbols}


@job_handler("verify_exchange_metadata")
def _verify_exchange_metadata(ctx: JobContext, params: dict) -> dict:
    """Apply operator-verified metadata from ``configs/metadata.yaml`` (Section 6
    verification workflow step 3): persist ``[VERIFIED]`` metadata rows the META
    gate reads. Idempotent."""
    from src.exchange import load_metadata_config, sync_verified_metadata

    cfg = load_metadata_config()
    ctx.log(
        f"applying [VERIFIED] metadata for {len(cfg.symbols())} symbols ({cfg.metadata_version})"
    )
    with session_scope() as session:
        written = sync_verified_metadata(session, cfg)
    ctx.progress(1, 1, f"{written} verified")
    return {
        "message": f"verified {written} symbols ({cfg.metadata_version})",
        "symbols": cfg.symbols(),
    }


@job_handler("build_symbol_universe")
def _build_symbol_universe(ctx: JobContext, params: dict) -> dict:
    """Build a versioned universe: filter every candidate (Section 9) and promote
    passing symbols to ``active``, logging entering/leaving symbols."""
    from src.universe import UniverseManager

    ctx.log("building dynamic universe (Phase 3 filters)")
    with session_scope() as session:
        result = UniverseManager().build(session)
        version = result.version
        active = result.active_symbols
        changes = len(result.changes)
    ctx.progress(1, 1, f"universe {version}: {len(active)} active, {changes} changes")
    return {
        "message": f"built universe {version} ({len(active)} active)",
        "version": version,
        "active": active,
        "changes": changes,
    }


@job_handler("build_feature_store")
def _build_feature_store(ctx: JobContext, params: dict) -> dict:
    """Build the immutable feature store from the current dataset snapshot for the
    active universe through the single feature code path (Section 10)."""
    from src.data import DataPlatform, load_data_config
    from src.features import FeatureStore
    from src.universe import UniverseManager, latest_active_symbols

    ctx.log("ensuring data snapshot + building feature store")
    platform = DataPlatform(cfg=load_data_config())
    run = platform.run_full(repair=True, source_jobs=["job:build_feature_store"])
    with session_scope() as session:
        uni = UniverseManager().build(session)
        active = latest_active_symbols(session) or uni.active_symbols
        build = FeatureStore().build(
            active,
            run.snapshot.snapshot_id,
            universe_version=uni.version,
            session=session,
            source_jobs=["job:build_feature_store"],
        )
        snapshot_id = build.feature_snapshot_id
        rows = build.total_rows
    ctx.progress(1, 1, f"features {snapshot_id} ({rows} rows)")
    return {
        "message": f"built features {snapshot_id} ({rows} rows)",
        "feature_snapshot_id": snapshot_id,
        "checksum": build.checksum,
        "rows": rows,
    }


@job_handler("run_feature_leakage_test")
def _run_feature_leakage_test(ctx: JobContext, params: dict) -> dict:
    """Run the synthetic-data leakage test (Section 16 / FEAT gate)."""
    from src.features import load_feature_config, synthetic_leakage_report

    ctx.log("running synthetic-data leakage test")
    report = synthetic_leakage_report(load_feature_config())
    ctx.progress(
        1, 1, f"leakage {'PASS' if report['passed'] else 'FAIL'} (|z|={abs(report['z']):.2f})"
    )
    return {"message": f"leakage {'PASS' if report['passed'] else 'FAIL'}", **report}


# --------------------------------------------------------------------------- #
# Data platform jobs (Appendix B.7 data jobs; Phase 2)                        #
# --------------------------------------------------------------------------- #
def _download_series(ctx: JobContext, data_types: list[str]) -> dict:
    """Shared body for the per-series historical download jobs.

    ``config_path`` (job param) selects the dataset config — e.g. ``configs/data.bybit.yaml``
    for REAL Bybit history (the ccxt source); absent → the default config."""
    from src.data import DataPlatform, load_data_config

    cfg_path = str(ctx.params.get("config_path")) if ctx.params.get("config_path") else None
    cfg = load_data_config(cfg_path)
    platform = DataPlatform(cfg=cfg)
    keys = [k for k in cfg.all_required_keys() if k.data_type in data_types]
    written = 0
    for i, key in enumerate(keys):
        ctx.check_cancelled()
        written += platform.download(key)
        ctx.progress(i + 1, len(keys), f"downloaded {key.label()}")
    ctx.log(f"downloaded {written} rows across {len(keys)} series ({', '.join(data_types)})")
    return {"message": f"downloaded {written} rows", "series": len(keys), "rows": written}


@job_handler("download_ohlcv_history")
def _download_ohlcv_history(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["ohlcv"])


@job_handler("download_mark_index_history")
def _download_mark_index_history(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["mark", "index"])


@job_handler("download_funding_history")
def _download_funding_history(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["funding"])


@job_handler("download_open_interest_history")
def _download_open_interest_history(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["open_interest"])


@job_handler("download_spread_snapshots")
def _download_spread_snapshots(ctx: JobContext, params: dict) -> dict:
    return _download_series(ctx, ["spread"])


@job_handler("download_liquidation_history")
def _download_liquidation_history(ctx: JobContext, params: dict) -> dict:
    """Liquidation data is "if available" (Section 8). The offline source does
    not provide it; the job records that it is unavailable rather than failing,
    so liquidation is simply not a required series this phase."""
    ctx.log("liquidation history unavailable for the skeleton source; skipping")
    ctx.progress(1, 1, "liquidation unavailable")
    return {"message": "liquidation history unavailable (not required this phase)", "rows": 0}


@job_handler("repair_missing_data")
def _repair_missing_data(ctx: JobContext, params: dict) -> dict:
    """Detect gaps and backfill only the missing ranges (safe gap repair)."""
    from src.data import DataPlatform, load_data_config

    cfg = load_data_config(str(params.get("config_path")) if params.get("config_path") else None)
    platform = DataPlatform(cfg=cfg)
    keys = cfg.all_required_keys()
    written = 0
    remaining = 0
    for i, key in enumerate(keys):
        ctx.check_cancelled()
        result = platform.ingestor.repair(key, cfg.window_start_ms, cfg.window_end_ms)
        written += result.rows_written
        remaining += result.gaps_after
        ctx.progress(i + 1, len(keys), f"repaired {key.label()}")
    ctx.log(f"repaired {written} rows; {remaining} timestamps still missing")
    return {"message": f"repaired {written} rows", "rows": written, "still_missing": remaining}


@job_handler("validate_data_quality")
def _validate_data_quality(ctx: JobContext, params: dict) -> dict:
    """Run the Section 23 data-quality checks and persist the report."""
    from src.data import DataPlatform, load_data_config

    cfg = load_data_config(str(params.get("config_path")) if params.get("config_path") else None)
    platform = DataPlatform(cfg=cfg)
    ctx.log("validating data quality")
    report = platform.validate()
    path = platform.write_quality_report(report, params.get("dataset_version"))
    ctx.progress(1, 1, f"data quality: {'PASS' if report.passed else 'FAIL'}")
    return {
        "message": f"data quality {'PASS' if report.passed else 'FAIL'}",
        "passed": report.passed,
        "critical": len(report.critical),
        "artifact_uri": path,
    }


@job_handler("build_dataset_version")
def _build_dataset_version(ctx: JobContext, params: dict) -> dict:
    """Download required series, validate, and produce an immutable dataset snapshot.

    Diagnoses the common silent failure: if the exchange is unreachable from the container (or
    the configured symbols don't exist there), the source treats every symbol as "no history",
    nothing downloads, and the snapshot is instantly INVALID. We now preflight reachability and
    fail with a clear, actionable message instead, stream per-series download progress, and log
    the concrete coverage/validation reasons when a snapshot is invalid."""
    from src.data import DataPlatform, load_data_config

    cfg_path = str(params.get("config_path")) if params.get("config_path") else None
    # Load ONCE: a config with `as_of: now` re-resolves the window on every load, so two loads
    # can land an hour apart — the download window would then differ from the validated/snapshotted
    # one (spurious trailing-gap failures, non-reproducible snapshot id at hour boundaries).
    cfg = load_data_config(cfg_path, as_of_ms=params.get("as_of_ms"))
    platform = DataPlatform(cfg=cfg)
    syms = cfg.active_symbols()
    # Surface the FROZEN window end so an operator can reproduce this exact snapshot later: pass
    # the same as_of_ms (or set as_of to this ISO in the data yaml). Under `as_of: now` the window
    # advances hourly, so without pinning a re-run yields a different snapshot id by design.
    from src.data.schema import ms_to_iso

    ctx.log(
        f"window frozen at as_of={ms_to_iso(cfg.window_end_ms)} (as_of_ms={cfg.window_end_ms}) — "
        "pin this for a reproducible re-snapshot"
    )

    # --- preflight: is the exchange reachable and do the symbols exist? --------------------- #
    ctx.log(
        f"checking {cfg.exchange_id} reachability for {len(syms)} symbols "
        f"({cfg_path or 'default'})"
    )
    available = [s for s in syms if platform.source.has_symbol(s)]
    if not available:
        raise RuntimeError(
            f"could not reach {cfg.exchange_id} (or none of {syms} exist there) — nothing was "
            "downloaded. Public market data needs no API keys, so this is almost always the "
            "container's outbound network: ensure it can reach the exchange API "
            "(e.g. api.bybit.com), or that the symbols in the config are valid. Fix that and "
            "re-run; no snapshot was created."
        )
    if len(available) < len(syms):
        ctx.log(
            f"only {len(available)}/{len(syms)} symbols available: {available} "
            f"(missing: {sorted(set(syms) - set(available))})",
            level="WARNING",
        )

    # --- download with progress (INCREMENTAL by default: only candles that appeared since the
    # last download; an empty store fetches the full window. ``full=true`` forces a re-fetch). --- #
    full = bool(params.get("full"))
    keys = [k for k in cfg.all_required_keys() if k.symbol in available]
    # ONE continuous progress bar across both stages: download (len(keys) steps) then input-build
    # (one step per symbol per prebuild timeframe). The message names the current stage so the chip
    # reads correctly throughout.
    prebuild_tfs = cfg.prebuild_timeframes
    n_syms = len(cfg.active_symbols())
    total = len(keys) + len(prebuild_tfs) * n_syms
    written = 0
    mode = "full" if full else "incremental (new candles since last download)"
    ctx.log(f"downloading {len(keys)} series — {mode}")
    for i, key in enumerate(keys):
        ctx.check_cancelled()
        written += platform.download(key) if full else platform.update_incremental(key)
        ctx.progress(i + 1, total, f"downloading {key.label()}: {written} new rows")
    ctx.log(f"downloaded {written} NEW rows across {len(keys)} series ({mode})")

    # --- validate + snapshot, with concrete diagnostics on failure -------------------------- #
    ctx.progress(len(keys), total, "validating coverage + building snapshot")
    run = platform.run_full(repair=True, source_jobs=["job:build_dataset_version"])
    if not run.coverage.covered:
        gaps = [g.key.label() for g in run.coverage.uncovered][:8]
        ctx.log(
            f"coverage INCOMPLETE — {len(run.coverage.uncovered)} series with gaps in the window "
            f"({cfg.active_symbols()} over {run.coverage.window}); e.g. {gaps}. Open-interest at "
            "5m has only ~16h retention on Bybit — keep the window recent / use a coarser OI grid.",
            level="WARNING",
        )
    if not run.validation.passed:
        crit = [f"{v.check}:{v.series}:{v.detail}" for v in run.validation.critical][:5]
        ctx.log(f"validation FAILED — {len(run.validation.critical)} critical issue(s): {crit}",
                level="WARNING")
    status = "VALID" if (run.coverage.covered and run.validation.passed) else "INVALID"
    ctx.log(f"snapshot {run.snapshot.snapshot_id}: {status}")

    # --- pre-build the engine inputs NOW (at download time), so validation/backtests load them
    # instantly instead of rebuilding (~hours on 4h, days on 5m for a 20-symbol universe).
    # Idempotent: an unchanged snapshot yields instant cache hits, so an incremental re-download
    # only rebuilds the timeframes whose data actually changed. Cancellable mid-build. --- #
    built_shapes: dict[str, int] = {}
    if available and prebuild_tfs:
        from src.backtest.service import prewarm_input_cache
        from src.config import get_settings
        from src.data.store import SeriesStore

        store = SeriesStore(get_settings().data_lake_path)
        tf_pos = {tf: j for j, tf in enumerate(prebuild_tfs)}

        def _build_progress(tf: str, done: int, tot: int, sym: str) -> None:
            ctx.check_cancelled()
            base = len(keys) + tf_pos[tf] * n_syms
            ctx.progress(base + done, total, f"building inputs ({tf}): {sym} ({done}/{tot})")

        ctx.log(f"building inputs for {prebuild_tfs} so validation loads them instantly")
        built_shapes = prewarm_input_cache(
            cfg, store, log=lambda m: ctx.log(m), progress=_build_progress
        )
        ctx.log(f"inputs cached: {built_shapes}")
    ctx.progress(total, total, f"ready — snapshot {run.snapshot.snapshot_id}: {status}")
    return {
        "message": f"dataset {run.snapshot.snapshot_id}: {status} "
        f"(rows={written}, covered={run.coverage.covered}, validated={run.validation.passed}; "
        f"inputs cached for {sorted(built_shapes)})",
        "dataset_version": run.snapshot.snapshot_id,
        "rows_downloaded": written,
        "symbols_available": available,
        "covered": run.coverage.covered,
        "validation_passed": run.validation.passed,
        "inputs_cached": built_shapes,
        "artifact_uri": run.report_path,
    }


# --------------------------------------------------------------------------- #
# Gate runner jobs                                                            #
# --------------------------------------------------------------------------- #
@job_handler("run_gate")
def _run_gate(ctx: JobContext, params: dict) -> dict:
    from src.gates import GateRunner

    gate_id = params["gate_id"]
    ctx.log(f"running gate {gate_id}")
    result = GateRunner().run(gate_id)
    ctx.progress(1, 1, f"{gate_id}: {result.overall}")
    return {"message": f"{gate_id}: {result.overall}", "artifact_uri": result.report_path}


@job_handler("run_all_gates")
def _run_all_gates(ctx: JobContext, params: dict) -> dict:
    from src.gates import GateRunner

    ctx.log("running all gates in dependency order")
    results = GateRunner().run_all()
    summary = {r.gate_id: r.overall for r in results}
    ctx.progress(len(results), len(results), "all gates evaluated")
    return {"message": "ran all gates", "summary": summary}


# --------------------------------------------------------------------------- #
# Backup / restore jobs (skeleton; full BACKUP gate in Phase 13)              #
# --------------------------------------------------------------------------- #
@job_handler("run_backup_check")
def _run_backup_check(ctx: JobContext, params: dict) -> dict:
    script = REPO_ROOT / "scripts" / "backup_db.sh"
    ctx.log(f"running backup script {script}")
    proc = subprocess.run(  # noqa: S603
        ["bash", str(script)], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    ctx.log(proc.stdout[-2000:] or "(no stdout)")
    if proc.returncode != 0:
        raise RuntimeError(f"backup failed: {proc.stderr[-500:]}")
    return {"message": "backup completed"}


@job_handler("run_restore_test_check")
def _run_restore_test_check(ctx: JobContext, params: dict) -> dict:
    script = REPO_ROOT / "scripts" / "restore_test.sh"
    ctx.log(f"running restore-test script {script}")
    proc = subprocess.run(  # noqa: S603
        ["bash", str(script)], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    ctx.log(proc.stdout[-2000:] or "(no stdout)")
    if proc.returncode != 0:
        raise RuntimeError(f"restore test failed: {proc.stderr[-500:]}")
    return {"message": "restore test passed"}


# --------------------------------------------------------------------------- #
# ML shadow jobs (Phase 9 — shadow mode only; no live influence)              #
# --------------------------------------------------------------------------- #
@job_handler("build_ml_dataset")
def _build_ml_dataset(ctx: JobContext, params: dict) -> dict:
    """Build a labeled dataset for meta-labeler training from paper trade outcomes.

    Phase 9 uses the deterministic synthetic reference dataset; once sufficient
    paper trade outcomes accumulate (Phase 10+), this job builds from the real
    shadow_log + paper_trades tables.
    """
    from src.ml.labels import build_reference_dataset

    n_good = int(params.get("n_good", 40))
    n_bad = int(params.get("n_bad", 30))
    n_neutral = int(params.get("n_neutral", 30))
    seed = int(params.get("seed", 42))

    ctx.log(f"building ML dataset: n_good={n_good} n_bad={n_bad} n_neutral={n_neutral}")
    samples = build_reference_dataset(n_good=n_good, n_bad=n_bad, n_neutral=n_neutral, seed=seed)
    ctx.progress(1, 1, f"{len(samples)} labeled samples")
    return {"message": f"built ML dataset: {len(samples)} samples", "n_samples": len(samples)}


@job_handler("train_ml_models")
def _train_ml_models(ctx: JobContext, params: dict) -> dict:
    """Train all five shadow ML models on the reference dataset.

    Shadow mode only — the trained models are stored in the artifact registry
    but never influence live trading decisions.
    """
    from src.ml import ShadowPredictor
    from src.ml.config import load_ml_config
    from src.ml.labels import build_reference_dataset, train_test_split

    ctx.log("loading ML config and building reference dataset")
    ml_cfg = load_ml_config()
    samples = build_reference_dataset(seed=42)
    train_samples, _ = train_test_split(samples, seed=42)

    ctx.log(f"training 5 shadow models on {len(train_samples)} samples")
    predictor = ShadowPredictor.from_config(ml_cfg)
    metrics = predictor.train(train_samples)
    ctx.progress(1, 1, "5 models trained")
    ctx.log(f"train metrics: {metrics}")
    return {
        "message": "5 shadow ML models trained (shadow mode; no live influence)",
        "model_version": ml_cfg.model_version,
        "ml_stage": ml_cfg.ml_stage,
        "metrics": metrics,
    }


@job_handler("evaluate_ml_models")
def _evaluate_ml_models(ctx: JobContext, params: dict) -> dict:
    """Evaluate trained shadow models against the ML-PROMO kill criteria.

    Produces a scoring report but does NOT promote models — promotion requires
    the ML-PROMO gate PASS and manual review (Section 20).
    """
    from src.config import get_settings
    from src.ml import ShadowPredictor, ShadowScorer
    from src.ml.config import load_ml_config
    from src.ml.labels import build_reference_dataset, train_test_split

    ctx.log("evaluating shadow ML models against ML-PROMO kill criteria")
    ml_cfg = load_ml_config()
    settings = get_settings()
    samples = build_reference_dataset(seed=42)
    train_samples, test_samples = train_test_split(samples, seed=42)

    predictor = ShadowPredictor.from_config(ml_cfg)
    predictor.train(train_samples)
    test_result = predictor.run(
        [s.candidate for s in test_samples], settings=settings, write_to_db=False
    )
    test_preds = [b.meta_label.label if b.meta_label else 1 for b in test_result.bundles]

    kc = ml_cfg.kill_criteria
    scorer = ShadowScorer(
        min_improvement=kc.min_improvement_over_baseline,
        min_pf_ratio=kc.min_profit_factor_ratio,
        max_tail_loss_ratio=kc.max_tail_loss_ratio,
        max_best_removed_pct=kc.max_best_trades_removed_pct,
    )
    score = scorer.score(test_samples, test_preds)
    ctx.progress(1, 1, f"scoring: passed={score.passed}")
    ctx.log(f"expectancy improvement: {score.expectancy_improvement:+.4f}R")
    return {
        "message": f"ML evaluation: {'PASS' if score.passed else 'FAIL'}",
        "passed": score.passed,
        "scoring": score.to_dict(),
        "fail_reasons": score.fail_reasons,
        "note": "promotion requires ML-PROMO gate PASS + manual_reviewed=True in registry",
    }


@job_handler("run_ml_shadow_pass")
def _run_ml_shadow_pass(ctx: JobContext, params: dict) -> dict:
    """Run the shadow ML predictor over a set of candidates and log results.

    Phase 9: runs over the reference dataset to prove the logging pipeline.
    Phase 10+: wired into the paper-trading loop for every candidate batch.
    """
    from src.config import get_settings
    from src.ml import ShadowPredictor
    from src.ml.config import load_ml_config
    from src.ml.labels import build_reference_dataset, train_test_split

    ctx.log("running ML shadow pass (shadow mode; applied=False on all log entries)")
    ml_cfg = load_ml_config()
    settings = get_settings()
    samples = build_reference_dataset(seed=42)
    train_samples, test_samples = train_test_split(samples, seed=42)

    predictor = ShadowPredictor.from_config(ml_cfg)
    predictor.train(train_samples)
    result = predictor.run(
        [s.candidate for s in test_samples[:20]],
        settings=settings,
        write_to_db=True,
    )
    ctx.progress(1, 1, f"{len(result.shadow_log_ids)} shadow log entries written")
    assert not result.applied, "shadow pass must never set applied=True"
    return {
        "message": f"shadow pass: {len(result.shadow_log_ids)} entries logged (applied=False)",
        "model_version": result.model_version,
        "shadow_log_ids": len(result.shadow_log_ids),
        "applied": result.applied,
    }


# --------------------------------------------------------------------------- #
# ML Phase 10 jobs — Recommendation + Constrained Filter                     #
# --------------------------------------------------------------------------- #
@job_handler("run_ml_recommendation_pass")
def _run_ml_recommendation_pass(ctx: JobContext, params: dict) -> dict:
    """Run Stage 3 Recommendation Mode over the reference dataset.

    Produces structured MLRecommendation objects and logs them to shadow_logs
    with mode=RECOMMEND, applied=False. Never affects trading behavior.
    """
    from src.config import get_settings
    from src.ml import RecommendationEngine
    from src.ml.config import load_ml_config
    from src.ml.labels import build_reference_dataset, train_test_split
    from src.ml.shadow import ShadowPredictor

    ctx.log("running ML Stage 3 recommendation pass (applied=False on all entries)")
    ml_cfg = load_ml_config()
    settings = get_settings()
    samples = build_reference_dataset(seed=42)
    train_samples, test_samples = train_test_split(samples, seed=42)

    predictor = ShadowPredictor.from_config(ml_cfg)
    predictor.train(train_samples)
    shadow_result = predictor.run(
        [s.candidate for s in test_samples[:20]],
        settings=settings,
        write_to_db=False,
    )

    engine = RecommendationEngine(
        model_version=ml_cfg.model_version,
        config_version=settings.config_version,
    )
    result = engine.run(shadow_result.bundles, write_to_db=True)
    assert not result.applied, "recommendation pass must never set applied=True"

    n_take = sum(1 for r in result.recommendations if r.recommend_take)
    n_skip = len(result.recommendations) - n_take
    ctx.progress(
        1, 1, f"{len(result.recommendations)} recommendations (take={n_take} skip={n_skip})"
    )
    return {
        "message": (
            f"Stage 3: {len(result.recommendations)} recommendations logged (applied=False)"
        ),
        "model_version": result.model_version,
        "n_recommendations": len(result.recommendations),
        "n_take": n_take,
        "n_skip": n_skip,
        "log_ids": len(result.log_ids),
        "applied": result.applied,
    }


@job_handler("run_ml_filter_evaluation")
def _run_ml_filter_evaluation(ctx: JobContext, params: dict) -> dict:
    """Evaluate the Stage 4 Constrained Live Filter on the reference dataset.

    Shows how many candidates would be blocked by the filter. This job is
    for evaluation — in production the filter runs inline in the paper/live
    loop after each candidate batch.
    """
    from src.config import get_settings
    from src.ml import MLFilter
    from src.ml.config import load_ml_config
    from src.ml.labels import build_reference_dataset, train_test_split
    from src.ml.shadow import ShadowPredictor

    ctx.log("evaluating ML Stage 4 constrained filter (evaluation only)")
    ml_cfg = load_ml_config()
    settings = get_settings()
    samples = build_reference_dataset(seed=42)
    train_samples, test_samples = train_test_split(samples, seed=42)

    predictor = ShadowPredictor.from_config(ml_cfg)
    predictor.train(train_samples)
    candidates = [s.candidate for s in test_samples]
    shadow_result = predictor.run(candidates, settings=settings, write_to_db=False)

    threshold = float(params.get("min_confidence_to_take", ml_cfg.filter.min_confidence_to_take))
    ml_filter = MLFilter(
        min_confidence_to_take=threshold,
        model_version=ml_cfg.model_version,
        config_version=settings.config_version,
    )
    result = ml_filter.apply(candidates, shadow_result.bundles, write_to_db=True)

    ctx.progress(
        1,
        1,
        f"filter: {result.pass_count} passed / {result.block_count} blocked "
        f"out of {len(candidates)}",
    )
    return {
        "message": (f"Stage 4 filter: {result.pass_count} passed / {result.block_count} blocked"),
        "total": len(candidates),
        "passed": result.pass_count,
        "blocked": result.block_count,
        "threshold": threshold,
        "block_reasons": result.block_reasons(),
        "note": "filter CANNOT create trades, increase risk, or override hard blockers",
    }


# --------------------------------------------------------------------------- #
# Research: validate strategy candidates → persist promote/shelve verdicts     #
# --------------------------------------------------------------------------- #
@job_handler("run_strategy_validation")
def _run_strategy_validation(ctx: JobContext, params: dict) -> dict:
    """Run the research harness over all enabled candidates and persist each promote/shelve
    verdict to the strategy_promotions registry, so paper/live can source only promoted ones."""
    from src.strategies.promotion import persist_validations
    from src.strategies.research import validate_all

    ctx.log("validating strategy candidates (backtest + walk-forward + stress + noise)")
    ctx.progress(0, 1, "validating candidates")
    validations = validate_all()
    written = persist_validations(validations)
    promoted = [v.candidate_id for v in validations if v.promoted]
    ctx.progress(1, 1, f"{len(promoted)}/{written} promoted")
    return {
        "message": f"{len(promoted)}/{written} candidates promoted; verdicts persisted",
        "promoted": promoted,
        "shelved": [v.candidate_id for v in validations if not v.promoted],
        "total": written,
    }


@job_handler("run_lake_strategy_validation")
def _run_lake_strategy_validation(ctx: JobContext, params: dict) -> dict:
    """Validate candidates on REAL downloaded data and persist promote/shelve verdicts.

    The real-data twin of run_strategy_validation: backtest + side decision + walk-forward +
    fee/slippage stress over a downloaded snapshot (the live market is the control, so the
    synthetic noise step is dropped). Promotions are tagged data_source='lake'. Requires the
    snapshot to be downloaded first (Data page → Download real history)."""
    from src.data.config import load_data_config
    from src.strategies.lake_research import validate_all_on_lake
    from src.strategies.promotion import persist_validations

    config_path = str(params.get("config_path") or "configs/data.bybit.yaml")
    data_cfg = load_data_config(config_path)
    timeframe = params.get("timeframe") or None
    tf = timeframe or data_cfg.base_timeframe
    ctx.log(
        f"validating candidates on REAL data ({data_cfg.exchange_id}/{data_cfg.data_version}) "
        f"on timeframe {tf}"
    )
    ctx.progress(0, 1, "running real-data validation (backtest + walk-forward + stress)")
    # Stream per-stage/per-candidate progress into the job log + progress bar so an hours-long
    # run is observable instead of a silent CPU spin (the operator asked for this explicitly).
    validations = validate_all_on_lake(
        data_cfg,
        timeframe=timeframe,
        emit=lambda msg: ctx.log(msg),
        progress=lambda done, total, msg: ctx.progress(done, max(total, 1), msg),
    )
    written = persist_validations(validations, data_source="lake")
    promoted = [v.candidate_id for v in validations if v.promoted]
    # Surface every verdict in the log — a shelve is a normal result, not a failure, and the
    # operator needs to SEE why nothing got promoted (e.g. "insufficient trades on real data").
    for v in validations:
        if v.promoted:
            ctx.log(f"PROMOTED {v.candidate_id} on real data")
        else:
            why = "; ".join(v.shelved_reasons) or "did not clear gates"
            ctx.log(f"SHELVED {v.candidate_id}: {why}", level="WARNING")
    if not promoted:
        ctx.log(
            f"0/{written} promoted on real data — the candidates did not clear the gates on this "
            "snapshot/timeframe (often too few trades or no real edge). Try more history (the "
            "window is now ~5.5y once re-downloaded) or a different timeframe; tune candidates in "
            "configs/strategies.yaml.",
            level="WARNING",
        )
    ctx.progress(1, 1, f"{len(promoted)}/{written} promoted on real data")
    return {
        "message": f"real-data validation: {len(promoted)}/{written} promoted",
        "promoted": promoted,
        "shelved": [
            {"id": v.candidate_id, "reasons": v.shelved_reasons}
            for v in validations
            if not v.promoted
        ],
        "data_source": "lake",
    }


# --------------------------------------------------------------------------- #
# Paper trading: run a session over PROMOTED strategies and persist it         #
# --------------------------------------------------------------------------- #
@job_handler("run_paper_session")
def _run_paper_session(ctx: JobContext, params: dict) -> dict:
    """Run a paper-trading session over candidates sourced from the promoted-strategy registry,
    then persist the trades + a run summary the dashboard Paper page reads."""
    from src.paper.run import run_paper_session

    name = str(params.get("session_name") or "dashboard_paper")
    ctx.log(f"running paper session over promoted strategies: {name}")
    ctx.progress(0, 1, "running paper pipeline")
    session, report, session_id = run_paper_session(session_name=name)
    net = sum(t.pnl for t in session.trades)
    ctx.progress(1, 1, f"{session.executed_count} trades, net_pnl={net:.2f}")
    return {
        "message": f"paper session {session_id}: {session.executed_count} executed / "
        f"{session.rejected_count} rejected, net_pnl={net:.2f}",
        "session_id": session_id,
        "executed": session.executed_count,
        "rejected": session.rejected_count,
    }


# --------------------------------------------------------------------------- #
# Basket (cross-sectional) PAPER loop — dashboard-triggered, rides the `live`   #
# queue (a long-lived continuous loop). The per-symbol `run_live_session` path  #
# can't run a basket strategy (funding_carry / residual_momentum) through its   #
# engine, so this drives the CrossSectionalEngine in real time (Section 12/19). #
# --------------------------------------------------------------------------- #
@job_handler("run_basket_paper_session")
def _run_basket_paper_session(ctx: JobContext, params: dict) -> dict:
    """Run + persist a continuous cross-sectional (basket) PAPER session for ONE strategy.

    PAPER only — simulated fills, no real orders/funds. Sources data from the live REST feed and
    drives :func:`src.live.basket.run_basket_paper_session` (which reuses the offline-proven
    ``BasketPaperLoop``). The loop polls the job's cancel flag every tick, so the dashboard Stop
    button halts it cleanly; whatever booked before the stop is still persisted. Unlike
    ``run_live_session`` this does NOT require promotion — a basket strategy is named explicitly and
    paper is simulated, so an unpromoted-but-validated carry/momentum edge can be paper-traded."""
    from src.config import get_settings
    from src.data.config import load_data_config
    from src.live.basket import run_basket_paper_session

    settings = get_settings()
    strategy = str(params.get("strategy") or params.get("candidate_id") or "").strip()
    if not strategy:
        raise ValueError("run_basket_paper_session requires a 'strategy' (cross-sectional id)")
    config_path = str(params.get("config_path") or "configs/data.bybit.yaml")
    data_cfg = load_data_config(config_path)
    timeframe = params.get("timeframe") or None
    poll_sec = float(params.get("poll_sec") or 60.0)
    max_ticks = int(params["max_ticks"]) if params.get("max_ticks") else None

    ctx.log(
        f"starting basket paper session: strategy={strategy} "
        f"timeframe={timeframe or data_cfg.base_timeframe} poll_sec={poll_sec} "
        f"max_ticks={'continuous' if max_ticks is None else max_ticks}"
    )
    ctx.progress(0, max_ticks or 0, "starting basket paper loop (press Stop to end)")

    n_trades = run_basket_paper_session(
        strategy,
        data_cfg=data_cfg,
        timeframe=timeframe,
        poll_sec=poll_sec,
        max_ticks=max_ticks,
        settings=settings,
        # Stop on a dashboard Stop (cancel flag) OR if this run was superseded (another worker
        # now owns the requeued job) — so two basket loops never run at once.
        should_stop=lambda: ctx.is_cancelled() or not ctx.still_owns(),
        # Surface rebalances / skips to the job log and per-tick progress to the dashboard, so a
        # basket that never forms (e.g. too little history for the feature window) is VISIBLE.
        on_event=ctx.log,
        on_tick=lambda i, msg: ctx.progress(i, max_ticks or 0, msg),
    )
    ctx.progress(max_ticks or 0, max_ticks or 0, f"stopped: {n_trades} paper legs booked")
    return {
        "message": f"basket paper session ({strategy}): {n_trades} legs booked",
        "strategy": strategy,
        "trades": n_trades,
    }


# --------------------------------------------------------------------------- #
# Live / demo trading loop (dashboard-triggered; runs on the dedicated `live`  #
# queue). Lets the operator start, watch, stop, and restart a demo/testnet run #
# entirely from the dashboard — no terminal command required.                  #
# --------------------------------------------------------------------------- #
def _live_loop_mode(exchange_env: str, override: str | None) -> str:
    """Map the EXCHANGE_ENV to the live-loop venue mode (the venue selector, Section 18).

    ``demo`` and ``testnet`` both use the real ccxt venue with virtual funds (no activation
    guard); only ``live`` is real-money and stays fully guarded. ``paper`` forces the offline
    SimulatedVenue. An explicit ``mode`` param overrides the mapping."""
    if override in ("paper", "testnet", "live"):
        return override
    if exchange_env == "live":
        return "live"
    if exchange_env in ("demo", "testnet"):
        return "testnet"  # real ccxt venue; EXCHANGE_ENV routes the endpoint (demo vs testnet)
    return "paper"


@job_handler("run_live_session")
def _run_live_session(ctx: JobContext, params: dict) -> dict:
    """Run + persist a live/demo/testnet trading session driven from the dashboard.

    The venue is chosen from EXCHANGE_ENV (demo/testnet = virtual funds; live = guarded
    real money); the session is tagged with the environment prefix (``demo:``…) so its
    statistics stay separated. Progress is streamed per tick to the Job row and the loop
    polls the job's cancel flag every tick, so the dashboard Stop button halts it cleanly.
    Whatever executed before a stop/cancel/error is still persisted."""
    from src.config import get_settings
    from src.data.config import load_data_config
    from src.live.loop import persist_live_run, run_replay_session

    settings = get_settings()
    config_path = str(params.get("config_path") or "configs/data.bybit.yaml")
    data_cfg = load_data_config(config_path)
    mode = _live_loop_mode(settings.exchange_env, params.get("mode"))
    realtime = bool(params.get("realtime", True))
    transport = params.get("transport") or ("rest" if realtime else None)
    timeframe = params.get("timeframe") or None
    symbols = params.get("symbols") or None
    candidate_id = params.get("candidate_id") or params.get("strategy") or None
    # By default run the ACTIVE PROMOTED ENSEMBLE (top-N strategies) — demo behaves exactly like
    # live. An explicit candidate_id pins the run to one strategy instead.
    multi_strategy = candidate_id is None
    # Continuous by default: max_ticks unset → run until the operator presses Stop. A realtime
    # run polls for new closed bars (poll_sec) instead of exiting when none are available.
    max_ticks = int(params["max_ticks"]) if params.get("max_ticks") else None
    poll_sec = float(params.get("poll_sec") or (10.0 if realtime else 0.0))
    prog_total = max_ticks or 0  # 0 = unbounded (the dashboard shows the live tick count)

    if multi_strategy:
        from src.paper.lake import resolve_active_strategies
        from src.strategies.promotion import reference_only_active_ids

        # Non-paper runs (testnet/demo/live) may ONLY run strategies validated on real lake
        # data; reference-only promotions are blocked (Section 13).
        require_real_data = mode != "paper"
        active, skipped = resolve_active_strategies(settings, require_real_data=require_real_data)
        ctx.log(
            f"active promoted strategies: {[sid for _s, sid, _v in active] or 'NONE'}"
            + (f" (skipped stale/unknown ids: {skipped})" if skipped else "")
        )
        blocked = reference_only_active_ids(settings.strategy_version) if require_real_data else []
        if blocked:
            ctx.log(
                f"blocked from {mode}: {blocked} — validated on synthetic/reference data only. "
                "Re-validate on downloaded lake data before they can trade a real account.",
                level="WARNING",
            )
        if not active:
            ctx.log(
                "no promoted strategies are active — run strategy validation on real lake data "
                "and promote at least one before trading (live trades nothing without one).",
                level="WARNING",
            )
    ctx.log(
        f"starting live session: env={settings.exchange_env} venue_mode={mode} "
        f"realtime={realtime} transport={transport} "
        f"max_ticks={'continuous' if max_ticks is None else max_ticks} poll_sec={poll_sec} "
        f"multi_strategy={multi_strategy}"
    )
    ctx.progress(0, prog_total, "starting live loop (press Stop to end)")

    # Liveness: a selective strategy (lead_lag) only emits a "tick" on a SIGNAL, so between setups
    # the session looks frozen at tick 0 even when healthy. Show an "alive" heartbeat — bars
    # evaluated + signals/exec/rej — updated each poll cycle (throttled), so quiet != dead.
    import time as _time

    hb = {"bars": 0, "signals": 0, "exec": 0, "rej": 0, "last": 0.0}

    def _write_progress(*, force: bool = False) -> None:
        now = _time.monotonic()
        if not force and now - hb["last"] < 15:
            return
        hb["last"] = now
        ctx.progress(
            hb["signals"], prog_total,
            f"alive · {hb['bars']} bars evaluated · {hb['signals']} signals · "
            f"{hb['exec']} exec / {hb['rej']} rej",
        )

    def _on_tick(tick, i: int) -> None:
        # tick.executed/rejected are PER-TICK deltas — accumulate so the message shows the running
        # TOTAL executions/rejections, not just the most recent tick.
        hb["exec"] += tick.executed
        hb["rej"] += tick.rejected
        hb["signals"] = i + 1
        _write_progress(force=True)  # a signal is notable → update immediately

    def _on_heartbeat(stats: dict) -> None:
        hb["bars"] += int(stats.get("advanced", 0))
        hb["signals"] = max(hb["signals"], int(stats.get("signals", 0)))
        _write_progress()  # throttled per-cycle liveness

    result = run_replay_session(
        data_cfg,
        mode=mode,
        timeframe=timeframe,
        symbols=symbols,
        candidate_id=candidate_id,
        multi_strategy=multi_strategy,
        transport=transport,
        realtime=realtime,
        max_ticks=max_ticks,
        poll_sec=poll_sec,
        on_tick=_on_tick,
        on_heartbeat=_on_heartbeat,
        # Stop on a dashboard Stop (cancel flag) OR if this run was superseded (a false-reap
        # requeued the job and another worker now owns it) — so two live loops never run at once.
        should_stop=lambda: ctx.is_cancelled() or not ctx.still_owns(),
    )
    session_id = persist_live_run(result, settings)
    net = sum(t.pnl for t in result.session.trades)
    status = "halted/stopped" if result.halted else "completed"
    ctx.progress(len(result.ticks), prog_total, f"{status}: {result.executed} executed")
    return {
        "message": f"live session {session_id} {status}: {result.executed} executed / "
        f"{result.rejected} rejected, net_pnl={net:.2f}",
        "session_id": session_id,
        "env": settings.exchange_env,
        "mode": mode,
        "executed": result.executed,
        "rejected": result.rejected,
        "ticks": len(result.ticks),
        "halted": result.halted,
    }


@job_handler("reset_env_stats")
def _reset_env_stats(ctx: JobContext, params: dict) -> dict:
    """Zero one environment's persisted statistics (default ``demo``) — runs+trades+decision
    logs+explainability — leaving other environments untouched (Section 26)."""
    from src.live.admin import reset_env_stats

    env = str(params.get("env") or "demo")
    ctx.log(f"resetting persisted statistics for environment {env!r}")
    removed = reset_env_stats(env)
    ctx.progress(1, 1, f"removed {removed.total} rows for {env}")
    return {
        "message": f"reset {env} stats: removed {removed.total} rows "
        f"({removed.runs} runs, {removed.trades} trades)",
        **removed.to_dict(),
    }


# --------------------------------------------------------------------------- #
# Backtesting (dashboard-triggered, runs on the dedicated `backtest` queue)    #
# --------------------------------------------------------------------------- #
@job_handler("run_backtest")
def _run_backtest(ctx: JobContext, params: dict) -> dict:
    """Run a full event-based reference backtest in the background and persist the result.

    Triggered from the dashboard (POST /api/backtests/run) and consumed by the dedicated
    ``backtest`` worker. The report JSON is written to the reports lake and an index row is
    upserted into ``backtest_runs`` (B.4) so the dashboard Backtests page can display the
    metrics. ``passed`` reflects a positive net expectancy (display only; the BT/WF/FEE/SLIP
    gates remain the authoritative profitability judgement)."""
    from src.backtest.service import (
        load_backtest_config,
        persist_backtest_run,
        run_reference_backtest,
        write_report,
    )
    from src.config import get_settings

    settings = get_settings()
    label = str(params.get("label") or "dashboard_backtest")
    ctx.log(f"starting backtest: {label}")
    ctx.progress(0, 1, "running event-based backtest")
    cfg = load_backtest_config()
    out = run_reference_backtest(cfg, label=label)
    report = out.report
    report_path = write_report(settings, report.payload, kind="backtest")
    run_id = persist_backtest_run(
        cfg,
        report,
        kind="backtest",
        report_path=report_path,
        settings=settings,
        passed=report.expectancy_r > 0,
        summary_extra={"label": label, "requested_by": str(params.get("requested_by", ""))},
    )
    ctx.progress(1, 1, f"backtest {run_id}: expectancy_r={report.expectancy_r:.4f}")
    return {
        "message": f"backtest {run_id}: expectancy_r={report.expectancy_r:.4f}, "
        f"PF={min(report.profit_factor, 1e9):.2f}, trades={report.trade_count}",
        "run_id": run_id,
        "expectancy_r": report.expectancy_r,
        "profit_factor": min(report.profit_factor, 1e9),
        "trade_count": report.trade_count,
        "report_path": report_path,
    }


@job_handler("run_lake_ml_shadow_pass")
def _run_lake_ml_shadow_pass(ctx: JobContext, params: dict) -> dict:
    """Run the shadow ML meta-labeler over REAL lake candidates (shadow-only, applied=False).

    Builds the candidate stream from a downloaded snapshot (the same real candidates the
    lake paper session uses) and scores them with the shadow predictor, logging every
    prediction with applied=False. The models themselves are trained on the reference
    dataset (training data is a separate concern); this proves the meta-labeler scores
    REAL setups in shadow mode without ever influencing trading."""
    from src.config import get_settings
    from src.data.config import load_data_config
    from src.ml import ShadowPredictor
    from src.ml.config import load_ml_config
    from src.ml.labels import build_reference_dataset, train_test_split
    from src.paper.lake import build_lake_paper_inputs

    config_path = str(params.get("config_path") or "configs/data.bybit.yaml")
    data_cfg = load_data_config(config_path)
    tf = params.get("timeframe") or data_cfg.base_timeframe
    symbols = params.get("symbols") or data_cfg.active_symbols()
    candidate_id = params.get("candidate_id") or params.get("strategy") or None
    settings = get_settings()
    ml_cfg = load_ml_config()

    ctx.log("training shadow models (reference dataset) then scoring REAL lake candidates")
    predictor = ShadowPredictor.from_config(ml_cfg)
    train_samples, _ = train_test_split(build_reference_dataset(seed=42), seed=42)
    predictor.train(train_samples)
    inputs, _, _ = build_lake_paper_inputs(
        data_cfg, timeframe=tf, symbols=symbols, candidate_id=candidate_id, settings=settings
    )
    candidates = [pin.candidate for pin in inputs]
    ctx.progress(0, 1, f"scoring {len(candidates)} real candidates (applied=False)")
    result = predictor.run(candidates, settings=settings, write_to_db=True)
    assert not result.applied, "shadow pass must never set applied=True"
    ctx.progress(1, 1, f"{len(result.shadow_log_ids)} shadow log entries")
    return {
        "message": f"lake ML shadow: {len(result.shadow_log_ids)} entries over "
        f"{len(candidates)} real candidates (applied=False)",
        "candidates": len(candidates),
        "shadow_log_ids": len(result.shadow_log_ids),
        "applied": result.applied,
    }


@job_handler("run_lake_paper_session")
def _run_lake_paper_session(ctx: JobContext, params: dict) -> dict:
    """Run + persist a REAL-DATA (replay) paper session over a downloaded snapshot.

    Derives the candidate stream from real lake data and runs it through the full paper
    pipeline (ranking → risk → execution → SimulatedVenue); trades persist to
    ``paper_trades`` (shadow-only). Requires the snapshot to be downloaded first."""
    from src.data.config import load_data_config
    from src.paper.lake import run_lake_paper_session

    config_path = str(params.get("config_path") or "configs/data.bybit.yaml")
    data_cfg = load_data_config(config_path)
    timeframe = params.get("timeframe") or None
    symbols = params.get("symbols") or None
    candidate_id = params.get("candidate_id") or params.get("strategy") or None
    # Ensemble = run ALL active promoted strategies in one run (the offline twin of the live
    # engine), tagged lakebt:…:ensemble so its combined stats are viewable on their own.
    multi_strategy = bool(params.get("multi_strategy")) or (
        candidate_id is None and bool(params.get("ensemble"))
    )
    dataset_version = params.get("dataset_version") or None
    ctx.log(
        f"real-data {'ensemble ' if multi_strategy else ''}session over "
        f"{data_cfg.exchange_id}/{data_cfg.data_version}"
    )
    ctx.progress(0, 1, "running lake paper session")
    session, _report, sid = run_lake_paper_session(
        data_cfg,
        timeframe=timeframe,
        symbols=symbols,
        candidate_id=candidate_id,
        multi_strategy=multi_strategy,
        dataset_version=dataset_version,
    )
    net = sum(t.pnl for t in session.trades)
    if session.executed_count + session.rejected_count == 0:
        ctx.log(
            "0 candidates derived from the lake — either no snapshot data exists for this "
            "window (download it on the Data page; check the window in configs/data.bybit.yaml) "
            "or the strategy produced no signals on real data.",
            level="WARNING",
        )
    ctx.progress(1, 1, f"{session.executed_count} trades, net_pnl={net:.2f}")
    return {
        "message": f"lake paper {sid}: {session.executed_count} executed / "
        f"{session.rejected_count} rejected, net_pnl={net:.2f}",
        "session_id": sid,
        "executed": session.executed_count,
        "rejected": session.rejected_count,
    }


@job_handler("run_lake_backtest")
def _run_lake_backtest(ctx: JobContext, params: dict) -> dict:
    """Run + persist ONE real-data backtest iteration on the ``backtest`` worker.

    The Parity-Rule real-data counterpart of ``run_backtest``: it reads a downloaded
    ``DATA_VERSION`` snapshot (default ``configs/data.bybit.yaml``), runs the event
    engine through the one feature pipeline, and upserts a ``backtest_runs`` row tagged
    with the snapshot so the iteration leaderboard can rank it. Requires the snapshot to
    have been downloaded first (``qbot download --config ...``)."""
    from src.backtest.service import run_and_persist_lake_backtest
    from src.data.config import load_data_config

    config_path = str(params.get("config_path") or "configs/data.bybit.yaml")
    data_cfg = load_data_config(config_path)
    timeframe = params.get("timeframe") or None
    symbols = params.get("symbols") or None
    candidate_id = params.get("candidate_id") or params.get("strategy") or None
    label = str(params.get("label") or "lake")
    dataset_version = params.get("dataset_version") or None
    ctx.log(f"real-data backtest over {data_cfg.exchange_id}/{data_cfg.data_version}")
    ctx.progress(0, 1, "running lake backtest")
    rid, out = run_and_persist_lake_backtest(
        data_cfg,
        timeframe=timeframe,
        symbols=symbols,
        candidate_id=candidate_id,
        dataset_version=dataset_version,
        label=label,
    )
    r = out.report
    if r.trade_count == 0:
        ctx.log(
            "0 trades — no snapshot data for this window (download it on the Data page) or the "
            "strategy produced no signals on real data.",
            level="WARNING",
        )
    ctx.progress(1, 1, f"{rid}: expectancy_r={r.expectancy_r:.4f}")
    return {
        "message": f"lake backtest {rid}: expectancy_r={r.expectancy_r:.4f}, "
        f"trades={r.trade_count}",
        "run_id": rid,
        "expectancy_r": r.expectancy_r,
        "trade_count": r.trade_count,
    }


ensure_handlers_registered()
