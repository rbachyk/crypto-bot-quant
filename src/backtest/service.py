"""Backtest orchestration (AGENTS.md Section 19).

The single entry point the BT/WF/FEE/SLIP gates and the backtest jobs call. It
builds the engine's per-symbol inputs through the ONE feature pipeline (the
Parity Rule, Section 10) from the deterministic reference series, runs the
event-based engine, builds the full report, and provides the contiguous,
re-based time windows the walk-forward harness evaluates as out-of-sample folds.

Reference inputs reuse features computed over the FULL series and then select a
contiguous segment, so every feature for a bar in a fold was still computed only
from data at or before that bar's decision time (causality preserved) without
re-introducing warmup edge effects per fold.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime

from src.backtest.config import BacktestConfig, load_backtest_config
from src.backtest.engine import BacktestEngine, BacktestResult, SymbolInput
from src.backtest.metrics import BacktestReport, build_report
from src.backtest.reference import ReferenceReader
from src.backtest.strategy import PortfolioStrategy, ReferenceMomentumStrategy, Strategy
from src.config import Settings, get_settings
from src.data.config import DataConfig, load_data_config
from src.data.schema import FUNDING, SPREAD, SeriesKey, timeframe_ms
from src.data.store import SeriesStore
from src.exchange.metadata import MetadataConfig, load_metadata_config
from src.features.config import FeatureConfig, load_feature_config
from src.features.pipeline import FeatureFrame, StoreReader, compute_features


@dataclass(slots=True)
class BacktestRunResult:
    result: BacktestResult
    report: BacktestReport


def make_strategy(cfg: BacktestConfig) -> Strategy:
    return ReferenceMomentumStrategy(cfg.reference_strategy)


_INPUTS_CACHE: dict[tuple, list[SymbolInput]] = {}


def _reference_signature(cfg: BacktestConfig, feat_cfg: FeatureConfig) -> tuple:
    r = cfg.reference
    return (
        r.edge,
        tuple(r.symbols),
        r.bars,
        r.timeframe,
        r.trend_drift,
        r.trend_period_bars,
        r.base_sigma,
        r.seed,
        tuple(sorted(r.activation_bar.items())),
        feat_cfg.timeframe,
        feat_cfg.feature_set_version,
        feat_cfg.windows.short,
        feat_cfg.windows.long,
        feat_cfg.windows.rank,
    )


def build_reference_inputs(
    cfg: BacktestConfig | None = None,
    feat_cfg: FeatureConfig | None = None,
) -> list[SymbolInput]:
    """Build per-symbol engine inputs from the deterministic reference series.

    Memoized on the reference + feature signature: the series is fully
    deterministic and the engine treats inputs as read-only, so repeated gate /
    stress / walk-forward runs reuse one feature build (which is the expensive
    step) instead of recomputing it.
    """
    cfg = cfg or load_backtest_config()
    feat_cfg = feat_cfg or _reference_feature_config(cfg)
    sig = _reference_signature(cfg, feat_cfg)
    cached = _INPUTS_CACHE.get(sig)
    if cached is not None:
        return cached
    iv = timeframe_ms(cfg.reference.timeframe)
    inputs: list[SymbolInput] = []
    for symbol in cfg.reference.symbols:
        reader = ReferenceReader(symbol, cfg.reference)
        frame = compute_features(symbol, reader, feat_cfg)
        bars = reader.ohlcv(symbol)
        spread = reader.series(symbol, SPREAD)
        funding = reader.funding_events()
        activation_bar = cfg.reference.activation_bar.get(symbol, 0)
        inputs.append(
            SymbolInput(
                symbol=symbol,
                bars=bars,
                frame=frame,
                spread_samples=[{"ts": s["ts"], "spread_bps": s["spread_bps"]} for s in spread],
                funding_events=[
                    {"ts": f["ts"], "funding_rate": f["funding_rate"]} for f in funding
                ],
                activation_ts=activation_bar * iv,
            )
        )
    _INPUTS_CACHE[sig] = inputs
    return inputs


def _reference_feature_config(cfg: BacktestConfig) -> FeatureConfig:
    """Feature config aligned to the reference timeframe (Parity Rule, Section 10)."""
    base = load_feature_config()
    if base.timeframe == cfg.reference.timeframe:
        return base
    from dataclasses import replace

    return replace(base, timeframe=cfg.reference.timeframe)


def _lake_feature_config(timeframe: str) -> FeatureConfig:
    """Feature config aligned to the lake decision timeframe (Parity Rule, Section 10)."""
    base = load_feature_config()
    if base.timeframe == timeframe:
        return base
    from dataclasses import replace

    return replace(base, timeframe=timeframe)


def build_lake_inputs(
    store: SeriesStore,
    *,
    exchange_id: str,
    symbols: list[str],
    timeframe: str,
    base_timeframe: str,
    funding_timeframe: str,
    start_ms: int,
    end_ms: int,
    oi_timeframe: str | None = None,
    feat_cfg: FeatureConfig | None = None,
) -> list[SymbolInput]:
    """Build per-symbol engine inputs from REAL downloaded lake series.

    The Parity-Rule twin of :func:`build_reference_inputs`: the SAME feature
    pipeline and the SAME :class:`SymbolInput` shape, but the data-reading
    adapter is the Parquet :class:`StoreReader` over a downloaded ``DATA_VERSION``
    snapshot instead of the synthetic reference. Mark/index/spread are read on
    ``base_timeframe``; OI may be coarser (``oi_timeframe``); funding on its own
    grid. Symbols with no OHLCV in the window are skipped (the caller validates
    coverage via the data platform before backtesting).
    """
    feat_cfg = feat_cfg or _lake_feature_config(timeframe)
    oi_tf = oi_timeframe or base_timeframe
    inputs: list[SymbolInput] = []
    for symbol in symbols:
        reader = StoreReader(
            store,
            exchange_id,
            timeframe,
            base_timeframe,
            funding_timeframe,
            start_ms,
            end_ms,
            oi_timeframe=oi_tf,
        )
        bars = reader.ohlcv(symbol)
        if not bars:
            continue  # no history in window — excluded from the run
        frame = compute_features(symbol, reader, feat_cfg)
        spread = store.read(
            SeriesKey(exchange_id, SPREAD, symbol, base_timeframe), start_ms, end_ms
        )
        funding = store.read(
            SeriesKey(exchange_id, FUNDING, symbol, funding_timeframe), start_ms, end_ms
        )
        inputs.append(
            SymbolInput(
                symbol=symbol,
                bars=bars,
                frame=frame,
                spread_samples=[{"ts": s["ts"], "spread_bps": s["spread_bps"]} for s in spread],
                funding_events=[
                    {"ts": f["ts"], "funding_rate": f["funding_rate"]} for f in funding
                ],
                activation_ts=0,
            )
        )
    if not inputs:
        return inputs
    # The engine indexes bars by EPOCH TIME, so rebasing is not required for correctness; we
    # still shift the window to a 0-based origin (aligned to the decision interval) so timestamps
    # are small and deterministic across snapshots and match the reference series' convention.
    # A symbol listed mid-window keeps its true offset from this origin (its first bar is NOT at
    # ts 0) — the engine handles that natively.
    iv = timeframe_ms(timeframe)
    lo = (start_ms // iv) * iv
    return rebase_window(inputs, lo, end_ms)


def lake_candidate_strategy(
    candidate_id: str,
) -> tuple[Strategy | PortfolioStrategy, str, str]:
    """Build a configured research candidate by id → (strategy, strategy_id, version).

    Lets real-data backtests run the actual strategy library (families A/B/G) rather
    than only the reference self-test. Family B is single-symbol; A/G are cross-asset
    (need a multi-symbol universe to produce signals)."""
    from src.strategies.candidates import build_strategy
    from src.strategies.config import load_strategies_config

    scfg = load_strategies_config()
    cand = scfg.candidate(candidate_id)
    if cand is None:
        known = ", ".join(c.id for c in scfg.candidates) or "(none)"
        raise ValueError(f"unknown candidate strategy {candidate_id!r}; known: {known}")
    strategy = build_strategy(cand, scfg.strategy_version, cand.params)
    return strategy, cand.id, scfg.strategy_version


def run_lake_backtest(
    data_cfg: DataConfig | None = None,
    cfg: BacktestConfig | None = None,
    meta: MetadataConfig | None = None,
    *,
    settings: Settings | None = None,
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    strategy: Strategy | PortfolioStrategy | None = None,
    label: str = "lake",
) -> BacktestRunResult:
    """Run the event-based engine over real lake data for a ``DATA_VERSION`` snapshot.

    Reads the downloaded series from the configured data lake (``data.bybit.yaml``
    et al.), builds inputs through the one feature pipeline and runs the engine.
    ``strategy`` defaults to the reference momentum self-test; pass a real candidate
    (see :func:`lake_candidate_strategy`) to research an actual edge. Fees fall back to
    the backtest-config defaults for symbols without verified exchange metadata
    (research-grade; verified metadata is a live/META concern).
    """
    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    cfg = cfg or load_backtest_config()
    meta = meta or load_metadata_config()
    tf = timeframe or data_cfg.base_timeframe
    syms = symbols or data_cfg.active_symbols()
    store = SeriesStore(settings.data_lake_path)
    inputs = build_lake_inputs(
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
    if not inputs:
        raise ValueError(
            "no lake inputs for the configured window — download a DATA_VERSION "
            "snapshot first (qbot download ...)"
        )
    return run_engine(cfg, meta, inputs, strategy=strategy, label=label)


def run_and_persist_lake_backtest(
    data_cfg: DataConfig | None = None,
    cfg: BacktestConfig | None = None,
    meta: MetadataConfig | None = None,
    *,
    settings: Settings | None = None,
    timeframe: str | None = None,
    symbols: list[str] | None = None,
    candidate_id: str | None = None,
    dataset_version: str | None = None,
    label: str = "lake",
    kind: str = "backtest",
) -> tuple[str, BacktestRunResult]:
    """Run ONE real-data backtest iteration and persist it as a comparable run.

    Writes the report to the reports lake and upserts a ``backtest_runs`` index row
    tagged with the ``DATA_VERSION`` it ran over, so the leaderboard can rank this
    iteration against others and no prior iteration is lost (each distinct
    snapshot/strategy/symbols combination is its own immutable row). ``candidate_id``
    selects a real research strategy (families A/B/G); default is the reference
    momentum self-test.
    """
    settings = settings or get_settings()
    data_cfg = data_cfg or load_data_config()
    cfg = cfg or load_backtest_config()
    tf = timeframe or data_cfg.base_timeframe
    syms = symbols or data_cfg.active_symbols()
    strategy: Strategy | PortfolioStrategy | None = None
    strat_id = cfg.reference_strategy.name
    strat_ver = cfg.reference_strategy.strategy_version
    if candidate_id:
        strategy, strat_id, strat_ver = lake_candidate_strategy(candidate_id)
    out = run_lake_backtest(
        data_cfg,
        cfg,
        meta,
        settings=settings,
        timeframe=tf,
        symbols=syms,
        strategy=strategy,
        label=label,
    )
    report_path = write_report(settings, out.report.payload, kind=kind)
    rid = persist_backtest_run(
        cfg,
        out.report,
        kind=kind,
        report_path=report_path,
        settings=settings,
        passed=out.report.expectancy_r > 0,  # display flag; the bar lives on the leaderboard
        strategy_id=strat_id,
        strategy_version=strat_ver,
        dataset_version=dataset_version or data_cfg.data_version,
        symbols=syms,
        summary_extra={
            "label": label,
            "timeframe": tf,
            "exchange_id": data_cfg.exchange_id,
            "data_version": data_cfg.data_version,
            "window": [data_cfg.window_start_ms, data_cfg.window_end_ms],
        },
    )
    return rid, out


def rebase_window(inputs: list[SymbolInput], lo_ts: int, hi_ts: int) -> list[SymbolInput]:
    """Return inputs restricted to ``[lo_ts, hi_ts)`` and re-based to start at ts=0.

    Bars, feature rows, spread samples, funding events and the activation cutoff
    are all shifted by ``-lo_ts`` so the engine sees a clean 0-based window. This
    is how walk-forward builds disjoint out-of-sample test folds.
    """
    out: list[SymbolInput] = []
    for s in inputs:
        bars = [_shift_ts(b, lo_ts) for b in s.bars if lo_ts <= b["ts"] < hi_ts]
        rows = [
            _shift_row(r, lo_ts)
            for r in s.frame.rows
            if lo_ts <= r["decision_ts"] < hi_ts and r["ts"] >= lo_ts
        ]
        frame = FeatureFrame(
            symbol=s.frame.symbol,
            timeframe=s.frame.timeframe,
            feature_names=list(s.frame.feature_names),
            rows=rows,
        )
        spread = [_shift_ts(x, lo_ts) for x in s.spread_samples if lo_ts <= x["ts"] < hi_ts]
        funding = [_shift_ts(x, lo_ts) for x in s.funding_events if lo_ts <= x["ts"] < hi_ts]
        out.append(
            SymbolInput(
                symbol=s.symbol,
                bars=bars,
                frame=frame,
                spread_samples=spread,
                funding_events=funding,
                activation_ts=max(0, s.activation_ts - lo_ts),
            )
        )
    return out


def _shift_ts(row: dict, lo_ts: int) -> dict:
    r = dict(row)
    r["ts"] = row["ts"] - lo_ts
    return r


def _shift_row(row: dict, lo_ts: int) -> dict:
    r = deepcopy(row)
    r["ts"] = row["ts"] - lo_ts
    r["decision_ts"] = row["decision_ts"] - lo_ts
    return r


def run_engine(
    cfg: BacktestConfig,
    meta: MetadataConfig,
    inputs: list[SymbolInput],
    strategy: Strategy | PortfolioStrategy | None = None,
    *,
    label: str = "",
) -> BacktestRunResult:
    engine = BacktestEngine(cfg, meta, strategy or make_strategy(cfg))
    result = engine.run(inputs)
    report = build_report(result, label=label)
    return BacktestRunResult(result=result, report=report)


def run_reference_backtest(
    cfg: BacktestConfig | None = None,
    meta: MetadataConfig | None = None,
    *,
    label: str = "reference_full",
) -> BacktestRunResult:
    """Full event-based backtest of the reference strategy on the reference series."""
    cfg = cfg or load_backtest_config()
    meta = meta or load_metadata_config()
    inputs = build_reference_inputs(cfg)
    return run_engine(cfg, meta, inputs, label=label)


# --------------------------------------------------------------------------- #
# Persistence (Section 24 reports; Appendix B.4 relational index)              #
# --------------------------------------------------------------------------- #
def run_id(cfg: BacktestConfig, kind: str, payload: dict) -> str:
    """Content-addressed run id: identical inputs ⇒ identical id (idempotent)."""
    digest = hashlib.sha256(
        json.dumps({"kind": kind, "report": payload}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"{cfg.backtest_version}_{kind}_{digest}"


def write_report(settings: Settings, payload: dict, *, kind: str = "backtest") -> str:
    """Persist a backtest/walk-forward/stress report to ``reports/backtest/``."""
    reports_dir = settings.reports_path / "backtest"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"{kind}_{stamp}.json"
    from src.reporting import wrap_report

    enveloped = wrap_report(
        payload,
        report_type=kind,
        methodology=(
            "Event-based backtest over the series through the single causal feature pipeline "
            "(Parity Rule, Section 10); costs from verified metadata; walk-forward + locked "
            "hold-out and fee/slippage stress where applicable (Section 16/19)."
        ),
        limitations=(
            "Modelled fills/costs on historical or synthetic data; past performance does not "
            "guarantee live results; spread is estimated where L1 history is unavailable."
        ),
        recommendations=payload.get("recommendations", ""),
        period=payload.get("period") or {"scope": payload.get("label", "full")},
        versions=settings.versions(),
    )
    path.write_text(json.dumps(enveloped, indent=2, default=str), encoding="utf-8")
    return str(path)


def persist_backtest_run(
    cfg: BacktestConfig,
    report: BacktestReport,
    *,
    kind: str,
    report_path: str,
    settings: Settings | None = None,
    passed: bool = True,
    strategy_id: str = "",
    strategy_version: str = "",
    dataset_version: str | None = None,
    symbols: list[str] | None = None,
    summary_extra: dict | None = None,
) -> str:
    """Upsert a :class:`~src.db.models.BacktestRun` index row for a run.

    ``dataset_version`` records the immutable ``DATA_VERSION`` snapshot a real-data
    (lake) run was computed over, so iterations are grouped + comparable on the
    leaderboard and prior runs are never lost. It also keys the content-addressed
    ``run_id`` so the same strategy on two different snapshots yields two rows.
    """
    settings = settings or get_settings()
    p = report.payload
    syms = list(symbols) if symbols is not None else list(cfg.reference.symbols)
    extra = summary_extra or {}
    # Fold the full iteration identity into the run id so distinct (snapshot, strategy,
    # symbols, timeframe, label) iterations are distinct rows even if their metrics
    # coincide (e.g. two timeframes that both produce zero trades on real data).
    rid = run_id(
        cfg,
        kind,
        {
            "report": p,
            "dataset_version": dataset_version,
            "strategy_id": strategy_id or cfg.reference_strategy.name,
            "symbols": syms,
            "timeframe": extra.get("timeframe"),
            "label": extra.get("label", p.get("label")),
        },
    )
    from src.db.base import session_scope
    from src.db.models import BacktestRun

    summary = {
        "label": p.get("label", ""),
        "win_rate": p.get("win_rate", 0.0),
        "cost_breakdown": p.get("cost_breakdown", {}),
        "rejected_candidates": p.get("rejected_candidates", {}),
        **(summary_extra or {}),
    }
    with session_scope() as session:
        row = session.query(BacktestRun).filter_by(run_id=rid).one_or_none()
        if row is None:
            row = BacktestRun(run_id=rid)
            session.add(row)
        row.kind = kind
        row.backtest_version = cfg.backtest_version
        row.strategy_id = strategy_id or cfg.reference_strategy.name
        row.strategy_version = strategy_version or cfg.reference_strategy.strategy_version
        row.dataset_version = dataset_version
        row.feature_set_version = settings.versions().get("feature_set_version")
        row.symbols = syms
        row.passed = passed
        row.trade_count = report.trade_count
        row.expectancy_r = report.expectancy_r
        row.profit_factor = min(report.profit_factor, 1e9)  # cap inf for storage
        row.total_return = report.total_return
        row.max_drawdown = report.max_drawdown
        row.summary = summary
        row.report_path = report_path
        row.related_versions = settings.versions()
    return rid
