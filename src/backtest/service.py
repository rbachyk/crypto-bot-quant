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
from src.backtest.strategy import ReferenceMomentumStrategy, Strategy
from src.config import Settings, get_settings
from src.data.schema import SPREAD, timeframe_ms
from src.exchange.metadata import MetadataConfig, load_metadata_config
from src.features.config import FeatureConfig, load_feature_config
from src.features.pipeline import FeatureFrame, compute_features


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
    strategy: Strategy | None = None,
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
    path.write_text(
        json.dumps({"versions": settings.versions(), **payload}, indent=2, default=str),
        encoding="utf-8",
    )
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
    summary_extra: dict | None = None,
) -> str:
    """Upsert a :class:`~src.db.models.BacktestRun` index row for a run."""
    settings = settings or get_settings()
    p = report.payload
    rid = run_id(cfg, kind, p)
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
        row.symbols = list(cfg.reference.symbols)
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
