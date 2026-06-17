"""Feature store — immutable, versioned feature builds from a dataset snapshot.

Builds the feature matrices for the active universe through the single feature
pipeline (``compute_features``) reading the frozen dataset snapshot, then writes
them as month-free per-symbol Parquet plus a manifest. The build id is
content-addressed (``feat_0001_<dataset>_<hash>``), so an identical rebuild
reuses it and produces a byte-identical checksum — the reproducibility the FEAT
gate requires. A ``feature_set_versions`` row indexes each build (Appendix B.4:
large matrices live in the lake, not Postgres).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.data.config import DataConfig, load_data_config
from src.data.store import SeriesStore
from src.db.models import FeatureSetVersion
from src.features.config import FeatureConfig, load_feature_config
from src.features.pipeline import FEATURE_NAMES, FeatureFrame, StoreReader, compute_features

_INT_COLS = {"ts", "decision_ts"}
_COLUMNS = ["ts", "decision_ts", "close", *FEATURE_NAMES]


@dataclass(slots=True)
class FeatureBuildResult:
    feature_snapshot_id: str
    created: bool
    checksum: str
    dataset_version: str
    frames: dict[str, FeatureFrame] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    manifest_path: str = ""

    @property
    def total_rows(self) -> int:
        return sum(self.row_counts.values())


class FeatureStore:
    def __init__(
        self,
        settings: Settings | None = None,
        data_cfg: DataConfig | None = None,
        feat_cfg: FeatureConfig | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.data_cfg = data_cfg or load_data_config()
        self.feat_cfg = feat_cfg or load_feature_config()
        self.store = SeriesStore(self.settings.data_lake_path)
        self.root = Path(self.settings.data_lake_path) / "features"

    def _reader(self, symbol: str) -> StoreReader:
        return StoreReader(
            self.store,
            exchange_id=self.data_cfg.exchange_id,
            timeframe=self.feat_cfg.timeframe,
            base_timeframe=self.data_cfg.base_timeframe,
            funding_timeframe=self.data_cfg.funding_timeframe,
            start_ms=self.data_cfg.window_start_ms,
            end_ms=self.data_cfg.window_end_ms,
        )

    def compute_all(self, symbols: list[str]) -> dict[str, FeatureFrame]:
        """Compute (without persisting) every symbol's feature frame."""
        return {
            symbol: compute_features(symbol, self._reader(symbol), self.feat_cfg)
            for symbol in symbols
        }

    # -- ids / checksums ------------------------------------------------- #
    def _config_fingerprint(self) -> dict:
        w = self.feat_cfg.windows
        return {
            "feature_set_version": self.feat_cfg.feature_set_version,
            "timeframe": self.feat_cfg.timeframe,
            "windows": {"short": w.short, "long": w.long, "rank": w.rank},
            "label_horizon": self.feat_cfg.label_horizon,
            "features": list(FEATURE_NAMES),
        }

    def combined_checksum(self, frames: dict[str, FeatureFrame]) -> str:
        payload = json.dumps(
            {
                "config": self._config_fingerprint(),
                "symbols": {sym: frames[sym].canonical() for sym in sorted(frames)},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _snapshot_id(self, dataset_version: str, checksum: str) -> str:
        return f"{self.feat_cfg.feature_set_version}_{dataset_version}_{checksum}"

    def feature_dir(self, snapshot_id: str) -> Path:
        return self.root / snapshot_id

    # -- build ----------------------------------------------------------- #
    def build(
        self,
        symbols: list[str],
        dataset_version: str,
        universe_version: str | None = None,
        session: Session | None = None,
        source_jobs: list[str] | None = None,
    ) -> FeatureBuildResult:
        frames = self.compute_all(symbols)
        checksum = self.combined_checksum(frames)
        snapshot_id = self._snapshot_id(dataset_version, checksum)
        row_counts = {sym: len(frames[sym].rows) for sym in frames}

        ddir = self.feature_dir(snapshot_id)
        created = not ddir.exists()
        if created:
            ddir.mkdir(parents=True, exist_ok=True)
            for sym in sorted(frames):
                self._write_parquet(ddir, sym, frames[sym])
            manifest = self._manifest(
                snapshot_id,
                dataset_version,
                universe_version,
                symbols,
                row_counts,
                checksum,
                source_jobs or [],
            )
            (ddir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
        manifest_path = str(ddir / "manifest.json")

        if session is not None:
            self._persist_row(
                session,
                snapshot_id,
                dataset_version,
                universe_version,
                symbols,
                row_counts,
                checksum,
                manifest_path,
                source_jobs or [],
            )

        return FeatureBuildResult(
            feature_snapshot_id=snapshot_id,
            created=created,
            checksum=checksum,
            dataset_version=dataset_version,
            frames=frames,
            row_counts=row_counts,
            manifest_path=manifest_path,
        )

    def _write_parquet(self, ddir: Path, symbol: str, frame: FeatureFrame) -> None:
        safe = symbol.replace("/", "_").replace(":", "_")
        cols: dict[str, list] = {c: [] for c in _COLUMNS}
        for row in frame.rows:
            for c in _COLUMNS:
                cols[c].append(row[c])
        arrays = [
            pa.array(cols[c], type=(pa.int64() if c in _INT_COLS else pa.float64()))
            for c in _COLUMNS
        ]
        table = pa.Table.from_arrays(arrays, names=_COLUMNS)
        pq.write_table(table, ddir / f"{safe}.parquet")

    def _manifest(
        self,
        snapshot_id: str,
        dataset_version: str,
        universe_version: str | None,
        symbols: list[str],
        row_counts: dict[str, int],
        checksum: str,
        source_jobs: list[str],
    ) -> dict:
        return {
            "feature_snapshot_id": snapshot_id,
            "feature_set_version": self.feat_cfg.feature_set_version,
            "dataset_version": dataset_version,
            "universe_version": universe_version,
            "exchange_id": self.data_cfg.exchange_id,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "timeframe": self.feat_cfg.timeframe,
            "symbols": list(symbols),
            "feature_names": list(FEATURE_NAMES),
            "label_horizon": self.feat_cfg.label_horizon,
            "row_counts": row_counts,
            "checksum": checksum,
            "config": self._config_fingerprint(),
            "source_jobs": source_jobs,
        }

    def _persist_row(
        self,
        session: Session,
        snapshot_id: str,
        dataset_version: str,
        universe_version: str | None,
        symbols: list[str],
        row_counts: dict[str, int],
        checksum: str,
        manifest_path: str,
        source_jobs: list[str],
    ) -> None:
        row = session.get(FeatureSetVersion, snapshot_id)
        if row is None:
            row = FeatureSetVersion(version=snapshot_id)
            session.add(row)
        row.feature_set_version = self.feat_cfg.feature_set_version
        row.dataset_version = dataset_version
        row.universe_version = universe_version
        row.exchange_id = self.data_cfg.exchange_id
        row.symbols = list(symbols)
        row.timeframe = self.feat_cfg.timeframe
        row.feature_names = list(FEATURE_NAMES)
        row.label_horizon = self.feat_cfg.label_horizon
        row.row_counts = row_counts
        row.checksum = checksum
        row.manifest_path = manifest_path
        row.source_jobs = source_jobs
