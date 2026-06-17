"""Dynamic Symbol Universe Manager (AGENTS.md Section 9).

Builds a **versioned** universe snapshot: every candidate symbol is scored
against the Section 9 filters (``src/universe/filters.py``) using owned data and
``[VERIFIED]`` metadata, then promoted to ``active`` or recorded
``research_only`` / ``quarantined`` with a per-filter reason. The build is:

* **content-addressed** — the version id ``univ_0001_<hash>`` is a pure function
  of the membership + statuses + filter policy, so an identical universe re-uses
  the same version (idempotent re-build), mirroring the dataset-snapshot split;
* **history-logged** — the diff against the previous version (symbols entering /
  leaving / changing status) is written to ``universe_changes`` (Section 9
  "store universe membership history").

The Manager is the single entry point the ``build_symbol_universe`` job and the
UNIV gate call, so they always reach the same recorded state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.data.config import DataConfig, load_data_config
from src.data.schema import FUNDING, OPEN_INTEREST, SeriesKey
from src.data.store import SeriesStore
from src.db.models import (
    SymbolStatus,
    UniverseChange,
    UniverseMember,
    UniverseVersion,
)
from src.exchange.metadata import MetadataConfig, load_metadata_config, sync_verified_metadata
from src.universe.config import UniverseConfig, load_universe_config
from src.universe.filters import SymbolEvaluation, SymbolMetaView, UniverseFilterEvaluator


@dataclass(slots=True)
class UniverseBuildResult:
    version: str
    created: bool
    evaluations: list[SymbolEvaluation]
    changes: list[dict] = field(default_factory=list)
    prev_version: str | None = None

    @property
    def active_symbols(self) -> list[str]:
        return [e.symbol for e in self.evaluations if e.status is SymbolStatus.ACTIVE]

    def filter_report(self) -> dict:
        return {e.symbol: e.to_dict() for e in self.evaluations}


class UniverseManager:
    def __init__(
        self,
        settings: Settings | None = None,
        uni_cfg: UniverseConfig | None = None,
        data_cfg: DataConfig | None = None,
        meta_cfg: MetadataConfig | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.uni_cfg = uni_cfg or load_universe_config()
        self.data_cfg = data_cfg or load_data_config()
        self.meta_cfg = meta_cfg or load_metadata_config()
        self.store = SeriesStore(self.settings.data_lake_path)
        self.evaluator = UniverseFilterEvaluator(self.store, self.data_cfg, self.uni_cfg)

    # -- metadata view --------------------------------------------------- #
    def _meta_view(self, symbol: str, verified_in_db: bool) -> SymbolMetaView:
        spec = self.meta_cfg.spec(symbol)
        fields = spec.fields if spec else {}
        verified = bool(verified_in_db and spec is not None and spec.is_complete_and_consistent())
        funding_key = SeriesKey(
            self.uni_cfg.exchange_id, FUNDING, symbol, self.data_cfg.funding_timeframe
        )
        oi_key = SeriesKey(
            self.uni_cfg.exchange_id, OPEN_INTEREST, symbol, self.data_cfg.base_timeframe
        )
        return SymbolMetaView(
            verified=verified,
            status=str(fields.get("status", "")),
            contract_type=str(fields.get("contract_type", "")),
            quote_currency=str(fields.get("quote_currency", "")),
            has_funding=self.store.count(funding_key) > 0,
            has_open_interest=self.store.count(oi_key) > 0,
        )

    def _verified_in_db(self, session: Session) -> dict[str, bool]:
        from src.db.models import ExchangeMetadata, VerificationStatus

        rows = session.execute(
            select(ExchangeMetadata.symbol).where(
                ExchangeMetadata.exchange_id == self.meta_cfg.exchange_id,
                ExchangeMetadata.metadata_version == self.meta_cfg.metadata_version,
                ExchangeMetadata.verification_status == VerificationStatus.VERIFIED,
            )
        ).all()
        return {symbol: True for (symbol,) in rows}

    # -- version id ------------------------------------------------------ #
    def _version_id(self, evaluations: list[SymbolEvaluation]) -> str:
        members = sorted((e.symbol, e.status.value) for e in evaluations)
        payload = json.dumps(
            {
                "policy": self.uni_cfg.universe_version,
                "metadata_version": self.meta_cfg.metadata_version,
                "filters": _filters_dict(self.uni_cfg),
                "members": members,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.uni_cfg.universe_version}_{digest}"

    # -- build ----------------------------------------------------------- #
    def build(self, session: Session, *, sync_metadata: bool = True) -> UniverseBuildResult:
        """Evaluate filters, persist a versioned universe + its change log."""
        if sync_metadata:
            sync_verified_metadata(session, self.meta_cfg)
            session.flush()

        verified = self._verified_in_db(session)
        evaluations = [
            self.evaluator.evaluate(symbol, self._meta_view(symbol, verified.get(symbol, False)))
            for symbol in self.uni_cfg.candidates
        ]
        version = self._version_id(evaluations)

        prev = self._latest_version(session, exclude=version)
        if session.get(UniverseVersion, version) is not None:
            # Idempotent re-build: this exact universe already exists.
            return UniverseBuildResult(
                version=version,
                created=False,
                evaluations=evaluations,
                changes=[],
                prev_version=prev.version if prev else None,
            )

        uv = UniverseVersion(
            version=version,
            exchange_id=self.uni_cfg.exchange_id,
            criteria={
                "phase": 3,
                "policy_version": self.uni_cfg.universe_version,
                "metadata_version": self.meta_cfg.metadata_version,
                "data_version": self.data_cfg.data_version,
                "filters": _filters_dict(self.uni_cfg),
                "filter_report": {e.symbol: e.to_dict() for e in evaluations},
                "active": sorted(e.symbol for e in evaluations if e.status is SymbolStatus.ACTIVE),
            },
            note=f"phase-3 universe build from {self.uni_cfg.universe_version}",
        )
        session.add(uv)
        session.flush()

        for ev in evaluations:
            session.add(
                UniverseMember(
                    universe_version=version,
                    symbol=ev.symbol,
                    status=ev.status,
                    reason=ev.reason(),
                )
            )

        changes = self._log_changes(session, prev, version, evaluations)
        return UniverseBuildResult(
            version=version,
            created=True,
            evaluations=evaluations,
            changes=changes,
            prev_version=prev.version if prev else None,
        )

    # -- change log ------------------------------------------------------ #
    def _latest_version(
        self, session: Session, exclude: str | None = None
    ) -> UniverseVersion | None:
        stmt = select(UniverseVersion).order_by(UniverseVersion.created_at.desc())
        for uv in session.execute(stmt).scalars():
            if exclude is not None and uv.version == exclude:
                continue
            return uv
        return None

    def _log_changes(
        self,
        session: Session,
        prev: UniverseVersion | None,
        version: str,
        evaluations: list[SymbolEvaluation],
    ) -> list[dict]:
        prev_status: dict[str, str] = {}
        if prev is not None:
            prev_status = {m.symbol: m.status.value for m in prev.members}
        new_status = {e.symbol: e.status.value for e in evaluations}
        reasons = {e.symbol: e.reason() for e in evaluations}

        changes: list[dict] = []

        def record(symbol: str, change_type: str, frm: str | None, to: str | None) -> None:
            session.add(
                UniverseChange(
                    universe_version=version,
                    prev_version=prev.version if prev else None,
                    symbol=symbol,
                    change_type=change_type,
                    from_status=frm,
                    to_status=to,
                    reason=reasons.get(symbol, ""),
                )
            )
            changes.append(
                {
                    "symbol": symbol,
                    "change_type": change_type,
                    "from_status": frm,
                    "to_status": to,
                }
            )

        for symbol, to in new_status.items():
            if symbol not in prev_status:
                record(symbol, "entered", None, to)
            elif prev_status[symbol] != to:
                record(symbol, "status_changed", prev_status[symbol], to)
        for symbol, frm in prev_status.items():
            if symbol not in new_status:
                record(symbol, "left", frm, None)
        return changes


def _filters_dict(uni_cfg: UniverseConfig) -> dict:
    f = uni_cfg.filters
    return {
        "quote_currency": f.quote_currency,
        "contract_type": f.contract_type,
        "min_daily_notional_usd": f.min_daily_notional_usd,
        "min_history_bars": f.min_history_bars,
        "min_listing_age_days": f.min_listing_age_days,
        "max_missing_data_pct": f.max_missing_data_pct,
        "max_median_spread_bps": f.max_median_spread_bps,
        "require_metadata_verified": f.require_metadata_verified,
        "require_stable_status": f.require_stable_status,
        "require_funding_history": f.require_funding_history,
        "require_open_interest": f.require_open_interest,
    }


def latest_active_symbols(session: Session, exchange_id: str | None = None) -> list[str]:
    """Active symbols of the most recent universe version (for META / features)."""
    stmt = select(UniverseVersion).order_by(UniverseVersion.created_at.desc())
    for uv in session.execute(stmt).scalars():
        if exchange_id is not None and uv.exchange_id != exchange_id:
            continue
        return sorted(m.symbol for m in uv.members if m.status is SymbolStatus.ACTIVE)
    return []
