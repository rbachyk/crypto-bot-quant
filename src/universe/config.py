"""Loader for ``configs/universe.yaml`` — the Dynamic Symbol Universe contract.

Turns the YAML into a typed :class:`UniverseConfig` read by the Universe
Manager, the per-symbol filters and the UNIV gate, so they never drift
(Section 4 config-driven behaviour; Section 9 universe responsibilities).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.config.settings import REPO_ROOT

UNIVERSE_YAML = REPO_ROOT / "configs" / "universe.yaml"


@dataclass(frozen=True, slots=True)
class UniverseFilters:
    quote_currency: str = "USDT"
    contract_type: str = "perpetual"
    min_daily_notional_usd: float = 10_000_000.0
    min_history_bars: int = 1000
    min_listing_age_days: float = 1.0
    max_missing_data_pct: float = 1.0
    max_median_spread_bps: float = 25.0
    require_metadata_verified: bool = True
    require_stable_status: bool = True
    require_funding_history: bool = True
    require_open_interest: bool = True


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    exchange_id: str
    universe_version: str
    candidates: list[str]
    eval_timeframe: str
    filters: UniverseFilters


@lru_cache
def load_universe_config(path: str | None = None) -> UniverseConfig:
    yaml_path = Path(path) if path else UNIVERSE_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data = raw["universe"]
    f = data.get("filters", {})
    filters = UniverseFilters(
        quote_currency=str(f.get("quote_currency", "USDT")),
        contract_type=str(f.get("contract_type", "perpetual")),
        min_daily_notional_usd=float(f.get("min_daily_notional_usd", 10_000_000.0)),
        min_history_bars=int(f.get("min_history_bars", 1000)),
        min_listing_age_days=float(f.get("min_listing_age_days", 1.0)),
        max_missing_data_pct=float(f.get("max_missing_data_pct", 1.0)),
        max_median_spread_bps=float(f.get("max_median_spread_bps", 25.0)),
        require_metadata_verified=bool(f.get("require_metadata_verified", True)),
        require_stable_status=bool(f.get("require_stable_status", True)),
        require_funding_history=bool(f.get("require_funding_history", True)),
        require_open_interest=bool(f.get("require_open_interest", True)),
    )
    return UniverseConfig(
        exchange_id=str(data["exchange_id"]),
        universe_version=str(data.get("universe_version", "univ_0001")),
        candidates=list(data["candidates"]),
        eval_timeframe=str(data.get("eval_timeframe", "1m")),
        filters=filters,
    )
