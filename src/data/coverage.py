"""Coverage computation for the DATA-COV gate (Appendix A DATA-COV).

DATA-COV passes when *every* active universe symbol has *all* required series
covering the window with zero unfilled gaps. Symbols the exchange genuinely
lacks history for are excluded (``insufficient_history``) rather than failing
the gate forever (DATA-COV remediation step 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.config import DataConfig
from src.data.gaps import GapReport, find_gaps
from src.data.schema import ms_to_iso
from src.data.store import SeriesStore


@dataclass(slots=True)
class CoverageReport:
    window: dict
    required_series: int = 0
    covered_series: int = 0
    uncovered: list[GapReport] = field(default_factory=list)
    insufficient_history: list[str] = field(default_factory=list)

    @property
    def covered(self) -> bool:
        return not self.uncovered

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "required_series": self.required_series,
            "covered_series": self.covered_series,
            "uncovered_count": len(self.uncovered),
            "uncovered": [g.to_dict() for g in self.uncovered],
            "insufficient_history": self.insufficient_history,
        }


def compute_coverage(store: SeriesStore, cfg: DataConfig) -> CoverageReport:
    start, end = cfg.window_start_ms, cfg.window_end_ms
    report = CoverageReport(
        window={"from": ms_to_iso(start), "to": ms_to_iso(end)},
        insufficient_history=list(cfg.insufficient_history),
    )
    for symbol in cfg.active_symbols():
        for key in cfg.required_keys(symbol):
            report.required_series += 1
            gap = find_gaps(store, key, start, end)
            if gap.covered:
                report.covered_series += 1
            else:
                report.uncovered.append(gap)
    return report
