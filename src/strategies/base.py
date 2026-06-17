"""Strategy hypothesis declaration (AGENTS.md Section 12/13).

Every research candidate must declare a full hypothesis BEFORE it trades
(Section 13 Stage 1 Draft): the edge it claims, the market condition it needs, its
data requirements, entry/exit/invalidation rules, risk + cost assumptions, the
failure modes it expects, the validation tests it must pass, and the criteria for
promotion. :class:`StrategyHypothesis` is that declaration, carried on the
strategy object and emitted verbatim into the Strategy Report so a reviewer sees
exactly what was hypothesised versus what the evidence showed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StrategyHypothesis:
    """Full pre-registration of a strategy's claim (Section 12/13)."""

    family: str  # "A".."H" (Section 12)
    name: str
    hypothesis: str
    market_condition: str
    edge_source: str
    data_requirements: tuple[str, ...]
    entry: str
    exit: str
    invalidation: str
    risk_assumptions: str
    cost_assumptions: str
    failure_modes: tuple[str, ...]
    validation_tests: tuple[str, ...]
    promotion_criteria: str
    exit_profile: str  # "mean_reversion" | "momentum" | "volatility" (Section 12)
    notes: str = ""
    references: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "market_condition": self.market_condition,
            "edge_source": self.edge_source,
            "data_requirements": list(self.data_requirements),
            "entry": self.entry,
            "exit": self.exit,
            "invalidation": self.invalidation,
            "risk_assumptions": self.risk_assumptions,
            "cost_assumptions": self.cost_assumptions,
            "failure_modes": list(self.failure_modes),
            "validation_tests": list(self.validation_tests),
            "promotion_criteria": self.promotion_criteria,
            "exit_profile": self.exit_profile,
            "notes": self.notes,
            "references": list(self.references),
        }
