"""ML Stage 4: Constrained Live Filter (AGENTS.md Section 20, ML Stage 4).

The filter sits between the deterministic candidate generator and the risk
manager. It may BLOCK weak deterministic candidates — and NOTHING else.

Safety invariants enforced unconditionally in code:
  1. output_len <= input_len — the filter cannot create new candidates.
  2. output ⊆ input — only original candidates pass; they are never modified.
  3. hard_blocked candidates are never unblocked (Section 15 hard blockers).
  4. Candidates with strategy_enabled=False are always blocked.
  5. risk_pct is never increased — the filter has no write access to candidates
     (they are frozen dataclasses).

NOT allowed (enforced here and checked by the gate):
  * creating a candidate not in the input list;
  * passing a candidate whose strategy_enabled=False;
  * passing a candidate with hard blockers active (data_fresh=False,
    metadata_verified=False, symbol_tradable=False);
  * modifying any field of the input candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ranking.candidate import Candidate

from .shadow import ShadowBundle

_MODE = "CONSTRAINED_FILTER"

# Hard-blocker flags from the Candidate that cannot be overridden.
_HARD_BLOCKER_FLAGS: tuple[str, ...] = (
    "data_fresh",
    "metadata_verified",
    "symbol_tradable",
    "strategy_enabled",
    "config_live_approved",
)


@dataclass(slots=True)
class FilterDecision:
    """One filter decision: whether a deterministic candidate was passed or blocked."""

    candidate: Candidate
    passed: bool
    block_reason: str = ""
    applied: bool = True  # Stage 4 decisions ARE applied (unlike Stage 2 shadow)


@dataclass
class FilterResult:
    """Outcome of one MLFilter pass over a batch of candidates."""

    passed: list[Candidate]
    blocked: list[Candidate]
    decisions: list[FilterDecision]
    model_version: str
    config_version: str

    @property
    def block_count(self) -> int:
        return len(self.blocked)

    @property
    def pass_count(self) -> int:
        return len(self.passed)

    def block_reasons(self) -> dict[str, str]:
        return {d.candidate.symbol: d.block_reason for d in self.decisions if not d.passed}


class MLFilter:
    """Constrained ML trade filter — ML Stage 4.

    Uses the meta-labeler confidence from :class:`~src.ml.shadow.ShadowBundle`
    to decide whether to pass or block each deterministic candidate.

    Rules:
    - If meta_labeler.probability < min_confidence_to_take → block.
    - Hard-blocked candidates are ALWAYS blocked (cannot unblock).
    - Disabled strategies are ALWAYS blocked.
    - The filter CANNOT create new candidates.
    - The filter CANNOT modify any candidate fields.

    Parameters
    ----------
    min_confidence_to_take:
        Meta-labeler probability threshold below which a candidate is blocked.
        Defaults to 0.4 — candidates below this are filtered out.
    model_version:
        Version tag for logging.
    config_version:
        Config version tag for logging.
    """

    def __init__(
        self,
        min_confidence_to_take: float = 0.4,
        model_version: str = "ml_shadow_0001",
        config_version: str = "cfg_0001",
    ) -> None:
        if not 0.0 <= min_confidence_to_take <= 1.0:
            raise ValueError(
                f"min_confidence_to_take must be in [0, 1], got {min_confidence_to_take}"
            )
        self.min_confidence_to_take = min_confidence_to_take
        self.model_version = model_version
        self.config_version = config_version

    def apply(
        self,
        candidates: list[Candidate],
        bundles: list[ShadowBundle],
        *,
        write_to_db: bool = True,
    ) -> FilterResult:
        """Apply the constrained filter.

        Parameters
        ----------
        candidates:
            Deterministic candidates from the strategy layer.
        bundles:
            One :class:`ShadowBundle` per candidate (same order).
        write_to_db:
            Persist filter decisions to ``shadow_logs`` with
            ``mode="CONSTRAINED_FILTER"`` and ``applied=True``.

        Returns
        -------
        FilterResult
            Candidates that passed the filter; blocked candidates with reasons.
        """
        if len(candidates) != len(bundles):
            raise ValueError(
                f"candidates ({len(candidates)}) and bundles ({len(bundles)}) "
                "must have equal length"
            )

        passed: list[Candidate] = []
        blocked: list[Candidate] = []
        decisions: list[FilterDecision] = []

        for cand, bundle in zip(candidates, bundles, strict=True):
            decision = self._decide(cand, bundle)
            decisions.append(decision)
            if decision.passed:
                passed.append(cand)
            else:
                blocked.append(cand)

        # Safety invariant 1: filter cannot create new candidates.
        assert len(passed) <= len(candidates), "BUG: filter produced more candidates than input"
        # Safety invariant 2: all passed candidates are from the original list.
        assert all(c in candidates for c in passed), "BUG: filter introduced a foreign candidate"

        result = FilterResult(
            passed=passed,
            blocked=blocked,
            decisions=decisions,
            model_version=self.model_version,
            config_version=self.config_version,
        )

        if write_to_db:
            _write_filter_logs(
                decisions, model_version=self.model_version, config_version=self.config_version
            )

        return result

    def _decide(self, cand: Candidate, bundle: ShadowBundle) -> FilterDecision:
        """Return a filter decision for one candidate."""
        # Hard-blocker check (Safety invariant 3 & 4).
        hard_block = _hard_blocker_reason(cand)
        if hard_block:
            return FilterDecision(
                candidate=cand,
                passed=False,
                block_reason=f"hard_blocker:{hard_block}",
                applied=True,
            )

        # ML filter: use meta-labeler confidence.
        meta = bundle.meta_label
        if meta is None:
            # No prediction available — pass through conservatively.
            return FilterDecision(candidate=cand, passed=True, block_reason="", applied=True)

        if meta.probability < self.min_confidence_to_take:
            reason = (
                f"meta_labeler_confidence={meta.probability:.3f} "
                f"< threshold={self.min_confidence_to_take}"
            )
            return FilterDecision(candidate=cand, passed=False, block_reason=reason, applied=True)

        return FilterDecision(candidate=cand, passed=True, block_reason="", applied=True)


def _hard_blocker_reason(cand: Candidate) -> str:
    """Return a non-empty reason string if the candidate has an active hard blocker."""
    if not cand.data_fresh:
        return "data_not_fresh"
    if not cand.metadata_verified:
        return "metadata_not_verified"
    if not cand.symbol_tradable:
        return "symbol_not_tradable"
    if not cand.strategy_enabled:
        return "strategy_disabled"
    if not cand.config_live_approved:
        return "config_not_approved"
    return ""


# --------------------------------------------------------------------------- #
# DB persistence                                                               #
# --------------------------------------------------------------------------- #


def _write_filter_logs(
    decisions: list[FilterDecision],
    *,
    model_version: str,
    config_version: str,
) -> list[int]:
    """Persist filter decisions as shadow_log entries with mode=CONSTRAINED_FILTER."""
    from datetime import UTC, datetime

    from src.db.base import session_scope
    from src.db.models import ShadowLog
    from src.ml.features import candidate_to_row

    ids: list[int] = []
    ts_now = datetime.now(UTC)

    with session_scope() as session:
        for dec in decisions:
            ctx = candidate_to_row(dec.candidate)
            row = ShadowLog(
                ts=ts_now,
                model_id=f"ml_filter_{model_version}",
                model_version=model_version,
                model_type="constrained_filter",
                mode=_MODE,
                symbol=dec.candidate.symbol,
                context_features=ctx,
                prediction={
                    "passed": dec.passed,
                    "block_reason": dec.block_reason,
                },
                confidence=None,
                deterministic_baseline=None,
                applied=dec.applied,
                config_version=config_version,
                created_at=ts_now,
            )
            session.add(row)
            session.flush()
            ids.append(row.id)

    return ids
