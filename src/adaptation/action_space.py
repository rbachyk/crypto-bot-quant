"""BoundedAction schema and validation (AGENTS.md Section 21.6).

Every action emitted by a learner must pass through :func:`validate` before it
reaches :mod:`envelope_guard`. Validation clamps or rejects out-of-range values
and logs what was adjusted in ``clamped_fields``.

Hard invariants (cannot be overridden by config):
- ``size_bucket`` ∈ {0.0, 0.25, 0.5, 1.0} — any other float is clamped to the
  nearest valid bucket or rejected (configurable; reject is the safe default).
- ``strategy_weights`` keys must be a subset of the provided allowed set.
- ``param_nudges`` keys must be in the ``registered_tunables`` allow-list.
- ``mode`` must be one of the three literal strings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

VALID_SIZE_BUCKETS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
VALID_MODES = ("SHADOW", "RECOMMEND", "LIVE_BOUNDED")
VALID_EXEC_STYLES = ("maker", "taker", "passive_then_taker")


@dataclass
class BoundedAction:
    """The only thing a learner may emit (Section 21.6).

    Instances are produced by :class:`~src.adaptation.policy_base.Policy` and
    must be passed through :func:`validate` (this module) then
    :func:`~src.adaptation.envelope_guard.enforce` before any downstream use.

    All fields carry their documented semantics from Section 21.6.
    """

    # 1) Strategy weighting among already-validated, enabled strategies only.
    strategy_weights: dict[str, float] = field(default_factory=dict)

    # 2) Bet-size multiplier (bucketed, never continuous-unbounded).
    size_bucket: float = 1.0  # Literal[0.0, 0.25, 0.5, 1.0]

    # 3) Trade filter — may only BLOCK a deterministic candidate, never create one.
    take: bool = True

    # 4) Execution routing (cosmetic to risk; cannot change price/size/stop).
    exec_style: str = "maker"  # Literal["maker", "taker", "passive_then_taker"]

    # 5) Bounded parameter nudges (only explicitly registered tunables).
    param_nudges: dict[str, float] = field(default_factory=dict)

    # Provenance (mandatory, Section 21.6).
    learner_id: str = "unknown"
    learner_version: str = "learner_0001"
    mode: str = "SHADOW"  # Literal["SHADOW", "RECOMMEND", "LIVE_BOUNDED"]
    rationale: str = ""


@dataclass
class ActionBounds:
    """Declared bounds for a policy (loaded from adaptation.yaml)."""

    w_min: float = 0.0
    w_max: float = 2.0
    size_buckets: tuple[float, ...] = VALID_SIZE_BUCKETS
    max_change_per_update: float = 0.10
    max_change_rate: float = 0.25
    registered_tunables: dict[str, dict[str, float]] = field(default_factory=dict)
    allowed_strategies: set[str] = field(default_factory=set)


@dataclass
class ValidationResult:
    """Result of :func:`validate`."""

    action: BoundedAction
    clamped_fields: list[str]
    rejected: bool
    rejection_reason: str | None


def validate(
    action: BoundedAction,
    bounds: ActionBounds,
    *,
    reject_on_bad_bucket: bool = True,
) -> ValidationResult:
    """Validate and clamp a :class:`BoundedAction` to declared bounds.

    Returns a :class:`ValidationResult` with the (possibly mutated) action and a
    list of clamped fields. If ``reject_on_bad_bucket`` is True and the
    ``size_bucket`` is not in ``bounds.size_buckets``, the action is rejected
    (safe default). Otherwise it is clamped to the nearest valid bucket.

    This function must be called before :func:`~src.adaptation.envelope_guard.enforce`.
    """
    result_action = copy.deepcopy(action)
    clamped: list[str] = []

    # --- mode ---------------------------------------------------------------- #
    if result_action.mode not in VALID_MODES:
        return ValidationResult(
            action=result_action,
            clamped_fields=[],
            rejected=True,
            rejection_reason=f"invalid mode={result_action.mode!r}; must be one of {VALID_MODES}",
        )

    # --- exec_style ---------------------------------------------------------- #
    if result_action.exec_style not in VALID_EXEC_STYLES:
        result_action.exec_style = "maker"
        clamped.append("exec_style")

    # --- size_bucket --------------------------------------------------------- #
    if result_action.size_bucket not in bounds.size_buckets:
        if reject_on_bad_bucket:
            return ValidationResult(
                action=result_action,
                clamped_fields=clamped,
                rejected=True,
                rejection_reason=(
                    f"size_bucket={result_action.size_bucket} not in {bounds.size_buckets}"
                ),
            )
        # Clamp to nearest.
        nearest = min(bounds.size_buckets, key=lambda b: abs(b - result_action.size_bucket))
        result_action.size_bucket = nearest
        clamped.append("size_bucket")

    # size_bucket must never exceed 1.0 (Section 21.6 hard invariant).
    if result_action.size_bucket > 1.0:
        result_action.size_bucket = 1.0
        clamped.append("size_bucket")

    # --- strategy_weights ---------------------------------------------------- #
    if bounds.allowed_strategies:
        bad_keys = set(result_action.strategy_weights) - bounds.allowed_strategies
        if bad_keys:
            for k in bad_keys:
                del result_action.strategy_weights[k]
            clamped.append("strategy_weights")

    # Clamp each weight to [w_min, w_max].
    for k, w in list(result_action.strategy_weights.items()):
        cw = max(bounds.w_min, min(bounds.w_max, w))
        if cw != w:
            result_action.strategy_weights[k] = cw
            if "strategy_weights" not in clamped:
                clamped.append("strategy_weights")

    # --- param_nudges -------------------------------------------------------- #
    bad_tunables = set(result_action.param_nudges) - set(bounds.registered_tunables)
    if bad_tunables:
        return ValidationResult(
            action=result_action,
            clamped_fields=clamped,
            rejected=True,
            rejection_reason=(
                f"param_nudges references unregistered tunables: {sorted(bad_tunables)}"
            ),
        )
    for k, v in list(result_action.param_nudges.items()):
        zone = bounds.registered_tunables[k]
        lo, hi = zone.get("lo", float("-inf")), zone.get("hi", float("inf"))
        cv = max(lo, min(hi, v))
        if cv != v:
            result_action.param_nudges[k] = cv
            clamped.append(f"param_nudges.{k}")

    return ValidationResult(
        action=result_action,
        clamped_fields=clamped,
        rejected=False,
        rejection_reason=None,
    )
