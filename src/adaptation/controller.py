"""Learner state machine — SHADOW → RECOMMEND → LIVE_BOUNDED (AGENTS.md Section 21.7).

The :class:`LearnerController` orchestrates the full decision path:

    deterministic candidate
        → policy.decide(ctx)
        → action_space.validate(action, bounds)
        → envelope_guard.enforce(action)
        → (LIVE_BOUNDED only) risk_manager.approve(order) [external]
        → store.write_learner_log(...)

In SHADOW and RECOMMEND modes ``update()`` is still called so the model trains
online; but the resulting action is logged with ``applied=False``.

Promotion (SHADOW → RECOMMEND, RECOMMEND → LIVE_BOUNDED) is MANUAL only:
a human operator must approve after the LEARN-PROMO-S / LEARN-PROMO-L gates
pass (Section 21.3, 27). This class never auto-promotes.

Freeze / frozen-fallback semantics (Section 21.7; audit H17)
------------------------------------------------------------
On :meth:`~LearnerController.freeze` the controller loads the frozen-fallback
policy snapshot (the last manually-approved policy, via
:mod:`~src.adaptation.versioning`) into ``frozen_policy`` and from then on runs
THAT policy in SHADOW mode only: its actions are validated/guard-enforced and
logged, but ``applied`` is always False — a frozen learner never influences
live orders again until manual recovery. If no fallback snapshot exists the
controller reports the truth: it is *learner-disabled* — every :meth:`run`
returns a rejected decision saying so, and ``fallback_active`` is False so
alerts/logs never claim a fallback is running when it is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from src.adaptation.action_space import ActionBounds, BoundedAction, validate
from src.adaptation.envelope_guard import enforce
from src.adaptation.policy_base import Context, Outcome, Policy

logger = logging.getLogger(__name__)


class LearnerMode(str, Enum):
    SHADOW = "SHADOW"
    RECOMMEND = "RECOMMEND"
    LIVE_BOUNDED = "LIVE_BOUNDED"
    FROZEN = "FROZEN"


@dataclass
class ControllerDecision:
    """Result of one controller.run() call."""

    action: BoundedAction | None
    applied: bool
    clamped_fields: list[str]
    rejected: bool
    rejection_reason: str | None
    mode: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class LearnerController:
    """Orchestrates the learner decision path (Section 21.7).

    Parameters
    ----------
    policy:
        The active learner policy (OnlineLogRegPolicy, GaussianTSBandit, etc.).
    bounds:
        The declared action bounds from adaptation.yaml.
    mode:
        Current operational mode (default SHADOW — never auto-promoted).
    frozen_policy:
        The fallback policy that becomes effective after :meth:`freeze`. May be
        injected directly (tests) or loaded lazily from
        ``fallback_snapshot_path`` when :meth:`freeze` fires.
    fallback_snapshot_path:
        Path to the frozen-fallback snapshot file (adaptation.yaml
        ``frozen_fallback_policy``). Loaded via
        :func:`~src.adaptation.versioning.load_frozen_fallback` on freeze.
    """

    policy: Policy
    bounds: ActionBounds
    mode: LearnerMode = LearnerMode.SHADOW
    frozen_policy: Policy | None = None
    fallback_snapshot_path: Path | str | None = None
    frozen_reason: str = field(default="", init=False)
    _decision_count: int = field(default=0, init=False, repr=False)
    _frozen: bool = field(default=False, init=False, repr=False)

    def run(
        self,
        ctx: Context,
        *,
        active_strategies: set[str] | None = None,
        hard_blocked: bool = False,
    ) -> ControllerDecision:
        """Run one decision cycle.

        The action produced is validated and guard-enforced in every mode.
        ``applied=True`` only in LIVE_BOUNDED; the Risk Layer independently
        approves before any order is placed (not modelled here).

        ``hard_blocked``: the caller's verdict that the deterministic system hard-blocked this
        candidate (setup-quality blocker / toxic spread / stale data). The immutable invariant is
        that a learner ``take=True`` can NEVER resurrect a hard-blocked candidate (Section 21.6);
        the envelope guard documents it but cannot see the candidate, so it is enforced HERE (M36).
        A no-trade regime on the context is treated as a hard block too, as a safety net.

        When frozen: with a fallback policy loaded, the fallback decides in
        SHADOW mode (never applied); with no fallback the learner is disabled
        and every call returns an honest rejected decision (audit H17).
        """
        self._decision_count += 1
        if self._frozen and self.frozen_policy is None:
            return ControllerDecision(
                action=None,
                applied=False,
                clamped_fields=[],
                rejected=True,
                rejection_reason=(
                    "learner frozen and DISABLED — no frozen-fallback snapshot available "
                    f"(freeze reason: {self.frozen_reason or 'unspecified'}); "
                    "restore a fallback snapshot and recover manually (Section 21.7)"
                ),
                mode=self.mode.value,
            )

        effective_policy = self._effective_policy()
        raw_action = effective_policy.decide(ctx)
        # Force mode field to match controller mode. While frozen the fallback
        # runs in SHADOW (a valid action mode — "FROZEN" is controller state,
        # not an action mode; audit H17) and is never applied.
        raw_action.mode = "SHADOW" if self._frozen else self.mode.value

        # Validate then guard.
        val = validate(raw_action, self.bounds)
        if val.rejected:
            return ControllerDecision(
                action=None,
                applied=False,
                clamped_fields=val.clamped_fields,
                rejected=True,
                rejection_reason=val.rejection_reason,
                mode=self.mode.value,
            )

        guard = enforce(
            val.action,
            active_strategies=active_strategies,
        )
        if guard.rejected:
            return ControllerDecision(
                action=None,
                applied=False,
                clamped_fields=guard.clamped_fields,
                rejected=True,
                rejection_reason=guard.rejection_reason,
                mode=self.mode.value,
            )

        applied = self.mode is LearnerMode.LIVE_BOUNDED and not self._frozen

        # Immutable invariant (Section 21.6, M36): a learner take=True cannot RESURRECT a candidate
        # the deterministic system hard-blocked. Only enforced on the APPLIED path (LIVE_BOUNDED) —
        # the invariant is about EXECUTION, and rejecting in SHADOW/RECOMMEND would drop the action
        # entirely, so the shadow log and the online-learning update never see those (hard-blocked,
        # take=True) rows the policy most needs to learn from. In shadow/recommend the action flows
        # with applied=False (it changes nothing); only when it would actually be applied is it
        # rejected. A no-trade regime on the context counts as a hard block.
        if applied and guard.action.take and (hard_blocked or self._is_no_trade_regime(ctx)):
            return ControllerDecision(
                action=None,
                applied=False,
                clamped_fields=guard.clamped_fields,
                rejected=True,
                rejection_reason=(
                    "take=True cannot resurrect a hard-blocked candidate "
                    f"(regime={ctx.regime!r}, hard_blocked={hard_blocked})"
                ),
                mode=self.mode.value,
            )
        return ControllerDecision(
            action=guard.action,
            applied=applied,
            clamped_fields=val.clamped_fields + guard.clamped_fields,
            rejected=False,
            rejection_reason=None,
            mode=self.mode.value,
        )

    def record_outcome(
        self,
        ctx: Context,
        decision: ControllerDecision,
        outcome: Outcome,
    ) -> None:
        """Feed realized outcome back to the policy for online updates.

        In SHADOW and RECOMMEND modes the call is forwarded so the model trains;
        the controller guarantees the resulting action was never applied.
        While frozen this is a no-op: the frozen fallback is the approved
        snapshot and must not drift from what was approved (Section 21.7).
        """
        if decision.action is None or self._frozen:
            return
        self._effective_policy().update(ctx, decision.action, outcome)

    def freeze(self, reason: str = "") -> None:
        """Freeze the controller (rollback trigger — Section 21.7).

        Records ``reason``, switches to FROZEN, and activates the frozen
        fallback policy: an already-injected ``frozen_policy`` wins; otherwise
        the snapshot at ``fallback_snapshot_path`` is loaded via versioning.
        If neither exists the controller is learner-disabled (see :meth:`run`).
        """
        self._frozen = True
        self.mode = LearnerMode.FROZEN
        self.frozen_reason = reason or "unspecified"
        if self.frozen_policy is None:
            self.frozen_policy = self._load_fallback_policy()
        if self.frozen_policy is None:
            logger.error(
                "learner frozen (reason=%s) with NO frozen-fallback snapshot — "
                "learner is disabled until a fallback is restored manually",
                self.frozen_reason,
            )
        else:
            logger.warning(
                "learner frozen (reason=%s); frozen-fallback policy active in SHADOW mode",
                self.frozen_reason,
            )

    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def fallback_active(self) -> bool:
        """True only when frozen AND a fallback policy is actually loaded."""
        return self._frozen and self.frozen_policy is not None

    @property
    def learner_disabled(self) -> bool:
        """True when frozen with no fallback: no learner decisions flow at all."""
        return self._frozen and self.frozen_policy is None

    def _load_fallback_policy(self) -> Policy | None:
        """Load the frozen-fallback snapshot into a fresh policy instance.

        Returns None (never raises) when the snapshot is missing or unloadable —
        the caller then reports the learner-disabled state honestly.
        """
        if self.fallback_snapshot_path is None:
            return None
        from src.adaptation.versioning import load_frozen_fallback

        path = Path(self.fallback_snapshot_path)
        try:
            blob = load_frozen_fallback(path.parent, path.name)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error("frozen-fallback snapshot unavailable at %s: %s", path, exc)
            return None
        try:
            fallback = type(self.policy)()
            fallback.load(blob)
        except Exception as exc:  # noqa: BLE001 — corrupt snapshot must not crash the freeze path
            logger.error("frozen-fallback snapshot at %s failed to load: %s", path, exc)
            return None
        return fallback

    @staticmethod
    def _is_no_trade_regime(ctx: Context) -> bool:
        """Whether the context's regime is a Section-11 no-trade regime (a hard block). Accepts
        both the R-code label and the lowercase name form."""
        regime = ctx.regime
        if not regime:
            return False
        from src.regime.detector import NO_TRADE_REGIMES

        r = str(regime)
        if r in NO_TRADE_REGIMES:
            return True
        # Tolerate lowercase name forms (e.g. "high_vol_chop" ↔ R4_HIGH_VOL_CHOP).
        ru = r.upper()
        return any(ru in code or code.split("_", 1)[-1] == ru for code in NO_TRADE_REGIMES)

    def _effective_policy(self) -> Policy:
        if self._frozen and self.frozen_policy is not None:
            return self.frozen_policy
        return self.policy
