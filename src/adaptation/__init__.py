"""Online Learning / Adaptation layer — Phase 11–12 (AGENTS.md Section 21).

Phase 11 delivers the bounded learner in SHADOW mode.
Phase 12 upgrades the RL policy stub to a full simulation-trained policy:

  * ``src/adaptation/action_space.py``   — BoundedAction + validation
  * ``src/adaptation/envelope_guard.py`` — immutable-envelope enforcement
  * ``src/adaptation/policy_base.py``    — Policy protocol + Context/Outcome
  * ``src/adaptation/policies/``         — concrete policy implementations
  * ``src/adaptation/scorer.py``         — shadow scoring + promotion metrics
  * ``src/adaptation/controller.py``     — state machine (SHADOW→RECOMMEND→LIVE)
  * ``src/adaptation/rollback.py``       — circuit breaker + revert-to-fallback
  * ``src/adaptation/versioning.py``     — snapshot / frozen-fallback persistence
  * ``src/adaptation/store.py``          — learner_log persistence
  * ``src/adaptation/config.py``         — adaptation.yaml loader
"""

from src.adaptation.action_space import (
    ActionBounds,
    BoundedAction,
    ValidationResult,
    validate,
)
from src.adaptation.config import AdaptationConfig, load_adaptation_config
from src.adaptation.controller import ControllerDecision, LearnerController, LearnerMode
from src.adaptation.envelope_guard import GuardResult, RiskEnvelope, enforce
from src.adaptation.policies.bandit import GaussianTSBandit
from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
from src.adaptation.policies.rl_policy import RLPolicy, RLPolicyStub
from src.adaptation.policy_base import Context, Outcome, Policy
from src.adaptation.rollback import RollbackEvent, RollbackGuard
from src.adaptation.scorer import ScorerResult, ShadowDecision, score_shadow_decisions
from src.adaptation.store import (
    InMemoryLearnerStore,
    LearnerLogEntry,
    get_memory_sink,
    reset_memory_sink,
    write_learner_log,
)
from src.adaptation.versioning import (
    SnapshotMeta,
    load_frozen_fallback,
    load_snapshot,
    make_frozen_fallback,
    save_snapshot,
)

__all__ = [
    # action_space
    "ActionBounds",
    "BoundedAction",
    "ValidationResult",
    "validate",
    # config
    "AdaptationConfig",
    "load_adaptation_config",
    # controller
    "ControllerDecision",
    "LearnerController",
    "LearnerMode",
    # envelope_guard
    "GuardResult",
    "RiskEnvelope",
    "enforce",
    # policy_base
    "Context",
    "Outcome",
    "Policy",
    # policies
    "GaussianTSBandit",
    "OnlineLogRegPolicy",
    "RLPolicy",
    "RLPolicyStub",
    # rollback
    "RollbackEvent",
    "RollbackGuard",
    # scorer
    "ScorerResult",
    "ShadowDecision",
    "score_shadow_decisions",
    # store
    "InMemoryLearnerStore",
    "LearnerLogEntry",
    "get_memory_sink",
    "reset_memory_sink",
    "write_learner_log",
    # versioning
    "SnapshotMeta",
    "load_frozen_fallback",
    "load_snapshot",
    "make_frozen_fallback",
    "save_snapshot",
]
