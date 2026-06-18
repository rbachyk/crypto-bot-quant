"""Section 21.7: rollback.revert() performs the full atomic revert, not just a freeze."""

from __future__ import annotations

from src.adaptation.action_space import ActionBounds
from src.adaptation.controller import LearnerController, LearnerMode
from src.adaptation.policies.online_logreg import OnlineLogRegPolicy
from src.adaptation.rollback import RollbackEvent, RollbackGuard, write_rollback_log


class _CapturingSink:
    def __init__(self) -> None:
        self.alerts: list = []

    def send(self, alert) -> bool:
        self.alerts.append(alert)
        return True


def _controller() -> LearnerController:
    return LearnerController(
        policy=OnlineLogRegPolicy(learner_id="live_v1"),
        bounds=ActionBounds(),
        frozen_policy=OnlineLogRegPolicy(learner_id="frozen_v1"),
        mode=LearnerMode.LIVE_BOUNDED,
    )


def test_revert_freezes_cancels_alerts_and_logs() -> None:
    sink = _CapturingSink()
    logs: list[RollbackEvent] = []
    guard = RollbackGuard(alert_sink=sink, cancel_orders=lambda: 2, log_writer=logs.append)
    controller = _controller()

    event = guard.revert(controller, trigger="manual_kill", detail="learner kill switch")

    assert controller.is_frozen() and controller.mode is LearnerMode.FROZEN
    assert event.fallback_active is True  # frozen fallback policy is now effective
    assert event.orders_cancelled == 2  # learner NEW orders cancelled
    assert len(sink.alerts) == 1 and sink.alerts[0].severity.value == "critical"
    assert sink.alerts[0].title.startswith("learner_rollback")
    assert logs == [event]  # written to the learner log
    assert event in guard.events()


def test_frozen_controller_serves_fallback_and_never_applies() -> None:
    controller = _controller()
    RollbackGuard(cancel_orders=lambda: 0).revert(controller, trigger="unsafe_regime", detail="R8")
    # _effective_policy() returns the frozen fallback once frozen.
    assert controller._effective_policy().learner_id == "frozen_v1"  # noqa: SLF001


def test_breaker_trigger_routes_through_full_revert() -> None:
    sink = _CapturingSink()
    logs: list[RollbackEvent] = []
    guard = RollbackGuard(alert_sink=sink, cancel_orders=lambda: 1, log_writer=logs.append)
    controller = _controller()
    guard.set_envelope_breaker(True)

    event = guard.check(controller)  # an envelope breaker must revert, not merely flag

    assert event is not None and event.trigger == "envelope_breaker"
    assert controller.is_frozen()
    assert len(sink.alerts) == 1 and len(logs) == 1  # alert + learner_log written


def test_write_rollback_log_persists_a_frozen_applied_false_row() -> None:
    from sqlalchemy import select
    from src.db.base import session_scope
    from src.db.models import LearnerLog

    ev = RollbackEvent(trigger="manual_kill", detail="persist test")
    write_rollback_log(ev, learner_id="rollback_test_learner", learner_version="learner_0001")
    with session_scope() as s:
        row = (
            s.execute(
                select(LearnerLog)
                .where(LearnerLog.learner_id == "rollback_test_learner")
                .order_by(LearnerLog.id.desc())
            )
            .scalars()
            .first()
        )
        assert row is not None
        assert row.mode == "FROZEN" and row.applied is False
        assert "manual_kill" in (row.rollback_event or "")
