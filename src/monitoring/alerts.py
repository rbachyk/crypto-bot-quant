"""Alerting skeleton (AGENTS.md Appendix B.14).

Phase 1 ships the alert model and a default log-based sink plus a dashboard
in-memory sink, so the Monitoring gate can verify an alert can be raised and
delivered end-to-end. Telegram/email transports are wired in Phase 13.

Every alert carries severity, timestamp, component, environment and a
recommended action (Appendix B.14).
"""

from __future__ import annotations

import abc
import enum
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import structlog

_log = structlog.get_logger("alerts")


class AlertSeverity(str, enum.Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass(slots=True)
class Alert:
    title: str
    severity: AlertSeverity
    component: str
    environment: str
    recommended_action: str = ""
    session_id: str | None = None
    dashboard_link: str | None = None
    escalation_path: str = "if unacknowledged in 15 min -> escalate"
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


class AlertSink(abc.ABC):
    @abc.abstractmethod
    def send(self, alert: Alert) -> bool:
        """Deliver an alert. Returns True on successful delivery."""


class LogAlertSink(AlertSink):
    """Default sink: structured-log delivery, always available locally.

    Keeps a bounded in-memory ring buffer so the dashboard alert center and the
    Monitoring gate's test-alert check can read recently delivered alerts.
    """

    def __init__(self, capacity: int = 256) -> None:
        self._buffer: deque[Alert] = deque(maxlen=capacity)

    def send(self, alert: Alert) -> bool:
        self._buffer.append(alert)
        _log.info("alert", **alert.to_dict())
        return True

    def recent(self, limit: int = 50) -> list[Alert]:
        return list(self._buffer)[-limit:]


_default_sink = LogAlertSink()


def get_alert_sink() -> LogAlertSink:
    """Return the process-wide alert sink."""
    return _default_sink
