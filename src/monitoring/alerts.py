"""Alerting (AGENTS.md Appendix B.14, Section 24).

The always-on :class:`LogAlertSink` is the structured-log + in-memory buffer the dashboard
alert center and the Monitoring gate read. Real push transports — :class:`TelegramAlertSink`
and :class:`EmailAlertSink` — are layered on by :class:`CompositeAlertSink` whenever they are
configured (``ALERT_TELEGRAM_*`` / ``ALERT_EMAIL_*``); they are fail-safe (a transport error
never raises into the caller, it just returns False and logs). With nothing configured the sink
behaves exactly like the log-only sink, so tests and gates are unaffected.

Every alert carries severity, timestamp, component, environment and a recommended action.
"""

from __future__ import annotations

import abc
import enum
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import structlog

_log = structlog.get_logger("alerts")

#: How long the same condition (severity+title+component+environment) waits before it is PUSHED
#: again. The log sink is never throttled — only the outbound transports.
_PUSH_COOLDOWN_S = 300.0
#: Upper bound on remembered conditions, so a long-lived process cannot grow the map without end.
_PUSH_KEY_CAP = 512


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


def _format_alert(alert: Alert) -> tuple[str, str]:
    """(subject, body) text shared by the push transports."""
    subject = f"[{alert.severity.value.upper()}] {alert.title}"
    body = (
        f"{alert.title}\n"
        f"severity: {alert.severity.value}\n"
        f"component: {alert.component} ({alert.environment})\n"
        f"action: {alert.recommended_action}\n"
        f"escalation: {alert.escalation_path}\n"
        f"ts: {alert.ts}"
    )
    return subject, body


class TelegramAlertSink(AlertSink):
    """Push alerts to a Telegram chat via the Bot API. Fail-safe (never raises)."""

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 5.0) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout

    def send(self, alert: Alert) -> bool:
        import httpx

        subject, body = _format_alert(alert)
        try:
            resp = httpx.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={"chat_id": self._chat_id, "text": f"{subject}\n{body}"},
                timeout=self._timeout,
            )
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            _log.warning("telegram_alert_failed", error=str(exc))
            return False


class EmailAlertSink(AlertSink):
    """Push alerts via SMTP email. Fail-safe (never raises)."""

    def __init__(
        self,
        host: str,
        port: int,
        sender: str,
        recipients: list[str],
        *,
        username: str = "",
        password: str = "",
        use_tls: bool = True,
        timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._sender = sender
        self._recipients = recipients
        self._username = username
        self._password = password
        self._use_tls = use_tls
        self._timeout = timeout

    def send(self, alert: Alert) -> bool:
        import smtplib
        import ssl
        from email.message import EmailMessage

        subject, body = _format_alert(alert)
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = self._sender
            msg["To"] = ", ".join(self._recipients)
            msg.set_content(body)
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as smtp:
                if self._use_tls:
                    smtp.starttls(context=ssl.create_default_context())
                if self._username:
                    smtp.login(self._username, self._password)
                smtp.send_message(msg)
            return True
        except Exception as exc:  # noqa: BLE001
            _log.warning("email_alert_failed", error=str(exc))
            return False


class CompositeAlertSink(AlertSink):
    """Fan an alert out to the always-on log sink plus any configured push transports.

    Delegates ``recent()`` to the log sink so the dashboard/Monitoring gate keep working.
    ``send()`` returns True only if EVERY sink delivered (a failed transport degrades the
    result but never raises)."""

    def __init__(self, log_sink: LogAlertSink, transports: list[AlertSink]) -> None:
        self._log = log_sink
        self._transports = transports
        self._last_push: dict[tuple[str, str, str, str], float] = {}

    def send(self, alert: Alert) -> bool:
        # The LOG sink always receives every alert: the dashboard alert center, the Monitoring
        # gate's test-alert check and the audit trail must stay complete.
        ok = self._log.send(alert)
        # PUSH transports are throttled per distinct condition. The conditions that alert here are
        # persistent by nature — a foreign position stays foreign until an operator acts — and the
        # per-tick reconciler re-raises them on every tick. Unthrottled, that is one blocking HTTP
        # POST (or SMTP session) per tick on the TRADING loop, forever, and Telegram rate-limits
        # the flood anyway. Re-notifying every _PUSH_COOLDOWN_S is escalation; every tick is noise
        # that buries the alert it is trying to raise.
        key = (alert.severity.value, alert.title, alert.component, alert.environment)
        now = time.monotonic()
        last = self._last_push.get(key)
        if last is not None and now - last < _PUSH_COOLDOWN_S:
            return ok
        self._last_push[key] = now
        if len(self._last_push) > _PUSH_KEY_CAP:  # bound the map on a long-lived process
            for stale, _ts in sorted(self._last_push.items(), key=lambda kv: kv[1])[:_PUSH_KEY_CAP
                                                                                    // 2]:
                self._last_push.pop(stale, None)
        for transport in self._transports:
            try:
                ok = transport.send(alert) and ok
            except Exception as exc:  # noqa: BLE001
                _log.warning("alert_transport_failed", error=str(exc))
                ok = False
        return ok

    def recent(self, limit: int = 50) -> list[Alert]:
        return self._log.recent(limit)

    @property
    def transports(self) -> list[AlertSink]:
        return list(self._transports)


def _build_transports() -> list[AlertSink]:
    from src.config import get_settings

    s = get_settings()
    out: list[AlertSink] = []
    if s.alert_telegram_bot_token and s.alert_telegram_chat_id:
        out.append(TelegramAlertSink(s.alert_telegram_bot_token, s.alert_telegram_chat_id))
    if s.alert_email_host and s.alert_email_from and s.alert_email_to:
        recipients = [r.strip() for r in s.alert_email_to.split(",") if r.strip()]
        out.append(
            EmailAlertSink(
                s.alert_email_host,
                s.alert_email_port,
                s.alert_email_from,
                recipients,
                username=s.alert_email_username,
                password=s.alert_email_password,
                use_tls=s.alert_email_use_tls,
            )
        )
    return out


_log_sink = LogAlertSink()
_composite: CompositeAlertSink | None = None


def get_alert_sink() -> CompositeAlertSink:
    """Return the process-wide alert sink (log buffer + any configured push transports)."""
    global _composite
    if _composite is None:
        _composite = CompositeAlertSink(_log_sink, _build_transports())
    return _composite


def reset_alert_sink() -> None:
    """Drop the cached sink so configured transports are rebuilt (tests / config reload)."""
    global _composite
    _composite = None
