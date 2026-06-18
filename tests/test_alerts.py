"""Alert transport tests (AGENTS.md Appendix B.14): the push sinks must be fail-safe and
the composite must fan out to transports while keeping the local buffer the dashboard reads."""

from __future__ import annotations

from src.monitoring import (
    Alert,
    AlertSeverity,
    AlertSink,
    CompositeAlertSink,
    EmailAlertSink,
    LogAlertSink,
    TelegramAlertSink,
    get_alert_sink,
    reset_alert_sink,
)


def _alert() -> Alert:
    return Alert(
        title="t",
        severity=AlertSeverity.WARNING,
        component="c",
        environment="local",
        recommended_action="x",
    )


class _Recorder(AlertSink):
    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return True


class _Failer(AlertSink):
    def send(self, alert: Alert) -> bool:
        return False


def test_composite_fans_out_and_keeps_local_buffer() -> None:
    log = LogAlertSink()
    rec = _Recorder()
    sink = CompositeAlertSink(log, [rec])
    assert sink.send(_alert()) is True
    assert len(rec.sent) == 1  # delivered to the transport
    assert len(sink.recent()) == 1  # and buffered locally for the dashboard/gate


def test_composite_degrades_on_transport_failure_but_never_raises() -> None:
    log = LogAlertSink()
    sink = CompositeAlertSink(log, [_Failer()])
    assert sink.send(_alert()) is False  # a failed transport degrades the result...
    assert len(sink.recent()) == 1  # ...but the alert is still recorded locally


def test_telegram_sink_is_failsafe(monkeypatch) -> None:
    import httpx

    def boom(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", boom)
    assert TelegramAlertSink("token", "chat").send(_alert()) is False  # no exception propagates


def test_email_sink_is_failsafe() -> None:
    # Connection refused (127.0.0.1:1) must be swallowed and reported as a failed delivery.
    sink = EmailAlertSink("127.0.0.1", 1, "from@x.test", ["to@x.test"], use_tls=False)
    assert sink.send(_alert()) is False


def test_get_alert_sink_is_composite_with_recent() -> None:
    reset_alert_sink()
    try:
        sink = get_alert_sink()
        assert isinstance(sink, CompositeAlertSink)
        sink.send(_alert())
        assert any(a.title == "t" for a in sink.recent())
    finally:
        reset_alert_sink()
