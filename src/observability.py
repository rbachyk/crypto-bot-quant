"""Logging configuration (structlog, JSON to stderr).

Logs go to **stderr** so that machine-readable JSON emitted on **stdout** by the
gate runner CLI stays clean for the orchestrator to parse.
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True
