"""Structured logging via loguru.

Every module should do::

    from agentic_rag.core.logging import get_logger
    log = get_logger(__name__)

Rather than configuring logging themselves. :func:`configure_logging` is
called once at application start (from the API, CLI, or MCP entrypoints).
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger as _loguru_logger

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the global loguru logger exactly once.

    Idempotent: calling twice has no additional effect. Emits JSON-ish
    structured records to stderr so downstream log shippers (Loki, CW,
    etc.) can parse them.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{extra} | {message}"
        ),
        backtrace=True,
        diagnose=False,  # never leak variable values into logs
        enqueue=False,
    )
    _CONFIGURED = True


def get_logger(name: str, **default_extra: Any) -> Any:
    """Return a logger bound to ``name`` with optional default extras."""
    if not _CONFIGURED:
        configure_logging()
    return _loguru_logger.bind(component=name, **default_extra)


__all__ = ["configure_logging", "get_logger"]
