"""Centralised exception hierarchy.

Using a single base class (:class:`AgenticRAGError`) lets higher layers
(API, CLI, MCP) catch all application errors with one ``except`` while
still distinguishing them by subtype for precise error reporting.
"""

from __future__ import annotations

from typing import Any


class AgenticRAGError(Exception):
    """Base class for every custom error raised by the platform."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "message": self.message,
            "context": self.context,
        }


class ConfigurationError(AgenticRAGError):
    """Raised when required configuration (e.g. an API key) is missing or invalid."""


class LLMProviderError(AgenticRAGError):
    """Raised on any failure communicating with an LLM provider."""


class RetrievalError(AgenticRAGError):
    """Raised by retrievers / vector stores when a lookup fails."""


class IngestionError(AgenticRAGError):
    """Raised when a document cannot be loaded, parsed, or indexed."""


class ToolExecutionError(AgenticRAGError):
    """Raised when an agent tool (web search, calculator, …) fails."""


class GuardrailViolationError(AgenticRAGError):
    """Raised when an input or output fails a guardrail check.

    Attributes
    ----------
    rule:
        The name of the guardrail rule that triggered.
    stage:
        ``"input"`` or ``"output"``.
    """

    def __init__(
        self,
        message: str,
        *,
        rule: str,
        stage: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.rule = rule
        self.stage = stage

    def to_dict(self) -> dict[str, Any]:  # type: ignore[override]
        data = super().to_dict()
        data["rule"] = self.rule
        data["stage"] = self.stage
        return data


__all__ = [
    "AgenticRAGError",
    "ConfigurationError",
    "LLMProviderError",
    "RetrievalError",
    "IngestionError",
    "ToolExecutionError",
    "GuardrailViolationError",
]
