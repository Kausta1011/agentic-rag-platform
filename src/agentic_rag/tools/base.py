"""Tool base class and singleton registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentic_rag.core.exceptions import ToolExecutionError
from agentic_rag.core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ToolResult:
    """Normalised return payload for every tool."""

    ok: bool
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """All agent tools inherit from this."""

    #: Canonical name used in the agent prompt / MCP registration.
    name: str
    description: str

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult: ...

    # ------------------------------------------------------------------
    async def safe_run(self, **kwargs: Any) -> ToolResult:
        """Wrap :meth:`run` so exceptions become :class:`ToolResult`.

        Called by the agent layer so a buggy / unavailable tool never
        crashes the entire graph — we just mark the call as failed and
        let the planner recover.
        """
        try:
            return await self.run(**kwargs)
        except ToolExecutionError as exc:
            log.bind(tool=self.name).warning(f"tool error: {exc}")
            return ToolResult(ok=False, error=exc.message, meta=exc.context)
        except Exception as exc:  # noqa: BLE001
            log.bind(tool=self.name).exception("unexpected tool failure")
            return ToolResult(ok=False, error=str(exc))


class ToolRegistry:
    """A small registry so the agent layer can look up tools by name.

    Kept deliberately simple — no dynamic discovery magic; register
    explicitly at startup so tool availability is trivially greppable.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolExecutionError(f"unknown tool: {name}") from exc

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


_REGISTRY: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ToolRegistry()
    return _REGISTRY


__all__ = ["BaseTool", "ToolRegistry", "ToolResult", "get_registry"]
