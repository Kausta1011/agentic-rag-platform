"""Small, shared, dependency-free enums used across layers."""

from __future__ import annotations

import sys
from enum import Enum

# We require Python 3.11+ in ``pyproject.toml`` (for ``enum.StrEnum``),
# but this shim keeps tooling and pre-3.11 sandboxes working.
if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum

    _StrEnumBase: type = StrEnum
else:  # pragma: no cover - 3.10 compatibility shim

    class _StrEnumBase(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
        """Backport of :class:`enum.StrEnum` for Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)


class RouteDecision(_StrEnumBase):
    """Decision emitted by the :class:`QueryRouter` agent.

    * ``VECTORSTORE`` — the question is answerable from the indexed corpus.
    * ``WEB_SEARCH``  — the question needs fresh / external knowledge.
    * ``DIRECT``      — small talk, maths, or otherwise not needing retrieval.
    * ``REFUSE``      — disallowed request; guardrails will short-circuit.
    """

    VECTORSTORE = "vectorstore"
    WEB_SEARCH = "web_search"
    DIRECT = "direct"
    REFUSE = "refuse"


class ToolName(_StrEnumBase):
    """Canonical names for tools the agent can invoke."""

    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"


__all__ = ["RouteDecision", "ToolName"]
