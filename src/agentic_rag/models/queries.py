"""Public request / response contracts exposed by the API and MCP server."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.core.types import RouteDecision
from agentic_rag.models.documents import Citation


class QueryRequest(BaseModel):
    """User-facing request."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4_000)
    session_id: str | None = Field(
        default=None, description="Optional conversation id for tracing / memory"
    )
    top_k: int | None = Field(default=None, ge=1, le=50)
    stream: bool = False


class AnswerResponse(BaseModel):
    """Final payload returned to the caller."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: RouteDecision
    reflection_steps: int = 0
    faithfulness_score: float | None = None
    latency_ms: float | None = None
    tokens: dict[str, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["QueryRequest", "AnswerResponse"]
