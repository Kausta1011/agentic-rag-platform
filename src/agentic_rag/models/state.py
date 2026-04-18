"""LangGraph state schema.

LangGraph state is a ``TypedDict`` — every node receives the full state
and returns a *partial* update which LangGraph merges in. We declare the
schema once here so every node has a single source of truth.

The design keeps **intermediate** artefacts (rewrites, retrieved chunks,
grader verdicts, reflection notes) on the state so they can be inspected
in traces and evaluation — critical for debuggability in production.
"""

from __future__ import annotations

from typing import Annotated, TypedDict
from uuid import uuid4

from langgraph.graph.message import add_messages  # type: ignore[import-not-found]

from agentic_rag.core.types import RouteDecision
from agentic_rag.models.documents import Citation, ScoredChunk


class AgentState(TypedDict, total=False):
    """The shape of the graph's runtime state.

    ``total=False`` — LangGraph only guarantees the keys a node writes;
    downstream nodes must tolerate missing keys.
    """

    # --- identity ---------------------------------------------------------
    session_id: str
    trace_id: str

    # --- inputs -----------------------------------------------------------
    question: str            # original user question
    rewritten_question: str  # post-rewrite, the query actually retrieved with

    # --- routing ----------------------------------------------------------
    route: RouteDecision

    # --- retrieval --------------------------------------------------------
    retrieved: list[ScoredChunk]
    reranked: list[ScoredChunk]
    web_results: list[dict]  # raw search-engine hits (if routed to web)

    # --- grading / reflection --------------------------------------------
    grade: str                  # "relevant" / "irrelevant"
    reflection_step: int        # incremented each time we loop back
    reflection_notes: list[str]

    # --- output -----------------------------------------------------------
    answer: str
    citations: list[Citation]
    faithfulness_score: float
    tokens: dict[str, int]

    # --- chat history / streaming ----------------------------------------
    # `add_messages` is the LangGraph reducer for LC messages — enables
    # multi-turn conversations without blowing away earlier turns.
    messages: Annotated[list, add_messages]


def build_initial_state(question: str, session_id: str | None = None) -> AgentState:
    """Factory for a freshly-initialised state dict.

    Centralising construction guarantees every graph invocation starts
    with consistent defaults (critical for reproducible evaluations).
    """
    return AgentState(
        session_id=session_id or str(uuid4()),
        trace_id=str(uuid4()),
        question=question,
        rewritten_question=question,
        reflection_step=0,
        reflection_notes=[],
        retrieved=[],
        reranked=[],
        web_results=[],
        citations=[],
        tokens={"input": 0, "output": 0},
        messages=[],
    )


__all__ = ["AgentState", "build_initial_state"]
