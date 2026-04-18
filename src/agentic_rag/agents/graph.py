"""LangGraph graph construction.

::

    question ──► router ──► (direct / refuse) ──► generate ──► END
                    │
                    ├─► rewrite ──► retrieve ──► rerank ──► grade
                    │                                         │
                    │                                         ├─ relevant ─► generate ─► reflect ─► END / retry
                    │                                         └─ irrelevant ─► web_search ─► generate ─► reflect ─► END
                    │
                    └─► rewrite ──► web_search ──► generate ──► reflect ──► END

The *reflect* step can loop back to ``rewrite`` up to
``MAX_REFLECTION_STEPS`` times; this is the self-correction hook.

We use :class:`GraphFactory` so dependencies (LLM, retriever, tools) can
be injected from any caller (tests, API, MCP, CLI).
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

from agentic_rag.agents.nodes import NodeFactory
from agentic_rag.config import get_settings
from agentic_rag.core.logging import get_logger
from agentic_rag.core.types import RouteDecision
from agentic_rag.llm import get_llm
from agentic_rag.llm.base import BaseLLMProvider
from agentic_rag.models.state import AgentState
from agentic_rag.retrieval.base import BaseRetriever
from agentic_rag.retrieval.reranker import CrossEncoderReranker
from agentic_rag.tools.base import ToolRegistry, get_registry
from agentic_rag.tools.calculator import CalculatorTool
from agentic_rag.tools.web_search import WebSearchTool

log = get_logger(__name__)


@dataclass(slots=True)
class GraphFactory:
    """Assembles an executable LangGraph instance.

    Call :meth:`compile` to get a runnable ``StateGraph`` you can invoke
    with ``await graph.ainvoke(initial_state)``.
    """

    llm: BaseLLMProvider
    retriever: BaseRetriever
    reranker: CrossEncoderReranker | None = None
    tools: ToolRegistry | None = None

    def compile(self):  # type: ignore[no-untyped-def]
        settings = get_settings()
        tools = self.tools or self._default_tools()
        nodes = NodeFactory(
            llm=self.llm,
            retriever=self.retriever,
            reranker=self.reranker,
            tools=tools,
        )

        workflow: StateGraph = StateGraph(AgentState)

        # -------- register nodes ---------------------------------------------
        workflow.add_node("router", nodes.router)
        workflow.add_node("rewriter", nodes.rewriter)
        workflow.add_node("retrieve", nodes.retrieve)
        workflow.add_node("rerank", nodes.rerank)
        workflow.add_node("grade", nodes.grade)
        workflow.add_node("web_search", nodes.web_search)
        workflow.add_node("generate", nodes.generate)
        workflow.add_node("reflect", nodes.reflect)

        # -------- edges ------------------------------------------------------
        workflow.add_edge(START, "router")

        workflow.add_conditional_edges(
            "router",
            _route_decision,
            {
                RouteDecision.VECTORSTORE.value: "rewriter",
                RouteDecision.WEB_SEARCH.value: "rewriter",
                RouteDecision.DIRECT.value: "generate",
                RouteDecision.REFUSE.value: "generate",
            },
        )

        # After rewrite, branch by the route decision again (captured in state)
        workflow.add_conditional_edges(
            "rewriter",
            _post_rewrite_branch,
            {
                "retrieve": "retrieve",
                "web_search": "web_search",
            },
        )

        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "grade")

        workflow.add_conditional_edges(
            "grade",
            _grade_branch,
            {
                "generate": "generate",
                "web_search": "web_search",
            },
        )

        workflow.add_edge("web_search", "generate")

        # Only RAG-routed answers are reflected-upon. DIRECT/REFUSE go straight
        # to END to avoid paying the extra LLM hop when there's no context.
        workflow.add_conditional_edges(
            "generate",
            _post_generate_branch,
            {
                "reflect": "reflect",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "reflect",
            _reflect_branch(settings.max_reflection_steps),
            {
                "retry": "rewriter",
                "end": END,
            },
        )

        return workflow.compile()

    # ------------------------------------------------------------------
    @staticmethod
    def _default_tools() -> ToolRegistry:
        reg = get_registry()
        if "web_search" not in reg:
            reg.register(WebSearchTool())
        if "calculator" not in reg:
            reg.register(CalculatorTool())
        return reg


# =============================================================================
# edge-routing functions
# =============================================================================
def _route_decision(state: AgentState) -> str:
    route = state.get("route") or RouteDecision.DIRECT
    return route.value


def _post_rewrite_branch(state: AgentState) -> str:
    route = state.get("route") or RouteDecision.DIRECT
    if route == RouteDecision.WEB_SEARCH:
        return "web_search"
    return "retrieve"


def _grade_branch(state: AgentState) -> str:
    grade = state.get("grade", "irrelevant")
    # If retrieval didn't find anything useful, fall back to the web.
    return "generate" if grade == "relevant" else "web_search"


def _post_generate_branch(state: AgentState) -> str:
    route = state.get("route") or RouteDecision.DIRECT
    # Only reflect on answers that were produced with retrieved context.
    if route in (RouteDecision.VECTORSTORE, RouteDecision.WEB_SEARCH):
        return "reflect"
    return "end"


def _reflect_branch(max_steps: int):
    def _inner(state: AgentState) -> str:
        if state.get("grade") == "relevant":
            return "end"
        if state.get("reflection_step", 0) >= max_steps:
            return "end"
        return "retry"

    return _inner


# =============================================================================
# public factory
# =============================================================================
def build_graph(
    *,
    retriever: BaseRetriever,
    llm: BaseLLMProvider | None = None,
    reranker: CrossEncoderReranker | None = None,
    tools: ToolRegistry | None = None,
):  # type: ignore[no-untyped-def]
    """One-liner for callers that don't need to customise the factory."""
    return GraphFactory(
        llm=llm or get_llm(),
        retriever=retriever,
        reranker=reranker,
        tools=tools,
    ).compile()


__all__ = ["GraphFactory", "build_graph"]
