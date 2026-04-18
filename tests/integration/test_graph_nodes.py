"""Integration-level tests that exercise the graph node implementations
against the FakeLLM / fake embeddings — no network, no real models.
"""

from __future__ import annotations

import pytest

from agentic_rag.agents.nodes import NodeFactory
from agentic_rag.core.types import RouteDecision
from agentic_rag.models.state import build_initial_state
from agentic_rag.retrieval.bm25_retriever import BM25Retriever
from agentic_rag.tools.base import ToolRegistry
from agentic_rag.tools.calculator import CalculatorTool
from agentic_rag.tools.web_search import WebSearchTool


@pytest.mark.asyncio
async def test_router_node_extracts_route(fake_llm, sample_chunks):
    fake_llm.canned = ['{"route": "vectorstore", "reason": "local corpus"}']
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    nodes = NodeFactory(llm=fake_llm, retriever=retriever, reranker=None, tools=ToolRegistry())

    state = build_initial_state("what is langgraph")
    update = await nodes.router(state)
    assert update["route"] == RouteDecision.VECTORSTORE


@pytest.mark.asyncio
async def test_router_defaults_to_vectorstore_on_garbage(fake_llm, sample_chunks):
    """Regression: when the LLM returns nonsense, the router must fall back
    to VECTORSTORE — never to DIRECT. Falling through to DIRECT was the
    cause of the 'answers with no retrieval' bug seen in production."""
    fake_llm.canned = ["absolute garbage, not even JSON"]
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    nodes = NodeFactory(llm=fake_llm, retriever=retriever, reranker=None, tools=ToolRegistry())

    state = build_initial_state("what is langgraph")
    update = await nodes.router(state)
    assert update["route"] == RouteDecision.VECTORSTORE


@pytest.mark.asyncio
async def test_router_respects_disable_flag(fake_llm, sample_chunks):
    """When DISABLE_ROUTER is true, the router must skip the LLM entirely
    and short-circuit to VECTORSTORE."""
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    nodes = NodeFactory(llm=fake_llm, retriever=retriever, reranker=None, tools=ToolRegistry())
    original = nodes.settings.disable_router
    nodes.settings.disable_router = True  # type: ignore[misc]
    try:
        state = build_initial_state("anything at all")
        update = await nodes.router(state)
        assert update["route"] == RouteDecision.VECTORSTORE
        assert fake_llm.prompts == []  # no LLM call made
    finally:
        # Reset — Settings is a cached singleton; leaking state across
        # tests causes flaky failures elsewhere.
        nodes.settings.disable_router = original  # type: ignore[misc]


@pytest.mark.asyncio
async def test_router_prompt_includes_corpus_description(fake_llm, sample_chunks):
    """The corpus description must make it into the system prompt so the
    LLM can route against it."""
    fake_llm.canned = ['{"route": "vectorstore"}']
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    nodes = NodeFactory(llm=fake_llm, retriever=retriever, reranker=None, tools=ToolRegistry())
    # Defensive: explicitly enable the router — the Settings singleton is
    # shared with other tests and may have been toggled off.
    nodes.settings.disable_router = False  # type: ignore[misc]
    nodes.settings.corpus_description = "MARKER_CORPUS_TOKEN_xyz"  # type: ignore[misc]

    # Capture the system prompt that gets passed in.
    captured: dict[str, str | None] = {"system": None}
    orig = fake_llm.generate

    async def _spy(prompt, *, system=None, **kw):
        captured["system"] = system
        return await orig(prompt, system=system, **kw)

    fake_llm.generate = _spy  # type: ignore[assignment]

    state = build_initial_state("what is in the corpus?")
    await nodes.router(state)

    assert captured["system"] is not None
    assert "MARKER_CORPUS_TOKEN_xyz" in captured["system"]


@pytest.mark.asyncio
async def test_retrieve_node_populates_state(fake_llm, sample_chunks):
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    nodes = NodeFactory(llm=fake_llm, retriever=retriever, reranker=None, tools=ToolRegistry())

    state = build_initial_state("cross encoder relevance")
    state["rewritten_question"] = "cross encoder relevance"
    update = await nodes.retrieve(state)
    assert update["retrieved"]


@pytest.mark.asyncio
async def test_grade_node_parses_json(fake_llm, sample_chunks):
    fake_llm.canned = ['{"relevant": true, "reason": "it matches"}']
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    nodes = NodeFactory(llm=fake_llm, retriever=retriever, reranker=None, tools=ToolRegistry())

    state = build_initial_state("langgraph")
    state["reranked"] = sample_chunks_to_scored(sample_chunks)
    update = await nodes.grade(state)
    assert update["grade"] == "relevant"


@pytest.mark.asyncio
async def test_generate_node_cites_context(fake_llm, sample_chunks):
    fake_llm.canned = ["LangGraph is a finite-state agent framework [1]."]
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    nodes = NodeFactory(llm=fake_llm, retriever=retriever, reranker=None, tools=ToolRegistry())

    state = build_initial_state("what is langgraph")
    state["reranked"] = sample_chunks_to_scored(sample_chunks)
    update = await nodes.generate(state)
    assert "LangGraph" in update["answer"]
    assert len(update["citations"]) == len(sample_chunks)


@pytest.mark.asyncio
async def test_tool_registry_dispatches():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(WebSearchTool())
    assert "calculator" in reg
    assert "web_search" in reg
    out = await reg.get("calculator").run(expression="1+1")
    assert out.ok and out.data == 2.0


# ---------------------------------------------------------------------------
def sample_chunks_to_scored(chunks):
    from agentic_rag.models.documents import ScoredChunk

    return [
        ScoredChunk(chunk=c, retrieval_score=1.0 - (i * 0.1), source_retriever="unit")
        for i, c in enumerate(chunks)
    ]
