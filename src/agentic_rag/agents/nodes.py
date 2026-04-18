"""LangGraph node implementations.

Each node is a pure ``async def`` that takes :class:`AgentState` and
returns a partial state update. Nodes never talk to each other directly
— all coupling is through the state schema. This keeps them trivially
unit-testable (pass a dict in, assert the returned dict).

The nodes themselves intentionally contain very little business logic
— they delegate to the ``retrieval`` / ``tools`` / ``llm`` packages and
translate results to state updates. This makes the graph file
declarative and readable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentic_rag.agents.prompts import (
    GENERATOR_SYSTEM,
    GRADER_SYSTEM,
    REFLECTOR_SYSTEM,
    REWRITER_SYSTEM,
    build_router_system,
)
from agentic_rag.config import get_settings
from agentic_rag.core.logging import get_logger
from agentic_rag.core.types import RouteDecision
from agentic_rag.llm.base import BaseLLMProvider
from agentic_rag.models.documents import Citation, ScoredChunk
from agentic_rag.models.state import AgentState
from agentic_rag.observability.metrics import get_metrics
from agentic_rag.observability.tracer import get_tracer
from agentic_rag.retrieval.base import BaseRetriever
from agentic_rag.retrieval.reranker import CrossEncoderReranker
from agentic_rag.tools.base import ToolRegistry

log = get_logger(__name__)
_tracer = get_tracer()
_metrics = get_metrics()

_JSON_RE = re.compile(r"\{[\s\S]*\}")


# =============================================================================
# helpers
# =============================================================================
def _extract_json(raw: str) -> dict[str, Any]:
    match = _JSON_RE.search(raw or "")
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _accumulate_tokens(state: AgentState, prompt_tokens: int, completion_tokens: int) -> dict:
    tokens = dict(state.get("tokens") or {"input": 0, "output": 0})
    tokens["input"] = tokens.get("input", 0) + prompt_tokens
    tokens["output"] = tokens.get("output", 0) + completion_tokens
    return tokens


# =============================================================================
# Nodes
# =============================================================================
class NodeFactory:
    """Builds closures bound to the shared dependencies.

    Using a class-as-namespace instead of free functions lets us inject
    the LLM / retriever / tools just once at graph construction time —
    each node returns a partial of :class:`AgentState`.
    """

    def __init__(
        self,
        *,
        llm: BaseLLMProvider,
        retriever: BaseRetriever,
        reranker: CrossEncoderReranker | None,
        tools: ToolRegistry,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.tools = tools
        self.settings = get_settings()

    # -----------------------------------------------------------------
    # ROUTER
    # -----------------------------------------------------------------
    async def router(self, state: AgentState) -> dict:
        question = state["question"]

        # Escape hatch — skip the router entirely for pure-RAG deployments.
        # Retrieval is cheap and the grader filters out bad hits, so for
        # corpora-only use cases this is both faster and more reliable.
        if self.settings.disable_router:
            log.bind(route="vectorstore", reason="DISABLE_ROUTER=true").info("router decision")
            _metrics.incr("route.vectorstore")
            return {"route": RouteDecision.VECTORSTORE}

        system_prompt = build_router_system(self.settings.corpus_description)

        with _tracer.start_as_current_span("node.router"), _metrics.timer("node.router"):
            resp = await self.llm.generate(
                f"USER QUESTION:\n{question}\n\nReturn the JSON only.",
                system=system_prompt,
                temperature=0.0,
                max_tokens=150,
            )
            data = _extract_json(resp.text)
            # SAFE DEFAULT: when the LLM is uncertain or returns garbage,
            # fall back to ``vectorstore`` rather than ``direct``. Skipping
            # retrieval is the worst failure mode — it silently hallucinates.
            raw = str(data.get("route", "vectorstore")).lower()
            try:
                route = RouteDecision(raw)
            except ValueError:
                route = RouteDecision.VECTORSTORE
            log.bind(route=route.value, reason=data.get("reason")).info("router decision")
            _metrics.incr(f"route.{route.value}")
            return {
                "route": route,
                "tokens": _accumulate_tokens(state, resp.input_tokens, resp.output_tokens),
            }

    # -----------------------------------------------------------------
    # REWRITER
    # -----------------------------------------------------------------
    async def rewriter(self, state: AgentState) -> dict:
        question = state.get("rewritten_question") or state["question"]
        with _tracer.start_as_current_span("node.rewriter"), _metrics.timer("node.rewriter"):
            resp = await self.llm.generate(
                question, system=REWRITER_SYSTEM, temperature=0.0, max_tokens=80
            )
            rewritten = resp.text.strip().strip('"') or question
            return {
                "rewritten_question": rewritten,
                "tokens": _accumulate_tokens(state, resp.input_tokens, resp.output_tokens),
            }

    # -----------------------------------------------------------------
    # RETRIEVE
    # -----------------------------------------------------------------
    async def retrieve(self, state: AgentState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        with _tracer.start_as_current_span("node.retrieve"), _metrics.timer("node.retrieve"):
            hits = await self.retriever.retrieve(query, top_k=self.settings.retrieval_top_k)
            log.bind(n=len(hits)).info("retrieved")
            _metrics.incr("retrieve.hits", value=len(hits))
            return {"retrieved": hits}

    # -----------------------------------------------------------------
    # RERANK
    # -----------------------------------------------------------------
    async def rerank(self, state: AgentState) -> dict:
        candidates: list[ScoredChunk] = state.get("retrieved") or []
        if not candidates or self.reranker is None:
            return {"reranked": candidates[: self.settings.rerank_top_k]}

        with _tracer.start_as_current_span("node.rerank"), _metrics.timer("node.rerank"):
            reranked = await self.reranker.rerank(
                state.get("rewritten_question") or state["question"],
                candidates,
                top_k=self.settings.rerank_top_k,
            )
            return {"reranked": reranked}

    # -----------------------------------------------------------------
    # GRADE
    # -----------------------------------------------------------------
    async def grade(self, state: AgentState) -> dict:
        candidates: list[ScoredChunk] = state.get("reranked") or []
        if not candidates:
            return {"grade": "irrelevant"}

        # Grade the top candidate only — cheap and usually sufficient.
        top = candidates[0]
        with _tracer.start_as_current_span("node.grade"), _metrics.timer("node.grade"):
            prompt = (
                f"USER QUESTION:\n{state['question']}\n\n"
                f"CANDIDATE PASSAGE:\n{top.chunk.content}\n\nReturn JSON only."
            )
            resp = await self.llm.generate(
                prompt, system=GRADER_SYSTEM, temperature=0.0, max_tokens=80
            )
            data = _extract_json(resp.text)
            verdict = "relevant" if bool(data.get("relevant")) else "irrelevant"
            _metrics.incr(f"grade.{verdict}")
            return {
                "grade": verdict,
                "tokens": _accumulate_tokens(state, resp.input_tokens, resp.output_tokens),
            }

    # -----------------------------------------------------------------
    # WEB SEARCH
    # -----------------------------------------------------------------
    async def web_search(self, state: AgentState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        with _tracer.start_as_current_span("node.web_search"), _metrics.timer("node.web_search"):
            tool = self.tools.get("web_search")
            result = await tool.safe_run(query=query)
            hits = result.data if result.ok and isinstance(result.data, list) else []
            log.bind(n=len(hits), ok=result.ok).info("web results")
            return {"web_results": hits}

    # -----------------------------------------------------------------
    # GENERATE
    # -----------------------------------------------------------------
    async def generate(self, state: AgentState) -> dict:
        question = state["question"]
        context_entries, citations = self._build_context(state)
        if not context_entries:
            prompt = f"USER QUESTION:\n{question}\n\n(There is no retrieved context.)"
        else:
            ctx = "\n\n".join(context_entries)
            prompt = f"CONTEXT:\n{ctx}\n\nUSER QUESTION:\n{question}"

        with _tracer.start_as_current_span("node.generate"), _metrics.timer("node.generate"):
            resp = await self.llm.generate(
                prompt,
                system=GENERATOR_SYSTEM,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
            return {
                "answer": resp.text.strip(),
                "citations": citations,
                "tokens": _accumulate_tokens(state, resp.input_tokens, resp.output_tokens),
            }

    # -----------------------------------------------------------------
    # REFLECT (self-correction)
    # -----------------------------------------------------------------
    async def reflect(self, state: AgentState) -> dict:
        step = int(state.get("reflection_step", 0))
        if step >= self.settings.max_reflection_steps:
            return {"reflection_step": step}  # forces END

        with _tracer.start_as_current_span("node.reflect"), _metrics.timer("node.reflect"):
            ctx_entries, _ = self._build_context(state)
            ctx_text = "\n\n".join(ctx_entries) if ctx_entries else "(none)"
            prompt = (
                f"USER QUESTION:\n{state['question']}\n\n"
                f"CONTEXT:\n{ctx_text}\n\n"
                f"DRAFT ANSWER:\n{state.get('answer', '')}\n\nReturn JSON only."
            )
            resp = await self.llm.generate(
                prompt, system=REFLECTOR_SYSTEM, temperature=0.0, max_tokens=200
            )
            data = _extract_json(resp.text)

        sufficient = bool(data.get("sufficient"))
        new_query = str(data.get("rewrite_query", "")).strip()
        notes = list(state.get("reflection_notes") or [])
        notes.append(str(data.get("missing", "")))

        update: dict[str, Any] = {
            "reflection_step": step + 1,
            "reflection_notes": notes,
            "tokens": _accumulate_tokens(state, resp.input_tokens, resp.output_tokens),
        }
        if not sufficient and new_query:
            update["rewritten_question"] = new_query
        update["grade"] = "relevant" if sufficient else "irrelevant"
        _metrics.incr("reflection.sufficient" if sufficient else "reflection.retry")
        return update

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _build_context(state: AgentState) -> tuple[list[str], list[Citation]]:
        """Compose the context block used by both ``generate`` and ``reflect``."""
        reranked: list[ScoredChunk] = state.get("reranked") or []
        web: list[dict] = state.get("web_results") or []

        entries: list[str] = []
        citations: list[Citation] = []

        for i, sc in enumerate(reranked, start=1):
            entries.append(
                f"[{i}] (source: {sc.chunk.metadata.get('source', 'unknown')})\n{sc.chunk.content}"
            )
            citations.append(
                Citation(
                    chunk_id=sc.chunk.id,
                    document_id=sc.chunk.document_id,
                    source=str(sc.chunk.metadata.get("source", "unknown")),
                    snippet=sc.chunk.content[:300],
                    score=sc.effective_score,
                )
            )

        offset = len(entries)
        for j, hit in enumerate(web, start=1):
            idx = offset + j
            title = hit.get("title", "")
            url = hit.get("url", "")
            content = hit.get("content", "")
            entries.append(f"[{idx}] (web: {url})\n{title}\n{content}")
            citations.append(
                Citation(
                    chunk_id=f"web::{idx}",
                    document_id=f"web::{url}",
                    source=url,
                    snippet=content[:300],
                )
            )
        return entries, citations


__all__ = ["NodeFactory"]
