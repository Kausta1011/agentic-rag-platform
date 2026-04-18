"""FastAPI dependency-injection wiring.

One place that assembles the full application graph at startup and
exposes it to endpoints via ``Depends(get_service)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from agentic_rag.agents.graph import build_graph
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.logging import configure_logging
from agentic_rag.guardrails.input_guard import InputGuard
from agentic_rag.guardrails.output_guard import OutputGuard
from agentic_rag.llm import get_embedding_provider, get_llm
from agentic_rag.observability.tracer import configure_tracing
from agentic_rag.retrieval.bm25_retriever import BM25Retriever
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.reranker import CrossEncoderReranker
from agentic_rag.retrieval.vector_store import ChromaVectorStore


@dataclass
class Service:
    """Container holding the fully-wired pipeline."""

    settings: Settings
    graph: object
    input_guard: InputGuard
    output_guard: OutputGuard


@lru_cache(maxsize=1)
def get_service() -> Service:
    """Singleton per process — FastAPI DI will re-use this."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing()

    embeddings = get_embedding_provider()
    vector = ChromaVectorStore(embeddings=embeddings)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever([vector, bm25], strategy="weighted", alpha=settings.hybrid_alpha)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    graph = build_graph(retriever=hybrid, reranker=reranker)

    llm = get_llm()
    return Service(
        settings=settings,
        graph=graph,
        input_guard=InputGuard(
            max_length=4000,
            redact_pii=True,
            block_injection=True,
        ) if settings.enable_input_guard else InputGuard(redact_pii=False, block_injection=False),
        output_guard=OutputGuard(llm, min_faithfulness=settings.min_faithfulness_score)
        if settings.enable_output_guard
        else OutputGuard(llm, min_faithfulness=0.0),
    )


__all__ = ["Service", "get_service"]
