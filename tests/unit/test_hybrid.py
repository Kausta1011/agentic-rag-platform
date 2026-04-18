"""Hybrid fusion tests."""

from __future__ import annotations

import pytest

from agentic_rag.models.documents import Chunk, ScoredChunk
from agentic_rag.retrieval.bm25_retriever import BM25Retriever
from agentic_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def _sc(chunk_id: str, score: float, src: str = "a") -> ScoredChunk:
    chunk = Chunk(id=chunk_id, document_id="d", ordinal=0, content=chunk_id)
    return ScoredChunk(chunk=chunk, retrieval_score=score, source_retriever=src)


def test_rrf_fuses_rankings_and_prefers_agreement():
    a = [_sc("x", 1.0, "a"), _sc("y", 0.9, "a"), _sc("z", 0.8, "a")]
    b = [_sc("y", 1.0, "b"), _sc("x", 0.95, "b"), _sc("w", 0.9, "b")]
    fused = reciprocal_rank_fusion([a, b])
    # x and y both appear in both rankings, so they should lead
    assert fused[0].chunk.id in {"x", "y"}
    assert fused[1].chunk.id in {"x", "y"}


def test_hybrid_retriever_composes_child_results():
    # build two BM25 retrievers with different corpora — exercises the
    # weighted fusion code path end-to-end.
    import asyncio

    async def _go():
        a = BM25Retriever()
        b = BM25Retriever()
        await a.add([Chunk(id="a::1", document_id="a", ordinal=0, content="apple banana cherry")])
        await b.add([Chunk(id="b::1", document_id="b", ordinal=0, content="apple durian elderberry")])

        hybrid = HybridRetriever([a, b], strategy="weighted", alpha=0.5)
        hits = await hybrid.retrieve("apple", top_k=5)
        ids = {h.chunk.id for h in hits}
        assert ids == {"a::1", "b::1"}

    asyncio.get_event_loop().run_until_complete(_go())


def test_hybrid_retriever_rejects_empty_children():
    with pytest.raises(ValueError):
        HybridRetriever([])
