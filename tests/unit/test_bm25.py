"""BM25 retriever tests."""

from __future__ import annotations

import pytest

from agentic_rag.retrieval.bm25_retriever import BM25Retriever


@pytest.mark.asyncio
async def test_bm25_retrieves_relevant_chunk(sample_chunks):
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    hits = await retriever.retrieve("reciprocal rank fusion", top_k=3)
    assert hits, "BM25 should return at least one hit"
    assert "Reciprocal rank fusion" in hits[0].chunk.content


@pytest.mark.asyncio
async def test_bm25_scores_normalised(sample_chunks):
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    hits = await retriever.retrieve("BM25 lexical ranking", top_k=3)
    for h in hits:
        assert 0.0 <= h.retrieval_score <= 1.0


@pytest.mark.asyncio
async def test_bm25_upsert_is_idempotent(sample_chunks):
    retriever = BM25Retriever()
    await retriever.add(sample_chunks)
    await retriever.add(sample_chunks)
    hits = await retriever.retrieve("LangGraph", top_k=5)
    # Should not return duplicates
    ids = [h.chunk.id for h in hits]
    assert len(ids) == len(set(ids))
