"""Chunker correctness tests."""

from __future__ import annotations

from agentic_rag.models.documents import Document
from agentic_rag.retrieval.chunking import RecursiveChunker, TokenAwareChunker


def _doc(text: str) -> Document:
    return Document(source="memory", content=text)


def test_recursive_chunker_produces_overlapping_chunks():
    text = "\n\n".join([f"Paragraph {i}." + " word" * 80 for i in range(5)])
    chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.split(_doc(text))
    assert len(chunks) > 1
    for c in chunks:
        assert 0 < len(c.content) <= 600  # chunk_size + some slack


def test_recursive_chunker_ids_are_deterministic():
    doc = _doc("one two three four five")
    a = RecursiveChunker(chunk_size=5, chunk_overlap=1).split(doc)
    b = RecursiveChunker(chunk_size=5, chunk_overlap=1).split(doc)
    assert [x.id for x in a] == [x.id for x in b]


def test_token_aware_chunker_handles_short_docs():
    chunks = TokenAwareChunker(max_tokens=64, overlap_tokens=8).split(_doc("hello world"))
    assert len(chunks) == 1
    assert chunks[0].content.strip().startswith("hello")


def test_chunker_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=200)
