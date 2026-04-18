"""Retriever interface.

Every retriever (vector, BM25, hybrid, reranker-wrapped) obeys the same
contract, so they compose freely: a ``HybridRetriever`` wraps two child
retrievers, a reranker wraps any retriever, and an eval harness can plug
any of them behind the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from agentic_rag.models.documents import Chunk, ScoredChunk


class BaseRetriever(ABC):
    """Strategy interface for all retrievers."""

    #: Short human id for logging / observability, e.g. ``"vector"``.
    name: str

    @abstractmethod
    async def add(self, chunks: Sequence[Chunk]) -> None:
        """Index a batch of chunks. Idempotent on chunk.id."""

    @abstractmethod
    async def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        """Return the top-k chunks for *query*, best first."""

    async def clear(self) -> None:  # pragma: no cover - optional
        """Optionally remove all indexed chunks. Default: no-op."""
        return None


__all__ = ["BaseRetriever"]
