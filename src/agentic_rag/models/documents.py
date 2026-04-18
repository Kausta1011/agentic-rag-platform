"""Document / chunk / citation contracts.

All retrieval, ingestion, and generation code speaks in terms of these
models. A :class:`Document` is the raw ingestion unit; a :class:`Chunk`
is a retrieval unit; a :class:`ScoredChunk` wraps a chunk with a score;
a :class:`Citation` is an answer-time pointer back into a chunk.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - 3.10 compatibility


class Document(BaseModel):
    """A raw source document before chunking."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str = Field(..., description="URI or filesystem path of origin")
    title: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=_utcnow)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """Deterministic hash used to de-duplicate across ingests."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """A retrieval-sized slice of a :class:`Document`.

    The ``id`` is deterministic (document_id + ordinal) so re-ingesting
    an unchanged document produces the same chunk IDs — useful for
    idempotent indexing and eval-harness reproducibility.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    ordinal: int = Field(..., ge=0, description="0-based index within the parent document")
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build_id(cls, document_id: str, ordinal: int) -> str:
        return f"{document_id}::{ordinal:05d}"


class ScoredChunk(BaseModel):
    """A chunk paired with retrieval and (optionally) rerank scores."""

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    retrieval_score: float = Field(..., description="Fused retriever score")
    rerank_score: float | None = None
    source_retriever: str = Field(..., description="Which retriever produced it")

    @property
    def effective_score(self) -> float:
        """Rerank score if present, otherwise the retrieval score."""
        return self.rerank_score if self.rerank_score is not None else self.retrieval_score


class Citation(BaseModel):
    """Pointer from an answer sentence back to the chunk it was grounded in."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    source: str
    snippet: str = Field(..., description="The exact quoted excerpt")
    score: float | None = None


__all__ = ["Document", "Chunk", "ScoredChunk", "Citation"]
