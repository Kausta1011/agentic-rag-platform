"""Chroma-backed dense vector retriever.

Chroma is a good default local vector DB — zero-ops, persistent,
embeddable. The Strategy interface means this can be swapped for
Qdrant / pgvector / Pinecone without touching callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import RetrievalError
from agentic_rag.core.logging import get_logger
from agentic_rag.llm.base import EmbeddingProvider
from agentic_rag.models.documents import Chunk, ScoredChunk
from agentic_rag.retrieval.base import BaseRetriever

log = get_logger(__name__)


class ChromaVectorStore(BaseRetriever):
    """Dense retriever backed by a persistent Chroma collection."""

    name = "vector"

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        persist_dir: Path | None = None,
        collection: str | None = None,
    ) -> None:
        try:
            import chromadb  # type: ignore[import-not-found]
            from chromadb.config import Settings as ChromaSettings  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RetrievalError("chromadb not installed") from exc

        settings = get_settings()
        persist_dir = persist_dir or settings.chroma_persist_dir
        collection = collection or settings.chroma_collection
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings = embeddings
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # We supply our own embeddings — pass ``embedding_function=None`` to
        # prevent Chroma from downloading its default model.
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    async def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.content for c in chunks]
        try:
            vectors = await self._embeddings.embed(texts)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"embedding failed during indexing: {exc}") from exc

        # Chroma's python client is sync — it's fine to call it from async.
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "document_id": c.document_id,
                    "ordinal": c.ordinal,
                    **{
                        k: v
                        for k, v in c.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                }
                for c in chunks
            ],
        )
        log.bind(n=len(chunks)).info("indexed chunks")

    # ------------------------------------------------------------------
    async def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        if not query.strip():
            return []
        try:
            [vector] = await self._embeddings.embed([query])
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"embedding failed at query time: {exc}") from exc

        try:
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"chroma query failed: {exc}") from exc

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        scored: list[ScoredChunk] = []
        for cid, content, meta, dist in zip(ids, docs, metas, distances, strict=False):
            meta = dict(meta or {})
            # cosine distance → similarity in [0, 1]
            similarity = max(0.0, 1.0 - float(dist))
            chunk = Chunk(
                id=cid,
                document_id=meta.pop("document_id", ""),
                ordinal=int(meta.pop("ordinal", 0)),
                content=content or "",
                metadata=meta,
            )
            scored.append(
                ScoredChunk(chunk=chunk, retrieval_score=similarity, source_retriever=self.name)
            )
        return scored

    async def clear(self) -> None:
        self._client.reset()


__all__ = ["ChromaVectorStore"]
