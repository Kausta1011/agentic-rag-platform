"""Cross-encoder reranker.

A cross-encoder reads the *query + passage* together and scores them
jointly — far more accurate than pure vector cosine. We only apply it to
the top-N candidates from the hybrid retriever to keep latency bounded.

Lazy import of sentence-transformers so importing this module is cheap
(important for eval harnesses and unit tests).
"""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.core.logging import get_logger
from agentic_rag.models.documents import ScoredChunk

log = get_logger(__name__)


class CrossEncoderReranker:
    """Wraps a HF cross-encoder and rescores retrieved chunks."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None  # loaded lazily on first call

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for reranking; pip install it"
            ) from exc
        log.info(f"loading cross-encoder model: {self.model_name}")
        self._model = CrossEncoder(self.model_name)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_k: int,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        self._ensure_loaded()
        assert self._model is not None

        pairs = [(query, c.chunk.content) for c in candidates]
        # ``predict`` is sync but fast — fine inside async here because
        # it's CPU-bound and reasonably brief. If latency matters, wrap
        # in ``asyncio.to_thread``.
        import asyncio

        scores = await asyncio.to_thread(self._model.predict, pairs)

        rescored = [
            ScoredChunk(
                chunk=c.chunk,
                retrieval_score=c.retrieval_score,
                rerank_score=float(s),
                source_retriever=c.source_retriever,
            )
            for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return rescored[:top_k]


__all__ = ["CrossEncoderReranker"]
