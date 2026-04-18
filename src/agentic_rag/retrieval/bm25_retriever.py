"""In-memory BM25 lexical retriever.

Combining BM25 with a dense retriever recovers a lot of the queries
where one alone fails — exact-match lookups, rare tokens, acronyms.
For a production corpus you'd likely swap this for OpenSearch or Vespa;
the contract is identical.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from agentic_rag.core.logging import get_logger
from agentic_rag.models.documents import Chunk, ScoredChunk
from agentic_rag.retrieval.base import BaseRetriever

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever(BaseRetriever):
    """BM25 (Okapi) lexical retriever using rank_bm25."""

    name = "bm25"

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._corpus: list[list[str]] = []
        self._ordered_ids: list[str] = []
        self._index = None  # built lazily

    async def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        for c in chunks:
            if c.id in self._chunks:
                # Upsert: replace the existing row.
                idx = self._ordered_ids.index(c.id)
                self._corpus[idx] = _tokenize(c.content)
                self._chunks[c.id] = c
            else:
                self._chunks[c.id] = c
                self._corpus.append(_tokenize(c.content))
                self._ordered_ids.append(c.id)
        self._index = None  # invalidate

    async def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        if not self._corpus:
            return []
        self._ensure_index()
        tokens = _tokenize(query)
        if not tokens:
            return []
        assert self._index is not None
        scores = self._index.get_scores(tokens)

        if not len(scores):
            return []

        # Keep only candidates that actually share a token with the query.
        # This bypasses BM25-Okapi's negative-IDF pathology on tiny
        # corpora where a term appears in every document (IDF goes
        # negative and every score becomes ≤ 0 despite the obvious match).
        query_set = set(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda pair: pair[1], reverse=True
        )
        kept: list[tuple[int, float]] = [
            (i, float(s))
            for i, s in ranked
            if query_set & set(self._corpus[i])
        ][:top_k]

        if not kept:
            return []

        # Min-max normalise to [0, 1] so the fusion layer can blend
        # lexical scores with cosine similarities safely.
        raw = [s for _, s in kept]
        lo, hi = min(raw), max(raw)
        if hi - lo < 1e-9:
            normed = [1.0] * len(raw)
        else:
            normed = [(s - lo) / (hi - lo) for s in raw]

        return [
            ScoredChunk(
                chunk=self._chunks[self._ordered_ids[i]],
                retrieval_score=float(n),
                source_retriever=self.name,
            )
            for (i, _), n in zip(kept, normed, strict=True)
        ]

    async def clear(self) -> None:
        self._chunks.clear()
        self._corpus.clear()
        self._ordered_ids.clear()
        self._index = None

    # ------------------------------------------------------------------
    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("rank_bm25 not installed") from exc
        self._index = BM25Okapi(self._corpus)


__all__ = ["BM25Retriever"]
