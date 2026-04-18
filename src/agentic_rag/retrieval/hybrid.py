"""Hybrid retrieval = weighted fusion of dense + lexical.

Two fusion strategies are provided:

* **Weighted sum** (parameterised by ``alpha``) — intuitive and tunable.
* **Reciprocal Rank Fusion** (RRF) — rank-based, score-free, very robust.

Both live behind the same ``HybridRetriever`` so you can A/B test fusion
strategies from config without touching call-sites.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from agentic_rag.core.logging import get_logger
from agentic_rag.models.documents import Chunk, ScoredChunk
from agentic_rag.retrieval.base import BaseRetriever

log = get_logger(__name__)

FusionStrategy = Literal["weighted", "rrf"]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[ScoredChunk]],
    *,
    k: int = 60,
) -> list[ScoredChunk]:
    """Classic RRF: ``score = Σ 1/(k + rank_i)``.

    *k=60* is the value recommended by the original paper (Cormack et al.,
    2009). Higher *k* flattens the contribution of top ranks.
    """
    fused: dict[str, tuple[Chunk, float, str]] = {}
    for ranking in rankings:
        for rank, sc in enumerate(ranking):
            contribution = 1.0 / (k + rank + 1)
            prev = fused.get(sc.chunk.id)
            if prev is None:
                fused[sc.chunk.id] = (sc.chunk, contribution, sc.source_retriever)
            else:
                chunk, score, src = prev
                fused[sc.chunk.id] = (chunk, score + contribution, f"{src}+{sc.source_retriever}")

    items = sorted(fused.values(), key=lambda t: t[1], reverse=True)
    return [
        ScoredChunk(chunk=ch, retrieval_score=score, source_retriever=src)
        for ch, score, src in items
    ]


class HybridRetriever(BaseRetriever):
    """Fuses results from any number of child retrievers."""

    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        *,
        strategy: FusionStrategy = "weighted",
        alpha: float = 0.5,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one child retriever")
        if strategy == "weighted" and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self._retrievers = list(retrievers)
        self.strategy = strategy
        self.alpha = alpha

    async def add(self, chunks: Sequence[Chunk]) -> None:
        for r in self._retrievers:
            await r.add(chunks)

    async def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        # Over-fetch from each child so fusion has material to work with.
        per_child_k = max(top_k * 2, 10)
        rankings: list[list[ScoredChunk]] = []
        for r in self._retrievers:
            rankings.append(await r.retrieve(query, top_k=per_child_k))

        if self.strategy == "rrf":
            fused = reciprocal_rank_fusion(rankings)
        else:
            fused = self._weighted_fusion(rankings)

        return fused[:top_k]

    async def clear(self) -> None:
        for r in self._retrievers:
            await r.clear()

    # ------------------------------------------------------------------
    def _weighted_fusion(self, rankings: list[list[ScoredChunk]]) -> list[ScoredChunk]:
        """Expects exactly 2 child rankings; first is dense, second lexical."""
        if len(rankings) != 2:
            log.warning(
                "weighted fusion expected 2 rankings; falling back to RRF",
                n=len(rankings),
            )
            return reciprocal_rank_fusion(rankings)

        dense, lex = rankings
        dense_scores = {sc.chunk.id: sc for sc in dense}
        lex_scores = {sc.chunk.id: sc for sc in lex}

        all_ids = set(dense_scores) | set(lex_scores)
        fused: list[ScoredChunk] = []
        for cid in all_ids:
            d = dense_scores.get(cid)
            lex_sc = lex_scores.get(cid)
            d_s = d.retrieval_score if d else 0.0
            l_s = lex_sc.retrieval_score if lex_sc else 0.0
            score = self.alpha * d_s + (1.0 - self.alpha) * l_s
            chunk = (d or lex_sc).chunk  # type: ignore[union-attr]
            src = "+".join(
                filter(None, [d and d.source_retriever, lex_sc and lex_sc.source_retriever])
            )
            fused.append(
                ScoredChunk(chunk=chunk, retrieval_score=score, source_retriever=src)
            )
        return sorted(fused, key=lambda s: s.retrieval_score, reverse=True)


__all__ = ["HybridRetriever", "reciprocal_rank_fusion", "FusionStrategy"]
