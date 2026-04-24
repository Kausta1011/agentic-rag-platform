"""RAG evaluation metrics.

These are RAGAS-flavoured metrics (Es, Shahul, et al., 2023), reproduced
in-house so the project has no runtime dependency on RAGAS. LLM-based
metrics call the project's own LLM factory.

Metrics provided
----------------
* **faithfulness**       — fraction of the answer's claims entailed by
  the retrieved context (LLM judge).
* **answer_relevance**   — cosine sim between the answer and query (via
  the configured embedding provider).
* **context_precision**  — |relevant ∩ retrieved| / |retrieved|.
* **context_recall**     — |relevant ∩ retrieved| / |relevant|.

All metrics return a float in [0, 1].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from agentic_rag.guardrails.output_guard import OutputGuard
from agentic_rag.llm.base import BaseLLMProvider, EmbeddingProvider
from agentic_rag.models.documents import ScoredChunk


@dataclass(slots=True)
class EvalResult:
    """Aggregated metric output for a single case."""

    case_id: str
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    latency_ms: float | None = None
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "case_id": self.case_id,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "latency_ms": self.latency_ms,
            **self.extras,
        }


# ---------------------------------------------------------------------------
# Retrieval metrics — cheap, deterministic
# ---------------------------------------------------------------------------
def context_precision(
    retrieved: Sequence[ScoredChunk], relevant_ids: Sequence[str]
) -> float | None:
    if not retrieved:
        return None
    if not relevant_ids:
        return None
    relevant_set = set(relevant_ids)
    hits = sum(1 for sc in retrieved if sc.chunk.id in relevant_set)
    return hits / len(retrieved)


def context_recall(retrieved: Sequence[ScoredChunk], relevant_ids: Sequence[str]) -> float | None:
    if not relevant_ids:
        return None
    retrieved_set = {sc.chunk.id for sc in retrieved}
    hits = sum(1 for r in relevant_ids if r in retrieved_set)
    return hits / len(relevant_ids)


# ---------------------------------------------------------------------------
# LLM-based metrics
# ---------------------------------------------------------------------------
async def faithfulness(
    llm: BaseLLMProvider,
    *,
    question: str,
    answer: str,
    context: Sequence[ScoredChunk],
) -> float:
    """Re-use :class:`OutputGuard`'s grader for consistency with runtime."""
    guard = OutputGuard(llm, min_faithfulness=0.0)
    verdict = await guard.check(question=question, answer=answer, context=context)
    return verdict.faithfulness or 0.0


# ---------------------------------------------------------------------------
# Embedding-based metric
# ---------------------------------------------------------------------------
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


async def answer_relevance(
    embeddings: EmbeddingProvider,
    *,
    question: str,
    answer: str,
) -> float:
    vecs = await embeddings.embed([question, answer])
    if len(vecs) != 2:
        return 0.0
    return max(0.0, _cosine(vecs[0], vecs[1]))


__all__ = [
    "EvalResult",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevance",
]
