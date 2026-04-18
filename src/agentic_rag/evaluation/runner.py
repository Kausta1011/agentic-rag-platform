"""End-to-end eval runner.

Drives the compiled LangGraph against every :class:`EvalCase` and
aggregates metric results. Emits both per-case rows and a summary table.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from agentic_rag.core.logging import get_logger
from agentic_rag.evaluation.dataset import EvalCase, EvalDataset
from agentic_rag.evaluation.metrics import (
    EvalResult,
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)
from agentic_rag.llm import get_embedding_provider, get_llm
from agentic_rag.models.documents import ScoredChunk
from agentic_rag.models.state import build_initial_state

log = get_logger(__name__)


@dataclass(slots=True)
class EvalReport:
    dataset: str
    results: list[EvalResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        def _avg(attr: str) -> float | None:
            xs = [getattr(r, attr) for r in self.results if getattr(r, attr) is not None]
            return mean(xs) if xs else None

        return {
            "n_cases": len(self.results),
            "faithfulness": _avg("faithfulness"),
            "answer_relevance": _avg("answer_relevance"),
            "context_precision": _avg("context_precision"),
            "context_recall": _avg("context_recall"),
            "avg_latency_ms": _avg("latency_ms"),
        }

    def to_rows(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.results]


class EvalRunner:
    """Drives the compiled graph and computes metrics per case."""

    def __init__(self, graph, *, concurrency: int = 4) -> None:  # type: ignore[no-untyped-def]
        self._graph = graph
        self._semaphore = asyncio.Semaphore(concurrency)
        self._llm = get_llm()
        self._embeddings = get_embedding_provider()

    async def run(self, dataset: EvalDataset) -> EvalReport:
        report = EvalReport(dataset=dataset.name)
        tasks = [asyncio.create_task(self._run_one(c)) for c in dataset.cases]
        for coro in asyncio.as_completed(tasks):
            report.results.append(await coro)
        log.bind(**report.summary()).info("eval complete")
        return report

    # ------------------------------------------------------------------
    async def _run_one(self, case: EvalCase) -> EvalResult:
        async with self._semaphore:
            started = time.perf_counter()
            state = build_initial_state(case.question, session_id=case.id)
            final: dict[str, Any] = await self._graph.ainvoke(state)
            elapsed = (time.perf_counter() - started) * 1_000

            reranked: list[ScoredChunk] = final.get("reranked") or []
            answer: str = str(final.get("answer", ""))

            # Metrics
            retr_precision = context_precision(reranked, case.relevant_chunk_ids)
            retr_recall = context_recall(reranked, case.relevant_chunk_ids)
            faith = await faithfulness(
                self._llm, question=case.question, answer=answer, context=reranked
            )
            rel = await answer_relevance(
                self._embeddings, question=case.question, answer=answer
            )

            return EvalResult(
                case_id=case.id,
                faithfulness=faith,
                answer_relevance=rel,
                context_precision=retr_precision,
                context_recall=retr_recall,
                latency_ms=elapsed,
            )


__all__ = ["EvalRunner", "EvalReport"]
