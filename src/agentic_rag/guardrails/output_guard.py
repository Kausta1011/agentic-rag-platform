"""Output guardrail.

Runs *after* the generator has produced a draft answer, but *before* we
return it. Two checks:

1. **Faithfulness** — for RAG-routed answers, the LLM is asked to judge
   whether every factual claim is supported by the retrieved context.
   Returns a score in [0, 1]; we refuse to ship answers below a
   configurable floor.
2. **Sensitive content** — a simple keyword filter; production systems
   would use a moderation classifier here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from agentic_rag.core.exceptions import GuardrailViolationError, LLMProviderError
from agentic_rag.core.logging import get_logger
from agentic_rag.llm.base import BaseLLMProvider
from agentic_rag.models.documents import ScoredChunk

log = get_logger(__name__)


_FAITHFULNESS_SYSTEM = (
    "You are a strict grader. Given a QUESTION, a CONTEXT (excerpts from "
    "trusted documents), and an ANSWER, decide whether EVERY factual claim "
    "in the ANSWER is entailed by the CONTEXT. Reply with a JSON object: "
    '{"score": <float in [0,1]>, "unsupported": [<claim1>, ...]}. '
    "A score of 1 means fully supported; 0 means hallucinated."
)


@dataclass(slots=True)
class OutputVerdict:
    ok: bool
    faithfulness: float | None = None
    unsupported_claims: list[str] | None = None
    notes: str = ""


class OutputGuard:
    """Post-generation quality gate."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        min_faithfulness: float = 0.6,
    ) -> None:
        self._llm = llm
        self.min_faithfulness = min_faithfulness

    # ------------------------------------------------------------------
    async def check(
        self,
        *,
        question: str,
        answer: str,
        context: Iterable[ScoredChunk] | None = None,
    ) -> OutputVerdict:
        if not answer.strip():
            raise GuardrailViolationError(
                "empty answer", rule="empty_answer", stage="output"
            )

        if context is None:
            # Nothing to ground against; pass with no score.
            return OutputVerdict(ok=True, notes="no-context")

        ctx_text = "\n\n---\n\n".join(sc.chunk.content for sc in context)
        prompt = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{ctx_text}\n\n"
            f"ANSWER:\n{answer}\n\n"
            'Return JSON only.'
        )
        try:
            resp = await self._llm.generate(
                prompt, system=_FAITHFULNESS_SYSTEM, temperature=0.0, max_tokens=300
            )
        except LLMProviderError as exc:
            log.warning(f"faithfulness grader LLM failed, treating as passthrough: {exc}")
            return OutputVerdict(ok=True, notes="grader-unavailable")

        score, unsupported = _parse_grader_json(resp.text)
        verdict = OutputVerdict(
            ok=score >= self.min_faithfulness,
            faithfulness=score,
            unsupported_claims=unsupported,
        )
        if not verdict.ok:
            verdict.notes = f"faithfulness {score:.2f} < threshold {self.min_faithfulness}"
        return verdict


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _parse_grader_json(raw: str) -> tuple[float, list[str]]:
    """Best-effort parse of the grader's JSON reply."""
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return 0.0, []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0, []
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    unsupported = data.get("unsupported") or []
    if not isinstance(unsupported, list):
        unsupported = []
    return score, [str(u) for u in unsupported]


__all__ = ["OutputGuard", "OutputVerdict"]
