"""Input guardrail.

Rules that look at the user's question *before* it enters the pipeline:

* PII detection (email / phone / credit-card / IBAN) — redacts in place.
* Prompt-injection heuristics — flags obvious "ignore previous
  instructions" style payloads.
* Length cap — rejects absurdly long queries.

Design: each rule is a small function returning ``(clean_text, verdict)``,
composed in a list. This is deliberately simpler than a full guard
framework (Guardrails-AI, NeMo Guardrails) because the main demo-value
lives in the LangGraph orchestration — but the extension point is here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_rag.core.exceptions import GuardrailViolationError

# ---------------------------------------------------------------------------
# Regexes — all intentionally conservative (false-negatives over false-positives)
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}(?!\d)")
_CC_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all|any|previous|the above).*instructions?",
        r"disregard (everything|all).*prior",
        r"you are (now|a).*dan",
        r"system prompt is",
    )
)


@dataclass(slots=True)
class GuardVerdict:
    ok: bool
    redactions: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    clean_text: str = ""


class InputGuard:
    """Apply input-stage guardrails to a raw user question."""

    def __init__(
        self,
        *,
        max_length: int = 4_000,
        redact_pii: bool = True,
        block_injection: bool = True,
    ) -> None:
        self.max_length = max_length
        self.redact_pii = redact_pii
        self.block_injection = block_injection

    # ------------------------------------------------------------------
    def check(self, text: str) -> GuardVerdict:
        verdict = GuardVerdict(ok=True, clean_text=text)

        if len(text) > self.max_length:
            raise GuardrailViolationError(
                f"input exceeds max length of {self.max_length} chars",
                rule="max_length",
                stage="input",
                context={"length": len(text)},
            )

        if self.redact_pii:
            text, redactions = self._redact_pii(text)
            verdict.redactions.extend(redactions)

        if self.block_injection:
            hits = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
            if hits:
                verdict.flags.extend(hits)
                raise GuardrailViolationError(
                    "prompt-injection heuristic triggered",
                    rule="prompt_injection",
                    stage="input",
                    context={"patterns": hits},
                )

        verdict.clean_text = text
        return verdict

    # ------------------------------------------------------------------
    @staticmethod
    def _redact_pii(text: str) -> tuple[str, list[str]]:
        redactions: list[str] = []
        for pattern, tag in (
            (_EMAIL_RE, "<EMAIL>"),
            (_IBAN_RE, "<IBAN>"),
            (_CC_RE, "<CC>"),
            (_PHONE_RE, "<PHONE>"),
        ):
            def _sub(match: re.Match[str], tag: str = tag) -> str:
                redactions.append(f"{tag}:{match.group(0)[:4]}***")
                return tag

            text = pattern.sub(_sub, text)
        return text, redactions


__all__ = ["InputGuard", "GuardVerdict"]
