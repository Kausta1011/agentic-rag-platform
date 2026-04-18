"""Centralised prompt library.

Keeping prompts in one file (a) avoids copy-paste drift, (b) makes
prompt-engineering diffs a clean standalone PR, (c) lets tests assert on
stable prompt contents.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Router — decides VECTORSTORE / WEB_SEARCH / DIRECT / REFUSE
#
# The router is the single most common source of RAG bugs: without a
# description of the actual corpus, the LLM guesses — and small models
# tend to guess "direct" (skip retrieval) far too often. We therefore
# (a) inject a concrete corpus description into the prompt and
# (b) bias the model toward ``vectorstore`` when uncertain.
# ---------------------------------------------------------------------------
def build_router_system(corpus_description: str) -> str:
    """Return the ROUTER system prompt parameterised by the corpus.

    Passing the corpus description at prompt-construction time lets one
    deployment route correctly against its own documents without editing
    source code.
    """
    desc = (corpus_description or "").strip() or ("Technical documents ingested by the operator.")
    return (
        "You are the router for an agentic RAG system. Given a USER QUESTION,\n"
        "choose exactly one route.\n\n"
        "KNOWLEDGE BASE CONTENTS:\n"
        f"{desc}\n\n"
        "Decision rules:\n"
        '- "vectorstore" : the question is plausibly answerable from the\n'
        "  knowledge base above. When in doubt, PREFER THIS — retrieval is\n"
        "  cheap and a downstream grader filters bad hits. Choose this for\n"
        "  definitions, how-tos, explanations, or any topic that overlaps\n"
        "  with the knowledge-base contents.\n"
        '- "web_search"  : the question needs fresh external information\n'
        '  (today\'s news, "latest" releases, real-time data, post-cutoff events).\n'
        '- "direct"      : pure small-talk, greetings, or meta questions\n'
        '  about the assistant itself (e.g. "who are you", "hi"). NEVER\n'
        "  choose this just because you think you already know the answer.\n"
        '- "refuse"      : disallowed content (malware, violence, doxing, etc.).\n\n'
        "Reply with a single JSON object on one line:\n"
        '{"route": "<one of the four>", "reason": "<short reason>"}\n'
    )


# Backwards-compat alias for tests / callers that used the old constant.
# Uses a generic corpus description so behaviour is still reasonable.
ROUTER_SYSTEM = build_router_system("Technical documents ingested by the operator.")

# ---------------------------------------------------------------------------
# Query rewriter — converts a raw question into a retrieval-friendly query
# ---------------------------------------------------------------------------
REWRITER_SYSTEM = """\
You rewrite user questions into retrieval queries for a hybrid
(BM25 + dense) retriever. Keep it short (≤ 20 tokens), keep important
entities and acronyms, drop filler words, expand one or two obvious
synonyms when they help recall. Return ONLY the rewritten query as
plain text — no quotes, no commentary.
"""

# ---------------------------------------------------------------------------
# Relevance grader — per-chunk yes/no
# ---------------------------------------------------------------------------
GRADER_SYSTEM = """\
You are a strict relevance grader. Given a USER QUESTION and a
CANDIDATE PASSAGE, decide whether the passage is sufficiently relevant
to contribute to an answer.

Reply with a single JSON object: {"relevant": true|false,
"reason": "<short>"}.
"""

# ---------------------------------------------------------------------------
# Final answer generator
# ---------------------------------------------------------------------------
GENERATOR_SYSTEM = """\
You are a careful technical assistant. Use ONLY the CONTEXT passages
below to answer. If the answer is not in the context, say so plainly
rather than guessing.

Rules:
- Be concise but complete.
- Cite the passages you used by their numeric id — use inline markers
  like [1], [2] that match the CONTEXT entries.
- Prefer short paragraphs over long ones.
- Never invent URLs or facts.
"""

# ---------------------------------------------------------------------------
# Reflection — the "am I done?" judge used for the self-correction loop
# ---------------------------------------------------------------------------
REFLECTOR_SYSTEM = """\
You are a reviewer. Given a USER QUESTION and a DRAFT ANSWER (produced
from the CONTEXT), decide whether the answer is (a) complete,
(b) grounded in the context, and (c) directly answers the question.

Reply with a JSON object:
{"sufficient": true|false,
 "missing": "<what is missing, if anything>",
 "rewrite_query": "<an improved retrieval query if sufficient=false, else empty>"}.
"""


__all__ = [
    "ROUTER_SYSTEM",
    "build_router_system",
    "REWRITER_SYSTEM",
    "GRADER_SYSTEM",
    "GENERATOR_SYSTEM",
    "REFLECTOR_SYSTEM",
]
