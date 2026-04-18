"""Shared pytest fixtures.

We use a :class:`FakeLLM` and :class:`FakeEmbeddings` so unit tests are
hermetic — no external API calls, fully deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from agentic_rag.llm.base import BaseLLMProvider, EmbeddingProvider, LLMResponse
from agentic_rag.models.documents import Chunk


class FakeLLM(BaseLLMProvider):
    name = "fake"

    def __init__(self, canned: list[str] | None = None) -> None:
        self.canned = canned or []
        self.prompts: list[str] = []

    async def generate(
        self, prompt: str, *, system: str | None = None,
        temperature: float = 0.0, max_tokens: int = 1024, stop=None,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        text = self.canned.pop(0) if self.canned else "ok"
        return LLMResponse(
            text=text, input_tokens=len(prompt.split()),
            output_tokens=len(text.split()), model="fake",
        )

    async def stream(
        self, prompt: str, *, system: str | None = None,
        temperature: float = 0.0, max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        resp = await self.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        for tok in resp.text.split():
            yield tok + " "


class FakeEmbeddings(EmbeddingProvider):
    """Deterministic bag-of-characters embedding — good enough for tests."""

    name = "fake"
    dimension = 32

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dimension
            for ch in t.lower():
                v[ord(ch) % self.dimension] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    texts = [
        "LangGraph models agents as stateful finite-state machines with conditional edges.",
        "BM25 is a lexical ranking function used in information retrieval for exact token matches.",
        "Reciprocal rank fusion combines rankings from multiple retrievers in a score-free way.",
        "Cross-encoders read the query and passage together, producing a high-quality relevance score.",
    ]
    return [
        Chunk(
            id=Chunk.build_id("doc-1", i),
            document_id="doc-1",
            ordinal=i,
            content=t,
            metadata={"source": "unit-test"},
        )
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# Make sure every test loop is fresh.
@pytest.fixture(autouse=True)
def _reset_event_loop():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield
    finally:
        loop.close()
