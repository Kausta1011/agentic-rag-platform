"""Abstract base classes for LLM and embedding providers.

Template-method / Strategy pattern: concrete providers implement the
protocol below. All callers depend on these interfaces, never on an SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Normalised output from every provider."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str | None = None
    raw: dict = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Strategy interface every text-LLM provider must implement."""

    #: Human-friendly provider id, e.g. ``"openai"``.
    name: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        """Return a complete (non-streaming) response."""

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they are produced."""


class EmbeddingProvider(ABC):
    """Strategy interface for embedding providers."""

    name: str
    dimension: int

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts. Must preserve input ordering."""


__all__ = ["LLMResponse", "BaseLLMProvider", "EmbeddingProvider"]
