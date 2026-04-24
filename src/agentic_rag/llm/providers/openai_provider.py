"""OpenAI provider implementation (chat + embeddings)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from agentic_rag.core.exceptions import LLMProviderError
from agentic_rag.core.logging import get_logger
from agentic_rag.llm.base import BaseLLMProvider, EmbeddingProvider, LLMResponse

log = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """Chat completion provider backed by the OpenAI Python SDK.

    Also works against any OpenAI-compatible gateway (OpenRouter, Azure
    OpenAI, local vLLM / Ollama, LiteLLM, …) by passing ``base_url``.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise LLMProviderError("openai package not installed") from exc

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=list(stop) if stop else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"openai generate failed: {exc}") from exc

        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=resp.model,
            stop_reason=choice.finish_reason,
            raw={"id": resp.id},
        )

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"openai stream failed: {exc}") from exc


class OpenAIEmbeddings(EmbeddingProvider):
    """Batched embedding provider for OpenAI."""

    name = "openai"

    # text-embedding-3-small → 1536; text-embedding-3-large → 3072
    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError("openai package not installed") from exc

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.dimension = self._DIMENSIONS.get(model, 1536)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = await self._client.embeddings.create(model=self._model, input=list(texts))
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"openai embed failed: {exc}") from exc
        return [d.embedding for d in resp.data]


__all__ = ["OpenAIProvider", "OpenAIEmbeddings"]
