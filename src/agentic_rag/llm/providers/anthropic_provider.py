"""Anthropic (Claude) provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from agentic_rag.core.exceptions import LLMProviderError
from agentic_rag.core.logging import get_logger
from agentic_rag.llm.base import BaseLLMProvider, LLMResponse

log = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Claude chat provider."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError("anthropic package not installed") from exc

        self._client = AsyncAnthropic(api_key=api_key)
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
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = list(stop)

        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"anthropic generate failed: {exc}") from exc

        # Anthropic returns a list of content blocks; concatenate text blocks.
        text = "".join(
            getattr(block, "text", "") for block in resp.content if block.type == "text"
        )
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens if resp.usage else 0,
            output_tokens=resp.usage.output_tokens if resp.usage else 0,
            model=resp.model,
            stop_reason=resp.stop_reason,
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
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"anthropic stream failed: {exc}") from exc


__all__ = ["AnthropicProvider"]
