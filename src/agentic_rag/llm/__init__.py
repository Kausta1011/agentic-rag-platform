"""LLM provider abstraction + factory.

Consumers always depend on :class:`BaseLLMProvider` (the Strategy), never
on a concrete SDK. This keeps the retrieval / agent / eval layers
provider-agnostic and makes swapping OpenAI ↔ Anthropic (or adding a new
provider such as Google / Mistral) a zero-change-to-callers operation.
"""

from agentic_rag.llm.base import BaseLLMProvider, EmbeddingProvider, LLMResponse
from agentic_rag.llm.factory import LLMFactory, get_embedding_provider, get_llm

__all__ = [
    "BaseLLMProvider",
    "EmbeddingProvider",
    "LLMResponse",
    "LLMFactory",
    "get_llm",
    "get_embedding_provider",
]
