"""LLM / embedding factories (Factory pattern).

Callers should do::

    from agentic_rag.llm import get_llm
    llm = get_llm()

…and never import a concrete provider directly. This guarantees the
codebase has exactly one place where provider selection happens.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.exceptions import ConfigurationError
from agentic_rag.llm.base import BaseLLMProvider, EmbeddingProvider


class LLMFactory:
    """Builds :class:`BaseLLMProvider` / :class:`EmbeddingProvider` instances.

    Isolated behind a class so it can be swapped or mocked wholesale in
    tests (e.g. an in-memory ``FakeLLM`` that records prompts).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_llm(self) -> BaseLLMProvider:
        provider = self._settings.llm_provider
        if provider == "openai":
            key = self._require_key(
                self._settings.openai_api_key, env_var="OPENAI_API_KEY", provider="openai"
            )
            from agentic_rag.llm.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(
                api_key=key,
                model=self._settings.openai_model,
                base_url=self._settings.openai_base_url,
            )

        if provider == "anthropic":
            key = self._require_key(
                self._settings.anthropic_api_key,
                env_var="ANTHROPIC_API_KEY",
                provider="anthropic",
            )
            from agentic_rag.llm.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(api_key=key, model=self._settings.anthropic_model)

        raise ConfigurationError(f"Unknown LLM_PROVIDER: {provider!r}")

    def build_embeddings(self) -> EmbeddingProvider:
        provider = self._settings.embedding_provider
        if provider == "openai":
            key = self._require_key(
                self._settings.openai_api_key,
                env_var="OPENAI_API_KEY",
                provider="openai-embeddings",
            )
            from agentic_rag.llm.providers.openai_provider import OpenAIEmbeddings
            return OpenAIEmbeddings(
                api_key=key,
                model=self._settings.openai_embedding_model,
                base_url=self._settings.openai_base_url,
            )

        if provider == "local":
            return _LocalSentenceTransformerEmbeddings(
                model_name=self._settings.local_embedding_model
            )

        raise ConfigurationError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _require_key(key, *, env_var: str, provider: str) -> str:
        if key is None or not key.get_secret_value():
            raise ConfigurationError(
                f"{provider} selected but {env_var} is not set. "
                f"Put it in `.env/variables.env`."
            )
        return key.get_secret_value()


# ---------------------------------------------------------------------------
# Local sentence-transformers fallback — kept here to avoid a module-level
# import of torch just because somebody read settings.
# ---------------------------------------------------------------------------
class _LocalSentenceTransformerEmbeddings(EmbeddingProvider):
    name = "local"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "sentence-transformers not installed; run `pip install sentence-transformers`"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # sentence-transformers is sync — run in default thread pool.
        import asyncio

        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            None, lambda: self._model.encode(list(texts), normalize_embeddings=True)
        )
        return vectors.tolist()


# ---------------------------------------------------------------------------
# Module-level convenience functions (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_llm() -> BaseLLMProvider:
    """Process-wide cached LLM provider."""
    return LLMFactory().build_llm()


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Process-wide cached embedding provider."""
    return LLMFactory().build_embeddings()


__all__ = ["LLMFactory", "get_llm", "get_embedding_provider"]
