"""Centralised application configuration.

All runtime configuration lives here and is loaded from environment
variables using ``pydantic-settings``. Every other module should read
config exclusively through :func:`get_settings` — never via ``os.environ``
directly. This keeps the dependency direction clean (single source of
truth, testable, trivially mockable).

Environment layout
------------------
The project stores env variables under ``.env/variables.env`` (a *folder*
rather than a file, by deliberate choice — see project README). At import
time we load every ``*.env`` inside ``.env/`` so secrets can be split by
concern (e.g. ``openai.env``, ``anthropic.env``, ``infra.env``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Load all .env files under the project's `.env/` folder, if present.
# This runs exactly once at import time.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_DIR = _PROJECT_ROOT / ".env"
if _ENV_DIR.is_dir():
    for _env_file in sorted(_ENV_DIR.glob("*.env")):
        load_dotenv(_env_file, override=False)


ProviderName = Literal["openai", "anthropic"]
EmbeddingProvider = Literal["openai", "local"]
TraceExport = Literal["console", "otlp", "none"]


class Settings(BaseSettings):
    """Immutable, validated settings object.

    Prefer ``get_settings()`` over instantiating this directly — it caches
    the object for the process lifetime so subsystems share state.
    """

    model_config = SettingsConfigDict(
        env_file=None,  # handled manually above
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM providers -----------------------------------------------------
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_provider: ProviderName = Field(default="anthropic", alias="LLM_PROVIDER")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")
    # Override the OpenAI endpoint — e.g. OpenRouter, Azure, local vLLM.
    # Leave empty to use the default OpenAI endpoint.
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    # --- Embeddings --------------------------------------------------------
    embedding_provider: EmbeddingProvider = Field(default="openai", alias="EMBEDDING_PROVIDER")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="LOCAL_EMBEDDING_MODEL"
    )

    # --- Tools -------------------------------------------------------------
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")

    # --- Vector store ------------------------------------------------------
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"), alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="agentic_rag", alias="CHROMA_COLLECTION")

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = Field(default=10, ge=1, le=100, alias="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=4, ge=1, le=50, alias="RERANK_TOP_K")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0, alias="HYBRID_ALPHA")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    # --- Generation --------------------------------------------------------
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, alias="TEMPERATURE")
    max_output_tokens: int = Field(default=1024, ge=1, alias="MAX_OUTPUT_TOKENS")
    max_reflection_steps: int = Field(default=2, ge=0, le=5, alias="MAX_REFLECTION_STEPS")

    # --- Router ------------------------------------------------------------
    # Description of what's in the indexed corpus. Passed into the router's
    # system prompt so the LLM can decide whether a question is on-topic.
    # Edit this whenever you change your corpus.
    corpus_description: str = Field(
        default=(
            "Technical documentation about the agentic RAG platform itself, "
            "plus essays on LangGraph (stateful agent orchestration), the "
            "Model Context Protocol (MCP), and hybrid retrieval with "
            "BM25, dense vectors, fusion strategies, and cross-encoder reranking."
        ),
        alias="CORPUS_DESCRIPTION",
    )
    # Escape hatch: skip the router entirely and always route to vectorstore.
    # Useful for pure-RAG demos where the router is just overhead.
    disable_router: bool = Field(default=False, alias="DISABLE_ROUTER")

    # --- Guardrails --------------------------------------------------------
    enable_input_guard: bool = Field(default=True, alias="ENABLE_INPUT_GUARD")
    enable_output_guard: bool = Field(default=True, alias="ENABLE_OUTPUT_GUARD")
    min_faithfulness_score: float = Field(
        default=0.6, ge=0.0, le=1.0, alias="MIN_FAITHFULNESS_SCORE"
    )

    # --- Observability -----------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_tracing: bool = Field(default=True, alias="ENABLE_TRACING")
    trace_export: TraceExport = Field(default="console", alias="TRACE_EXPORT")
    otlp_endpoint: str | None = Field(default=None, alias="OTLP_ENDPOINT")

    # --- API ---------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")
    api_cors_origins: str = Field(default="*", alias="API_CORS_ORIGINS")

    # --- MCP ---------------------------------------------------------------
    mcp_server_name: str = Field(default="agentic-rag", alias="MCP_SERVER_NAME")

    # ------------------------------------------------------------------
    # Cross-field validation
    # ------------------------------------------------------------------
    @field_validator("llm_provider", mode="after")
    @classmethod
    def _provider_has_key(cls, v: ProviderName) -> ProviderName:  # noqa: D401
        # Validation of *presence* of a key is deferred to the factory so
        # that importing settings in tests never fails. The factory raises
        # a clean ConfigurationError if the selected provider lacks a key.
        return v

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Using an ``lru_cache`` gives us a single, lazily-constructed singleton
    without global mutable state and without making tests painful — tests
    can call ``get_settings.cache_clear()`` or override via FastAPI DI.
    """
    return Settings()  # type: ignore[call-arg]


__all__ = ["Settings", "get_settings", "ProviderName", "EmbeddingProvider", "TraceExport"]
