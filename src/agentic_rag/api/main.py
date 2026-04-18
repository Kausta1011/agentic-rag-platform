"""FastAPI application factory + ``uvicorn`` entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag.api.routes import router
from agentic_rag.config import get_settings
from agentic_rag.core.logging import configure_logging, get_logger
from agentic_rag.observability.tracer import configure_tracing

log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing()
    log.info("agentic-rag api starting")
    yield
    log.info("agentic-rag api stopping")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Agentic RAG Platform",
        version="0.1.0",
        description=(
            "Production-grade Agentic RAG powered by LangGraph, hybrid retrieval, "
            "cross-encoder reranking, guardrails, reflection, and evaluation."
        ),
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    return app


# ``uvicorn`` re-importable entrypoint
app = create_app()


def run() -> None:
    """Console-script shim: ``agentic-rag-api``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "agentic_rag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()


__all__ = ["create_app", "app", "run"]
