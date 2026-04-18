"""MCP server implementation (stdio transport).

Exposes two tools:

* ``ask_corpus``      — run the full agentic-RAG pipeline.
* ``ingest_document`` — stream a document into the index.

Using the reference ``mcp`` Python SDK. Run via::

    agentic-rag-mcp
    # or: python -m mcp_server.server
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agentic_rag.api.dependencies import get_service
from agentic_rag.config import get_settings
from agentic_rag.core.logging import configure_logging, get_logger
from agentic_rag.ingestion.loaders import TextLoader
from agentic_rag.ingestion.pipeline import IngestionPipeline
from agentic_rag.models.state import build_initial_state

log = get_logger(__name__)


def build_server():  # type: ignore[no-untyped-def]
    """Construct an MCP :class:`Server` instance with our two tools."""
    try:
        from mcp.server import Server  # type: ignore[import-not-found]
        from mcp.types import TextContent, Tool  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "`mcp` package not installed; `pip install mcp`"
        ) from exc

    settings = get_settings()
    server: Server = Server(settings.mcp_server_name)

    # ------------------------------------------------------------------
    # list_tools
    # ------------------------------------------------------------------
    @server.list_tools()  # type: ignore[misc]
    async def _list_tools() -> list[Tool]:  # noqa: ANN001
        return [
            Tool(
                name="ask_corpus",
                description=(
                    "Answer a natural-language question using the indexed corpus. "
                    "Runs query routing, hybrid retrieval, cross-encoder "
                    "rerank, grounded generation, and a reflection loop."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question": {"type": "string", "description": "User question"},
                        "session_id": {"type": "string"},
                    },
                },
            ),
            Tool(
                name="ingest_document",
                description="Ingest a plain-text document into the corpus.",
                inputSchema={
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Filesystem path to a .txt or .md file",
                        }
                    },
                },
            ),
        ]

    # ------------------------------------------------------------------
    # call_tool
    # ------------------------------------------------------------------
    @server.call_tool()  # type: ignore[misc]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:  # noqa: ANN001
        if name == "ask_corpus":
            return await _ask_corpus(arguments)
        if name == "ingest_document":
            return await _ingest(arguments)
        return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]

    async def _ask_corpus(arguments: dict[str, Any]) -> list[TextContent]:
        question = arguments.get("question", "").strip()
        if not question:
            return [TextContent(type="text", text=json.dumps({"error": "question required"}))]

        service = get_service()
        state = build_initial_state(question, session_id=arguments.get("session_id"))
        final = await service.graph.ainvoke(state)  # type: ignore[attr-defined]
        payload = {
            "answer": final.get("answer", ""),
            "route": str(final.get("route")),
            "citations": [c.model_dump() for c in final.get("citations", [])],
            "reflection_step": final.get("reflection_step", 0),
            "tokens": final.get("tokens"),
        }
        return [TextContent(type="text", text=json.dumps(payload, default=str))]

    async def _ingest(arguments: dict[str, Any]) -> list[TextContent]:
        path = arguments.get("path")
        if not path:
            return [TextContent(type="text", text=json.dumps({"error": "path required"}))]

        # Re-build a pipeline against a fresh retriever. A cleaner design
        # would expose the retriever on ``Service``; kept minimal here to
        # avoid widening the public surface.
        get_service()
        from agentic_rag.llm import get_embedding_provider
        from agentic_rag.retrieval.vector_store import ChromaVectorStore

        embeddings = get_embedding_provider()
        vector = ChromaVectorStore(embeddings=embeddings)
        pipeline = IngestionPipeline(retriever=vector)
        report = await pipeline.ingest(TextLoader(path))
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "documents_loaded": report.documents_loaded,
                        "chunks_indexed": report.chunks_indexed,
                        "errors": report.errors,
                    }
                ),
            )
        ]

    return server


def main() -> None:
    """Run the MCP server over stdio (the canonical MCP transport)."""
    configure_logging(get_settings().log_level)

    try:
        from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("`mcp` package not installed; `pip install mcp`") from exc

    server = build_server()

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_server", "main"]
