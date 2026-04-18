"""Web search tool.

Tries Tavily (best LLM-tuned search API) if an API key is available;
falls back to DuckDuckGo (key-free) otherwise. Failures do not crash the
graph — they surface as a ``ToolResult(ok=False, …)``.
"""

from __future__ import annotations

from typing import Any

from agentic_rag.config import get_settings
from agentic_rag.core.exceptions import ToolExecutionError
from agentic_rag.core.logging import get_logger
from agentic_rag.core.types import ToolName
from agentic_rag.tools.base import BaseTool, ToolResult

log = get_logger(__name__)


class WebSearchTool(BaseTool):
    name = ToolName.WEB_SEARCH.value
    description = (
        "Search the public web for up-to-date information on a topic. "
        "Use when the question needs fresh, external knowledge."
    )

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        self._settings = get_settings()

    async def run(self, query: str, **_: Any) -> ToolResult:
        if not query.strip():
            raise ToolExecutionError("empty web search query")

        if self._settings.tavily_api_key:
            return await self._tavily(query)
        return await self._duckduckgo(query)

    # ------------------------------------------------------------------
    async def _tavily(self, query: str) -> ToolResult:
        try:
            from tavily import TavilyClient  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ToolExecutionError("tavily-python not installed") from exc

        assert self._settings.tavily_api_key is not None
        client = TavilyClient(api_key=self._settings.tavily_api_key.get_secret_value())
        import asyncio

        try:
            raw = await asyncio.to_thread(
                client.search,
                query=query,
                max_results=self.max_results,
                search_depth="advanced",
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(f"tavily search failed: {exc}") from exc

        hits = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score"),
            }
            for r in raw.get("results", [])
        ]
        return ToolResult(ok=True, data=hits, meta={"provider": "tavily"})

    async def _duckduckgo(self, query: str) -> ToolResult:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ToolExecutionError("duckduckgo_search not installed") from exc

        import asyncio

        def _search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=self.max_results))

        try:
            raw = await asyncio.to_thread(_search)
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(f"duckduckgo search failed: {exc}") from exc

        hits = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "content": r.get("body", ""),
            }
            for r in raw
        ]
        return ToolResult(ok=True, data=hits, meta={"provider": "duckduckgo"})


__all__ = ["WebSearchTool"]
