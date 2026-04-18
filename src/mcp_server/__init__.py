"""Model Context Protocol (MCP) server exposing the RAG pipeline as a tool.

Running this server lets any MCP-aware client (Claude Desktop, Cursor,
Windsurf, Cowork, etc.) ask questions against the indexed corpus through
the same pipeline as the HTTP API — a direct demonstration of the
2026-era "tools everywhere" integration pattern.
"""

from mcp_server.server import build_server, main

__all__ = ["build_server", "main"]
