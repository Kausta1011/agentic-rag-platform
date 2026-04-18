# Model Context Protocol (MCP)

MCP is an open protocol for connecting LLM-based clients to tools and data sources. Think of it as "LSP for AI agents": a small, stable, JSON-RPC-based surface that lets any MCP-aware client discover and invoke tools exposed by any MCP-aware server.

## Why it matters

Before MCP, every agent framework invented its own tool-calling format. Every tool had to be re-implemented per-framework, and every client had to maintain a bespoke plugin system. MCP standardises:

- **Tool discovery** — the server advertises its tools and their JSON schemas.
- **Tool invocation** — the client sends a `call_tool` request; the server returns structured content.
- **Resources** — exposing static or dynamic data behind URIs the client can read.
- **Prompts** — reusable prompt templates the client can surface to users.

## Transports

MCP is transport-agnostic. The two canonical transports are **stdio** (for locally-spawned servers, ideal for desktop clients) and **HTTP + SSE** (for hosted, shared servers). A server implemented against the SDK supports both without changes.

## When to build an MCP server

If your product has a pipeline (search, database, internal tool, RAG system) that LLMs could usefully call, exposing it as an MCP server is often the right move. It lets Claude Desktop, Cursor, Windsurf, Cowork, and anything else MCP-aware consume it for free — no per-client integration work.
