"""Agentic RAG Platform.

A production-grade Retrieval-Augmented Generation platform built with LangGraph.
Implements:

* Hybrid retrieval (dense + BM25) with reciprocal-rank-fusion
* Cross-encoder re-ranking
* LangGraph-based stateful multi-step agent with self-reflection
* Query routing (RAG / web search / direct)
* Pluggable LLM providers (OpenAI, Anthropic) via factory pattern
* Guardrails (input & output)
* Evaluation harness (faithfulness, relevance, context precision/recall)
* OpenTelemetry tracing + structured metrics
* FastAPI backend with server-sent-event streaming
* Model Context Protocol (MCP) server exposing the pipeline as a tool

Design principles
-----------------
* **SOLID**: each module has a single responsibility, depends on abstractions.
* **Factory / Strategy / Template-Method** patterns for LLMs, retrievers, tools.
* **Dependency injection** via ``pydantic-settings`` and FastAPI DI.
* **Contract-first**: all I/O flows through typed Pydantic models.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("agentic-rag-platform")
except _metadata.PackageNotFoundError:  # local / editable install before build
    __version__ = "0.1.0"

__all__ = ["__version__"]
