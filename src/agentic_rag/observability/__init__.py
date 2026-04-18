"""Tracing + metrics (OpenTelemetry, with a graceful console fallback)."""

from agentic_rag.observability.metrics import Metrics, get_metrics
from agentic_rag.observability.tracer import configure_tracing, get_tracer

__all__ = ["configure_tracing", "get_tracer", "Metrics", "get_metrics"]
