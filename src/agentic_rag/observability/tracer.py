"""OpenTelemetry tracing setup.

Nodes in the LangGraph call ``with tracer.start_as_current_span(...)``
to produce a well-structured, vendor-neutral trace. Export is configured
via ``TRACE_EXPORT``:

* ``console`` — dev-friendly, prints spans to stdout.
* ``otlp``    — ship to any OTLP-compatible collector (Jaeger, Tempo,
                Grafana Cloud, Honeycomb, etc.).
* ``none``    — tracing disabled; all calls become no-ops.
"""

from __future__ import annotations

from functools import lru_cache

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from agentic_rag.config import get_settings

_CONFIGURED = False


def configure_tracing() -> None:
    """Configure the global tracer provider once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    if not settings.enable_tracing or settings.trace_export == "none":
        _CONFIGURED = True
        return

    resource = Resource.create({"service.name": "agentic-rag", "service.version": "0.1.0"})
    provider = TracerProvider(resource=resource)

    if settings.trace_export == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif settings.trace_export == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError:  # pragma: no cover - optional dep
            # Soft fall back to console if exporter not installed.
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        else:
            endpoint = settings.otlp_endpoint or "http://localhost:4318/v1/traces"
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    trace.set_tracer_provider(provider)
    _CONFIGURED = True


@lru_cache(maxsize=16)
def get_tracer(name: str = "agentic-rag") -> trace.Tracer:
    configure_tracing()
    return trace.get_tracer(name)


__all__ = ["configure_tracing", "get_tracer"]
