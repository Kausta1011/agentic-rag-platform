"""In-process metrics.

Deliberately dependency-free: this is a minimal counter/histogram store
that the API exposes at ``/metrics``. For a real deployment swap in
``opentelemetry.metrics`` or ``prometheus_client``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _Histogram:
    count: int = 0
    sum_ms: float = 0.0
    max_ms: float = 0.0
    samples: list[float] = field(default_factory=list)

    def observe(self, value_ms: float) -> None:
        self.count += 1
        self.sum_ms += value_ms
        self.max_ms = max(self.max_ms, value_ms)
        if len(self.samples) < 1024:
            self.samples.append(value_ms)

    def snapshot(self) -> dict[str, Any]:
        avg = self.sum_ms / self.count if self.count else 0.0
        sorted_samples = sorted(self.samples)

        def p(q: float) -> float:
            if not sorted_samples:
                return 0.0
            return sorted_samples[int(q * (len(sorted_samples) - 1))]
        return {
            "count": self.count,
            "avg_ms": round(avg, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(p(0.5), 2),
            "p95_ms": round(p(0.95), 2),
            "p99_ms": round(p(0.99), 2),
        }


class Metrics:
    """Thread-safe counter + histogram registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._histos: dict[str, _Histogram] = defaultdict(_Histogram)

    # ------------------------------------------------------------------
    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            self._histos[name].observe(ms)

    def timer(self, name: str) -> _Timer:
        return _Timer(self, name)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {k: h.snapshot() for k, h in self._histos.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histos.clear()


class _Timer:
    __slots__ = ("metrics", "name", "_start")

    def __init__(self, metrics: Metrics, name: str) -> None:
        self.metrics = metrics
        self.name = name
        self._start = 0.0

    def __enter__(self) -> _Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.metrics.observe(self.name, (time.perf_counter() - self._start) * 1_000)


_METRICS: Metrics | None = None


def get_metrics() -> Metrics:
    global _METRICS
    if _METRICS is None:
        _METRICS = Metrics()
    return _METRICS


__all__ = ["Metrics", "get_metrics"]
