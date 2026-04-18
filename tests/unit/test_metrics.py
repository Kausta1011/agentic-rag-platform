"""Observability metrics tests."""

from __future__ import annotations

import time

from agentic_rag.observability.metrics import Metrics


def test_counter_and_histogram_roundtrip():
    m = Metrics()
    m.incr("foo", 2)
    m.incr("foo")
    m.observe("lat", 10.0)
    m.observe("lat", 20.0)
    snap = m.snapshot()

    assert snap["counters"]["foo"] == 3
    h = snap["histograms"]["lat"]
    assert h["count"] == 2
    assert h["max_ms"] == 20.0
    assert 10.0 <= h["avg_ms"] <= 20.0


def test_timer_measures_elapsed():
    m = Metrics()
    with m.timer("sleep"):
        time.sleep(0.01)
    h = m.snapshot()["histograms"]["sleep"]
    assert h["count"] == 1
    assert h["max_ms"] >= 5.0
