"""Calculator tool tests (no LLM required)."""

from __future__ import annotations

import math

import pytest

from agentic_rag.tools.calculator import CalculatorTool


@pytest.mark.asyncio
async def test_calculator_basic_arithmetic():
    r = await CalculatorTool().run(expression="2 + 3 * 4")
    assert r.ok is True
    assert r.data == 14.0


@pytest.mark.asyncio
async def test_calculator_functions_and_constants():
    r = await CalculatorTool().run(expression="sqrt(4) + pi")
    assert r.ok is True
    assert math.isclose(r.data, 2.0 + math.pi, rel_tol=1e-9)


@pytest.mark.asyncio
async def test_calculator_rejects_unknown_identifier():
    r = await CalculatorTool().safe_run(expression="dangerous_var")
    assert r.ok is False
    assert "unknown identifier" in (r.error or "")


@pytest.mark.asyncio
async def test_calculator_rejects_unsafe_call():
    r = await CalculatorTool().safe_run(expression="__import__('os').system('ls')")
    assert r.ok is False
