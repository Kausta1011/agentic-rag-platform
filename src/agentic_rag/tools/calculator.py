"""Safe arithmetic tool.

LLMs are routinely poor at arithmetic; exposing a calculator tool
materially improves numerical questions. We hand-roll a safe evaluator
using Python's ``ast`` module rather than ``eval()``.
"""

from __future__ import annotations

import ast
import math
import operator as op
from typing import Any

from agentic_rag.core.exceptions import ToolExecutionError
from agentic_rag.core.types import ToolName
from agentic_rag.tools.base import BaseTool, ToolResult

# Supported AST node types and their Python operators.
_BIN_OPS: dict[type[ast.AST], Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS: dict[type[ast.AST], Any] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return float(_ALLOWED_NAMES[node.id])
        raise ToolExecutionError(f"unknown identifier: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _ALLOWED_FUNCS.get(node.func.id)
        if func is None:
            raise ToolExecutionError(f"function not allowed: {node.func.id}")
        args = [_eval_node(a) for a in node.args]
        return float(func(*args))
    raise ToolExecutionError(f"unsupported expression node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = ToolName.CALCULATOR.value
    description = (
        "Evaluate a maths expression. Supports + - * / ** %, parentheses, "
        "sqrt/log/sin/cos/tan/abs/round, and the constants pi and e."
    )

    async def run(self, expression: str, **_: Any) -> ToolResult:
        if not expression or not expression.strip():
            raise ToolExecutionError("empty expression")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError(f"invalid syntax: {exc.msg}") from exc
        value = _eval_node(tree)
        return ToolResult(ok=True, data=value, meta={"expression": expression})


__all__ = ["CalculatorTool"]
