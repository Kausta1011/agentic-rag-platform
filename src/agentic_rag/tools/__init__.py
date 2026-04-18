"""Agent-callable tools with a simple registry."""

from agentic_rag.tools.base import BaseTool, ToolRegistry, ToolResult, get_registry
from agentic_rag.tools.calculator import CalculatorTool
from agentic_rag.tools.web_search import WebSearchTool

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "get_registry",
    "WebSearchTool",
    "CalculatorTool",
]
