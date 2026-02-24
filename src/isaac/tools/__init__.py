"""Isaac tools — secure toolbox for agent capabilities.

Import this module to auto-register all built-in tools with the
global :class:`ToolRegistry`.
"""

from __future__ import annotations

from isaac.tools.base import ToolRegistry, get_tool_registry
from isaac.tools.browser import BrowserTool
from isaac.tools.file import FileReadTool, FileWriteTool, FileListTool, FileDeleteTool
from isaac.tools.search import WebSearchTool
from isaac.tools.email import EmailReadTool, EmailSendTool
from isaac.tools.calendar import CalendarReadTool, CalendarWriteTool
from isaac.tools.code import CodeTool


def register_all_tools() -> ToolRegistry:
    """Instantiate and register every built-in tool."""
    registry = get_tool_registry()
    for tool_cls in (
        BrowserTool,
        FileReadTool,
        FileWriteTool,
        FileListTool,
        FileDeleteTool,
        WebSearchTool,
        EmailReadTool,
        EmailSendTool,
        CalendarReadTool,
        CalendarWriteTool,
        CodeTool,
    ):
        try:
            registry.register(tool_cls())
        except Exception:
            pass  # graceful — tool may have missing deps
    return registry


__all__ = [
    "register_all_tools",
    "get_tool_registry",
    "BrowserTool",
    "FileReadTool",
    "FileWriteTool",
    "FileListTool",
    "FileDeleteTool",
    "WebSearchTool",
    "EmailReadTool",
    "EmailSendTool",
    "CalendarReadTool",
    "CalendarWriteTool",
    "CodeTool",
]
