"""Isaac tools — secure toolbox for agent capabilities.

Import this module to auto-register all built-in tools with the
global :class:`ToolRegistry`.
"""

from __future__ import annotations

import contextlib

from isaac.tools.base import ToolRegistry, get_tool_registry
from isaac.tools.browser import BrowserTool
from isaac.tools.calendar import CalendarReadTool, CalendarWriteTool
from isaac.tools.code import CodeTool
from isaac.tools.desktop import ComputerControlTool, ComputerDescribeTool, ComputerViewTool
from isaac.tools.email import EmailReadTool, EmailSendTool
from isaac.tools.file import FileDeleteTool, FileListTool, FileReadTool, FileWriteTool
from isaac.tools.fileops import (
    FsCopyTool,
    FsInfoTool,
    FsListTool,
    FsMkdirTool,
    FsMoveTool,
    FsReadTool,
    FsWriteTool,
)
from isaac.tools.search import WebSearchTool
from isaac.tools.shell import ShellTool
from isaac.tools.system import SystemInfoTool


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
        ComputerViewTool,
        ComputerDescribeTool,
        ComputerControlTool,
        # Host-reach tools (operate on the real machine; gated by risk/constitution)
        ShellTool,
        SystemInfoTool,
        FsListTool,
        FsInfoTool,
        FsReadTool,
        FsWriteTool,
        FsMkdirTool,
        FsMoveTool,
        FsCopyTool,
    ):
        # graceful — tool may have missing deps
        with contextlib.suppress(Exception):
            registry.register(tool_cls())
    return registry


__all__ = [
    "BrowserTool",
    "CalendarReadTool",
    "CalendarWriteTool",
    "CodeTool",
    "ComputerControlTool",
    "ComputerDescribeTool",
    "ComputerViewTool",
    "EmailReadTool",
    "EmailSendTool",
    "FileDeleteTool",
    "FileListTool",
    "FileReadTool",
    "FileWriteTool",
    "FsCopyTool",
    "FsInfoTool",
    "FsListTool",
    "FsMkdirTool",
    "FsMoveTool",
    "FsReadTool",
    "FsWriteTool",
    "ShellTool",
    "SystemInfoTool",
    "WebSearchTool",
    "get_tool_registry",
    "register_all_tools",
]
