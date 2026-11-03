"""Isaac tools — secure toolbox for agent capabilities.

Import this module to auto-register all built-in tools with the
global :class:`ToolRegistry`.
"""

from __future__ import annotations

import logging

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
from isaac.tools.kanban import (
    KANBAN_TOOL_CLASSES,
    KanbanBlockTool,
    KanbanCommentTool,
    KanbanCompleteTool,
    KanbanCreateTool,
    KanbanHeartbeatTool,
    KanbanLinkTool,
    KanbanShowTool,
)
from isaac.tools.search import WebSearchTool
from isaac.tools.shell import ShellTool
from isaac.tools.system import SystemInfoTool

logger = logging.getLogger(__name__)


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
        # Kanban multi-agent queue tools
        *KANBAN_TOOL_CLASSES,
    ):
        try:
            registry.register(tool_cls())
        except Exception as exc:
            logger.warning("Could not register tool %s: %s", tool_cls.__name__, exc)
    return registry


__all__ = [
    "KANBAN_TOOL_CLASSES",
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
    "KanbanBlockTool",
    "KanbanCommentTool",
    "KanbanCompleteTool",
    "KanbanCreateTool",
    "KanbanHeartbeatTool",
    "KanbanLinkTool",
    "KanbanShowTool",
    "ShellTool",
    "SystemInfoTool",
    "WebSearchTool",
    "get_tool_registry",
    "register_all_tools",
]
