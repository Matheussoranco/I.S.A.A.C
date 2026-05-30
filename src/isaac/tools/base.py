"""Abstract base tool — all Isaac tools inherit from this.

Every tool declares its risk level, whether it requires approval,
and whether it must run inside a sandbox.  The capability token
system checks these fields before execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Outcome of a tool invocation."""

    success: bool = False
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class IsaacTool(ABC):
    """Abstract base class for all Isaac tools.

    Attributes
    ----------
    name:
        Human-readable tool name.
    description:
        What the tool does (shown to the LLM).
    risk_level:
        Risk level 1–5.  Level 4+ auto-requires approval.
    requires_approval:
        Whether human approval is needed before execution.
    sandbox_required:
        Whether the tool must run inside a Docker container.
    """

    name: str = ""
    description: str = ""
    risk_level: int = 1
    requires_approval: bool = False
    sandbox_required: bool = False

    #: JSON-Schema describing the keyword arguments accepted by ``execute``.
    #: Used to build native LLM tool-calling schemas (``bind_tools``).  The
    #: default is a permissive empty object; concrete tools should override it.
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-set requires_approval for high-risk tools."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "risk_level") and cls.risk_level >= 4:
            cls.requires_approval = True

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given arguments.

        All concrete subclasses must implement this as ``async def`` so
        they can be uniformly awaited by the approval and connector nodes.

        Parameters
        ----------
        **kwargs:
            Tool-specific keyword arguments.

        Returns
        -------
        ToolResult
            The outcome of the tool invocation.
        """

    async def aclose(self) -> None:
        """Release any long-lived resources held by the tool.

        No-op by default.  Tools that hold persistent state (e.g. a browser
        session) override this; the agent loop calls it when a run finishes.
        """
        return None

    def to_schema(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
            "sandbox_required": self.sandbox_required,
            "parameters": self.parameters,
        }

    def to_function_schema(self) -> dict[str, Any]:
        """Return an OpenAI/Anthropic-style function schema for native tool calling.

        The returned dict is accepted directly by LangChain's
        :meth:`BaseChatModel.bind_tools`, which makes the tool callable by any
        provider that supports function calling (Anthropic, OpenAI, Ollama).
        """
        params = self.parameters or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


class ToolRegistry:
    """Registry of all available tools.

    Tools are registered at startup and looked up by name during
    the Synthesis/Planner nodes' tool selection.
    """

    def __init__(self) -> None:
        self._tools: dict[str, IsaacTool] = {}

    def register(self, tool: IsaacTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (risk=%d)", tool.name, tool.risk_level)

    def get(self, name: str) -> IsaacTool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return schemas for all registered tools."""
        return [t.to_schema() for t in self._tools.values()]

    def list_all(self) -> list[IsaacTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def filter_by_max_risk(self, max_risk: int) -> list[IsaacTool]:
        """Return tools at or below the given risk level."""
        return [t for t in self._tools.values() if t.risk_level <= max_risk]


# Module-level registry singleton
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Return the global tool registry singleton."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
