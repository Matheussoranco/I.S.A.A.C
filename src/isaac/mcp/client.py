from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None  # For HTTP/SSE servers


class MCPClient:
    """
    Client for connecting to Model Context Protocol (MCP) servers.
    Supports both stdio (local processes) and HTTP (SSE) transports.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        """Connect to the MCP server using the configured transport."""
        stack = AsyncExitStack()
        try:
            await stack.__aenter__()
            if self.config.url:
                read, write = await stack.enter_async_context(sse_client(self.config.url))
            else:
                server_params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env,
                )
                read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            self._exit_stack = stack

            logger.info(f"Connected to MCP server: {self.config.name}")
        except Exception as e:
            await stack.aclose()
            logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            raise

    async def list_tools(self) -> list[dict[str, Any]]:
        """Fetch the list of available tools from the MCP server."""
        if not self._session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        result = await self._session.list_tools()
        # Map MCP tool format to a generic dict for Isaac integration
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool on the MCP server."""
        if not self._session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        try:
            result = await self._session.call_tool(tool_name, arguments)
            return result.content
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
        logger.info(f"Disconnected from MCP server: {self.config.name}")
