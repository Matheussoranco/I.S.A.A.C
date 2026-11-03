from __future__ import annotations

import logging
from typing import Any

from isaac.mcp.client import MCPClient, MCPServerConfig
from isaac.tools.base import IsaacTool, ToolResult, get_tool_registry

logger = logging.getLogger(__name__)


class MCPToolWrapper(IsaacTool):
    """
    Wraps an MCP server tool as a native IsaacTool.
    """

    def __init__(self, client: MCPClient, tool_def: dict[str, Any]):
        super().__init__()
        self.client = client
        self.name = tool_def["name"]
        self.description = tool_def.get("description") or ""
        # MCP tools are generally treated as risk 1 unless they are known to be destructive.
        # In a full implementation, this could be mapped from MCP metadata.
        self.risk_level = 1
        self.parameters = tool_def.get("input_schema") or {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            # Call the remote MCP tool
            content = await self.client.call_tool(self.name, kwargs)

            # MCP content is usually a list of TextContent/ImageContent.
            # We flatten it to a string for Isaac's ToolResult.
            output_text = ""
            for item in content:
                if hasattr(item, "text"):
                    output_text += item.text + "\n"
                else:
                    output_text += str(item) + "\n"

            return ToolResult(success=True, output=output_text.strip())
        except Exception as e:
            logger.error(f"MCP tool {self.name} execution failed: {e}")
            return ToolResult(success=False, error=str(e))


async def load_mcp_servers_from_config(config_path: str | None = None):
    """
    Discovers and registers MCP servers from .mcp.json or config.yaml.
    """
    import pathlib

    candidates = (
        [pathlib.Path(config_path)]
        if config_path
        else [
            pathlib.Path.cwd() / ".mcp.json",
            pathlib.Path.home() / ".mcp.json",
        ]
    )
    mcp_json_path = next((p for p in candidates if p.exists()), candidates[0])

    servers_to_load: dict[str, Any] = {}

    if mcp_json_path.exists():
        try:
            import json

            with open(mcp_json_path, encoding="utf-8") as f:
                data = json.load(f)
                servers_to_load.update(data.get("mcpServers", {}))
        except Exception as e:
            logger.warning(f"Could not read {mcp_json_path}: {e}")

    registry = get_tool_registry()

    for name, cfg in servers_to_load.items():
        # Map raw config to MCPServerConfig
        server_cfg = MCPServerConfig(
            name=name,
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
            env=cfg.get("env", {}),
            url=cfg.get("url"),
        )

        client: MCPClient | None = None
        try:
            client = MCPClient(server_cfg)
            await client.connect()

            # Discover tools
            tools_defs = await client.list_tools()
            if not tools_defs:
                await client.disconnect()
                continue
            for t_def in tools_defs:
                wrapper = MCPToolWrapper(client, t_def)
                registry.register(wrapper)
                logger.info(f"Registered MCP tool: {wrapper.name} from server {name}")

        except Exception as e:
            logger.error(f"Failed to load MCP server {name}: {e}")
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:
                    logger.debug("MCP client cleanup failed for %s.", name, exc_info=True)
