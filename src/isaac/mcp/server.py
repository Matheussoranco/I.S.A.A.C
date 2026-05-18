"""MCP stdio server — I.S.A.A.C. as a Model Context Protocol tool provider.

Implements MCP protocol version 2024-11-05 over stdio transport (JSON-RPC 2.0).
Claude Code connects by running: ``isaac mcp-serve``

Supported MCP methods:
    initialize          Capability handshake
    notifications/initialized  Post-init notification (no-op)
    tools/list          Enumerate available tools
    tools/call          Invoke a tool by name
    ping                Health check

Tools exposed:
    isaac_ask           Full cognitive-loop task execution
    isaac_memory_search Multi-layer memory recall
    isaac_skill_search  Semantic skill library lookup
    isaac_code_execute  Docker-sandboxed Python execution
    isaac_web_search    DuckDuckGo search
    isaac_knowledge_query  Semantic knowledge graph query
    isaac_spawn_subagent   Claude API sub-agent delegation
    isaac_meta_stats    Self-improvement statistics
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from isaac.mcp.tools import TOOL_SCHEMAS, call_tool

logger = logging.getLogger(__name__)

MCP_VERSION = "2024-11-05"
SERVER_NAME = "I.S.A.A.C."
SERVER_VERSION = "0.2.0"


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _ok(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _send(obj: dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


def _handle_initialize(req: dict[str, Any]) -> dict[str, Any]:
    return _ok(
        req.get("id"),
        {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        },
    )


def _handle_tools_list(req: dict[str, Any]) -> dict[str, Any]:
    return _ok(req.get("id"), {"tools": TOOL_SCHEMAS})


def _handle_tools_call(req: dict[str, Any]) -> dict[str, Any]:
    params = req.get("params", {})
    name: str = params.get("name", "")
    arguments: dict[str, Any] = params.get("arguments", {})

    if not name:
        return _err(req.get("id"), -32602, "Missing tool name")

    try:
        result = call_tool(name, arguments)
        # MCP expects content array
        content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
        return _ok(req.get("id"), {"content": content, "isError": "error" in result})
    except ValueError as exc:
        return _err(req.get("id"), -32601, str(exc))
    except Exception as exc:
        logger.exception("Tool call failed: %s", exc)
        return _err(req.get("id"), -32603, "Internal error", str(exc))


def _handle_ping(req: dict[str, Any]) -> dict[str, Any]:
    return _ok(req.get("id"), {})


_DISPATCH = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "ping": _handle_ping,
    # Notifications (no response needed)
    "notifications/initialized": None,
    "notifications/cancelled": None,
}


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------


def run_server() -> None:
    """Run the MCP stdio server until EOF."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Windows UTF-8 fix
    if sys.platform == "win32":
        import io

        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

    logger.info("I.S.A.A.C. MCP server started (protocol %s)", MCP_VERSION)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _send(_err(None, -32700, f"Parse error: {exc}"))
            continue

        method: str = req.get("method", "")
        handler = _DISPATCH.get(method)

        if handler is None:
            if method in _DISPATCH:
                # It's a notification — no response
                continue
            _send(_err(req.get("id"), -32601, f"Method not found: {method!r}"))
            continue

        try:
            response = handler(req)
            _send(response)
        except Exception as exc:
            logger.exception("Handler for %r raised: %s", method, exc)
            _send(_err(req.get("id"), -32603, "Internal error", str(exc)))
