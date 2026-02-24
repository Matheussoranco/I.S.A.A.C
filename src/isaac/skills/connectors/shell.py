"""ShellConnector — Execute allow-listed shell commands on the host.

Enforces a strict allowlist of commands and blocks shell metacharacters
(pipes, redirections, semicolons, etc.) to prevent abuse.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

_DANGEROUS_CHARS = re.compile(r"[|;&`$><\n\r]")

_DEFAULT_ALLOWED = frozenset(
    [
        "ls",
        "dir",
        "cat",
        "head",
        "tail",
        "wc",
        "find",
        "grep",
        "echo",
        "date",
        "whoami",
        "hostname",
        "pwd",
        "uname",
        "df",
        "du",
        "uptime",
        "python",
        "pip",
        "git",
        "curl",
    ]
)


class ShellConnector(BaseConnector):
    """Run allow-listed shell commands on the host."""

    name = "shell"
    description = (
        "Execute allow-listed shell commands with timeout protection. "
        "Blocks pipes, redirections, and dangerous metacharacters."
    )
    requires_env: list[str] = []

    def _allowed_commands(self) -> frozenset[str]:
        """Return the set of allowed commands."""
        try:
            from isaac.config.settings import get_settings

            cmds = get_settings().shell_allowed_commands
            if cmds:
                return frozenset(cmds)
        except Exception:
            pass
        return _DEFAULT_ALLOWED

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a shell command.

        Parameters
        ----------
        command : str
            The command string to execute.
        timeout : int
            Timeout in seconds (default 10, max 60).
        cwd : str | None
            Working directory for the command.
        """
        command: str = kwargs.get("command", "").strip()
        timeout: int = min(int(kwargs.get("timeout", 10)), 60)
        cwd: str | None = kwargs.get("cwd")

        if not command:
            return {"error": "No command provided"}

        # --- safety checks ---
        if _DANGEROUS_CHARS.search(command):
            return {"error": "Command contains blocked metacharacters (|;&`$><)"}

        parts = command.split()
        executable = parts[0].lower()
        allowed = self._allowed_commands()
        if executable not in allowed:
            return {"error": f"Command '{executable}' not in allowlist: {sorted(allowed)}"}

        try:
            start = time.perf_counter()
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env={**os.environ},
            )
            duration_ms = round((time.perf_counter() - start) * 1000)
            return {
                "command": command,
                "stdout": result.stdout[:10_000],
                "stderr": result.stderr[:5_000],
                "exit_code": result.returncode,
                "duration_ms": duration_ms,
            }
        except subprocess.TimeoutExpired:
            return {"command": command, "error": f"Timeout after {timeout}s"}
        except Exception as exc:
            logger.error("shell run failed: %s", exc)
            return {"command": command, "error": str(exc)}
