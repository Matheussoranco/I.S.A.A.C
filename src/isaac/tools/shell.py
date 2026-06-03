"""Shell Tool — run commands on the host, gated by the constitutional critic.

This is the capability that lets a specialist actually *operate the PC*: run
``git``, invoke build tools, move files, query the system.  Because that power
is dangerous, every invocation passes through three gates:

1. **Constitutional review** (:func:`isaac.security.constitution.review`) —
   a critical-severity match (``rm -rf /``, fork bomb, raw disk write, …) is a
   hard deny that nothing can override.
2. **Risk gating** — the tool is risk level 4, so the :class:`AgentLoop`
   refuses to run it unless the caller opted in (``auto_approve`` / human
   approval).  An OS-operator specialist opts in explicitly.
3. **Execution mode** — by default commands run through the strict allow-list
   + metacharacter block of :class:`~isaac.skills.connectors.shell.ShellConnector`.
   Setting ``ISAAC_SHELL_UNRESTRICTED=true`` switches to a full platform shell
   (PowerShell on Windows, ``/bin/sh`` elsewhere) for power users on a trusted
   machine — the constitutional gate still applies.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)

_MAX_STDOUT = 12_000
_MAX_STDERR = 4_000


class ShellTool(IsaacTool):
    """Execute a shell command on the host machine (constitution-gated)."""

    name = "shell"
    description = (
        "Run a command on the host operating system and return its stdout/stderr/"
        "exit code. Use for git, build tools, package managers, file management, "
        "and system queries. Provide 'command' (the full command line) and "
        "optionally 'cwd' (working directory) and 'timeout' (seconds). High risk: "
        "destructive commands are blocked by the safety critic."
    )
    risk_level = 4  # __init_subclass__ auto-sets requires_approval = True
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The full command line to execute on the host.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory (absolute path).",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default from settings, max 600).",
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        command: str = str(kwargs.get("command", "")).strip()
        cwd: str | None = kwargs.get("cwd") or None
        if not command:
            return ToolResult(success=False, error="Missing 'command' parameter.")

        try:
            from isaac.config.settings import get_settings

            settings = get_settings()
            default_timeout = int(getattr(settings, "shell_tool_timeout", 30))
            unrestricted = bool(getattr(settings, "shell_unrestricted", False))
        except Exception:
            default_timeout, unrestricted = 30, False

        timeout = max(1, min(int(kwargs.get("timeout", default_timeout)), 600))

        # ── Gate 1: constitutional review (hard-denies critical patterns) ──
        decision = self._review(command, cwd)
        if decision is not None and not decision.allow:
            violated = ", ".join(v.rule for v in decision.violations) or decision.reason
            return ToolResult(
                success=False,
                error=f"BLOCKED by safety critic: {violated}",
                metadata={"constitution": decision.to_dict()},
            )

        # ── Gate 3: execution mode ─────────────────────────────────────────
        if unrestricted:
            return self._run_unrestricted(command, cwd, timeout)
        return self._run_allowlisted(command, cwd, timeout)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _review(self, command: str, cwd: str | None) -> Any:
        try:
            from isaac.security.constitution import review

            # use_llm=False keeps the tool fast and fully offline; the symbolic
            # deny-list already covers the catastrophic cases.
            return review("shell", command, context={"cwd": cwd or os.getcwd()}, use_llm=False)
        except Exception as exc:  # pragma: no cover - safety layer is optional
            logger.debug("Constitution review unavailable: %s", exc)
            return None

    def _run_allowlisted(self, command: str, cwd: str | None, timeout: int) -> ToolResult:
        """Delegate to the strict ShellConnector (allow-list + metachar block)."""
        try:
            from isaac.skills.connectors.shell import ShellConnector

            result = ShellConnector().run(command=command, timeout=timeout, cwd=cwd)
        except Exception as exc:
            return ToolResult(success=False, error=f"Shell execution failed: {exc}")

        if "error" in result and "exit_code" not in result:
            return ToolResult(success=False, error=str(result["error"]))
        return self._format(
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            exit_code=int(result.get("exit_code", 0)),
            command=command,
        )

    def _run_unrestricted(self, command: str, cwd: str | None, timeout: int) -> ToolResult:
        """Run through the platform shell (PowerShell on Windows, /bin/sh else)."""
        is_windows = platform.system() == "Windows"
        try:
            if is_windows:
                args = [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ]
                completed = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=cwd,
                    env={**os.environ},
                )
            else:
                completed = subprocess.run(
                    command,
                    shell=True,  # noqa: S602 - intentional power-user mode, constitution-gated
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=cwd,
                    env={**os.environ},
                    executable="/bin/sh",
                )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"Command timed out after {timeout}s.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Shell execution failed: {exc}")

        return self._format(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            exit_code=completed.returncode,
            command=command,
        )

    @staticmethod
    def _format(stdout: str, stderr: str, exit_code: int, command: str) -> ToolResult:
        body = stdout[:_MAX_STDOUT]
        err = stderr[:_MAX_STDERR]
        ok = exit_code == 0
        summary = body if body else "(no stdout)"
        if err:
            summary = f"{summary}\n[stderr]\n{err}"
        summary = f"$ {command}\n[exit {exit_code}]\n{summary}"
        return ToolResult(
            success=ok,
            output=summary,
            error="" if ok else (err or f"Command exited with code {exit_code}."),
            metadata={"exit_code": exit_code},
        )
