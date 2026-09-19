"""Approval-gated, read-only host commands with a fail-closed safety review."""

from __future__ import annotations

import logging
import os
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)

_MAX_STDOUT = 12_000
_MAX_STDERR = 4_000

#: Minimal env inherited by child shells.  Everything else — in particular
#: secrets — is stripped unless ``ISAAC_SHELL_PASS_SECRETS=true`` opts in.
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "PATHEXT",
        "COMSPEC",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TERM",
    }
)

#: Secret markers — stripped from the child env unless explicitly opted in.
_SECRET_SUBSTRINGS = ("TOKEN", "KEY", "PASSWORD", "SECRET", "CREDENTIALS")
_SECRET_PREFIXES = (
    "OPENAI_",
    "ANTHROPIC_",
    "TELEGRAM_",
    "GITHUB_",
    "ISAAC_EMAIL_",
    "ISAAC_CALDAV_",
)


def build_child_env(pass_secrets: bool = False) -> dict[str, str]:
    """Build a minimal child env (secrets-safe by default).

    Never logs values — only key counts.  With ``pass_secrets=False``
    (default) any key matching :data:`_SECRET_PREFIXES` /
    :data:`_SECRET_SUBSTRINGS` is dropped.
    """
    env: dict[str, str] = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
    if pass_secrets:
        for k, v in os.environ.items():
            if k not in env:
                env[k] = v
        return env
    dropped = 0
    for k in list(os.environ):
        uk = k.upper()
        if uk in _SAFE_ENV_KEYS:
            continue
        if uk.startswith(_SECRET_PREFIXES) or any(s in uk for s in _SECRET_SUBSTRINGS):
            dropped += 1
            continue
        # Non-secret, non-allow-listed vars are also dropped (minimal env).
    if dropped:
        logger.debug("Stripped %d secret env var(s) from child shell env", dropped)
    # Ensure PATH exists so allow-listed binaries resolve.
    if "PATH" not in env and os.environ.get("PATH"):
        env["PATH"] = os.environ["PATH"]
    return env


class ShellTool(IsaacTool):
    """Execute a shell command on the host machine (constitution-gated)."""

    name = "shell"
    description = (
        "Run a bounded read-only command within allowed roots. Supports echo, pwd, "
        "whoami, hostname, date, uname, ls/dir [path], cat/head/tail/wc file, "
        "and grep literal-text file. No external programs, scripts, pipes or chains. "
        "Use file tools for changes and the Docker code tool for computation."
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
            pass_secrets = bool(getattr(settings, "shell_pass_secrets", False))
            allowlist_cfg: list[str] = list(getattr(settings, "shell_allowed_commands", []) or [])
        except Exception:
            return ToolResult(success=False, error="BLOCKED: shell configuration unavailable")

        try:
            timeout = max(1, min(int(kwargs.get("timeout", default_timeout)), 600))
        except (ValueError, TypeError, OverflowError):
            return ToolResult(success=False, error="Invalid timeout: expected seconds as int.")

        # ── Gate 1: constitutional review (fail-closed) ────────────────────
        try:
            decision = self._review_or_raise(command, cwd)
        except PermissionError as exc:
            return ToolResult(
                success=False,
                error=f"BLOCKED by safety critic: {exc}",
                metadata={"constitution": {"fail_closed": True}},
            )
        if decision is not None and not decision.allow:
            violated = ", ".join(v.rule for v in decision.violations) or decision.reason
            return ToolResult(
                success=False,
                error=f"BLOCKED by safety critic: {violated}",
                metadata={"constitution": decision.to_dict()},
            )

        # ── Gate 3: execution mode ─────────────────────────────────────────
        if unrestricted:
            return self._run_unrestricted_confined(
                command, cwd, timeout, pass_secrets, allowlist_cfg
            )
        return self._run_allowlisted(command, cwd, timeout, pass_secrets)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _review_or_raise(self, command: str, cwd: str | None) -> Any:
        """Constitutional review — fail-closed on infrastructure errors."""
        try:
            from isaac.security.constitution import review

            # use_llm=False keeps the tool fast and fully offline; the symbolic
            # deny-list already covers the catastrophic cases.
            return review("shell", command, context={"cwd": cwd or os.getcwd()}, use_llm=False)
        except Exception as exc:
            logger.error("Constitution review unavailable — blocking shell execution: %s", exc)
            raise PermissionError(f"safety review unavailable ({exc})") from exc

    def _review(self, command: str, cwd: str | None) -> Any:  # backwards-compat
        """Deprecated: use :meth:`_review_or_raise` (fail-closed)."""
        return self._review_or_raise(command, cwd)

    def _run_allowlisted(
        self, command: str, cwd: str | None, timeout: int, pass_secrets: bool = False
    ) -> ToolResult:
        """Delegate to the strict ShellConnector (allow-list + metachar block)."""
        try:
            from isaac.skills.connectors.shell import ShellConnector

            result = ShellConnector().run(
                command=command, timeout=timeout, cwd=cwd, pass_secrets=pass_secrets
            )
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

    def _run_unrestricted_confined(self, *args: Any, **kwargs: Any) -> ToolResult:
        return ToolResult(
            success=False,
            error="BLOCKED: unrestricted host execution is disabled; use the Docker code tool.",
        )

    def _run_unrestricted(self, *args: Any, **kwargs: Any) -> ToolResult:
        return self._run_unrestricted_confined(*args, **kwargs)

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
