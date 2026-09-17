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

import contextlib
import logging
import os
import platform
import subprocess
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
            pass_secrets = bool(getattr(settings, "shell_pass_secrets", False))
            allowlist_cfg: list[str] = list(getattr(settings, "shell_allowed_commands", []) or [])
        except Exception:
            default_timeout, unrestricted, pass_secrets, allowlist_cfg = 30, False, False, []

        timeout = max(1, min(int(kwargs.get("timeout", default_timeout)), 600))

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

    def _run_unrestricted_confined(
        self,
        command: str,
        cwd: str | None,
        timeout: int,
        pass_secrets: bool = False,
        allowlist_cfg: list[str] | None = None,
    ) -> ToolResult:
        """Confined unrestricted shell: allow-list + audit log required.

        ``ISAAC_SHELL_UNRESTRICTED=true`` alone is NOT enough: an explicit
        ``ISAAC_SHELL_ALLOWED_COMMANDS`` allow-list must be configured and
        every invocation is audit-logged.  The constitutional gate (fail-closed)
        already ran before this point.
        """
        from isaac.skills.connectors.shell import _DEFAULT_ALLOWED

        allowed = {c.lower() for c in (allowlist_cfg or [])} or set(_DEFAULT_ALLOWED)
        if not allowlist_cfg:
            logger.error("Unrestricted shell BLOCKED: no ISAAC_SHELL_ALLOWED_COMMANDS configured")
            return ToolResult(
                success=False,
                error="BLOCKED: ISAAC_SHELL_UNRESTRICTED requires an explicit "
                "ISAAC_SHELL_ALLOWED_COMMANDS allow-list.",
            )
        first = (command.strip().split() or [""])[0].lower().rstrip(";,&|")
        if first not in allowed:
            return ToolResult(
                success=False,
                error=f"BLOCKED: command '{first}' not in unrestricted allow-list.",
            )
        try:
            from isaac.skills.connectors.registry import audit_connector
        except Exception:
            audit_connector = None  # type: ignore[assignment]
        if audit_connector is not None:
            with contextlib.suppress(Exception):
                audit_connector("shell", "unrestricted-invoke", command[:200])
        result = self._run_unrestricted(command, cwd, timeout, pass_secrets)
        if audit_connector is not None:
            with contextlib.suppress(Exception):
                audit_connector(
                    "shell",
                    "unrestricted-success" if result.success else "unrestricted-error",
                    command[:200],
                )
        return result

    def _run_unrestricted(
        self, command: str, cwd: str | None, timeout: int, pass_secrets: bool = False
    ) -> ToolResult:
        """Run without the metacharacter block, but still confined.

        POSIX path avoids ``shell=True``: the command is parsed with shlex
        and executed as argv (``shell=False``), so ``$(...)``, backticks,
        pipes and redirections never reach ``/bin/sh``. Anything that is
        not a plain argv invocation is denied rather than reinterpreted.
        """
        # Command substitution / expansion never allowed, even unrestricted:
        # it escapes the first-word allow-list ("git $(rm -rf ~)").
        if "$(" in command or "`" in command or "${" in command:
            return ToolResult(
                success=False,
                error="BLOCKED: command substitution ($(), ``, ${}) is never allowed.",
            )
        is_windows = platform.system() == "Windows"
        child_env = build_child_env(pass_secrets=pass_secrets)
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
                    env=child_env,
                )
            else:
                import shlex as _shlex

                try:
                    argv = _shlex.split(command, posix=True)
                except ValueError as exc:
                    return ToolResult(
                        success=False,
                        error=f"BLOCKED: could not parse command safely ({exc}).",
                    )
                if not argv:
                    return ToolResult(success=False, error="Missing 'command' parameter.")
                # shell=False: no pipes, redirections, chains, globs or
                # expansions. Multi-command input is denied, not half-run.
                if any(tok in {"|", "&", ";", ">", "<", "&&", "||", "$", "~"} for tok in argv):
                    return ToolResult(
                        success=False,
                        error="BLOCKED: shell operators are not allowed, even unrestricted. "
                        "Run one plain command per invocation.",
                    )
                completed = subprocess.run(
                    argv,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=cwd,
                    env=child_env,
                    executable=None,
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
