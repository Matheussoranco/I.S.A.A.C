"""Read-only host command connector; no interpreters or external processes."""

from __future__ import annotations

from typing import Any, ClassVar

from isaac.security.safe_shell import COMMANDS, run_readonly
from isaac.skills.connectors.base import BaseConnector

_DEFAULT_ALLOWED = COMMANDS


class ShellConnector(BaseConnector):
    name = "shell"
    description = "Bounded read-only commands within allowed paths; no host code execution."
    requires_env: ClassVar[list[str]] = []

    def run(self, **kwargs: Any) -> dict[str, Any]:
        from isaac.config.settings import get_settings

        try:
            settings = get_settings()
            if settings.shell_unrestricted or kwargs.get("pass_secrets"):
                raise ValueError("Unrestricted host shell is disabled; use the Docker code tool")
            configured = settings.shell_allowed_commands
            allowed = frozenset(c.lower() for c in configured) if configured else COMMANDS
            command = str(kwargs.get("command", ""))
            stdout = run_readonly(command, kwargs.get("cwd"), allowed)
            return {"command": command, "stdout": stdout, "stderr": "", "exit_code": 0}
        except Exception as exc:
            return {"error": str(exc)}
