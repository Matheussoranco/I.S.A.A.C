"""Code executor — injects Python into the Docker sandbox and captures output.

This is the **only** bridge between the cognitive graph and the outside world.
The host process never ``exec``s user-generated code directly.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

from isaac.core.state import ExecutionResult
from isaac.sandbox.manager import SandboxManager
from isaac.sandbox.security import SecurityPolicy, default_policy

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Execute a Python code string inside an ephemeral Docker container.

    Parameters
    ----------
    image:
        Docker image tag for the sandbox.
    policy:
        Security policy (defaults to settings-derived policy).
    """

    def __init__(
        self,
        image: str | None = None,
        policy: SecurityPolicy | None = None,
    ) -> None:
        from isaac.config.settings import settings

        self._image = image or settings.sandbox.image
        self._policy = policy or default_policy()
        self._manager = SandboxManager(self._image, self._policy)

    def execute(
        self,
        code: str,
        *,
        extra_files: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run *code* in a fresh sandbox and return the captured output.

        ``extra_files`` maps sandbox-relative paths (for example
        ``"isaac_arc/dsl.py"``) to file contents; they are mounted read-only
        next to ``task.py`` so imports resolve without touching ``sys.path``.

        Lifecycle
        ---------
        1. Validate `code` with `ast.parse` to catch immediate errors.
        2. Write ``code`` (and any extra files) to a temporary host directory.
        3. Bind-mount the directory read-only at ``/input``.
        4. Create and start an ephemeral container.
        5. Wait for completion (or timeout).
        6. Capture stdout / stderr / exit code.
        7. Destroy the container and clean up temp files.
        """
        import ast

        try:
            tree = ast.parse(code)
            # Basic AST walk to reject out-of-scope modules
            # Modules that must never be imported — even inside Docker,
            # the host-side pre-check catches obvious injections early.
            _BLOCKED_MODULES = frozenset(
                {
                    "os",
                    "socket",
                    "subprocess",
                    "ctypes",
                    "importlib",
                    "sys",
                    "multiprocessing",
                    "pty",
                    "fcntl",
                    "signal",
                    "resource",
                    "platform",
                    "shutil",
                    "tempfile",
                    "glob",
                    "builtins",
                    "imp",
                    "runpy",
                    "code",
                    "codeop",
                    "posix",
                    "nt",
                    "mmap",
                    "select",
                    "selectors",
                    "threading",
                    "_thread",
                    "concurrent",
                }
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in _BLOCKED_MODULES:
                            return ExecutionResult(
                                stdout="",
                                stderr=(
                                    f"SecurityError: Import of module '{alias.name}' is blocked."
                                ),
                                exit_code=1,
                                duration_ms=0.0,
                            )
                elif isinstance(node, ast.ImportFrom):
                    top = (node.module or "").split(".")[0]
                    if top in _BLOCKED_MODULES:
                        return ExecutionResult(
                            stdout="",
                            stderr=f"SecurityError: Import from module '{node.module}' is blocked.",
                            exit_code=1,
                            duration_ms=0.0,
                        )
        except SyntaxError as e:
            return ExecutionResult(
                stdout="",
                stderr=f"SyntaxError: {e.msg} at line {e.lineno}",
                exit_code=1,
                duration_ms=0.0,
            )

        tmp_dir = tempfile.mkdtemp(prefix="isaac_sandbox_")
        root = Path(tmp_dir).resolve()
        task_file = root / "task.py"
        task_file.write_text(code, encoding="utf-8")

        for relative, content in (extra_files or {}).items():
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return ExecutionResult(
                    stdout="",
                    stderr=f"SecurityError: extra file path escapes sandbox: {relative!r}",
                    exit_code=1,
                    duration_ms=0.0,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        volumes = {
            str(tmp_dir): {"bind": "/input", "mode": "ro"},
        }

        t0 = time.perf_counter()
        container = None
        try:
            container = self._manager.create_container(
                command=["python", "/input/task.py"],
                volumes=volumes,
            )
            self._manager.start(container)
            exit_code = self._manager.wait(container)
            stdout, stderr = self._manager.logs(container)
            duration_ms = (time.perf_counter() - t0) * 1000
        finally:
            if container is not None:
                self._manager.destroy(container)
            # Clean up host temp files
            shutil.rmtree(tmp_dir, ignore_errors=True)

        result = ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=round(duration_ms, 2),
        )
        logger.info(
            "Sandbox execution finished: exit=%d  duration=%.1fms",
            exit_code,
            duration_ms,
        )
        return result

    def close(self) -> None:
        """Release the Docker client."""
        self._manager.close()
