"""Code executor — injects Python into the Docker sandbox and captures output.

This is the **only** bridge between the cognitive graph and the outside world.
The host process never ``exec``s user-generated code directly.
"""

from __future__ import annotations

import logging
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
        from isaac.config.settings import settings  # noqa: PLC0415

        self._image = image or settings.sandbox.image
        self._policy = policy or default_policy()
        self._manager = SandboxManager(self._image, self._policy)

    def execute(self, code: str) -> ExecutionResult:
        """Run *code* in a fresh sandbox and return the captured output.

        Lifecycle
        ---------
        1. Write ``code`` to a temporary file on the host.
        2. Bind-mount the file read-only at ``/input/task.py``.
        3. Create and start an ephemeral container.
        4. Wait for completion (or timeout).
        5. Capture stdout / stderr / exit code.
        6. Destroy the container and clean up temp files.
        """
        tmp_dir = tempfile.mkdtemp(prefix="isaac_sandbox_")
        task_file = Path(tmp_dir) / "task.py"
        task_file.write_text(code, encoding="utf-8")

        volumes = {
            str(tmp_dir): {"bind": "/input", "mode": "ro"},
        }

        container = self._manager.create_container(
            command=["python", "/input/task.py"],
            volumes=volumes,
        )

        t0 = time.perf_counter()
        try:
            self._manager.start(container)
            exit_code = self._manager.wait(container)
            stdout, stderr = self._manager.logs(container)
            duration_ms = (time.perf_counter() - t0) * 1000
        finally:
            self._manager.destroy(container)
            # Clean up host temp files
            try:
                task_file.unlink(missing_ok=True)
                Path(tmp_dir).rmdir()
            except OSError:
                logger.debug("Temp dir cleanup failed for %s", tmp_dir)

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
