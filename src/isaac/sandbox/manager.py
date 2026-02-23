"""Docker container lifecycle manager.

Handles image verification, container creation, and tear-down.  All
operations go through the Docker SDK — never through ``subprocess``.

Computer-Use additions
----------------------
* ``exec_command``       — run an arbitrary command in a *running* container
* ``take_screenshot``   — capture the virtual display via ``scrot``
* ``execute_ui_action`` — map a ``UIAction`` to ``xdotool`` commands and run
"""

from __future__ import annotations

import base64
import io
import logging
import tarfile
import time
from typing import TYPE_CHECKING

import docker
from docker.errors import DockerException, ImageNotFound

if TYPE_CHECKING:
    from docker.models.containers import Container

    from isaac.core.state import UIAction, UIActionResult
    from isaac.sandbox.security import SecurityPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# xdotool command builder (pure function — no I/O)
# ---------------------------------------------------------------------------

def _build_xdotool_command(action: UIAction) -> list[str]:  # type: ignore[name-defined]
    """Convert a ``UIAction`` into an xdotool argv list."""
    t = action.type
    if t == "click":
        return ["xdotool", "mousemove", "--sync",
                str(action.x), str(action.y), "click", "1"]
    if t == "double_click":
        return ["xdotool", "mousemove", "--sync",
                str(action.x), str(action.y), "click", "--repeat", "2", "1"]
    if t == "right_click":
        return ["xdotool", "mousemove", "--sync",
                str(action.x), str(action.y), "click", "3"]
    if t == "move":
        return ["xdotool", "mousemove", "--sync", str(action.x), str(action.y)]
    if t == "type":
        return ["xdotool", "type", "--clearmodifiers", action.text or ""]
    if t == "key":
        return ["xdotool", "key", action.key or "Return"]
    if t == "scroll":
        btn = "4" if action.scroll_direction in ("up", "left") else "5"
        cmds: list[str] = []
        for _ in range(action.scroll_amount):
            cmds += ["xdotool", "click", btn, ";"]
        return cmds[:-1]  # strip trailing semicolon token
    if t == "drag":
        return [
            "xdotool", "mousemove", "--sync", str(action.x), str(action.y),
            "mousedown", "1",
            "mousemove", "--sync", str(action.target_x), str(action.target_y),
            "mouseup", "1",
        ]
    if t == "wait":
        ms = action.duration_ms or 500
        return ["sleep", str(ms / 1000)]
    # screenshot or unknown — no-op placeholder (scrot called separately)
    return ["true"]


class SandboxManager:
    """Create, track, and destroy ephemeral Docker containers.

    Parameters
    ----------
    image:
        Docker image tag used for sandbox containers.
    policy:
        Security policy governing resource limits and isolation.
    """

    def __init__(self, image: str, policy: SecurityPolicy) -> None:
        self._image = image
        self._policy = policy
        self._client = docker.from_env()
        self._verify_image()

    # -- internal -----------------------------------------------------------

    def _verify_image(self) -> None:
        """Ensure the sandbox image exists locally."""
        try:
            self._client.images.get(self._image)
            logger.info("Sandbox image '%s' verified.", self._image)
        except ImageNotFound:
            logger.warning(
                "Sandbox image '%s' not found locally.  "
                "Build it with: docker build -t %s sandbox_image/",
                self._image,
                self._image,
            )

    # -- public API ---------------------------------------------------------

    def create_container(
        self,
        command: list[str],
        *,
        volumes: dict[str, dict[str, str]] | None = None,
        environment: dict[str, str] | None = None,
    ) -> Container:
        """Create (but do not start) an ephemeral container."""
        kwargs = self._policy.to_container_kwargs()
        container: Container = self._client.containers.create(
            image=self._image,
            command=command,
            volumes=volumes or {},
            environment=environment or {},
            detach=True,
            auto_remove=False,
            **kwargs,
        )
        logger.debug("Container %s created (image=%s).", container.short_id, self._image)
        return container

    def start(self, container: Container) -> None:
        """Start a previously created container."""
        container.start()
        logger.debug("Container %s started.", container.short_id)

    def wait(self, container: Container, timeout: int | None = None) -> int:
        """Block until the container exits.  Returns the exit code."""
        t = timeout or self._policy.timeout_seconds
        try:
            result = container.wait(timeout=t)
            exit_code: int = result.get("StatusCode", -1)
        except Exception:
            logger.warning("Container %s timed out after %ds — killing.", container.short_id, t)
            container.kill()
            exit_code = -1
        return exit_code

    def logs(self, container: Container) -> tuple[str, str]:
        """Return ``(stdout, stderr)`` from the container."""
        stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")
        return stdout, stderr

    def destroy(self, container: Container) -> None:
        """Force-remove the container and reclaim resources."""
        try:
            container.remove(force=True)
            logger.debug("Container %s destroyed.", container.short_id)
        except DockerException as exc:
            logger.error("Failed to destroy container %s: %s", container.short_id, exc)

    def close(self) -> None:
        """Close the underlying Docker client."""
        self._client.close()

    # -- Computer-Use API ---------------------------------------------------

    def exec_command(
        self,
        container: Container,
        cmd: list[str],
        *,
        environment: dict[str, str] | None = None,
        workdir: str = "/workspace",
    ) -> tuple[int, str, str]:
        """Execute *cmd* inside a *running* container.

        Returns
        -------
        tuple[int, str, str]
            ``(exit_code, stdout, stderr)``
        """
        env = environment or {}
        result = container.exec_run(
            cmd,
            environment=env,
            workdir=workdir,
            demux=True,
        )
        exit_code: int = result.exit_code or 0
        raw_out, raw_err = result.output or (b"", b"")
        stdout = (raw_out or b"").decode(errors="replace")
        stderr = (raw_err or b"").decode(errors="replace")
        return exit_code, stdout, stderr

    def take_screenshot(
        self,
        container: Container,
        display: str = ":99",
    ) -> bytes:
        """Capture the virtual display as PNG bytes via ``scrot``.

        The PNG is written to ``/tmp/isaac_screen.png`` inside the container
        and streamed back to the host using the Docker archive API.

        Returns
        -------
        bytes
            Raw PNG bytes.  Empty bytes on failure.
        """
        sc_path = "/tmp/isaac_screen.png"
        exit_code, _, stderr = self.exec_command(
            container,
            ["scrot", "-o", sc_path],
            environment={"DISPLAY": display},
        )
        if exit_code != 0:
            logger.error("scrot failed (exit=%d): %s", exit_code, stderr[:300])
            return b""

        # Stream the file out of the container
        try:
            bits, _ = container.get_archive(sc_path)
            buf = io.BytesIO(b"".join(bits))
            with tarfile.open(fileobj=buf) as tf:
                member = tf.getmembers()[0]
                f = tf.extractfile(member)
                return f.read() if f else b""
        except Exception as exc:
            logger.error("Failed to retrieve screenshot from container: %s", exc)
            return b""

    def execute_ui_action(
        self,
        container: Container,
        action: UIAction,  # type: ignore[name-defined]
        display: str = ":99",
    ) -> UIActionResult:  # type: ignore[name-defined]
        """Map *action* to xdotool commands, execute, capture before/after screenshots.

        Returns
        -------
        UIActionResult
            Populated with success flag and before/after screenshots.
        """
        from isaac.core.state import UIActionResult

        t0 = time.perf_counter()

        # Before screenshot
        before_png = self.take_screenshot(container, display)
        before_b64 = base64.b64encode(before_png).decode() if before_png else ""

        # Build and run xdotool command
        cmd = _build_xdotool_command(action)
        exit_code, _stdout, stderr = self.exec_command(
            container,
            cmd,
            environment={"DISPLAY": display},
        )

        # Small settle delay for UI to update
        time.sleep(0.4)

        # After screenshot
        after_png = self.take_screenshot(container, display)
        after_b64 = base64.b64encode(after_png).decode() if after_png else ""

        duration_ms = (time.perf_counter() - t0) * 1000

        result = UIActionResult(
            action=action,
            success=exit_code == 0,
            screenshot_before_b64=before_b64,
            screenshot_after_b64=after_b64,
            error=stderr.strip() if exit_code != 0 else "",
            duration_ms=round(duration_ms, 2),
        )
        logger.info(
            "UIAction '%s' at (%s,%s): exit=%d  %.0fms",
            action.type,
            action.x,
            action.y,
            exit_code,
            duration_ms,
        )
        return result

