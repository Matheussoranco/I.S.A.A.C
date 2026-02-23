"""UI Executor — manages long-lived virtual-desktop containers for Computer-Use.

Unlike the ephemeral ``CodeExecutor`` (fire-and-forget), ``UIExecutor``
keeps the container alive across multiple screenshot→action cycles,
because Xvfb and the browser state must persist between steps.

Typical lifecycle::

    executor = UIExecutor()
    executor.start()                          # boots Xvfb + openbox
    screenshot = executor.screenshot()       # PNG bytes
    result = executor.act(UIAction(type="click", x=320, y=200))
    ...
    executor.stop()                           # tears down container
"""

from __future__ import annotations

import base64
import logging

from docker.models.containers import Container

from isaac.core.state import GUIState, UIAction, UIActionResult
from isaac.sandbox.manager import SandboxManager
from isaac.sandbox.security import ui_policy

logger = logging.getLogger(__name__)


class UIExecutor:
    """Long-lived container executor for virtual-desktop Computer-Use tasks.

    Parameters
    ----------
    image:
        Docker image tag for the UI sandbox (default: ``isaac-ui-sandbox:latest``).
    display:
        X display identifier inside the container (default: ``:99``).
    """

    def __init__(
        self,
        image: str | None = None,
        display: str = ":99",
    ) -> None:
        from isaac.config.settings import settings

        self._image = image or settings.ui_sandbox.image
        self._display = display
        self._policy = ui_policy()
        self._manager = SandboxManager(self._image, self._policy)
        self._container: Container | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Create and start the virtual-desktop container.

        The entrypoint boots Xvfb + openbox and then idles, waiting for
        ``docker exec`` calls from the host.
        """
        from isaac.config.settings import settings

        cfg = settings.ui_sandbox
        env = {
            "DISPLAY": self._display,
            "SCREEN_WIDTH": str(cfg.screen_width),
            "SCREEN_HEIGHT": str(cfg.screen_height),
            "SCREEN_DEPTH": str(cfg.screen_depth),
            "ISAAC_VNC_ENABLED": "1" if cfg.vnc_enabled else "0",
        }
        # Long-lived: no command override; entrypoint idles via wait
        self._container = self._manager.create_container(
            command=[],   # entrypoint handles idle loop
            environment=env,
        )
        self._manager.start(self._container)

        # Wait for Xvfb to be ready
        import time
        timeout = 10.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            ec, _, _ = self._manager.exec_command(
                self._container,
                ["xdpyinfo", "-display", self._display],
                environment={"DISPLAY": self._display},
            )
            if ec == 0:
                logger.info("UIExecutor: virtual display %s is ready.", self._display)
                return
            time.sleep(0.5)

        logger.warning("UIExecutor: display did not become ready within %.0fs.", timeout)

    def stop(self) -> None:
        """Destroy the container and release resources."""
        if self._container is not None:
            self._manager.destroy(self._container)
            self._container = None
            logger.info("UIExecutor: container destroyed.")

    def __enter__(self) -> UIExecutor:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # -- actions ------------------------------------------------------------

    def screenshot(self) -> bytes:
        """Return the current virtual display as raw PNG bytes."""
        self._require_running()
        return self._manager.take_screenshot(self._container, self._display)  # type: ignore[arg-type]

    def screenshot_b64(self) -> str:
        """Return the current virtual display as a base64-encoded PNG string."""
        raw = self.screenshot()
        return base64.b64encode(raw).decode() if raw else ""

    def act(self, action: UIAction) -> UIActionResult:
        """Execute a single ``UIAction`` and return the before/after result."""
        self._require_running()
        return self._manager.execute_ui_action(
            self._container,  # type: ignore[arg-type]
            action,
            self._display,
        )

    def exec_python(self, code: str) -> tuple[int, str, str]:
        """Run a Python script string inside the container (hybrid mode).

        The script is written to ``/tmp/isaac_task.py`` via a heredoc-style
        echo, then executed with python3.

        Returns
        -------
        tuple[int, str, str]
            ``(exit_code, stdout, stderr)``
        """
        self._require_running()

        # Write script to a temp file on the HOST, then copy into container
        script_path = "/tmp/isaac_task.py"

        # Use docker cp via SDK: we put the file in a tar stream
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            encoded = code.encode()
            info = tarfile.TarInfo(name="isaac_task.py")
            info.size = len(encoded)
            tf.addfile(info, io.BytesIO(encoded))
        buf.seek(0)
        self._container.put_archive("/tmp", buf)  # type: ignore[union-attr]

        return self._manager.exec_command(
            self._container,  # type: ignore[arg-type]
            ["python3", script_path],
            environment={"DISPLAY": self._display},
        )

    def get_gui_state(self) -> GUIState:
        """Return a ``GUIState`` snapshot populated from the current screenshot.

        Element detection requires a vision LLM and is handled by the
        ComputerUse node.  This method only fills the ``screenshot_b64`` field
        and basic display metadata.
        """
        from isaac.config.settings import settings

        cfg = settings.ui_sandbox
        return GUIState(
            screenshot_b64=self.screenshot_b64(),
            screen_width=cfg.screen_width,
            screen_height=cfg.screen_height,
            display=self._display,
        )

    # -- internal -----------------------------------------------------------

    def _require_running(self) -> None:
        if self._container is None:
            raise RuntimeError(
                "UIExecutor has no running container.  Call start() first."
            )
