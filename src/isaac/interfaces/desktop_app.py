"""Native desktop shell for the I.S.A.A.C. web interface.

PyWebView hosts the existing local FastAPI UI inside the operating system's
native WebView.  The HTTP server binds to an ephemeral loopback port and stops
when the native window closes.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from isaac.interfaces.web_app import create_app

logger = logging.getLogger(__name__)


@dataclass
class NativeAppServer:
    host: str = "127.0.0.1"
    port: int = 0
    verbose: bool = False

    def __post_init__(self) -> None:
        self.port = self.port or _free_loopback_port(self.host)
        self._thread: threading.Thread | None = None
        self._server: Any | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout: float = 10.0) -> None:
        import uvicorn

        config = uvicorn.Config(
            create_app(),
            host=self.host,
            port=self.port,
            log_level="debug" if self.verbose else "warning",
            access_log=self.verbose,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(target=server.run, name="isaac-native-api", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if server.started:
                return
            if self._thread is not None and not self._thread.is_alive():
                break
            time.sleep(0.05)
        self.stop()
        raise RuntimeError("The native UI server did not start in time.")

    def stop(self) -> None:
        server = self._server
        if server is not None:
            server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        self._server = None


def run_desktop_app(*, verbose: bool = False) -> int:
    """Open I.S.A.A.C. as a native desktop window and block until it closes."""
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(
            "Native desktop dependencies are missing. Run: pip install -e '.[desktop]'"
        ) from exc

    server = NativeAppServer(verbose=verbose)
    server.start()
    try:
        window = webview.create_window(
            "I.S.A.A.C.",
            server.url,
            width=1500,
            height=930,
            min_size=(1080, 680),
            background_color="#09090d",
            text_select=True,
        )
        if window is None:
            raise RuntimeError("The native UI window could not be created.")
        window.events.closed += server.stop
        webview.start(debug=verbose, private_mode=True)
    finally:
        server.stop()
    return 0


def _free_loopback_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


__all__ = ["NativeAppServer", "run_desktop_app"]
