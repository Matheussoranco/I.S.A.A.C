"""Exercise the installed doctor CLI against a loopback Ollama fixture."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sysconfig
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import TemporaryDirectory
from threading import Thread

from isaac.llm.providers.ollama import DEFAULT_MODEL


class TagsHandler(BaseHTTPRequestHandler):
    models: tuple[str, ...] = (f"{DEFAULT_MODEL}:latest",)

    def do_GET(self) -> None:
        if self.path != "/api/tags":
            self.send_error(404)
            return
        body = json.dumps({"models": [{"name": name} for name in self.models]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    """Assert both doctor outcomes from an isolated installed CLI."""
    exe = shutil.which("isaac", path=sysconfig.get_path("scripts"))
    if not exe:
        raise SystemExit("Install I.S.A.A.C. in this Python environment before running this test.")

    with (
        TemporaryDirectory(prefix="isaac-doctor-") as home,
        HTTPServer(("127.0.0.1", 0), TagsHandler) as server,
    ):
        env = {
            name: os.environ[name]
            for name in ("PATH", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH")
            if name in os.environ
        }
        endpoint = f"127.0.0.1:{server.server_port}"
        env.update(
            HOME=home,
            USERPROFILE=home,
            ISAAC_HOME=home,
            ISAAC_PROFILE="doctor-smoke",
            ISAAC_OLLAMA_BASE_URL=f"http://{endpoint}",
            DOCKER_HOST=f"tcp://{endpoint}",
            PYTHONIOENCODING="utf-8",
        )
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            for models, expected_code, diagnostic in (
                ((f"{DEFAULT_MODEL}:latest",), 0, "All core checks passed."),
                ((), 1, f"ollama pull {DEFAULT_MODEL}"),
            ):
                TagsHandler.models = models
                result = subprocess.run(
                    [exe, "doctor"],
                    cwd=home,
                    env=env,
                    capture_output=True,
                    encoding="utf-8",
                    timeout=30,
                    check=False,
                )
                if result.returncode != expected_code or diagnostic not in result.stdout:
                    raise SystemExit(
                        f"Expected exit {expected_code} and {diagnostic!r}; "
                        f"got exit {result.returncode}.\n{result.stdout}{result.stderr}"
                    )
                print(f"PASS: isaac doctor exits {expected_code} and reports {diagnostic!r}")
        finally:
            server.shutdown()
            server_thread.join()


if __name__ == "__main__":
    main()
