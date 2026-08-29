"""Native shell server lifecycle tests."""

from __future__ import annotations

import httpx

from isaac.interfaces import desktop_entry
from isaac.interfaces.desktop_app import NativeAppServer


def test_native_server_uses_loopback_and_stops_cleanly() -> None:
    server = NativeAppServer()
    server.start()
    try:
        response = httpx.get(f"{server.url}/api/status", timeout=3.0)
        assert response.status_code == 200
        assert response.json()["name"]
        assert server.url.startswith("http://127.0.0.1:")
    finally:
        server.stop()

    assert server._thread is None


def test_packaged_entry_opens_desktop_directly(monkeypatch) -> None:
    launched: list[bool] = []
    monkeypatch.setattr(
        desktop_entry,
        "run_desktop_app",
        lambda: launched.append(True) or 0,
    )

    assert desktop_entry.main() == 0
    assert launched == [True]
