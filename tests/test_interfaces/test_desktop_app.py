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


def test_packaged_entry_pins_the_profile_before_starting_server(tmp_path, monkeypatch) -> None:
    import subprocess
    import sys
    from textwrap import dedent

    monkeypatch.setenv("ISAAC_HOME", str(tmp_path))
    monkeypatch.delenv("ISAAC_PROFILE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    probe = dedent("""\
        import sys
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from isaac.config import profiles
        from isaac.config.settings import get_settings
        from isaac.security.credentials import get_credential, set_credential

        keys = {}
        sys.modules['keyring'] = SimpleNamespace(
            get_password=lambda service, provider: keys.get((service, provider)),
            set_password=lambda service, provider, key: keys.update({(service, provider): key}),
        )
        for name in ('coding', 'research'):
            profiles.save_profile_overrides(name, {'agent_name': name})
            profiles.switch_profile(name)
            set_credential('openai', f'{name}-key')

        from isaac.interfaces import desktop_app, desktop_entry

        def start_server(self):
            assert get_settings().agent_name == 'research'
            profiles.switch_profile('coding')
            assert profiles.get_active_profile() == 'research'
            assert get_credential('openai') == 'research-key'

        desktop_app.NativeAppServer.start = start_server
        sys.modules['webview'] = SimpleNamespace(create_window=MagicMock(), start=MagicMock())
        assert desktop_entry.main() == 0
        """)
    child = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
    )
    assert child.returncode == 0, child.stderr
