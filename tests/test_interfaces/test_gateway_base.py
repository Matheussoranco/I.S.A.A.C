"""Tests for the generic Gateway base, registry, telegram refactor, and discord adapter."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from isaac.interfaces.gateway_base import Gateway
from isaac.interfaces.telegram_gateway import TelegramGateway, start_bot


class FakeGateway(Gateway):
    """Concrete fake gateway for exercising base-class behaviour."""

    def __init__(self, agent_runner=None) -> None:
        super().__init__(agent_runner)
        self.sent: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "fake"

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send_message(self, channel_id: str, text: str) -> None:
        self.sent.append((channel_id, text))


def test_registry_decorator_registers_subclass() -> None:
    @Gateway.register("test-fake")
    class AnotherFake(FakeGateway):
        pass

    assert Gateway.GATEWAY_REGISTRY["test-fake"] is AnotherFake


def test_telegram_registered() -> None:
    assert Gateway.GATEWAY_REGISTRY["telegram"] is TelegramGateway


def test_telegram_subclass_exposes_start_bot() -> None:
    # start_bot remains as a module-level async function (backwards compat).
    assert callable(start_bot)
    gw = TelegramGateway()
    assert isinstance(gw, Gateway)
    assert gw.name == "telegram"
    assert gw.running is False


@pytest.mark.asyncio
async def test_handle_incoming_dispatches_to_agent_runner() -> None:
    calls: list[str] = []

    def fake_agent(text: str) -> str:
        calls.append(text)
        return f"echo:{text}"

    gw = FakeGateway(agent_runner=fake_agent)
    reply = await gw._handle_incoming("chan-1", "user-1", "hello")

    assert calls == ["hello"]
    assert reply == "echo:hello"
    assert gw.sent == [("chan-1", "echo:hello")]


@pytest.mark.asyncio
async def test_handle_incoming_agent_error_returns_friendly_message() -> None:
    def boom(text: str) -> str:
        raise ValueError("nope")

    gw = FakeGateway(agent_runner=boom)
    reply = await gw._handle_incoming("chan-2", "user-2", "hi")

    assert "nope" in reply
    assert gw.sent and "nope" in gw.sent[0][1]


def test_running_flag_tracks_lifecycle() -> None:
    gw = FakeGateway()
    assert gw.running is False


# ── Discord adapter ──────────────────────────────────────────────

def test_discord_construct_fails_cleanly_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    from isaac.interfaces.discord_gateway import DiscordGateway

    with pytest.raises(RuntimeError, match="DISCORD_BOT_TOKEN"):
        DiscordGateway()


def test_discord_module_imports_without_discord_py() -> None:
    # The module must import cleanly even when discord.py is absent.
    saved = sys.modules.pop("discord", None)
    saved_ext = {k: v for k, v in sys.modules.items() if k.startswith("discord")}
    for k in saved_ext:
        sys.modules.pop(k)
    try:
        import importlib

        import isaac.interfaces.discord_gateway as dg

        importlib.reload(dg)
        assert issubclass(dg.DiscordGateway, Gateway)
        # Registered in the shared registry.
        assert Gateway.GATEWAY_REGISTRY["discord"] is dg.DiscordGateway
    finally:
        if saved is not None:
            sys.modules["discord"] = saved
        sys.modules.update(saved_ext)


def test_discord_lazy_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """start() raises a helpful error when discord.py is missing."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-token")

    # Simulate discord.py absence.
    monkeypatch.setitem(sys.modules, "discord", None)

    from isaac.interfaces.discord_gateway import DiscordGateway

    gw = DiscordGateway()

    async def run_start() -> None:
        await gw.start()

    with pytest.raises(RuntimeError, match=r"discord\.py is not installed"):
        import asyncio

        asyncio.run(run_start())


def test_discord_should_respond(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-token")
    from isaac.interfaces.discord_gateway import DiscordGateway

    gw = DiscordGateway(channel_ids={"42"})

    bot_user = types.SimpleNamespace(id=999)

    def make_msg(*, author_bot: bool, dm: bool, mentioned: bool, channel_id: int) -> Any:
        author = types.SimpleNamespace(bot=author_bot, id=7)
        guild = None if dm else object()
        client_user_ref: list[Any] = [999]
        _ = client_user_ref
        return types.SimpleNamespace(
            author=author,
            guild=guild,
            mentions=[bot_user] if mentioned else [],
            channel=types.SimpleNamespace(id=channel_id),
        )

    gw._client = types.SimpleNamespace(user=bot_user)

    def check(*, author_bot: bool = False, dm: bool = False, mentioned: bool = False,
              channel_id: int = 1) -> bool:
        return gw._should_respond(
            make_msg(author_bot=author_bot, dm=dm, mentioned=mentioned, channel_id=channel_id)
        )

    # Ignore bots
    assert check(author_bot=True, dm=True) is False
    # DMs
    assert check(dm=True) is True
    # Mention in guild channel
    assert check(mentioned=True) is True
    # Allowlisted channel
    assert check(channel_id=42) is True
    # Regular channel without mention
    assert check(channel_id=5) is False


# ── Dispatcher ───────────────────────────────────────────────────

def test_dispatcher_load_enabled_gateways_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAAC_GATEWAYS", raising=False)
    from isaac.interfaces.gateway_dispatcher import load_enabled_gateways

    gateways = load_enabled_gateways(agent_runner=lambda t: t)
    assert [g.name for g in gateways] == ["telegram"]


def test_dispatcher_unknown_gateway_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAAC_GATEWAYS", "nosuchgateway")
    from isaac.interfaces.gateway_dispatcher import load_enabled_gateways

    with pytest.raises(ValueError, match="nosuchgateway"):
        load_enabled_gateways()


def test_dispatcher_discord_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAAC_GATEWAYS", "discord")
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    from isaac.interfaces.gateway_dispatcher import load_enabled_gateways

    with pytest.raises(RuntimeError, match="DISCORD_BOT_TOKEN"):
        load_enabled_gateways()
