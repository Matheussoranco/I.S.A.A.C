"""Discord Gateway — bidirectional operator interface via discord.py (optional dep).

Install with ``pip install isaac[gateway-discord]``.

Environment
-----------
DISCORD_BOT_TOKEN   — required; bot token from the Discord developer portal.
DISCORD_CHANNEL_IDS — optional comma-separated allowlist of channel IDs the
                      bot responds in (in addition to DMs and @mentions).

Behaviour
---------
* Bots (including itself) are ignored.
* Always responds in DMs.
* In guild channels, responds when mentioned, or when the channel is in the
  ``DISCORD_CHANNEL_IDS`` allowlist.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from isaac.interfaces.gateway_base import AgentRunner, Gateway

logger = logging.getLogger(__name__)


@Gateway.register("discord")
class DiscordGateway(Gateway):
    """Discord gateway built on discord.py (lazily imported)."""

    def __init__(
        self,
        agent_runner: AgentRunner | None = None,
        *,
        token: str | None = None,
        channel_ids: set[str] | None = None,
    ) -> None:
        super().__init__(agent_runner)
        self._token = token if token is not None else os.environ.get("DISCORD_BOT_TOKEN", "")
        if not self._token:
            raise RuntimeError(
                "DISCORD_BOT_TOKEN is not set. Create a bot in the Discord "
                "developer portal, copy its token, and export it as "
                "DISCORD_BOT_TOKEN before enabling the discord gateway."
            )
        if channel_ids is not None:
            self._channel_ids = {str(c) for c in channel_ids}
        else:
            raw = os.environ.get("DISCORD_CHANNEL_IDS", "")
            self._channel_ids = {c.strip() for c in raw.split(",") if c.strip()}
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return "discord"

    async def send_message(self, channel_id: str, text: str) -> None:
        """Send a message to a Discord channel (discord.py is lazy-loaded)."""
        import discord  # noqa: F401 — ensures the optional dep is present

        if self._client is None:
            raise RuntimeError("Discord gateway is not running.")
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        # Discord messages are capped at 2000 chars.
        for chunk in [text[i : i + 1999] for i in range(0, len(text), 1999)] or [""]:
            await channel.send(chunk)

    def _should_respond(self, message: Any) -> bool:
        """True when the gateway should reply to this message."""
        if message.author.bot:
            return False
        # DMs — message.guild is None
        if message.guild is None:
            return True
        if (
            self._client is not None
            and self._client.user is not None
            and (self._client.user in message.mentions)
        ):
            return True
        return str(message.channel.id) in self._channel_ids

    async def start(self) -> None:
        """Connect to Discord and run the event loop until stopped."""
        try:
            import discord
        except ImportError as exc:
            raise RuntimeError(
                "discord.py is not installed. Install it with "
                "'pip install isaac[gateway-discord]' to use the discord gateway."
            ) from exc

        intents = discord.Intents.default()
        intents.message_content = True

        client = discord.Client(intents=intents)
        self._client = client
        self._running = True

        @client.event
        async def on_message(message: Any) -> None:
            if not self._should_respond(message):
                return
            text = message.content or ""
            # Strip the bot mention from guild messages.
            if client.user is not None:
                for mention in (f"<@{client.user.id}>", f"<@!{client.user.id}>"):
                    text = text.replace(mention, "").strip()
            if not text:
                return
            attachments = [a.url for a in message.attachments] or None
            try:
                await self._handle_incoming(
                    str(message.channel.id),
                    str(message.author.id),
                    text,
                    attachments,
                )
            except Exception as exc:  # pragma: no cover — network dependent
                logger.exception("Discord on_message handler failed: %s", exc)

        logger.info("Discord gateway connecting...")
        try:
            await client.start(self._token)
        finally:
            self._running = False
            self._client = None

    async def stop(self) -> None:
        """Gracefully disconnect from Discord."""
        self._running = False
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:
                logger.debug("Error closing Discord client: %s", exc)
            finally:
                self._client = None
