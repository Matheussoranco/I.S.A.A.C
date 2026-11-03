"""Generic Gateway base — pluggable bidirectional operator interfaces.

A Gateway is any chat-style front-end (Telegram, Discord, …) that receives
messages on channels, routes them through an agent, and replies.

Gateways are registered by name via ``@Gateway.register("name")`` so the
dispatcher (``gateway_dispatcher``) can instantiate them from the
``ISAAC_GATEWAYS`` environment variable without hard-coding imports.

The agent is injected as a plain callable ``agent_runner(text) -> str`` so
gateways are testable without a real LLM.  When none is provided, a default
runner lazily delegates to the ISAAC agent loop.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

AgentRunner = Callable[[str], str]

_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac"})


def _default_agent_runner(text: str) -> str:
    """Default agent runner — delegates to the ISAAC agent loop."""
    from isaac.agents.agent_loop import build_default_agent  # lazy: heavy import

    return build_default_agent().run(text).output


class Gateway(ABC):
    """Abstract base class for bidirectional chat gateways."""

    GATEWAY_REGISTRY: ClassVar[dict[str, type[Gateway]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Gateway]], type[Gateway]]:
        """Class decorator: register a Gateway subclass under ``name``."""

        def decorator(subclass: type[Gateway]) -> type[Gateway]:
            cls.GATEWAY_REGISTRY[name] = subclass
            return subclass

        return decorator

    def __init__(self, agent_runner: AgentRunner | None = None) -> None:
        self._agent_runner: AgentRunner = agent_runner or _default_agent_runner
        self._running: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human/CLI name of this gateway (matches its registry key)."""

    @property
    def running(self) -> bool:
        """True while the gateway's event loop is active."""
        return self._running

    @abstractmethod
    async def start(self) -> None:
        """Start the gateway's event loop (blocks until stopped)."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the gateway."""

    @abstractmethod
    async def send_message(self, channel_id: str, text: str) -> None:
        """Send ``text`` to ``channel_id``."""

    def _route_to_agent(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        attachments: list[Any] | None = None,
    ) -> str:
        """Run the injected agent on an incoming message and return a reply."""
        try:
            return self._agent_runner(text)
        except Exception as exc:
            logger.exception("%s gateway: agent error", self.name)
            return f"⚠️ Agent error: {exc}"

    async def _handle_incoming(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        attachments: list[Any] | None = None,
    ) -> str:
        """Route an incoming message to the agent and send the reply.

        Returns the reply text (useful for tests).
        """
        if attachments:
            text = await asyncio.to_thread(self._transcribe_audio, text, attachments)

        reply = await asyncio.to_thread(
            self._route_to_agent, channel_id, user_id, text, attachments
        )
        await self.send_message(channel_id, reply)
        return reply

    @staticmethod
    def _attachment_path(attachment: Any) -> Path | None:
        candidate = None
        if isinstance(attachment, dict):
            candidate = attachment.get("file_path") or attachment.get("path")
        elif isinstance(attachment, str):
            candidate = attachment
        else:
            candidate = getattr(attachment, "file_path", None) or getattr(attachment, "path", None)
        if not candidate:
            return None
        path = Path(str(candidate)).expanduser()
        return path if path.is_file() else None

    def _transcribe_audio(self, text: str, attachments: list[Any]) -> str:
        """Transcribe local audio attachments and append the text."""
        paths = [
            path
            for path in (self._attachment_path(item) for item in attachments)
            if path is not None and path.suffix.lower() in _AUDIO_SUFFIXES
        ]
        if not paths:
            return text

        from isaac.multimodal.voice.stt import STTManager

        for path in paths:
            try:
                transcribed = STTManager().transcribe(path)
            except Exception as exc:
                logger.error("STT transcription failed for %s: %s", path, exc)
                continue
            text = f"{text}\n[Audio]: {transcribed}" if text else transcribed
        return text
