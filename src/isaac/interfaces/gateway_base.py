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

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

AgentRunner = Callable[[str], str]


def _default_agent_runner(text: str) -> str:
    """Default agent runner — delegates to the ISAAC agent loop."""
    from isaac.agents.agent_loop import build_default_agent  # lazy: heavy import

    result = build_default_agent().run(text)
    return str(result)


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
        # Handle audio attachments via STT
        if attachments:
            from pathlib import Path

            from isaac.multimodal.voice.stt import STTManager

            stt = STTManager()
            for attach in attachments:
                # Check if attachment is an audio file (simplistic check for path/extension)
                path_attr = getattr(attach, "file_path", None) or getattr(attach, "path", None)
                if path_attr and any(
                    str(path_attr).endswith(ext) for ext in [".wav", ".mp3", ".ogg", ".m4a"]
                ):
                    try:
                        transcribed_text = stt.transcribe(Path(path_attr))
                        text = f"{text}\n[Audio]: {transcribed_text}" if text else transcribed_text
                    except Exception as e:
                        logger.error("STT transcription failed for %s: %s", path_attr, e)

        reply = self._route_to_agent(channel_id, user_id, text, attachments)
        await self.send_message(channel_id, reply)
        return reply
