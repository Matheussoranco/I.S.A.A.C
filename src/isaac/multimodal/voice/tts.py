"""Text-to-Speech Manager with Provider Matrix.

Supports multiple backends: edge-tts (default), OpenAI, ElevenLabs, MiniMax, Mistral,
Gemini, NeuTTS, Piper, and KittenTTS.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """Base adapter for TTS providers."""

    @abstractmethod
    async def synthesize(self, text: str, output_path: Path) -> Path:
        pass


class EdgeTTSProvider(TTSProvider):
    """Free, streaming TTS using edge-tts."""

    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
            await cast(Any, communicate.save(output_path))
            return output_path
        except ImportError as err:
            raise RuntimeError("edge-tts not installed. Run `pip install edge-tts`.") from err


class OpenAIProvider(TTSProvider):
    """OpenAI TTS API."""

    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            from openai import AsyncOpenAI

            from isaac.config.settings import settings

            client = AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.audio.speech.create(model="tts-1", voice="alloy", input=text)
            response.stream_to_file(output_path)
            return output_path
        except ImportError as err:
            raise RuntimeError("openai not installed.") from err


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS API."""

    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            import httpx

            from isaac.config.settings import settings

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.elevenlabs.io/v1/text-to-speech/21k0xGdgS7vWzZpQ95Wp",
                    headers={"xi-api-key": settings.elevenlabs_api_key},
                    json={"text": text},
                )
                resp.raise_for_status()
                output_path.write_bytes(resp.content)
            return output_path
        except ImportError as err:
            raise RuntimeError("httpx not installed.") from err


class GenericAPIProvider(TTSProvider):
    """Placeholder for other providers."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    async def synthesize(self, text: str, output_path: Path) -> Path:
        logger.warning(
            "TTS provider %s is not yet fully implemented. Falling back to EdgeTTS.",
            self.provider_name,
        )
        return await EdgeTTSProvider().synthesize(text, output_path)


class PiperProvider(TTSProvider):
    """Local Piper TTS."""

    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            from piper import PiperVoice

            from isaac.config.settings import settings

            voice = PiperVoice.load(settings.voice_tts_voice)
            import wave

            with wave.open(str(output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(voice.config.sample_rate)
                voice.synthesize_wav(text, wav)
            return output_path
        except ImportError as err:
            raise RuntimeError("piper-tts not installed.") from err


class TTSManager:
    """Coordinates TTS synthesis across multiple providers."""

    PROVIDERS: dict[str, type[TTSProvider]] = {
        "edge": EdgeTTSProvider,
        "openai": OpenAIProvider,
        "elevenlabs": ElevenLabsProvider,
        "piper": PiperProvider,
    }

    def __init__(self):
        self._current_provider_name = "edge"
        self._provider: TTSProvider = self.PROVIDERS[self._current_provider_name]()

    def _load(self) -> None:
        """Compatibility hook; providers initialize on first synthesis."""
        return None

    def set_provider(self, name: str):
        if name not in self.PROVIDERS:
            if name in ["minimax", "mistral", "gemini", "neutts", "kittentts"]:
                self._provider = GenericAPIProvider(name)
            else:
                raise ValueError(f"Unsupported TTS provider: {name}")
        else:
            self._provider = self.PROVIDERS[name]()
        self._current_provider_name = name
        logger.info("TTS provider switched to %s", name)

    async def synthesize(self, text: str, output_path: Path | None = None) -> Path:
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_path = Path(tmp.name)
        return await self._provider.synthesize(text, output_path)

    async def speak(self, text: str) -> Path:
        """Synthesize speech and return the generated audio path."""
        return await self.synthesize(text)


_manager: TTSManager | None = None


def get_tts_manager() -> TTSManager:
    global _manager
    if _manager is None:
        _manager = TTSManager()
    return _manager


# Compatibility API used by REPL and web integrations.
class TextToSpeech:
    def __init__(self) -> None:
        self._manager = TTSManager()

    def synthesize(self, text: str, out_path: Path | None = None) -> Path:
        """Synchronous compatibility wrapper for legacy REPL integrations."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._manager.synthesize(text, output_path=out_path))
        raise RuntimeError("Use TTSManager.synthesize from async code.")


def get_tts() -> TTSManager:
    return get_tts_manager()


def is_tts_available() -> bool:
    return True
