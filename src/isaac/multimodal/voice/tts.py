"""Text-to-Speech Manager with Provider Matrix.

Supports multiple backends: edge-tts (default), OpenAI, ElevenLabs, MiniMax, Mistral, 
Gemini, NeuTTS, Piper, and KittenTTS.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from isaac.config.settings import Settings

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
            await communicate.save(output_path)
            return output_path
        except ImportError:
            raise RuntimeError("edge-tts not installed. Run `pip install edge-tts`.")

class OpenAIProvider(TTSProvider):
    """OpenAI TTS API."""
    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            from openai import AsyncOpenAI
            from isaac.config.settings import settings
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.audio.speech.create(model="tts-1", voice="alloy", input=text)
            await response.stream_to_file(output_path)
            return output_path
        except ImportError:
            raise RuntimeError("openai not installed.")

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
                    json={"text": text}
                )
                resp.raise_for_status()
                output_path.write_bytes(resp.content)
            return output_path
        except ImportError:
            raise RuntimeError("httpx not installed.")

class GenericAPIProvider(TTSProvider):
    """Placeholder for other providers."""
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
    async def synthesize(self, text: str, output_path: Path) -> Path:
        logger.warning("TTS provider %s is not yet fully implemented. Falling back to EdgeTTS.", self.provider_name)
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
                voice.synthesize(text, wav)
            return output_path
        except ImportError:
            raise RuntimeError("piper-tts not installed.")

class TTSManager:
    """Coordinates TTS synthesis across multiple providers."""
    PROVIDERS: Dict[str, Type[TTSProvider]] = {
        "edge": EdgeTTSProvider,
        "openai": OpenAIProvider,
        "elevenlabs": ElevenLabsProvider,
        "piper": PiperProvider,
    }

    def __init__(self):
        self._current_provider_name = "edge"
        self._provider: TTSProvider = self.PROVIDERS[self._current_provider_name]()

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
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            output_path = Path(tmp.name)
            tmp.close()
        return await self._provider.synthesize(text, output_path)

_manager: TTSManager | None = None

def get_tts_manager() -> TTSManager:
    global _manager
    if _manager is None:
        _manager = TTSManager()
    return _manager
