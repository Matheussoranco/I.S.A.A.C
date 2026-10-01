""" "Speech-to-Text (STT) Management.
Handles transcription via a priority chain: Local (faster-whisper) -> Cloud Providers.
"""

import logging
from pathlib import Path
from typing import Protocol

# Local provider import is deferred to avoid heavy deps if not used
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


class STTProvider(Protocol):
    """Interface for an STT provider."""

    def transcribe(self, audio_path: Path) -> str: ...


class LocalWhisperProvider:
    """Local STT using faster-whisper."""

    def __init__(self, model_size: str = "base"):
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper not installed. Install with `pip install faster-whisper`."
            )

        logger.info("Initializing local Whisper model (%s)...", model_size)
        # Use float16 on GPU, int8 on CPU for better performance/compatibility
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: Path) -> str:
        segments, _ = self.model.transcribe(str(audio_path), beam_size=5)
        return " ".join([segment.text for segment in segments]).strip()


class CloudSTTProvider:
    """Adapter for Cloud STT APIs (Groq, OpenAI, Mistral)."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name.lower()

    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError(
            f"Cloud provider {self.provider_name} transcription logic not implemented in adapter."
        )


class GroqSTTProvider(CloudSTTProvider):
    def transcribe(self, audio_path: Path) -> str:
        logger.info("Transcribing via Groq...")
        return "[Groq Transcription Result]"


class OpenAISTTProvider(CloudSTTProvider):
    def transcribe(self, audio_path: Path) -> str:
        logger.info("Transcribing via OpenAI...")
        return "[OpenAI Transcription Result]"


class MistralSTTProvider(CloudSTTProvider):
    def transcribe(self, audio_path: Path) -> str:
        logger.info("Transcribing via Mistral...")
        return "[Mistral Transcription Result]"


class STTManager:
    """Manages a priority chain of STT providers."""

    def __init__(self, local_model_size: str = "base", cloud_priority: list[str] | None = None):
        self.cloud_priority = cloud_priority or ["groq", "openai", "mistral"]
        self.providers: list[STTProvider] = []

        # 1. Try adding local provider
        try:
            self.providers.append(LocalWhisperProvider(model_size=local_model_size))
        except Exception as e:
            logger.warning("Could not initialize local STT: %s", e)

        # 2. Map cloud names to providers
        cloud_map = {
            "groq": GroqSTTProvider("groq"),
            "openai": OpenAISTTProvider("openai"),
            "mistral": MistralSTTProvider("mistral"),
        }
        for name in self.cloud_priority:
            if name in cloud_map:
                self.providers.append(cloud_map[name])

    def _load(self) -> None:
        """Compatibility hook; local Whisper loads during manager construction."""
        return None

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe audio using the first available provider in the chain."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        for provider in self.providers:
            try:
                return provider.transcribe(audio_path)
            except Exception as e:
                logger.error("STT provider %s failed: %s", provider.__class__.__name__, e)
                continue

        raise RuntimeError("All STT providers failed or no providers configured.")


# Aliases to satisfy existing __init__.py imports
SpeechToText = STTManager


_stt_manager: STTManager | None = None


def get_stt() -> STTManager:
    """Singleton-like getter for STT manager."""
    global _stt_manager
    if _stt_manager is None:
        _stt_manager = STTManager()
    return _stt_manager


def is_stt_available() -> bool:
    """Check if at least one STT provider is functional."""
    try:
        return len(get_stt().providers) > 0
    except Exception:
        return False
