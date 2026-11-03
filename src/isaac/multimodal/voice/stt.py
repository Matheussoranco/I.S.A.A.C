"""Speech-to-Text (STT) management.

Priority chain: local faster-whisper (when installed) followed by cloud
providers that actually have a configured credential.
"""

import logging
from pathlib import Path
from typing import Protocol

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

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "",
    ):
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper not installed. Install with `pip install faster-whisper`."
            )

        logger.info("Initializing local Whisper model (%s) on %s...", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language or None

    def transcribe(self, audio_path: Path) -> str:
        segments, _ = self.model.transcribe(str(audio_path), language=self.language, beam_size=5)
        return " ".join([segment.text for segment in segments]).strip()


class OpenAISTTProvider:
    """Cloud STT through the OpenAI audio-transcriptions endpoint."""

    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self._api_key = api_key
        self._model = model

    def transcribe(self, audio_path: Path) -> str:
        from openai import OpenAI

        with OpenAI(api_key=self._api_key) as client, audio_path.open("rb") as handle:
            result = client.audio.transcriptions.create(model=self._model, file=handle)
        return str(getattr(result, "text", "")).strip()


class STTManager:
    """Manages a priority chain of STT providers."""

    def __init__(
        self,
        local_model_size: str = "base",
        cloud_priority: list[str] | None = None,
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "",
    ):
        self.cloud_priority = cloud_priority or ["openai"]
        self.providers: list[STTProvider] = []

        try:
            self.providers.append(
                LocalWhisperProvider(
                    model_size=local_model_size,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                )
            )
        except Exception as exc:
            logger.warning("Could not initialize local STT: %s", exc)

        if "openai" in {name.lower() for name in self.cloud_priority}:
            from isaac.security.credentials import get_credential

            api_key = get_credential("openai")
            if api_key:
                self.providers.append(OpenAISTTProvider(api_key))

    def _load(self) -> None:
        """Compatibility hook; providers are prepared during construction."""
        return None

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe audio using the first available provider in the chain."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        for provider in self.providers:
            try:
                text = provider.transcribe(audio_path)
            except Exception as exc:
                logger.error("STT provider %s failed: %s", provider.__class__.__name__, exc)
                continue
            if text.strip():
                return text
            logger.warning("STT provider %s returned no text.", provider.__class__.__name__)

        raise RuntimeError("All STT providers failed or no providers configured.")


# Aliases to satisfy existing __init__.py imports
SpeechToText = STTManager


_stt_manager: STTManager | None = None


def get_stt() -> STTManager:
    """Singleton-like getter for STT manager, honoring the voice settings."""
    global _stt_manager
    if _stt_manager is None:
        from isaac.config.settings import get_settings

        settings = get_settings()
        if not settings.voice_enabled:
            raise RuntimeError("Voice subsystem is disabled (ISAAC_VOICE_ENABLED=false).")
        _stt_manager = STTManager(
            local_model_size=settings.voice_stt_model,
            device=settings.voice_device,
            compute_type=settings.voice_stt_compute_type,
            language=settings.voice_stt_language,
        )
    return _stt_manager


def is_stt_available() -> bool:
    """Check whether at least one real STT backend is configured."""
    from isaac.config.settings import get_settings

    if not get_settings().voice_enabled:
        return False
    try:
        return len(get_stt().providers) > 0
    except Exception:
        return False
