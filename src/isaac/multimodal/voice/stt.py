"""Speech-to-Text — local Whisper backend.

Backends (auto-selected, in order):

    1. ``faster-whisper``  — recommended (CTranslate2, fastest CPU/GPU)
    2. ``openai-whisper``  — reference implementation
    3. Cloud fallback (OpenAI ``whisper-1``)  — only if API key + opt-in

All backends accept either:

* a path to a WAV/MP3/FLAC file, or
* a numpy ``float32`` array sampled at 16 kHz (mono).

The chosen backend is loaded lazily on first use; weights stay in RAM
for the lifetime of the process.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "base"
"""Whisper model size — ``tiny``, ``base``, ``small``, ``medium``, ``large-v3``."""
DEFAULT_LANGUAGE = ""  # auto-detect


class SpeechToText:
    """Lazy STT engine wrapping faster-whisper or openai-whisper.

    Parameters
    ----------
    model_name:
        Whisper model size identifier.
    language:
        ISO 639-1 code (``en``, ``pt``, ...) or empty for auto-detect.
    device:
        ``"cpu"``, ``"cuda"``, or ``"auto"``.
    compute_type:
        For faster-whisper: ``"int8"``, ``"float16"``, ``"float32"``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._engine: Any = None
        self._engine_kind: str = ""

    # -- Lazy load ----------------------------------------------------------

    def _load(self) -> None:
        if self._engine is not None:
            return
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

            device = self.device if self.device != "auto" else "cpu"
            self._engine = WhisperModel(
                self.model_name, device=device, compute_type=self.compute_type
            )
            self._engine_kind = "faster_whisper"
            logger.info("STT: loaded faster-whisper '%s' on %s.", self.model_name, device)
            return
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover
            logger.warning("STT: faster-whisper load failed: %s", exc)

        # Fall back to openai-whisper
        try:
            import whisper  # type: ignore[import-not-found]

            self._engine = whisper.load_model(self.model_name)
            self._engine_kind = "openai_whisper"
            logger.info("STT: loaded openai-whisper '%s'.", self.model_name)
            return
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover
            logger.warning("STT: openai-whisper load failed: %s", exc)

        raise RuntimeError(
            "No STT backend available. Install one of: faster-whisper, openai-whisper. "
            "(`pip install faster-whisper` is recommended.)"
        )

    # -- Public API ---------------------------------------------------------

    def transcribe(
        self,
        audio: str | os.PathLike[str] | "np.ndarray",
        sample_rate: int = 16000,
    ) -> str:
        """Return transcribed text from a path or 16-kHz mono float32 array."""
        self._load()
        if isinstance(audio, (str, Path)):
            audio_input: Any = str(audio)
        else:
            audio_input = audio  # numpy array

        if self._engine_kind == "faster_whisper":
            segments, _info = self._engine.transcribe(
                audio_input,
                language=self.language or None,
                beam_size=1,
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        # openai-whisper
        if self._engine_kind == "openai_whisper":
            kwargs: dict[str, Any] = {"fp16": False}
            if self.language:
                kwargs["language"] = self.language
            result = self._engine.transcribe(audio_input, **kwargs)
            return str(result.get("text", "")).strip()

        raise RuntimeError("STT engine not initialised.")

    def is_loaded(self) -> bool:
        return self._engine is not None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_stt: SpeechToText | None = None


def get_stt() -> SpeechToText:
    """Return the singleton STT engine, configured from settings."""
    global _stt  # noqa: PLW0603
    if _stt is None:
        try:
            from isaac.config.settings import settings
            _stt = SpeechToText(
                model_name=settings.voice_stt_model,
                language=settings.voice_stt_language,
                device=settings.voice_device,
                compute_type=settings.voice_stt_compute_type,
            )
        except Exception:
            _stt = SpeechToText()
    return _stt


def is_stt_available() -> bool:
    """Return True if any STT backend can be imported."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def reset_stt() -> None:
    """Reset the singleton (used in tests)."""
    global _stt  # noqa: PLW0603
    _stt = None
