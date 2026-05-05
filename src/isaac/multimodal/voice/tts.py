"""Text-to-Speech — local Piper / Coqui / pyttsx3.

Backends (auto-selected, in order):

    1. ``piper-tts``  — recommended (fast neural TTS, runs on CPU)
    2. ``TTS`` (Coqui) — higher quality, slower
    3. ``pyttsx3``    — system TTS (Windows SAPI / macOS NSSpeech / espeak)

The chosen backend is loaded lazily and reused.  ``synthesize()`` returns
a path to a temporary WAV file or, when ``return_audio=True``, a numpy
array suitable for direct playback.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Lazy TTS engine, backend chosen automatically.

    Parameters
    ----------
    voice:
        Voice / model name.  Meaning depends on backend:
        * piper: model file path or built-in name (``en_US-lessac-medium``).
        * coqui: model name (``tts_models/en/ljspeech/tacotron2-DDC``).
        * pyttsx3: voice id (system-specific).
    rate:
        Speech rate (words/min for pyttsx3; ignored by neural backends).
    sample_rate:
        Output sample rate hint.
    """

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        rate: int = 175,
        sample_rate: int = 22050,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.sample_rate = sample_rate
        self._engine: Any = None
        self._engine_kind: str = ""

    # -- Lazy load ----------------------------------------------------------

    def _load(self) -> None:
        if self._engine is not None:
            return

        # ── Piper (fast neural) ────────────────────────────────────────────
        try:
            from piper import PiperVoice  # type: ignore[import-not-found]

            voice_path = self.voice
            if not Path(voice_path).is_file():
                voice_path = self._resolve_piper_voice(self.voice)
            if voice_path and Path(voice_path).is_file():
                self._engine = PiperVoice.load(voice_path)
                self._engine_kind = "piper"
                logger.info("TTS: loaded piper voice '%s'.", voice_path)
                return
            logger.debug("Piper voice '%s' not found locally — skipping.", self.voice)
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover
            logger.warning("TTS: piper load failed: %s", exc)

        # ── Coqui TTS (higher quality, heavier) ────────────────────────────
        try:
            from TTS.api import TTS  # type: ignore[import-not-found]

            model_name = self.voice if "/" in self.voice else "tts_models/en/ljspeech/tacotron2-DDC"
            self._engine = TTS(model_name)
            self._engine_kind = "coqui"
            logger.info("TTS: loaded Coqui model '%s'.", model_name)
            return
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover
            logger.warning("TTS: Coqui load failed: %s", exc)

        # ── pyttsx3 (system TTS) ───────────────────────────────────────────
        try:
            import pyttsx3  # type: ignore[import-not-found]

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            self._engine = engine
            self._engine_kind = "pyttsx3"
            logger.info("TTS: using pyttsx3 system voice.")
            return
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover
            logger.warning("TTS: pyttsx3 init failed: %s", exc)

        raise RuntimeError(
            "No TTS backend available. Install one of: piper-tts, TTS (coqui), pyttsx3. "
            "(`pip install piper-tts pyttsx3` is the cheapest combo.)"
        )

    @staticmethod
    def _resolve_piper_voice(name: str) -> str:
        """Find a piper .onnx voice file by name in common locations."""
        candidates: list[Path] = []
        env_path = os.environ.get("PIPER_VOICE_DIR")
        if env_path:
            candidates.append(Path(env_path) / f"{name}.onnx")
        candidates.append(Path.home() / ".isaac" / "voices" / f"{name}.onnx")
        candidates.append(Path("./voices") / f"{name}.onnx")
        for c in candidates:
            if c.is_file():
                return str(c)
        return ""

    # -- Public API ---------------------------------------------------------

    def synthesize(
        self,
        text: str,
        out_path: str | os.PathLike[str] | None = None,
    ) -> str:
        """Synthesise *text* to a WAV file and return the path."""
        self._load()
        if not text.strip():
            raise ValueError("TextToSpeech.synthesize: empty text.")

        if out_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            out_path = tmp.name
        out_path = str(out_path)

        if self._engine_kind == "piper":
            import wave

            with wave.open(out_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self._engine.config.sample_rate)
                self._engine.synthesize(text, wav)
            return out_path

        if self._engine_kind == "coqui":
            self._engine.tts_to_file(text=text, file_path=out_path)
            return out_path

        if self._engine_kind == "pyttsx3":
            self._engine.save_to_file(text, out_path)
            self._engine.runAndWait()
            return out_path

        raise RuntimeError("TTS engine not initialised.")

    def speak(self, text: str) -> None:
        """Synthesise *text* and play it through the default speaker."""
        path = self.synthesize(text)
        try:
            from isaac.multimodal.voice.audio_io import play_wav
            play_wav(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def is_loaded(self) -> bool:
        return self._engine is not None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tts: TextToSpeech | None = None


def get_tts() -> TextToSpeech:
    """Return the singleton TTS engine, configured from settings."""
    global _tts  # noqa: PLW0603
    if _tts is None:
        try:
            from isaac.config.settings import settings
            _tts = TextToSpeech(
                voice=settings.voice_tts_voice,
                rate=settings.voice_tts_rate,
                sample_rate=settings.voice_tts_sample_rate,
            )
        except Exception:
            _tts = TextToSpeech()
    return _tts


def is_tts_available() -> bool:
    """Return True if any TTS backend can be imported."""
    for mod in ("piper", "TTS", "pyttsx3"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def reset_tts() -> None:
    """Reset the singleton (used in tests)."""
    global _tts  # noqa: PLW0603
    _tts = None
