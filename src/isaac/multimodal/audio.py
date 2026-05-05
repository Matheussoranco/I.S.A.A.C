"""Multimodal Audio — local-first Speech-to-Text and Text-to-Speech.

STT: faster-whisper (local, no API key needed) → OpenAI Whisper API fallback
TTS: pyttsx3 (local, offline) → kokoro (local neural) → OpenAI TTS fallback

Both paths are lazy-imported so the module loads even without optional deps.

Usage
-----
    from isaac.multimodal.audio import transcribe, speak

    text = transcribe("recording.wav")          # STT
    speak("Hello, I am I.S.A.A.C.")            # TTS
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

WhisperModel = Literal["tiny", "base", "small", "medium", "large-v3"]

# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str | Path,
    model: WhisperModel = "base",
    language: str | None = None,
) -> str:
    """Transcribe audio to text.

    Tries faster-whisper (local) first, then falls back to OpenAI Whisper API.

    Parameters
    ----------
    audio_path:
        Path to audio file (.wav, .mp3, .m4a, .ogg, .flac).
    model:
        Whisper model size. Larger = slower but more accurate.
    language:
        ISO 639-1 language code (e.g. 'en', 'pt'). ``None`` = auto-detect.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # 1. Try faster-whisper (local)
    try:
        return _transcribe_faster_whisper(path, model, language)
    except ImportError:
        logger.info("faster-whisper not installed; falling back to OpenAI API")
    except Exception as exc:
        logger.warning("faster-whisper failed (%s); falling back to OpenAI API", exc)

    # 2. Fallback: OpenAI Whisper API
    return _transcribe_openai(path, language)


def _transcribe_faster_whisper(path: Path, model: str, language: str | None) -> str:
    from faster_whisper import WhisperModel as FWModel  # type: ignore[import-untyped]

    wm = FWModel(model, device="cpu", compute_type="int8")
    segments, info = wm.transcribe(str(path), language=language, beam_size=5)
    logger.debug("Detected language: %s (%.0f%%)", info.language, info.language_probability * 100)
    return " ".join(seg.text.strip() for seg in segments)


def _transcribe_openai(path: Path, language: str | None) -> str:
    import openai
    client = openai.OpenAI()
    with path.open("rb") as f:
        extra = {"language": language} if language else {}
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f, **extra)  # type: ignore[arg-type]
    return transcript.text


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------


def speak(
    text: str,
    output_path: str | Path | None = None,
    engine: Literal["pyttsx3", "kokoro", "openai", "auto"] = "auto",
    voice: str = "default",
    speed: float = 1.0,
) -> Path | None:
    """Convert text to speech.

    Parameters
    ----------
    text:
        Text to synthesize.
    output_path:
        Save audio to this file. If ``None``, plays directly (pyttsx3 only).
    engine:
        TTS engine. ``'auto'`` tries pyttsx3 → kokoro → openai in order.
    voice:
        Voice name/ID (engine-specific).
    speed:
        Speech rate multiplier (0.5–2.0).

    Returns
    -------
    Path | None
        Path to the saved audio file if ``output_path`` was given, else ``None``.
    """
    if engine == "auto":
        for _engine in ("pyttsx3", "kokoro", "openai"):
            try:
                return _speak_dispatch(text, output_path, _engine, voice, speed)
            except ImportError:
                continue
            except Exception as exc:
                logger.warning("TTS engine %r failed: %s", _engine, exc)
        raise RuntimeError("No TTS engine available")

    return _speak_dispatch(text, output_path, engine, voice, speed)


def _speak_dispatch(
    text: str,
    output_path: str | Path | None,
    engine: str,
    voice: str,
    speed: float,
) -> Path | None:
    if engine == "pyttsx3":
        return _speak_pyttsx3(text, output_path, voice, speed)
    if engine == "kokoro":
        return _speak_kokoro(text, output_path, voice, speed)
    if engine == "openai":
        return _speak_openai(text, output_path, voice, speed)
    raise ValueError(f"Unknown TTS engine: {engine!r}")


def _speak_pyttsx3(text: str, output_path: str | Path | None, voice: str, speed: float) -> Path | None:
    import pyttsx3  # type: ignore[import-untyped]

    engine = pyttsx3.init()
    engine.setProperty("rate", int(engine.getProperty("rate") * speed))

    if voice != "default":
        voices = engine.getProperty("voices")
        for v in voices:
            if voice.lower() in v.name.lower():
                engine.setProperty("voice", v.id)
                break

    if output_path:
        out = Path(output_path)
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        return out
    else:
        engine.say(text)
        engine.runAndWait()
        return None


def _speak_kokoro(text: str, output_path: str | Path | None, voice: str, speed: float) -> Path | None:
    import kokoro  # type: ignore[import-untyped]
    import soundfile as sf  # type: ignore[import-untyped]
    import numpy as np

    pipeline = kokoro.KPipeline(lang_code="en-us")
    voice_name = voice if voice != "default" else "af_heart"

    out_path = Path(output_path) if output_path else Path(tempfile.mktemp(suffix=".wav"))
    samples = []
    for _gs, _ps, audio in pipeline(text, voice=voice_name, speed=speed):
        samples.append(audio)

    audio_np = np.concatenate(samples) if samples else np.array([], dtype=np.float32)
    sf.write(str(out_path), audio_np, 24000)
    return out_path


def _speak_openai(text: str, output_path: str | Path | None, voice: str, speed: float) -> Path | None:
    import openai

    client = openai.OpenAI()
    oai_voice = voice if voice in ("alloy", "echo", "fable", "onyx", "nova", "shimmer") else "nova"
    out_path = Path(output_path) if output_path else Path(tempfile.mktemp(suffix=".mp3"))

    response = client.audio.speech.create(
        model="tts-1", voice=oai_voice, input=text, speed=speed  # type: ignore[arg-type]
    )
    response.stream_to_file(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Microphone capture (local only)
# ---------------------------------------------------------------------------


def record_microphone(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
) -> Path:
    """Record from the default microphone and return a .wav file path.

    Requires ``sounddevice`` and ``soundfile``.
    """
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
        import soundfile as sf    # type: ignore[import-untyped]
        import numpy as np
    except ImportError:
        raise ImportError("Install sounddevice and soundfile: pip install sounddevice soundfile")

    logger.info("Recording %.1f seconds at %d Hz…", duration_seconds, sample_rate)
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    out = Path(tempfile.mktemp(suffix=".wav"))
    sf.write(str(out), audio, sample_rate)
    logger.info("Saved recording to %s", out)
    return out
