"""Audio I/O — microphone capture, speaker playback, and VAD-based recording.

Backends:
* ``sounddevice`` (preferred — cross-platform PortAudio)
* ``simpleaudio`` (playback only fallback)

The :func:`record_until_silence` helper implements voice-activity-detection
(VAD) using ``webrtcvad`` when available, falling back to a simple energy
gate otherwise.  This is what the voice REPL uses for push-to-talk and
hands-free conversational mode.
"""

from __future__ import annotations

import io
import logging
import os
import time
import wave
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 30
DEFAULT_SILENCE_TIMEOUT_S = 1.2
DEFAULT_MAX_RECORD_S = 30.0


def is_audio_available() -> bool:
    """Return True if microphone capture is supported."""
    try:
        import sounddevice  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def play_wav(path: str | os.PathLike[str]) -> None:
    """Block-play a WAV file through the default output device."""
    path = str(path)
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        sd.play(data, sr)
        sd.wait()
        return
    except ImportError:
        pass

    try:
        import simpleaudio as sa  # type: ignore[import-not-found]

        wave_obj = sa.WaveObject.from_wave_file(path)
        play = wave_obj.play()
        play.wait_done()
        return
    except ImportError:
        pass

    # Last-resort: hand off to the OS
    if os.name == "nt":
        import winsound  # type: ignore[import-not-found]
        winsound.PlaySound(path, winsound.SND_FILENAME)
        return

    raise RuntimeError(
        "No audio playback backend available. "
        "Install one of: sounddevice+soundfile, simpleaudio."
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def record_fixed(
    duration_s: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> "np.ndarray":
    """Record exactly *duration_s* seconds of mono audio."""
    import numpy as np
    import sounddevice as sd

    frames = int(duration_s * sample_rate)
    data = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return data.flatten()


def record_until_silence(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    silence_timeout_s: float = DEFAULT_SILENCE_TIMEOUT_S,
    max_record_s: float = DEFAULT_MAX_RECORD_S,
    energy_threshold: float = 0.005,
    on_chunk: Any = None,
) -> "np.ndarray":
    """Record until a stretch of silence is detected.

    Parameters
    ----------
    sample_rate:
        Capture sample rate.  16 kHz is the standard for STT.
    silence_timeout_s:
        How long the audio must remain below ``energy_threshold`` before
        recording stops.
    max_record_s:
        Hard ceiling on recording length.
    energy_threshold:
        RMS amplitude (0..1) below which a frame counts as silent.
        Used as the fallback when ``webrtcvad`` is unavailable.
    on_chunk:
        Optional callback ``f(rms_level: float)`` called once per frame
        — useful for drawing a level meter in the REPL.
    """
    import numpy as np
    import sounddevice as sd

    frame_size = int(sample_rate * DEFAULT_FRAME_MS / 1000)
    silence_frames_needed = int(silence_timeout_s * 1000 / DEFAULT_FRAME_MS)
    max_frames = int(max_record_s * 1000 / DEFAULT_FRAME_MS)

    vad: Any = None
    try:
        import webrtcvad  # type: ignore[import-not-found]
        vad = webrtcvad.Vad(2)  # 0 = least aggressive, 3 = most
    except ImportError:
        pass

    def _is_silent(frame: "np.ndarray") -> bool:
        if vad is not None:
            pcm16 = (frame * 32767).astype(np.int16).tobytes()
            try:
                return not vad.is_speech(pcm16, sample_rate)
            except Exception:
                pass  # fall through to energy gate
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        return rms < energy_threshold

    chunks: list["np.ndarray"] = []
    silent_streak = 0
    started_speaking = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        for _ in range(max_frames):
            frame, _overflow = stream.read(frame_size)
            frame = frame.flatten()
            chunks.append(frame)
            silent = _is_silent(frame)
            if on_chunk is not None:
                try:
                    rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
                    on_chunk(rms)
                except Exception:
                    pass
            if silent:
                if started_speaking:
                    silent_streak += 1
                    if silent_streak >= silence_frames_needed:
                        break
            else:
                started_speaking = True
                silent_streak = 0

    if not chunks:
        return np.zeros(0, dtype="float32")
    import numpy as np
    return np.concatenate(chunks).astype("float32")


def save_wav(
    samples: "np.ndarray",
    path: str | os.PathLike[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """Write a float32 numpy array to disk as a 16-bit PCM WAV file."""
    import numpy as np

    path = str(path)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return path


def samples_to_wav_bytes(
    samples: "np.ndarray",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """Return WAV-encoded bytes for an in-memory audio buffer."""
    import numpy as np

    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()
