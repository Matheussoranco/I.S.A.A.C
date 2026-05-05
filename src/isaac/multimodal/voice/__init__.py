"""Voice I/O subsystem — STT, TTS, microphone & speaker handling.

The submodules guard their imports so the whole package can be imported
even when no audio dependencies are installed.
"""

from __future__ import annotations

from isaac.multimodal.voice.stt import (
    SpeechToText,
    get_stt,
    is_stt_available,
)
from isaac.multimodal.voice.tts import (
    TextToSpeech,
    get_tts,
    is_tts_available,
)

__all__ = [
    "SpeechToText",
    "TextToSpeech",
    "get_stt",
    "get_tts",
    "is_stt_available",
    "is_tts_available",
]
