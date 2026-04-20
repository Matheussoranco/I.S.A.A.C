"""I.S.A.A.C. multimodal subsystem.

Modules
-------
* :pymod:`isaac.multimodal.voice`   — STT + TTS + audio I/O
* :pymod:`isaac.multimodal.vision`  — Vision-language model inference
* :pymod:`isaac.multimodal.input`   — Unified multimodal input handler

Each submodule is independently importable and degrades gracefully when
its optional dependencies (whisper, piper, sounddevice, pillow, ...) are
missing.  Callers should query ``is_available()`` before using a feature.
"""

from __future__ import annotations

__all__ = ["voice", "vision", "input"]
