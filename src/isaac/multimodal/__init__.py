"""I.S.A.A.C. multimodal subsystem.

Modules
-------
* :pymod:`isaac.multimodal.audio`    — STT (faster-whisper) + TTS
* :pymod:`isaac.multimodal.document` — PDF / DOCX / PPTX / OCR
* :pymod:`isaac.multimodal.voice`    — voice I/O pipeline
* :pymod:`isaac.multimodal.vision`   — vision-language model inference
* :pymod:`isaac.multimodal.input`    — unified multimodal input handler

Each submodule is independently importable and degrades gracefully when
its optional dependencies (whisper, piper, sounddevice, pillow, ...) are
missing. Callers should query ``is_available()`` before using a feature.
"""

from __future__ import annotations
