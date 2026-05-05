"""Unified multimodal input — converts heterogeneous user inputs into a
single ``HumanMessage`` for the cognitive graph.

Accepts any combination of:

* ``text``           — plain string
* ``image_paths``    — list of file paths or URLs
* ``audio_path``     — WAV/MP3 file → transcribed via STT
* ``screenshot``     — capture current screen → attached as image
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def build_multimodal_message(
    text: str = "",
    image_paths: list[str | Path] | None = None,
    audio_path: str | Path | None = None,
    screenshot: bool = False,
) -> HumanMessage:
    """Combine inputs into a single multimodal HumanMessage.

    Parameters
    ----------
    text:
        Optional textual prompt.
    image_paths:
        List of image files (local paths or http(s) URLs).
    audio_path:
        Optional audio file — transcribed and prepended to ``text``.
    screenshot:
        If True, capture the current screen and attach as an image.

    Returns
    -------
    HumanMessage
        With either string content (text-only) or a list of content blocks
        compatible with multimodal LLMs.
    """
    image_paths = list(image_paths or [])

    # 1) Resolve audio → text
    if audio_path is not None:
        try:
            from isaac.multimodal.voice.stt import get_stt
            transcribed = get_stt().transcribe(str(audio_path))
            if transcribed:
                text = (text + "\n" + transcribed).strip() if text else transcribed
        except Exception as exc:
            logger.warning("Multimodal: STT failed: %s", exc)

    # 2) Resolve screenshot → image
    if screenshot:
        try:
            from isaac.multimodal.vision.screen_capture import capture_screen_b64
            data_url = f"data:image/png;base64,{capture_screen_b64()}"
            image_paths.append(data_url)
        except Exception as exc:
            logger.warning("Multimodal: screen capture failed: %s", exc)

    # 3) Build the message
    if not image_paths:
        return HumanMessage(content=text)

    from isaac.multimodal.vision.vision_lm import _image_to_data_url

    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for img in image_paths:
        try:
            data_url = _image_to_data_url(img) if not str(img).startswith("data:") else str(img)
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        except Exception as exc:
            logger.warning("Multimodal: failed to attach image %s: %s", img, exc)
    return HumanMessage(content=blocks)
