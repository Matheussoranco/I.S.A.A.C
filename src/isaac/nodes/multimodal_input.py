"""Multimodal Input Node — transcribe audio, extract document text, analyse images.

Runs after Guard, before Perception, when the incoming state carries multimodal
attachments (audio file paths, document paths, or raw image base64).

Attachments are passed via ``world_model.resources["_attachments"]``:

    [
        {"type": "audio",    "path": "/tmp/recording.wav"},
        {"type": "document", "path": "/tmp/report.pdf"},
        {"type": "image",    "path": "/tmp/diagram.png"},
        {"type": "image_b64","data": "<base64>", "mime": "image/png"},
    ]

The node enriches the user message and ``world_model.observations`` with the
extracted content so the rest of the graph treats it as plain text/context.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from isaac.core.state import IsaacState

logger = logging.getLogger(__name__)


def multimodal_input_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: extract and inject multimodal content into the state."""
    wm = state.get("world_model")
    resources = wm.resources if wm else {}
    attachments: list[dict[str, Any]] = resources.get("_attachments", [])

    if not attachments:
        return {}

    extracted_parts: list[str] = []

    for att in attachments:
        att_type = att.get("type", "")
        try:
            if att_type == "audio":
                text = _handle_audio(att)
                extracted_parts.append(f"[AUDIO TRANSCRIPT]\n{text}")

            elif att_type == "document":
                text = _handle_document(att)
                extracted_parts.append(f"[DOCUMENT: {att.get('path', '')}]\n{text}")

            elif att_type == "image":
                result = _handle_image_path(att)
                if result.get("ocr_text"):
                    extracted_parts.append(f"[IMAGE OCR]\n{result['ocr_text']}")
                if result.get("description"):
                    extracted_parts.append(f"[IMAGE DESCRIPTION]\n{result['description']}")

            elif att_type == "image_b64":
                result = _handle_image_b64(att)
                if result.get("description"):
                    extracted_parts.append(f"[IMAGE DESCRIPTION]\n{result['description']}")

        except Exception as exc:
            logger.warning("Multimodal(%s) failed: %s", att_type, exc)
            extracted_parts.append(f"[{att_type.upper()} ERROR: {exc}]")

    if not extracted_parts:
        return {}

    # Inject extracted content into the conversation as a system-level context message
    combined = "\n\n".join(extracted_parts)
    injection = HumanMessage(content=f"<multimodal_context>\n{combined}\n</multimodal_context>")

    # Also update world_model observations
    if wm:
        wm.observations.extend(extracted_parts[:3])  # cap to avoid bloat
        resources.pop("_attachments", None)  # consumed

    return {
        "messages": [injection],
        "world_model": wm,
        "current_phase": "multimodal_input",
    }


def _handle_audio(att: dict[str, Any]) -> str:
    from isaac.multimodal.audio import transcribe
    path = att.get("path", "")
    model = att.get("model", "base")
    language = att.get("language")
    return transcribe(path, model=model, language=language)


def _handle_document(att: dict[str, Any]) -> str:
    from isaac.multimodal.document import extract_text
    path = att.get("path", "")
    max_pages = att.get("max_pages", 50)
    return extract_text(path, max_pages=max_pages)[:8000]  # cap to avoid context overflow


def _handle_image_path(att: dict[str, Any]) -> dict[str, Any]:
    from isaac.multimodal.document import analyse_image
    path = att.get("path", "")
    prompt = att.get("prompt", "Describe this image. Extract any visible text, data, or diagrams.")
    return analyse_image(path, prompt=prompt)


def _handle_image_b64(att: dict[str, Any]) -> dict[str, Any]:
    data = att.get("data", "")
    mime = att.get("mime", "image/png")
    prompt = att.get("prompt", "Describe this image. Extract any visible text, data, or diagrams.")

    # Save to temp file and call analyse_image
    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(mime, ".png")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(base64.b64decode(data))
        tmp_path = f.name

    from isaac.multimodal.document import analyse_image
    return analyse_image(tmp_path, prompt=prompt)
