"""Vision-language model wrapper.

Routes image+text prompts through a multimodal model (defaults to Ollama
``llava`` or ``qwen2.5-vl``).  Falls back to cloud VLMs (Claude, GPT-4o)
when configured.

The interface accepts either a path / URL / base64 string for the image
and returns a plain-text answer.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def is_vision_available() -> bool:
    """Return True if a VLM is available (local Ollama or cloud key)."""
    try:
        from isaac.config.settings import settings
        from isaac.llm.providers.ollama import health_check

        if settings.vision_enabled and health_check(settings.ollama_base_url):
            return True
        if settings.openai_api_key or settings.anthropic_api_key:
            return True
    except Exception:
        pass
    return False


def _image_to_data_url(image: str | Path | bytes) -> str:
    """Return a ``data:image/...;base64,...`` URL for the input image."""
    if isinstance(image, bytes):
        b64 = base64.b64encode(image).decode("ascii")
        return f"data:image/png;base64,{b64}"

    s = str(image)
    if s.startswith(("http://", "https://", "data:image")):
        return s

    p = Path(s)
    raw = p.read_bytes()
    suffix = p.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{suffix};base64,{b64}"


class VisionLM:
    """Lazy wrapper around a multimodal chat model."""

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self._model: BaseChatModel | None = model

    def _ensure_model(self) -> BaseChatModel:
        if self._model is not None:
            return self._model
        from isaac.llm.multimodal_router import (
            Complexity,
            Modality,
            get_multimodal_router,
        )

        router = get_multimodal_router()
        try:
            self._model = router.route(Modality.VISION, Complexity.DEFAULT)
        except Exception as exc:
            logger.warning("VisionLM: router fallback to text model: %s", exc)
            self._model = router.route(Modality.TEXT, Complexity.DEFAULT)
        return self._model

    # -- Public API ---------------------------------------------------------

    def ask(
        self,
        prompt: str,
        image: str | Path | bytes,
    ) -> str:
        """Ask a question about an image.  Returns the model's text reply."""
        from langchain_core.messages import HumanMessage

        data_url = _image_to_data_url(image)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        model = self._ensure_model()
        response = model.invoke([message])
        content = response.content
        return content if isinstance(content, str) else str(content)

    def describe(self, image: str | Path | bytes) -> str:
        return self.ask(
            "Describe this image in detail. Include any text visible, "
            "objects, people, layout, and any notable context.",
            image,
        )

    def extract_text(self, image: str | Path | bytes) -> str:
        return self.ask(
            "Extract all readable text from this image, preserving layout. "
            "If there is no text, reply 'NO_TEXT'.",
            image,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_vision: VisionLM | None = None


def get_vision_lm() -> VisionLM:
    global _vision  # noqa: PLW0603
    if _vision is None:
        _vision = VisionLM()
    return _vision


def describe_image(image: str | Path | bytes) -> str:
    """Convenience wrapper — describe an image with the default VLM."""
    return get_vision_lm().describe(image)


def reset_vision_lm() -> None:
    """Reset the singleton (used in tests)."""
    global _vision  # noqa: PLW0603
    _vision = None
