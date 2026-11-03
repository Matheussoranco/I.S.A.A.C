import base64
import logging
import mimetypes
from pathlib import Path

import httpx

from isaac.config.settings import settings

logger = logging.getLogger(__name__)


class VisionManager:
    """Handles image-to-text prompts using local (Ollama) and cloud providers."""

    def __init__(self) -> None:
        self.local_provider = settings.vision_model or "llava:7b"
        self.cloud_provider = settings.vision_cloud_provider or "gpt-4o"
        self._cloud_kind, self._cloud_model, self._cloud_key = self._resolve_cloud()

    def _resolve_cloud(self) -> tuple[str, str, str]:
        value = self.cloud_provider.strip()
        lowered = value.lower()
        if "claude" in lowered or lowered == "anthropic":
            model = "claude-3-5-sonnet-20240620" if lowered in {"claude", "anthropic"} else value
            return "anthropic", model, settings.anthropic_api_key.strip()
        if "gemini" in lowered:
            model = "gemini-1.5-pro" if lowered == "gemini" else value
            return "gemini", model, settings.gemini_api_key.strip()
        model = "gpt-4o" if lowered == "openai" else value
        return "openai", model, settings.openai_api_key.strip()

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    @staticmethod
    def _media_type(image_path: Path) -> str:
        return mimetypes.guess_type(image_path.name)[0] or "image/jpeg"

    async def analyze(self, image_path: Path, prompt: str) -> str:
        """Analyze an image locally, falling back to a configured cloud VLM."""
        try:
            return await self._analyze_local(image_path, prompt)
        except Exception as exc:
            if not self._cloud_key:
                logger.debug("Local VLM failed (%s); no cloud credential configured", exc)
                raise
            logger.debug("Local VLM failed (%s); falling back to cloud", exc)
            return await self._analyze_cloud(image_path, prompt)

    async def _analyze_local(self, image_path: Path, prompt: str) -> str:
        base64_image = self._encode_image(image_path)
        url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.local_provider,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "No response from local VLM.")

    async def _analyze_cloud(self, image_path: Path, prompt: str) -> str:
        data = self._encode_image(image_path)
        media_type = self._media_type(image_path)
        if self._cloud_kind == "anthropic":
            return await self._anthropic_analyze(data, prompt, media_type)
        if self._cloud_kind == "gemini":
            return await self._gemini_analyze(data, prompt, media_type)
        return await self._openai_analyze(data, prompt, media_type)

    async def _openai_analyze(self, data: str, prompt: str, media_type: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._cloud_key}"}
        payload = {
            "model": self._cloud_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        },
                    ],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _anthropic_analyze(self, data: str, prompt: str, media_type: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self._cloud_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self._cloud_model,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    async def _gemini_analyze(self, data: str, prompt: str, media_type: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._cloud_model}:generateContent?key={self._cloud_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": media_type, "data": data}},
                    ]
                }
            ]
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
