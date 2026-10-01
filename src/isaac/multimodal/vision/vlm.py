import base64
from pathlib import Path

import httpx

from isaac.config.settings import settings


class VisionManager:
    """Handles image-to-text prompts using local (Ollama) and cloud providers."""

    def __init__(self):
        self.local_provider = settings.vision_model or "llava:7b"
        self.cloud_provider = getattr(settings, "vision_cloud_provider", "gpt-4o") or "gpt-4o"

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    async def analyze(self, image_path: Path, prompt: str) -> str:
        """
        Analyzes an image based on the prompt.
        Attempts local VLM first, falls back to cloud.
        """
        try:
            return await self._analyze_local(image_path, prompt)
        except Exception as e:
            print(f"Local VLM failed: {e}. Falling back to cloud...")
            return await self._analyze_cloud(image_path, prompt)

    async def _analyze_local(self, image_path: Path, prompt: str) -> str:
        # Ollama implementation (llava, qwen-vl)
        base64_image = self._encode_image(image_path)
        url = f"{settings.ollama_base_url}/api/generate"

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
        base64_image = self._encode_image(image_path)

        if "gpt-4o" in self.cloud_provider:
            return await self._gpt4o_analyze(base64_image, prompt)
        elif "claude" in self.cloud_provider:
            return await self._claude_analyze(base64_image, prompt)
        elif "gemini" in self.cloud_provider:
            return await self._gemini_analyze(base64_image, prompt)
        else:
            raise ValueError(f"Unsupported cloud provider: {self.cloud_provider}")

    async def _gpt4o_analyze(self, base64_image: str, prompt: str) -> str:
        # Mocked/Simplified OpenAI VLM call
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _claude_analyze(self, base64_image: str, prompt: str) -> str:
        # Mocked/Simplified Anthropic VLM call
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"}
        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    async def _gemini_analyze(self, base64_image: str, prompt: str) -> str:
        # Mocked/Simplified Google Gemini VLM call
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
                    ]
                }
            ]
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
