
import asyncio
import httpx
import pytest
from isaac.config.settings import settings

# We assume the proxy is running on http://127.0.0.1:8000
BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = getattr(settings, "ISAAC_API_KEY", "test-key")

@pytest.mark.asyncio
async def test_api_auth_failure():
    async with httpx.AsyncClient() as client:
        # No auth header
        resp = await client.post(f"{BASE_URL}/chat/completions", json={
            "model": "isaac-strong",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert resp.status_code == 403

@pytest.mark.asyncio
async def test_api_auth_success():
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        resp = await client.get(f"{BASE_URL}/models", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert any(m["id"] == "isaac-strong" for m in data["data"])

@pytest.mark.asyncio
async def test_chat_completion():
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        payload = {
            "model": "isaac-strong",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False
        }
        resp = await client.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] is not None

@pytest.mark.asyncio
async def test_chat_completion_streaming():
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        payload = {
            "model": "isaac-strong",
            "messages": [{"role": "user", "content": "Write a short poem about AI."}],
            "stream": True
        }
        async with client.stream("POST", f"{BASE_URL}/chat/completions", json=payload, headers=headers) as response:
            assert response.status_code == 200
            full_content = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    content = line[6:]
                    if content == "[DONE]":
                        break
                    try:
                        chunk = __import__('json').loads(content)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        full_content.append(delta)
                    except:
                        pass
            assert "".join(full_content).strip() != ""
