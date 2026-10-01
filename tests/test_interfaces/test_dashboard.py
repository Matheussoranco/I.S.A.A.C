"""Tests for Dashboard Management UI backend.
Verifies config updates and memory retrieval endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from isaac.interfaces.desktop_api import router
from fastapi import FastAPI
from unittest.mock import MagicMock, patch

# Setup a dummy app for testing the router
app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_config_get_post():
    """Test retrieving and updating profile configuration."""
    with patch("isaac.interfaces.dashboard.config_editor.get_config_content") as mock_get, \
         patch("isaac.interfaces.dashboard.config_editor.update_config") as mock_set:
        
        mock_get.return_value = {"llm": {"model": "gpt-4"}}
        resp = client.get("/api/desktop/dashboard/config?profile=test")
        assert resp.status_code == 200
        assert resp.json() == {"llm": {"model": "gpt-4"}}
        
        payload = {"llm": {"model": "claude-3"}}
        mock_set.return_value = {"ok": True, "profile": "test"}
        resp = client.post("/api/desktop/dashboard/config?profile=test", json=payload)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

def test_memory_search_endpoint():
    """Test the unified memory search endpoint."""
    with patch("isaac.interfaces.dashboard.memory_browser.search_memory") as mock_search:
        mock_search.return_value = {
            "episodic": "Recent experience...",
            "semantic": [{"subject": "Isaac", "predicate": "is", "object": "Agent"}],
            "procedural": ["skill_a"],
            "combined": "Context string"
        }
        resp = client.get("/api/desktop/dashboard/memory/search?query=Who is Isaac?")
        assert resp.status_code == 200
        data = resp.json()
        assert "episodic" in data
        assert data["combined"] == "Context string"

def test_api_keys_management():
    """Test API key listing and setting."""
    with patch("isaac.interfaces.dashboard.api_keys.list_keys") as mock_list, \
         patch("isaac.interfaces.dashboard.api_keys.set_key") as mock_set:
        
        mock_list.return_value = {"openai": True, "anthropic": False}
        resp = client.get("/api/desktop/dashboard/keys")
        assert resp.status_code == 200
        assert resp.json()["openai"] is True
        
        mock_set.return_value = {"ok": True, "provider": "anthropic"}
        resp = client.post("/api/desktop/dashboard/keys", json={"provider": "anthropic", "value": "sk-..."})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
