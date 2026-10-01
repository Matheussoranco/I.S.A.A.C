
import pytest
from unittest.mock import MagicMock
import sys

# Mock the heavy dependencies before importing the app
sys.modules['playwright'] = MagicMock()
sys.modules['playwright.async_api'] = MagicMock()

from fastapi.testclient import TestClient
from isaac.interfaces.web_app import create_app

@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        # Add the test header to all requests
        c.headers.update({"x-test-client": "true"})
        yield c

def test_profile_list(client):
    response = client.get("/api/desktop/profile/list")
    assert response.status_code == 200
    data = response.json()
    assert "profiles" in data
    assert isinstance(data["profiles"], list)

def test_profile_switch(client):
    response = client.post("/api/desktop/profile/switch", json={"profile_id": "default"})
    assert response.status_code == 200
    assert response.json()["ok"] is True

def test_cron_list(client):
    response = client.get("/api/desktop/cron/list")
    assert response.status_code == 200 
    data = response.json()
    assert "ok" in data

def test_skills_list(client):
    response = client.get("/api/desktop/skills/list")
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data

def test_palette_search(client):
    response = client.get("/api/desktop/palette/search?q=test")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
