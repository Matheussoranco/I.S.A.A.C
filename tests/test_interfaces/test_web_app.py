"""Local graphical interface transport tests."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from isaac.agents.agent_loop import AgentRunResult
from isaac.interfaces import web_app as web_app_module
from isaac.interfaces.conversation_store import ConversationStore
from isaac.interfaces.web_app import create_app


class _FakeAgent:
    def run(self, prompt: str, context: str = "") -> AgentRunResult:
        assert prompt == "hello"
        assert context == ""
        return AgentRunResult(output="hello back", iterations=1, stopped_reason="final")


def _builder(**kwargs: Any) -> _FakeAgent:
    assert kwargs["llm"] is _FAKE_LLM
    assert callable(kwargs["on_event"])
    assert callable(kwargs["browser_event_callback"])
    assert callable(kwargs["desktop_event_callback"])
    assert callable(kwargs["approval_callback"])
    assert callable(kwargs["should_stop"])
    return _FakeAgent()


_FAKE_LLM = object()


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "credential_available", lambda provider: False)
    monkeypatch.setattr(web_app_module, "set_credential", lambda provider, value: None)


def _app(tmp_path):
    return create_app(
        agent_builder=_builder,
        conversation_store=ConversationStore(tmp_path / "conversations.sqlite3"),
        llm_builder=lambda *args, **kwargs: _FAKE_LLM,
    )


def test_index_and_status_do_not_expose_secrets(tmp_path) -> None:
    client = TestClient(_app(tmp_path))

    index = client.get("/")
    status = client.get("/api/status")

    assert index.status_code == 200
    assert "Área do agente" in index.text
    assert status.status_code == 200
    assert set(status.json()) == {
        "name",
        "provider",
        "model",
        "version",
        "local_first",
        "profiles",
        "openai_configured",
        "anthropic_configured",
    }
    assert "api_key" not in status.text.lower()


def test_websocket_runs_agent_and_streams_completion(tmp_path) -> None:
    client = TestClient(_app(tmp_path))

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "connected"
        assert websocket.receive_json()["type"] == "conversation_list"
        assert websocket.receive_json()["type"] == "history_loaded"
        websocket.send_json({"action": "run", "message": "hello"})
        events = []
        while not events or events[-1]["type"] != "run_complete":
            events.append(websocket.receive_json())

    started = next(event for event in events if event["type"] == "run_started")
    completed = events[-1]
    assert started == {"type": "run_started", "data": {"message": "hello"}}
    assert completed["type"] == "run_complete"
    assert completed["data"]["output"] == "hello back"
    assert completed["data"]["success"] is True


def test_websocket_configures_model_without_exposing_credentials(tmp_path) -> None:
    client = TestClient(_app(tmp_path))

    with client.websocket_connect("/ws") as websocket:
        for _ in range(3):
            websocket.receive_json()
        websocket.send_json(
            {
                "action": "configure",
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
            }
        )
        configured = websocket.receive_json()

    assert configured == {
        "type": "configured",
        "data": {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "credential_configured": False,
        },
    }


def test_websocket_persists_completed_conversation(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    client = TestClient(
        create_app(
            agent_builder=_builder,
            conversation_store=store,
            llm_builder=lambda *args, **kwargs: _FAKE_LLM,
        )
    )

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        listed = websocket.receive_json()
        websocket.receive_json()
        conversation_id = listed["data"]["active_id"]
        websocket.send_json({"action": "run", "message": "hello"})
        while websocket.receive_json()["type"] != "run_complete":
            pass

    assert store.load(conversation_id) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hello back"},
    ]


def test_computer_mode_uses_first_party_runner(tmp_path) -> None:
    created: list[dict[str, Any]] = []

    class _Runner:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

        def run(self, prompt: str, context: str = "") -> AgentRunResult:
            assert prompt == "use the desktop"
            return AgentRunResult(output="desktop done", iterations=2, stopped_reason="final")

    client = TestClient(
        create_app(
            agent_builder=_builder,
            computer_runner_builder=_Runner,
            conversation_store=ConversationStore(tmp_path / "conversations.sqlite3"),
            llm_builder=lambda *args, **kwargs: _FAKE_LLM,
        )
    )

    with client.websocket_connect("/ws") as websocket:
        for _ in range(3):
            websocket.receive_json()
        websocket.send_json(
            {
                "action": "configure",
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
            }
        )
        websocket.receive_json()
        websocket.send_json({"action": "run", "mode": "computer", "message": "use the desktop"})
        completed = None
        while completed is None or completed["type"] != "run_complete":
            completed = websocket.receive_json()

    assert completed["data"]["output"] == "desktop done"
    assert created[0]["model"] == "gpt-5.6-sol"
    assert created[0]["reasoning_effort"] == "high"
