"""Shared test fixtures for I.S.A.A.C.

Provides mock LLM responses, mock Docker containers, and pre-built state
objects so individual test modules stay focused.

The suite runs **fully offline**: :func:`stub_ollama` is autouse, so no test
ever contacts a real Ollama daemon (or any other network service) even though
Ollama is now the default provider.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from isaac.core.state import (
    ErrorEntry,
    ExecutionResult,
    IsaacState,
    PlanStep,
    SkillCandidate,
    make_initial_state,
)
from isaac.llm.providers.ollama import DEFAULT_MODEL
from isaac.security.capabilities import TokenStore

# ---------------------------------------------------------------------------
# Offline guarantees
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_capability_store(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TokenStore]:
    """Keep authorization state deterministic and out of the user's profile."""
    from isaac.security import capabilities

    store_root = tmp_path_factory.mktemp("capability-store")
    store = TokenStore(store_root / "tokens.json")
    monkeypatch.setattr(capabilities, "_instance", store)
    yield store
    monkeypatch.setattr(capabilities, "_instance", None)


@pytest.fixture(autouse=True)
def stub_ollama():
    """Simulate a healthy local Ollama with the default model installed.

    Ollama is the default provider, so without this the suite would probe
    ``http://localhost:11434`` on any machine that runs it — flaky locally and
    a forbidden network call in CI.  Tests that need the *unhealthy* path
    patch these same symbols themselves; the innermost patch wins.
    """
    from isaac.llm import provider as provider_mod

    with (
        patch("isaac.llm.providers.ollama.health_check", return_value=True),
        patch("isaac.llm.providers.ollama.list_models", return_value=[DEFAULT_MODEL]),
        patch("isaac.llm.providers.ollama.preflight", return_value=None),
        patch("isaac.llm.providers.llamacpp.health_check", return_value=False),
    ):
        provider_mod._preflight_once.cache_clear()
        yield
        provider_mod._preflight_once.cache_clear()


# ---------------------------------------------------------------------------
# State fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def blank_state() -> IsaacState:
    """A fully initialised blank state."""
    return make_initial_state()


@pytest.fixture()
def state_with_plan() -> IsaacState:
    """State that already has a two-step plan with step-1 active."""
    state = make_initial_state()
    state["hypothesis"] = "Test hypothesis"
    state["plan"] = [
        PlanStep(id="s1", description="First step", status="active"),
        PlanStep(id="s2", description="Second step", status="pending", depends_on=["s1"]),
    ]
    state["iteration"] = 1
    return state


@pytest.fixture()
def state_after_success(state_with_plan: IsaacState) -> IsaacState:
    """State after a successful sandbox execution + reflection."""
    state_with_plan["execution_logs"] = [
        ExecutionResult(stdout="42\n", stderr="", exit_code=0, duration_ms=150.0)
    ]
    state_with_plan["code_buffer"] = "print(42)"
    state_with_plan["skill_candidate"] = SkillCandidate(
        name="add_numbers",
        code="print(42)",
        task_context="First step",
        success_count=1,
    )
    state_with_plan["plan"][0].status = "done"
    return state_with_plan


@pytest.fixture()
def state_after_failure(state_with_plan: IsaacState) -> IsaacState:
    """State after a failed sandbox execution + reflection."""
    state_with_plan["execution_logs"] = [
        ExecutionResult(
            stdout="", stderr="NameError: name 'x' is not defined", exit_code=1, duration_ms=80.0
        )
    ]
    state_with_plan["code_buffer"] = "print(x)"
    state_with_plan["errors"] = [
        ErrorEntry(
            node="reflection",
            message="NameError in generated code",
            attempt=1,
            timestamp="2026-01-01T00:00:00Z",
        )
    ]
    return state_with_plan


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLMResponse:
    """Mimics a ``langchain_core`` AIMessage for prompt testing."""

    def __init__(self, content: str) -> None:
        self.content = content


class MockLLM:
    """Deterministic LLM stub that returns pre-configured responses."""

    def __init__(self, response: str = '{"observations": [], "hypothesis": "mock"}') -> None:
        self._response = response

    def invoke(self, messages: Any) -> MockLLMResponse:
        return MockLLMResponse(self._response)


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture()
def patch_llm(mock_llm: MockLLM):
    """Patch ``get_llm`` globally to return the mock."""
    with patch("isaac.llm.provider.get_llm", return_value=mock_llm):
        yield mock_llm


# ---------------------------------------------------------------------------
# Mock Docker
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_docker():
    """Patch the ``docker`` module to avoid real container operations."""
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.short_id = "abc123"
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = b"mock output"

    mock_client.containers.create.return_value = mock_container
    mock_client.images.get.return_value = True

    with patch("docker.from_env", return_value=mock_client):
        yield mock_client, mock_container
