"""Tests for the Synthesis node."""

from __future__ import annotations

from unittest.mock import patch

from isaac.core.state import PlanStep, make_initial_state
from isaac.nodes.synthesis import synthesis_node, _extract_code
from tests.conftest import MockLLM


class TestExtractCode:
    def test_fenced_python(self) -> None:
        text = 'Here is code:\n```python\nprint("hi")\n```\nDone.'
        assert _extract_code(text) == 'print("hi")'

    def test_fenced_no_language(self) -> None:
        text = '```\nx = 1\n```'
        assert _extract_code(text) == "x = 1"

    def test_no_fence(self) -> None:
        text = "print('hello')"
        assert _extract_code(text) == "print('hello')"


class TestSynthesisNode:
    def test_generates_code_buffer(self) -> None:
        state = make_initial_state()
        state["plan"] = [PlanStep(id="s1", description="print 42", status="active")]
        state["hypothesis"] = "compute 42"

        mock = MockLLM('```python\nprint(42)\n```')
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = synthesis_node(state)

        assert result["code_buffer"] == "print(42)"
        assert result["current_phase"] == "synthesis"

    def test_no_active_step(self) -> None:
        state = make_initial_state()
        state["plan"] = [PlanStep(id="s1", description="done", status="done")]

        result = synthesis_node(state)
        assert "NOOP" in result["code_buffer"]
