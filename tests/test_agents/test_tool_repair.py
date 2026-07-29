"""Tests for malformed tool-call salvage.

The positive cases are the dialects small local models actually emit; the
negative cases matter just as much, because a parser that fires on ordinary
prose would turn every answer containing a brace into a spurious tool call.
"""

from __future__ import annotations

import pytest

from isaac.agents.tool_repair import (
    looks_like_attempted_call,
    reflexion_prompt,
    repair_json,
    salvage_tool_calls,
)

KNOWN = {"web_search", "code", "read_file"}


class TestRepairJson:
    def test_clean_json(self) -> None:
        assert repair_json('{"a": 1}') == {"a": 1}

    def test_python_literal(self) -> None:
        assert repair_json("{'a': True, 'b': None}") == {"a": True, "b": None}

    def test_trailing_comma(self) -> None:
        assert repair_json('{"a": 1,}') == {"a": 1}

    def test_unquoted_keys(self) -> None:
        assert repair_json('{name: "x"}') == {"name": "x"}

    def test_smart_quotes(self) -> None:
        assert repair_json("{“a”: “b”}") == {"a": "b"}

    def test_unparseable_returns_none(self) -> None:
        assert repair_json("this is not json at all") is None

    def test_empty_returns_none(self) -> None:
        assert repair_json("   ") is None


class TestSalvageDialects:
    @pytest.mark.parametrize(
        ("text", "name", "args"),
        [
            (
                '```json\n{"name": "web_search", "arguments": {"query": "cats"}}\n```',
                "web_search",
                {"query": "cats"},
            ),
            (
                '<tool_call>{"name": "code", "arguments": {"source": "1+1"}}</tool_call>',
                "code",
                {"source": "1+1"},
            ),
            ('{"tool": "code", "args": {"source": "x"}}', "code", {"source": "x"}),
            (
                '{"name": "code", "arguments": "{\\"source\\": \\"y\\"}"}',
                "code",
                {"source": "y"},
            ),
            (
                "{'name': 'read_file', 'parameters': {'path': 'a.txt'}}",
                "read_file",
                {"path": "a.txt"},
            ),
            (
                '{"name": "web_search", "arguments": {"query": "q",},}',
                "web_search",
                {"query": "q"},
            ),
            ('{"web_search": {"query": "z"}}', "web_search", {"query": "z"}),
            (
                '{"function": {"name": "code", "arguments": {"source": "s"}}}',
                "code",
                {"source": "s"},
            ),
            (
                'Let me look.\n{"name":"web_search","arguments":{"query":"a"}}',
                "web_search",
                {"query": "a"},
            ),
        ],
        ids=[
            "fenced-json",
            "hermes-tag",
            "args-alias",
            "double-encoded-args",
            "python-dict-syntax",
            "trailing-commas",
            "name-as-key",
            "openai-wire-shape",
            "prose-preamble",
        ],
    )
    def test_dialect_is_recovered(self, text: str, name: str, args: dict) -> None:
        calls = salvage_tool_calls(text, KNOWN)
        assert len(calls) == 1
        assert calls[0]["name"] == name
        assert calls[0]["args"] == args

    def test_python_call_expression(self) -> None:
        calls = salvage_tool_calls('I will call web_search(query="cats", limit=3)', KNOWN)
        assert calls[0]["name"] == "web_search"
        assert calls[0]["args"] == {"query": "cats", "limit": 3}

    def test_nested_braces_in_argument_value(self) -> None:
        text = '{"name":"code","arguments":{"source":"d={\\"a\\":1}"}}'
        calls = salvage_tool_calls(text, KNOWN)
        assert calls[0]["args"]["source"] == 'd={"a":1}'

    def test_braces_inside_string_do_not_split_the_object(self) -> None:
        text = '{"name": "web_search", "arguments": {"query": "a } b"}}'
        calls = salvage_tool_calls(text, KNOWN)
        assert calls[0]["args"] == {"query": "a } b"}

    def test_every_call_gets_a_unique_id(self) -> None:
        text = (
            '<tool_call>{"name":"code","arguments":{"source":"1"}}</tool_call>'
            '<tool_call>{"name":"code","arguments":{"source":"2"}}</tool_call>'
        )
        calls = salvage_tool_calls(text, KNOWN)
        assert len({c["id"] for c in calls}) == len(calls) == 2

    def test_identical_duplicate_calls_are_collapsed(self) -> None:
        text = (
            '<tool_call>{"name":"code","arguments":{"source":"1"}}</tool_call>'
            '<tool_call>{"name":"code","arguments":{"source":"1"}}</tool_call>'
        )
        assert len(salvage_tool_calls(text, KNOWN)) == 1


class TestSalvageNegatives:
    @pytest.mark.parametrize(
        "text",
        [
            "The capital of France is Paris.",
            'In Python you write a dict as {"a": 1}.',
            "",
            "   ",
            'Here is JSON: {"unrelated": "data"}',
        ],
        ids=["plain", "code-in-prose", "empty", "whitespace", "unrelated-json"],
    )
    def test_prose_is_not_a_tool_call(self, text: str) -> None:
        assert salvage_tool_calls(text, KNOWN) == []

    def test_unknown_tool_name_is_rejected(self) -> None:
        assert salvage_tool_calls('{"name": "rm_rf", "arguments": {}}', KNOWN) == []

    def test_without_known_tools_python_calls_are_not_guessed(self) -> None:
        # No allow-list means no way to tell a call from a sentence; the risky
        # dialects must stay off rather than fire on arbitrary text.
        assert salvage_tool_calls('print("hello")', None) == []


class TestIntentDetection:
    @pytest.mark.parametrize(
        "text",
        [
            '{"name": "not_a_tool", "arguments": {}}',
            "I will call the web_search tool now.",
            "<tool_call>garbage that will not parse",
            "web_search(query=",
        ],
        ids=["hallucinated-tool", "stated-intent", "broken-tag", "truncated-call"],
    )
    def test_failed_attempts_are_flagged(self, text: str) -> None:
        assert looks_like_attempted_call(text, KNOWN) is True

    @pytest.mark.parametrize(
        "text",
        ["Paris is the capital of France.", "", "The answer is 42."],
        ids=["answer", "empty", "numeric-answer"],
    )
    def test_answers_are_not_flagged(self, text: str) -> None:
        assert looks_like_attempted_call(text, KNOWN) is False


class TestReflexionPrompt:
    def test_lists_tools_and_quotes_the_failure(self) -> None:
        prompt = reflexion_prompt('{"broken": ', KNOWN)
        assert "web_search" in prompt
        assert "could not be parsed" in prompt
        assert '{"broken":' in prompt

    def test_long_output_is_truncated(self) -> None:
        prompt = reflexion_prompt("x" * 5000, KNOWN)
        assert len(prompt) < 1200

    def test_includes_parser_error_when_given(self) -> None:
        assert "Unexpected token" in reflexion_prompt("{", KNOWN, error="Unexpected token")
