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


class TestProseAroundTheCall:
    """Small models narrate before they act, and they use contractions.

    ``_extract_balanced`` used to track quotes at every depth, so the
    apostrophe in "I'll" opened a string literal that never closed and the
    JSON call after it was never seen — silently turning the model's intended
    call into a final answer, the exact failure 1.4.0 set out to fix.
    """

    CALL = '{"name": "web_search", "arguments": {"query": "cats"}}'

    @pytest.mark.parametrize(
        "preamble",
        [
            "I'll use the search tool now: ",
            "Don't worry, I can look that up: ",
            "Let's search for it: ",
            "It's easier to just search: ",
            "I will search: ",
        ],
        ids=["ill", "dont", "lets", "its", "no-contraction"],
    )
    def test_contractions_do_not_hide_the_call(self, preamble: str) -> None:
        calls = salvage_tool_calls(preamble + self.CALL, KNOWN)
        assert [c["name"] for c in calls] == ["web_search"]
        assert calls[0]["args"] == {"query": "cats"}

    def test_braces_inside_strings_still_survive(self) -> None:
        # The reason quote tracking exists at all: a brace inside an argument
        # value must not end the object early.
        text = '{"name": "web_search", "arguments": {"query": "a } b"}}'
        assert salvage_tool_calls(text, KNOWN)[0]["args"] == {"query": "a } b"}

    def test_single_quoted_python_dialect_still_survives(self) -> None:
        text = "{'name': 'web_search', 'args': {'query': 'x } y'}}"
        assert salvage_tool_calls(text, KNOWN)[0]["args"] == {"query": "x } y"}


class TestZeroArgumentCalls:
    """``system_info()`` and ``file_list()`` are real, genuinely argument-less
    tools. Requiring at least one keyword made them unrecoverable, while
    accepting a bare ``name()`` anywhere in prose would fire on discussion of a
    tool rather than a call. Whole-message calls are accepted; mentions are not.
    """

    ZERO_ARG = {"system_info", "file_list", "web_search"}

    @pytest.mark.parametrize(
        "text",
        ["system_info()", "  system_info()  ", "```python\nsystem_info()\n```"],
        ids=["bare", "padded", "fenced"],
    )
    def test_whole_message_zero_arg_call_is_recovered(self, text: str) -> None:
        calls = salvage_tool_calls(text, self.ZERO_ARG)
        assert calls and calls[0]["name"] == "system_info"
        assert calls[0]["args"] == {}

    @pytest.mark.parametrize(
        "text",
        [
            "the file_list() helper returns a list of paths",
            "You could call system_info() but I already know the answer.",
        ],
        ids=["prose-mention", "prose-aside"],
    )
    def test_zero_arg_mentions_in_prose_are_ignored(self, text: str) -> None:
        assert salvage_tool_calls(text, self.ZERO_ARG) == []

    def test_positional_arguments_are_refused(self) -> None:
        # Without the tool signature a positional cannot be mapped to a name;
        # guessing is worse than letting Reflexion ask again.
        assert salvage_tool_calls('web_search("cats")', self.ZERO_ARG) == []

    def test_keyword_arguments_still_parse(self) -> None:
        calls = salvage_tool_calls('web_search(query="cats", limit=3)', self.ZERO_ARG)
        assert calls[0]["args"] == {"query": "cats", "limit": 3}
