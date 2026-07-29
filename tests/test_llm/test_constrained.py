"""Tests for schema- and grammar-constrained tool calling."""

from __future__ import annotations

import json

from isaac.llm.constrained import (
    NO_TOOL,
    apply_constraint,
    gbnf_for_tools,
    parse_envelope,
    supports_constrained_decoding,
    tool_envelope_schema,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code",
            "description": "Run Python.",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string"}},
                "required": ["source"],
            },
        },
    },
]
KNOWN = {"web_search", "code"}


class TestEnvelopeSchema:
    def test_branches_per_tool_by_default(self) -> None:
        # Measured: branching is what constrains argument keys, and a flat
        # schema lets a small model invent them. See docs/MODELS.md.
        schema = tool_envelope_schema(TOOLS)
        assert "oneOf" in schema

    def test_flat_schema_enumerates_tools_plus_none(self) -> None:
        schema = tool_envelope_schema(TOOLS, per_tool=False)
        assert set(schema["properties"]["tool"]["enum"]) == {"web_search", "code", NO_TOOL}
        assert schema["required"] == ["tool"]

    def test_schema_is_json_serialisable(self) -> None:
        # Ollama sends this over the wire as the `format` field.
        json.dumps(tool_envelope_schema(TOOLS))
        json.dumps(tool_envelope_schema(TOOLS, per_tool=False))

    def test_per_tool_schema_branches_and_carries_real_params(self) -> None:
        schema = tool_envelope_schema(TOOLS, per_tool=True)
        branches = schema["oneOf"]
        assert len(branches) == 3  # two tools + the final-answer branch
        search = next(b for b in branches if b["properties"]["tool"].get("const") == "web_search")
        assert search["properties"]["arguments"]["required"] == ["query"]

    def test_each_branch_pins_its_own_argument_schema(self) -> None:
        branches = tool_envelope_schema(TOOLS, per_tool=True)["oneOf"]
        code = next(b for b in branches if b["properties"]["tool"].get("const") == "code")
        assert code["properties"]["arguments"]["required"] == ["source"]

    def test_answer_branch_requires_final_answer(self) -> None:
        branches = tool_envelope_schema(TOOLS, per_tool=True)["oneOf"]
        answer = next(b for b in branches if b["properties"]["tool"].get("const") == NO_TOOL)
        assert "final_answer" in answer["required"]

    def test_accepts_bare_name_parameters_dicts(self) -> None:
        schema = tool_envelope_schema(
            [{"name": "plain", "parameters": {"type": "object"}}], per_tool=False
        )
        assert "plain" in schema["properties"]["tool"]["enum"]

    def test_empty_tool_list_still_produces_valid_schema(self) -> None:
        schema = tool_envelope_schema([], per_tool=False)
        assert schema["properties"]["tool"]["enum"] == [NO_TOOL]

    def test_empty_tool_list_per_tool_keeps_the_answer_branch(self) -> None:
        assert len(tool_envelope_schema([], per_tool=True)["oneOf"]) == 1


class TestGbnf:
    def test_contains_a_root_rule(self) -> None:
        assert "root ::=" in gbnf_for_tools(TOOLS)

    def test_every_tool_name_appears_as_a_terminal(self) -> None:
        grammar = gbnf_for_tools(TOOLS)
        assert "web_search" in grammar
        assert "code" in grammar

    def test_answer_branch_is_reachable_from_root(self) -> None:
        grammar = gbnf_for_tools(TOOLS)
        root = next(ln for ln in grammar.splitlines() if ln.startswith("root ::="))
        assert "answer" in root

    def test_flat_mode_emits_a_toolname_alternation(self) -> None:
        grammar = gbnf_for_tools(TOOLS, per_tool=False)
        assert "toolname ::=" in grammar

    def test_empty_tools_degrades_to_generic_object(self) -> None:
        assert "root ::=" in gbnf_for_tools([])


class TestParseEnvelope:
    def test_tool_call_envelope(self) -> None:
        calls, answer = parse_envelope('{"tool": "web_search", "arguments": {"query": "x"}}', KNOWN)
        assert answer == ""
        assert calls[0]["name"] == "web_search"
        assert calls[0]["args"] == {"query": "x"}

    def test_final_answer_envelope(self) -> None:
        calls, answer = parse_envelope('{"tool": "none", "final_answer": "Paris"}', KNOWN)
        assert calls == []
        assert answer == "Paris"

    def test_survives_a_stray_code_fence(self) -> None:
        calls, _ = parse_envelope(
            '```json\n{"tool": "code", "arguments": {"source": "1"}}\n```', KNOWN
        )
        assert calls[0]["name"] == "code"

    def test_double_encoded_arguments(self) -> None:
        calls, _ = parse_envelope('{"tool": "code", "arguments": "{\\"source\\": \\"2\\"}"}', KNOWN)
        assert calls[0]["args"] == {"source": "2"}

    def test_falls_back_to_salvage_for_non_envelope_output(self) -> None:
        # A provider that ignored the constraint still has to be handled.
        calls, _ = parse_envelope('{"name": "code", "arguments": {"source": "3"}}', KNOWN)
        assert calls[0]["name"] == "code"

    def test_unknown_tool_is_not_returned_as_a_call(self) -> None:
        calls, answer = parse_envelope('{"tool": "evil", "arguments": {}}', KNOWN)
        assert calls == []
        assert answer  # falls through to being treated as text

    def test_plain_text_is_returned_as_the_answer(self) -> None:
        calls, answer = parse_envelope("The answer is Paris.", KNOWN)
        assert calls == []
        assert answer == "The answer is Paris."

    def test_empty_input(self) -> None:
        assert parse_envelope("", KNOWN) == ([], "")


class _FakeOllama:
    """Minimal stand-in exposing the `.bind()` surface used by apply_constraint."""

    def __init__(self) -> None:
        self.bound: dict = {}

    def bind(self, **kwargs):
        self.bound = kwargs
        return self


class _NoBind:
    """A client whose bind() fails — the constraint must degrade, not raise."""

    def bind(self, **kwargs):
        raise RuntimeError("unsupported")


class TestApplyConstraint:
    def test_ollama_detected_by_class_name(self) -> None:
        assert supports_constrained_decoding(_FakeOllama()) == "ollama"

    def test_llamacpp_detected_by_port(self) -> None:
        class Client:
            base_url = "http://localhost:8080/v1"

        assert supports_constrained_decoding(Client()) == "grammar"

    def test_unknown_provider_reports_no_channel(self) -> None:
        assert supports_constrained_decoding(object()) == ""

    def test_ollama_binds_a_json_schema(self) -> None:
        llm = _FakeOllama()
        apply_constraint(llm, TOOLS, channel="ollama")
        assert "oneOf" in llm.bound["format"]

    def test_flat_schema_can_be_requested(self) -> None:
        llm = _FakeOllama()
        apply_constraint(llm, TOOLS, channel="ollama", per_tool=False)
        assert "tool" in llm.bound["format"]["properties"]

    def test_grammar_channel_binds_gbnf(self) -> None:
        llm = _FakeOllama()
        apply_constraint(llm, TOOLS, channel="grammar")
        assert "root ::=" in llm.bound["extra_body"]["grammar"]

    def test_unbindable_client_is_returned_unchanged(self) -> None:
        llm = _NoBind()
        assert apply_constraint(llm, TOOLS, channel="ollama") is llm

    def test_no_channel_is_a_passthrough(self) -> None:
        obj = object()
        assert apply_constraint(obj, TOOLS) is obj
