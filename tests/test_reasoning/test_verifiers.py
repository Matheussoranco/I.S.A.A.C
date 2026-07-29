"""Tests for the cheap verifiers used by best-of-N."""

from __future__ import annotations

from isaac.reasoning.verifiers import (
    all_of,
    json_verifier,
    non_empty_verifier,
    numeric_range_verifier,
    python_syntax_verifier,
    regex_verifier,
    schema_verifier,
)


class TestNonEmpty:
    def test_accepts_text(self) -> None:
        assert non_empty_verifier()("hello") == 1.0

    def test_rejects_whitespace_only(self) -> None:
        assert non_empty_verifier()("   \n ") == 0.0

    def test_rejects_none(self) -> None:
        assert non_empty_verifier()(None) == 0.0

    def test_min_chars_rejects_truncated_output(self) -> None:
        assert non_empty_verifier(min_chars=10)("short") == 0.0


class TestPythonSyntax:
    def test_accepts_valid_code(self) -> None:
        assert python_syntax_verifier()("x = 1\nprint(x)") == 1.0

    def test_rejects_syntax_error(self) -> None:
        assert python_syntax_verifier()("def broken(:") == 0.0

    def test_strips_markdown_fence(self) -> None:
        assert python_syntax_verifier()("```python\nx = 1\n```") == 1.0

    def test_require_expr_rejects_statements(self) -> None:
        assert python_syntax_verifier(require_expr=True)("x = 1") == 0.0

    def test_require_expr_accepts_an_expression(self) -> None:
        assert python_syntax_verifier(require_expr=True)("1 + 2") == 1.0

    def test_empty_is_rejected(self) -> None:
        assert python_syntax_verifier()("") == 0.0

    def test_reads_message_content_attribute(self) -> None:
        class Msg:
            content = "y = 2"

        assert python_syntax_verifier()(Msg()) == 1.0


class TestJsonVerifier:
    def test_accepts_valid_json(self) -> None:
        assert json_verifier()('{"a": 1}') == 1.0

    def test_rejects_invalid_json(self) -> None:
        assert json_verifier()("{not json") == 0.0

    def test_strips_fence(self) -> None:
        assert json_verifier()('```json\n{"a": 1}\n```') == 1.0

    def test_all_required_keys_present(self) -> None:
        assert json_verifier(["a", "b"])('{"a": 1, "b": 2}') == 1.0

    def test_partial_keys_score_between_zero_and_one(self) -> None:
        score = json_verifier(["a", "b"])('{"a": 1}')
        assert 0.0 < score < 1.0

    def test_missing_all_keys_scores_zero(self) -> None:
        assert json_verifier(["a", "b"])('{"z": 1}') == 0.0


class TestSchemaVerifier:
    SCHEMA = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def test_accepts_conforming_object(self) -> None:
        assert schema_verifier(self.SCHEMA)('{"name": "x"}') == 1.0

    def test_rejects_non_json(self) -> None:
        assert schema_verifier(self.SCHEMA)("nope") == 0.0

    def test_missing_required_key_is_rejected(self) -> None:
        # Whether jsonschema is installed or not, a missing required key must
        # not score a full pass.
        assert schema_verifier(self.SCHEMA)('{"other": 1}') < 1.0


class TestNumericRange:
    def test_accepts_in_range(self) -> None:
        assert numeric_range_verifier(0, 100)("The answer is 42") == 1.0

    def test_rejects_out_of_range(self) -> None:
        assert numeric_range_verifier(0, 10)("The answer is 42") == 0.0

    def test_rejects_when_no_number_present(self) -> None:
        assert numeric_range_verifier(0, 10)("no digits here") == 0.0

    def test_uses_the_last_number_in_chain_of_thought(self) -> None:
        # "2 + 3 = 5" — the result is the final figure, not the operands.
        assert numeric_range_verifier(4, 6)("2 + 3 = 5") == 1.0

    def test_handles_thousands_separators(self) -> None:
        assert numeric_range_verifier(1000, 2000)("total: 1,500") == 1.0

    def test_open_ended_bounds(self) -> None:
        assert numeric_range_verifier(low=0)("999999") == 1.0


class TestRegexVerifier:
    def test_match(self) -> None:
        assert regex_verifier(r"\bParis\b")("The capital is Paris.") == 1.0

    def test_no_match(self) -> None:
        assert regex_verifier(r"\bParis\b")("The capital is Lyon.") == 0.0


class TestAllOf:
    def test_averages_component_scores(self) -> None:
        combined = all_of(lambda v: 1.0, lambda v: 0.0)
        assert combined("x") == 0.5

    def test_unanimous_pass(self) -> None:
        combined = all_of(non_empty_verifier(), python_syntax_verifier())
        assert combined("x = 1") == 1.0

    def test_empty_combination_scores_zero(self) -> None:
        assert all_of()("x") == 0.0

    def test_a_raising_component_contributes_zero(self) -> None:
        def boom(_v):
            raise RuntimeError("bad")

        assert all_of(lambda v: 1.0, boom)("x") == 0.5
