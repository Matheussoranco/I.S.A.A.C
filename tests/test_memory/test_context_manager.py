"""Tests for the context window manager."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from isaac.memory.context_manager import (
    _estimate_tokens,
    _extract_text,
    _summarise_extractive,
    compress_messages,
    summarise_messages,
)


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert _estimate_tokens([]) == 0

    def test_rough_heuristic(self) -> None:
        msgs = [HumanMessage(content="hello world")]  # 11 chars → ~2 tokens
        tokens = _estimate_tokens(msgs)
        assert tokens >= 1


class TestExtractText:
    def test_string_content(self) -> None:
        msg = HumanMessage(content="hello")
        assert _extract_text(msg) == "hello"

    def test_multimodal_content(self) -> None:
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": "..."},
                {"type": "text", "text": "second"},
            ]
        )
        assert "first" in _extract_text(msg)
        assert "second" in _extract_text(msg)


class TestSummariseExtractive:
    def test_concatenates_excerpts(self) -> None:
        msgs = [
            HumanMessage(content="Write a function"),
            AIMessage(content="Here is the code"),
        ]
        summary = _summarise_extractive(msgs)
        assert "Write a function" in summary
        assert "Here is the code" in summary

    def test_truncates_long_messages(self) -> None:
        long_text = "x" * 500
        msgs = [HumanMessage(content=long_text)]
        summary = _summarise_extractive(msgs)
        assert "..." in summary
        assert len(summary) < 500


class TestSummariseMessages:
    def test_extractive_without_llm(self) -> None:
        msgs = [HumanMessage(content="Hello")]
        result = summarise_messages(msgs, llm=None)
        assert "Hello" in result

    def test_abstractive_with_mock_llm(self) -> None:
        class _MockLLM:
            def invoke(self, messages):
                class _R:
                    content = "Summary: user asked for hello."
                return _R()

        msgs = [HumanMessage(content="Hello")]
        result = summarise_messages(msgs, llm=_MockLLM())
        assert "Summary" in result


class TestCompressMessages:
    def test_under_threshold_no_change(self) -> None:
        msgs = [HumanMessage(content=f"msg {i}") for i in range(5)]
        result = compress_messages(msgs, max_messages=10, keep_recent=3)
        assert result == msgs

    def test_over_threshold_compresses(self) -> None:
        msgs = [HumanMessage(content=f"msg {i}") for i in range(20)]
        result = compress_messages(msgs, max_messages=10, keep_recent=5)
        # Should be: 1 summary + 5 recent = 6
        assert len(result) < 20
        assert len(result) == 6
        assert isinstance(result[0], SystemMessage)
        assert "[Context Summary]" in result[0].content

    def test_preserves_system_prefix(self) -> None:
        msgs = [SystemMessage(content="You are an assistant")] + [
            HumanMessage(content=f"msg {i}") for i in range(20)
        ]
        result = compress_messages(msgs, max_messages=10, keep_recent=5)
        # System prefix preserved + summary + 5 recent
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are an assistant"
        assert isinstance(result[1], SystemMessage)  # summary
        assert "[Context Summary]" in result[1].content

    def test_exact_threshold_no_change(self) -> None:
        msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
        result = compress_messages(msgs, max_messages=10, keep_recent=5)
        assert result == msgs

    def test_few_non_system_messages_no_crash(self) -> None:
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
        ]
        # keep_recent >= non-system count → no compression
        result = compress_messages(msgs, max_messages=1, keep_recent=5)
        assert result == msgs
