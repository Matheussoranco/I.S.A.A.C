"""Tests for the Tool Registry and base tool infrastructure."""

from __future__ import annotations

from typing import Any

import pytest

from isaac.tools.base import IsaacTool, ToolRegistry, ToolResult


class DummyTool(IsaacTool):
    """Minimal tool for testing."""

    name = "dummy"
    description = "A dummy tool for testing"
    risk_level = 1

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="dummy output")


class HighRiskTool(IsaacTool):
    """Tool with risk level >= 4 (auto-requires approval)."""

    name = "high_risk"
    description = "Dangerous tool"
    risk_level = 4

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="executed")


class TestToolResult:
    def test_defaults(self) -> None:
        r = ToolResult()
        assert r.success is False
        assert r.output == ""
        assert r.error == ""
        assert r.metadata == {}


class TestIsaacTool:
    def test_auto_approval_for_high_risk(self) -> None:
        tool = HighRiskTool()
        assert tool.requires_approval is True

    def test_no_auto_approval_for_low_risk(self) -> None:
        tool = DummyTool()
        assert tool.requires_approval is False

    def test_to_schema(self) -> None:
        tool = DummyTool()
        schema = tool.to_schema()
        assert schema["name"] == "dummy"
        assert schema["risk_level"] == 1
        assert schema["requires_approval"] is False

    @pytest.mark.asyncio
    async def test_execute(self) -> None:
        tool = DummyTool()
        result = await tool.execute()
        assert result.success is True
        assert result.output == "dummy output"


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        tool = DummyTool()
        reg.register(tool)
        assert reg.get("dummy") is tool

    def test_get_missing(self) -> None:
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_list_tools(self) -> None:
        reg = ToolRegistry()
        reg.register(DummyTool())
        reg.register(HighRiskTool())
        schemas = reg.list_tools()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"dummy", "high_risk"}

    def test_list_names(self) -> None:
        reg = ToolRegistry()
        reg.register(DummyTool())
        names = reg.list_names()
        assert "dummy" in names

    def test_filter_by_max_risk(self) -> None:
        reg = ToolRegistry()
        reg.register(DummyTool())
        reg.register(HighRiskTool())
        safe = reg.filter_by_max_risk(2)
        assert len(safe) == 1
        assert safe[0].name == "dummy"
