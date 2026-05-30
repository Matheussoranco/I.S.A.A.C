"""Every built-in tool must expose a valid function-calling schema."""

from __future__ import annotations

import pytest

from isaac.tools import register_all_tools
from isaac.tools.base import get_tool_registry


@pytest.fixture(scope="module")
def tools() -> list:
    register_all_tools()
    return get_tool_registry().list_all()


def test_all_tools_have_function_schema(tools: list) -> None:
    assert tools, "expected built-in tools to be registered"
    for tool in tools:
        schema = tool.to_function_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == tool.name
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        # 'required' keys, if present, must exist in properties.
        for req in params.get("required", []):
            assert req in params["properties"], f"{tool.name}: required '{req}' not in properties"


def test_to_schema_includes_parameters(tools: list) -> None:
    for tool in tools:
        assert "parameters" in tool.to_schema()


@pytest.mark.parametrize(
    "name,required",
    [
        ("file_read", ["path"]),
        ("file_write", ["path", "content"]),
        ("web_search", ["query"]),
        ("browser", ["action"]),
        ("email_send", ["to", "body"]),
    ],
)
def test_specific_required_params(tools: list, name: str, required: list[str]) -> None:
    by_name = {t.name: t for t in tools}
    if name not in by_name:
        pytest.skip(f"{name} not registered in this environment")
    params = by_name[name].to_function_schema()["function"]["parameters"]
    assert params.get("required") == required
