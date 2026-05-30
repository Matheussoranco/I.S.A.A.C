"""BrowserTool behaviour — schema, validation, and graceful degradation."""

from __future__ import annotations

import importlib.util

import pytest

from isaac.tools.browser import BrowserTool

_HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None


@pytest.mark.asyncio
async def test_missing_action_errors() -> None:
    result = await BrowserTool().execute()
    assert result.success is False
    assert "action" in result.error.lower()


@pytest.mark.asyncio
@pytest.mark.skipif(_HAS_PLAYWRIGHT, reason="Playwright is installed in this env")
async def test_graceful_without_playwright() -> None:
    # When Playwright is absent, the tool must report it cleanly, not crash.
    result = await BrowserTool().execute(action="navigate", url="https://example.com")
    assert result.success is False
    assert "playwright" in result.error.lower()


def test_browser_schema_enumerates_actions() -> None:
    schema = BrowserTool().to_function_schema()["function"]["parameters"]
    actions = schema["properties"]["action"]["enum"]
    for expected in ("navigate", "extract_text", "click", "type", "screenshot"):
        assert expected in actions


@pytest.mark.asyncio
async def test_aclose_is_safe_before_launch() -> None:
    # Closing a never-launched session must not raise.
    await BrowserTool().aclose()
