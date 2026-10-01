from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from isaac.tools.browser import BrowserTool


@pytest.mark.asyncio
async def test_browser_automation_actions(monkeypatch):
    """
    End-to-end test: navigate to example.com, extract title, and take a screenshot.
    """
    page = SimpleNamespace(
        url="https://example.com",
        goto=AsyncMock(),
        title=AsyncMock(return_value="Example Domain"),
        inner_text=AsyncMock(return_value="Example Domain"),
        screenshot=AsyncMock(),
    )
    tool = BrowserTool()
    monkeypatch.setattr(tool, "_navigation_url", AsyncMock(return_value="https://example.com"))
    monkeypatch.setattr(tool, "_ensure_page", AsyncMock(return_value=page))

    # 1. Navigate
    nav_result = await tool.execute(action="navigate", url="https://example.com")
    assert nav_result.success is True
    assert "Example Domain" in nav_result.output

    # 2. Extract
    extract_result = await tool.execute(action="extract", selector="h1")
    assert extract_result.success is True
    assert "Example Domain" in extract_result.output

    # 3. Screenshot
    shot_result = await tool.execute(action="screenshot")
    assert shot_result.success is True
    assert "screenshot" in shot_result.output
