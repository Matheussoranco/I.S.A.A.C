import asyncio
import pytest
from isaac.tools.browser import BrowserTool
from isaac.interfaces.browser_manager import browser_manager

@pytest.mark.asyncio
async def test_browser_automation_e2e():
    """
    End-to-end test: navigate to example.com, extract title, and take a screenshot.
    """
    tool = BrowserTool()
    
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
    
    await browser_manager.shutdown()
