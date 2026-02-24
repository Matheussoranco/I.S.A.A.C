"""Browser Tool — Playwright-based web automation inside Docker.

All browser operations run inside the Docker sandbox container,
never on the host machine.  Network is allowed only within the
browser container.
"""

from __future__ import annotations

import logging
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)


class BrowserTool(IsaacTool):
    """Web browser automation via Playwright inside Docker.

    Methods: navigate, extract_text, click, screenshot.
    Network is allowed only in the browser container.
    """

    name = "browser"
    description = "Web browser automation: navigate URLs, extract text, click elements, take screenshots."
    risk_level = 3
    requires_approval = False
    sandbox_required = True

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a browser action.

        Parameters
        ----------
        action:
            One of ``"navigate"``, ``"extract_text"``, ``"click"``, ``"screenshot"``.
        url:
            URL for navigate action.
        selector:
            CSS selector for click action.
        """
        action = kwargs.get("action", "screenshot")

        if action == "navigate":
            return await self._navigate(kwargs.get("url", ""))
        elif action == "extract_text":
            return await self._extract_text()
        elif action == "click":
            return await self._click(kwargs.get("selector", ""))
        elif action == "screenshot":
            return await self._screenshot()
        else:
            return ToolResult(success=False, error=f"Unknown browser action: {action}")

    async def _navigate(self, url: str) -> ToolResult:
        """Navigate to a URL inside the sandbox browser."""
        if not url:
            return ToolResult(success=False, error="No URL provided.")

        code = f"""
from playwright.sync_api import sync_playwright
import json

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page()
    page.goto("{url}", wait_until="domcontentloaded", timeout=15000)
    result = {{"title": page.title(), "url": page.url}}
    print(json.dumps(result))
    browser.close()
"""
        return await self._run_in_sandbox(code)

    async def _extract_text(self) -> ToolResult:
        """Extract visible text from the current page."""
        code = """
from playwright.sync_api import sync_playwright
import json

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page()
    text = page.inner_text('body')
    print(text[:5000])
    browser.close()
"""
        return await self._run_in_sandbox(code)

    async def _click(self, selector: str) -> ToolResult:
        """Click an element by CSS selector."""
        if not selector:
            return ToolResult(success=False, error="No selector provided.")

        code = f"""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page()
    page.click("{selector}", timeout=5000)
    print("Clicked: {selector}")
    browser.close()
"""
        return await self._run_in_sandbox(code)

    async def _screenshot(self) -> ToolResult:
        """Take a screenshot of the current page."""
        code = """
import base64
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page()
    screenshot = page.screenshot()
    print(base64.b64encode(screenshot).decode())
    browser.close()
"""
        return await self._run_in_sandbox(code)

    async def _run_in_sandbox(self, code: str) -> ToolResult:
        """Execute Python code in the sandbox container."""
        try:
            from isaac.sandbox.executor import CodeExecutor

            executor = CodeExecutor()
            try:
                result = executor.execute(code)
            finally:
                executor.close()

            return ToolResult(
                success=result.exit_code == 0,
                output=result.stdout,
                error=result.stderr,
            )
        except Exception as exc:
            logger.error("Browser tool sandbox execution failed: %s", exc)
            return ToolResult(success=False, error=str(exc))
