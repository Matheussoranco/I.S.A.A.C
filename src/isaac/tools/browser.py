from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from isaac.interfaces.browser_manager import browser_manager
from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)

class BrowserTool(IsaacTool):
    name = "browser"
    description = (
        "Real-time web browser automation. Use these tools to navigate URLs, "
        "click elements, type text, extract content, and capture screenshots."
    )
    risk_level = 3
    requires_approval = False
    sandbox_required = False
    
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate",
                    "click",
                    "type",
                    "extract",
                    "screenshot",
                    "pdf",
                ],
                "description": "The browser action to perform.",
            },
            "url": {"type": "string", "description": "URL for 'navigate'."},
            "selector": {
                "type": "string", 
                "description": "CSS selector for 'click', 'type', or 'extract'.",
            },
            "text": {"type": "string", "description": "Text to type for 'type'."},
            "regex": {"type": "string", "description": "Regex for 'extract' to filter text content."},
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = (kwargs.get("action") or "").strip()
        if not action:
            return ToolResult(success=False, error="Missing 'action' parameter.")
        
        try:
            # Use default session; could be expanded to use a session_id from kwargs
            context = await browser_manager.get_context("default")
            page = (context.pages[0] if context.pages else await context.new_page())
            
            return await self._dispatch(page, action, kwargs)
        except Exception as exc:
            logger.error("BrowserTool action '%s' failed: %s", action, exc)
            return ToolResult(success=False, error=f"{action} failed: {exc}")

    async def _dispatch(self, page: Any, action: str, kwargs: dict[str, Any]) -> ToolResult:
        if action == "navigate":
            url = kwargs.get("url")
            if not url:
                return ToolResult(success=False, error="URL is required for navigate.")
            await page.goto(url, wait_until="domcontentloaded")
            return ToolResult(success=True, output=f"Navigated to {page.url}. Title: {await page.title()}")

        if action == "click":
            selector = kwargs.get("selector")
            if not selector:
                return ToolResult(success=False, error="Selector is required for click.")
            await page.click(selector)
            return ToolResult(success=True, output=f"Clicked element {selector}.")

        if action == "type":
            selector = kwargs.get("selector")
            text = kwargs.get("text", "")
            if not selector:
                return ToolResult(success=False, error="Selector is required for type.")
            await page.fill(selector, text)
            return ToolResult(success=True, output=f"Typed '{text}' into {selector}.")

        if action == "extract":
            selector = kwargs.get("selector")
            if not selector:
                return ToolResult(success=False, error="Selector is required for extract.")
            content = await page.inner_text(selector)
            
            regex = kwargs.get("regex")
            if regex:
                import re
                match = re.search(regex, content)
                content = match.group(0) if match else "No regex match found."
                
            return ToolResult(success=True, output=content)

        if action == "screenshot":
            path = f"screenshot_{int(asyncio.get_event_loop().time())}.png"
            await page.screenshot(path=path)
            return ToolResult(success=True, output=f"Screenshot saved to {path}", metadata={"path": path})

        if action == "pdf":
            path = f"page_{int(asyncio.get_event_loop().time())}.pdf"
            await page.pdf(path=path)
            return ToolResult(success=True, output=f"PDF saved to {path}", metadata={"path": path})

        return ToolResult(success=False, error=f"Unknown action: {action}")
