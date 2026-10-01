from __future__ import annotations

import asyncio
import ipaddress
import logging
import shutil
import socket
import tempfile
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
            "regex": {
                "type": "string",
                "description": "Regex for 'extract' to filter text content.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, engine: str | None = None, channel: str | None = None) -> None:
        aliases = {"chrome": ("chromium", "chrome"), "edge": ("chromium", "msedge")}
        requested = (engine or "chromium").lower()
        if requested in aliases:
            selected, alias_channel = aliases[requested]
            if channel is not None:
                raise ValueError(f"Browser {requested!r} does not accept a channel override.")
            channel = alias_channel
        elif requested in {"chromium", "firefox", "webkit"}:
            selected = requested
        else:
            raise ValueError(f"Unsupported browser engine: {requested}")
        if requested == "chromium" and channel == "edge":
            channel = "msedge"
        if channel not in {None, "chrome", "msedge", "chromium"}:
            raise ValueError(f"Unsupported browser channel: {channel}")
        if selected != "chromium" and channel is not None:
            raise ValueError("Browser channels are only supported by Chromium.")
        self.engine = selected
        self.channel = channel
        self._pw: Any = None
        self._context: Any = None
        self._page: Any = None
        self._profile: Any = None

    async def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        from playwright.async_api import async_playwright

        self._profile = __import__("pathlib").Path(tempfile.mkdtemp(prefix="isaac-browser-"))
        try:
            self._pw = await async_playwright().start()
            options: dict[str, Any] = {
                "service_workers": "block",
                "accept_downloads": False,
            }
            if self.channel is not None:
                options["channel"] = self.channel
            if self.engine == "chromium":
                options["chromium_sandbox"] = True
            browser = getattr(self._pw, self.engine)
            self._context = await browser.launch_persistent_context(str(self._profile), **options)
            await self._context.route("**/*", self._route_request)
            await self._context.route_web_socket("**/*", self._block_websocket)
            pages = self._context.pages
            self._page = pages[0] if pages else await self._context.new_page()
            return self._page
        except Exception:
            await self.aclose()
            raise

    async def _navigation_url(self, url: str) -> str:
        value = url.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("Invalid URL")
        if "://" not in value:
            value = f"https://{value}"
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only public HTTP and HTTPS URLs are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URLs containing credentials are not allowed")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            addresses = socket.getaddrinfo(parsed.hostname, port)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not resolve URL host: {parsed.hostname}") from exc
        for address in addresses:
            ip = ipaddress.ip_address(str(address[4][0]).split("%", 1)[0])
            if not ip.is_global:
                raise ValueError(f"private or local address is blocked: {ip}")
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc,
                parsed.path or "/",
                parsed.query,
                parsed.fragment,
            )
        )

    async def _route_request(self, route: Any) -> None:
        try:
            await self._navigation_url(route.request.url)
            response = await route.fetch(max_redirects=0)
            await route.fulfill(response=response)
            await response.dispose()
        except ValueError:
            await route.abort("blockedbyclient")
        except Exception:
            await route.abort("failed")

    async def _block_websocket(self, route: Any) -> None:
        await route.close()

    async def aclose(self) -> None:
        context, playwright, profile = self._context, self._pw, self._profile
        self._context = self._pw = self._page = self._profile = None
        if context is not None:
            await context.close()
        if playwright is not None:
            await playwright.stop()
        if profile is not None:
            shutil.rmtree(profile, ignore_errors=True)

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = (kwargs.get("action") or "").strip()
        if not action:
            return ToolResult(success=False, error="Missing 'action' parameter.")

        try:
            if action == "navigate":
                kwargs["url"] = await self._navigation_url(kwargs.get("url", ""))
            page = await self._ensure_page()

            return await self._dispatch(page, action, kwargs)
        except Exception as exc:
            logger.error("BrowserTool action '%s' failed: %s", action, exc)
            if "executable" in str(exc).lower() or "browser" in str(exc).lower():
                return ToolResult(
                    success=False,
                    error=(
                        f"{action} failed: {exc}. Install the selected browser with "
                        f"playwright install {self.engine}."
                    ),
                )
            return ToolResult(success=False, error=f"{action} failed: {exc}")

    async def _dispatch(self, page: Any, action: str, kwargs: dict[str, Any]) -> ToolResult:
        if action == "navigate":
            url = kwargs.get("url")
            if not url:
                return ToolResult(success=False, error="URL is required for navigate.")
            await page.goto(url, wait_until="domcontentloaded")
            return ToolResult(
                success=True, output=f"Navigated to {page.url}. Title: {await page.title()}"
            )

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
            return ToolResult(
                success=True, output=f"Screenshot saved to {path}", metadata={"path": path}
            )

        if action == "pdf":
            path = f"page_{int(asyncio.get_event_loop().time())}.pdf"
            await page.pdf(path=path)
            return ToolResult(success=True, output=f"PDF saved to {path}", metadata={"path": path})

        return ToolResult(success=False, error=f"Unknown action: {action}")
