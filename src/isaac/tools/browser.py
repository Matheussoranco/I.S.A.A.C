"""Browser Tool — persistent Playwright browser session for web automation.

Unlike a stateless fetch, this tool keeps a single Chromium page **alive
across actions** so the agent can drive a real multi-step browsing session:
navigate → read → click → type → navigate again, all on the same page with
cookies, history and DOM state preserved (the "Claude for Chrome" capability).

The browser runs in-process via Playwright's async API.  A single page is
lazily launched on first use and reused for every subsequent action within
the same agent run; the agent loop calls :meth:`aclose` when the run ends.

Requires the optional ``browser`` extra::

    pip install playwright && python -m playwright install chromium
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)

_MAX_TEXT = 8000
_DEFAULT_TIMEOUT_MS = 20000


class BrowserTool(IsaacTool):
    """Drive a persistent Chromium page via Playwright.

    Actions
    -------
    navigate      Go to a URL (``url``).
    extract_text  Return the visible text of the current page.
    get_html      Return the current page's HTML.
    get_links     List the anchors (text → href) on the current page.
    click         Click an element by CSS selector (``selector``).
    type          Fill an input by CSS selector (``selector``, ``text``).
    press         Press a key, e.g. ``Enter`` (``key``).
    eval          Evaluate a JavaScript expression (``script``).
    screenshot    Save a PNG of the current page and return its path.
    back          Navigate back in history.
    current       Report the current URL and title.
    """

    name = "browser"
    description = (
        "Drive a persistent web browser to accomplish a task: navigate to URLs, read "
        "page text/links, click elements, type into fields, run JavaScript, take "
        "screenshots, and go back. The page (cookies, login, scroll position) is kept "
        "alive across calls, so you can browse step by step like a human."
    )
    risk_level = 3
    requires_approval = False
    # In-process persistent session — not the Docker sandbox (which cannot hold
    # a page open across separate executions).
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate",
                    "extract_text",
                    "get_html",
                    "get_links",
                    "click",
                    "type",
                    "press",
                    "eval",
                    "screenshot",
                    "back",
                    "current",
                ],
                "description": "The browser action to perform.",
            },
            "url": {"type": "string", "description": "URL for the 'navigate' action."},
            "selector": {
                "type": "string",
                "description": "CSS selector for 'click' / 'type'.",
            },
            "text": {"type": "string", "description": "Text to type for the 'type' action."},
            "key": {"type": "string", "description": "Key to press for 'press' (e.g. 'Enter')."},
            "script": {
                "type": "string",
                "description": "JavaScript expression for the 'eval' action.",
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _ensure_page(self) -> Any:
        """Launch the browser/page on first use; reuse it afterwards."""
        if self._page is not None:
            return self._page

        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )
        self._page = await self._context.new_page()
        logger.info("BrowserTool: launched persistent Chromium session.")
        return self._page

    async def aclose(self) -> None:
        """Tear down the browser session (called by the agent loop on finish)."""
        for closer in (
            getattr(self._context, "close", None),
            getattr(self._browser, "close", None),
            getattr(self._pw, "stop", None),
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception as exc:  # pragma: no cover - best-effort teardown
                logger.debug("BrowserTool teardown step failed: %s", exc)
        self._pw = self._browser = self._context = self._page = None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = (kwargs.get("action") or "").strip()
        if not action:
            return ToolResult(success=False, error="Missing 'action' parameter.")

        try:
            import playwright  # noqa: F401
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "Playwright is not installed. Run: pip install playwright && "
                    "python -m playwright install chromium"
                ),
            )

        try:
            page = await self._ensure_page()
        except Exception as exc:
            logger.error("BrowserTool: failed to launch browser: %s", exc)
            return ToolResult(
                success=False,
                error=(
                    f"Could not launch browser ({exc}). Chromium may be missing — run: "
                    "python -m playwright install chromium"
                ),
            )

        try:
            return await self._dispatch(page, action, kwargs)
        except Exception as exc:
            logger.error("BrowserTool action '%s' failed: %s", action, exc)
            return ToolResult(success=False, error=f"{action} failed: {exc}")

    async def _dispatch(self, page: Any, action: str, kwargs: dict[str, Any]) -> ToolResult:
        if action == "navigate":
            url = kwargs.get("url", "")
            if not url:
                return ToolResult(success=False, error="No URL provided.")
            if "://" not in url:
                url = "https://" + url
            await page.goto(url, wait_until="domcontentloaded", timeout=_DEFAULT_TIMEOUT_MS)
            title = await page.title()
            return ToolResult(
                success=True,
                output=f"Loaded '{title}' at {page.url}",
                metadata={"url": page.url, "title": title},
            )

        if action == "extract_text":
            text = await page.inner_text("body")
            return ToolResult(success=True, output=_truncate(text))

        if action == "get_html":
            html = await page.content()
            return ToolResult(success=True, output=_truncate(html))

        if action == "get_links":
            links = await page.eval_on_selector_all(
                "a[href]",
                "els => els.slice(0, 100).map(e => "
                "({text: (e.innerText||'').trim().slice(0,80), href: e.href}))",
            )
            lines = [f"- {ln['text'] or '(no text)'} -> {ln['href']}" for ln in links if ln["href"]]
            return ToolResult(success=True, output="\n".join(lines) or "(no links found)")

        if action == "click":
            selector = kwargs.get("selector", "")
            if not selector:
                return ToolResult(success=False, error="No selector provided.")
            await page.click(selector, timeout=8000)
            return ToolResult(success=True, output=f"Clicked '{selector}'. Now at {page.url}")

        if action == "type":
            selector = kwargs.get("selector", "")
            text = kwargs.get("text", "")
            if not selector:
                return ToolResult(success=False, error="No selector provided.")
            await page.fill(selector, text, timeout=8000)
            return ToolResult(success=True, output=f"Typed into '{selector}'.")

        if action == "press":
            key = kwargs.get("key", "Enter")
            await page.keyboard.press(key)
            return ToolResult(success=True, output=f"Pressed '{key}'. Now at {page.url}")

        if action == "eval":
            script = kwargs.get("script", "")
            if not script:
                return ToolResult(success=False, error="No script provided.")
            result = await page.evaluate(script)
            return ToolResult(success=True, output=_truncate(str(result)))

        if action == "screenshot":
            out_dir = _screenshot_dir()
            count = len(list(out_dir.glob("shot_*.png")))
            path = out_dir / f"shot_{count:03d}.png"
            await page.screenshot(path=str(path), full_page=False)
            return ToolResult(
                success=True,
                output=f"Saved screenshot to {path}",
                metadata={"path": str(path)},
            )

        if action == "back":
            await page.go_back(timeout=_DEFAULT_TIMEOUT_MS)
            return ToolResult(success=True, output=f"Went back. Now at {page.url}")

        if action == "current":
            title = await page.title()
            return ToolResult(
                success=True,
                output=f"Current page: '{title}' at {page.url}",
                metadata={"url": page.url, "title": title},
            )

        return ToolResult(success=False, error=f"Unknown browser action: {action}")


def _truncate(text: str, limit: int = _MAX_TEXT) -> str:
    text = text or ""
    if len(text) > limit:
        return text[:limit] + f"\n\n... truncated at {limit} chars ..."
    return text


def _screenshot_dir() -> Path:
    """Return (and create) the directory where screenshots are saved."""
    try:
        from isaac.config.settings import get_settings

        root = get_settings().isaac_home / "browser"
    except Exception:
        root = Path.home() / ".isaac" / "browser"
    root.mkdir(parents=True, exist_ok=True)
    return root
