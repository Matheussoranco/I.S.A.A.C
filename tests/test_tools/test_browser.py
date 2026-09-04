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


def test_browser_mutating_actions_require_approval() -> None:
    browser = BrowserTool()

    assert browser.approval_required(action="extract_text") is False
    assert browser.approval_required(action="click", selector="#submit") is True
    assert browser.approval_required(action="eval", script="document.body.innerHTML='x'") is True
    assert browser.effective_risk_level(action="click") == 4


@pytest.mark.asyncio
async def test_aclose_is_safe_before_launch() -> None:
    # Closing a never-launched session must not raise.
    await BrowserTool().aclose()


class _Locator:
    @property
    def first(self) -> _Locator:
        return self

    async def bounding_box(self, timeout: int) -> dict[str, float]:
        assert timeout == 8000
        return {"x": 100, "y": 50, "width": 80, "height": 30}


class _VisualPage:
    url = "https://example.com/next"

    def locator(self, selector: str) -> _Locator:
        assert selector == "#continue"
        return _Locator()

    async def click(self, selector: str, timeout: int) -> None:
        assert selector == "#continue"
        assert timeout == 8000

    async def wait_for_timeout(self, timeout: int) -> None:
        assert timeout == 180

    async def screenshot(self, **kwargs: object) -> bytes:
        assert kwargs == {"type": "png", "full_page": False}
        return b"png"

    async def title(self) -> str:
        return "Example"


@pytest.mark.asyncio
async def test_visual_callback_receives_cursor_and_frame() -> None:
    events: list[tuple[str, dict]] = []
    browser = BrowserTool(visual_callback=lambda kind, data: events.append((kind, data)))

    result = await browser._dispatch(_VisualPage(), "click", {"selector": "#continue"})

    assert result.success is True
    assert [kind for kind, _ in events] == ["browser_cursor", "browser_frame"]
    cursor = events[0][1]
    assert cursor["x"] == 140
    assert cursor["y"] == 65
    assert cursor["action"] == "click"
    frame = events[1][1]
    assert frame["image_base64"] == "cG5n"
    assert frame["url"] == "https://example.com/next"
