"""Real-desktop tools are bounded, observable, and approval-gated."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import isaac.tools.desktop as desktop_module
from isaac.tools.desktop import (
    ComputerControlTool,
    ComputerDescribeTool,
    ComputerViewTool,
    ScreenFrame,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.actions: list[tuple[str, dict[str, Any]]] = []

    def capture(self) -> ScreenFrame:
        return ScreenFrame(
            png=b"png",
            left=0,
            top=0,
            width=1920,
            height=1080,
            cursor_x=400,
            cursor_y=300,
        )

    def perform(self, action: str, args: dict[str, Any]) -> None:
        self.actions.append((action, args))


def _patch_desktop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeBackend:
    backend = _FakeBackend()
    monkeypatch.setattr(desktop_module, "get_desktop_backend", lambda: backend)
    monkeypatch.setattr(desktop_module, "_save_frame", lambda png: tmp_path / "screen.png")
    return backend


@pytest.mark.asyncio
async def test_computer_view_publishes_local_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_desktop(monkeypatch, tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []
    tool = ComputerViewTool(visual_callback=lambda kind, data: events.append((kind, data)))

    result = await tool.execute()

    assert result.success is True
    assert result.metadata["width"] == 1920
    assert events[0][0] == "desktop_frame"
    assert events[0][1]["image_base64"] == "cG5n"
    assert tool.risk_level == 2
    assert tool.requires_approval is False


@pytest.mark.asyncio
async def test_computer_control_is_gated_and_emits_resulting_cursor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = _patch_desktop(monkeypatch, tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []
    tool = ComputerControlTool(visual_callback=lambda kind, data: events.append((kind, data)))

    result = await tool.execute(action="click", x=400, y=300)

    assert result.success is True
    assert backend.actions == [("click", {"action": "click", "x": 400, "y": 300})]
    assert tool.risk_level == 4
    assert tool.requires_approval is True
    assert [kind for kind, _ in events] == ["desktop_frame", "desktop_cursor"]


@pytest.mark.asyncio
async def test_computer_describe_uses_vision_only_after_tool_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_desktop(monkeypatch, tmp_path)

    class _Vision:
        def ask(self, question: str, image: bytes) -> str:
            assert question == "What is open?"
            assert image == b"png"
            return "A settings window is open."

    from isaac.multimodal.vision import vision_lm

    monkeypatch.setattr(vision_lm, "get_vision_lm", lambda: _Vision())
    tool = ComputerDescribeTool()

    result = await tool.execute(question="What is open?")

    assert result.success is True
    assert "settings window" in result.output
    assert tool.requires_approval is True


def test_control_schema_contains_only_bounded_actions() -> None:
    schema = ComputerControlTool().to_function_schema()["function"]["parameters"]
    assert schema["properties"]["action"]["enum"] == [
        "move",
        "click",
        "double_click",
        "type",
        "press",
        "hotkey",
        "scroll",
        "wait",
    ]
    assert "script" not in schema["properties"]


def test_responses_action_translates_virtual_desktop_coordinates(monkeypatch) -> None:
    calls: list[tuple[Any, ...]] = []

    class _Gui:
        KEYBOARD_KEYS = ["ctrl", "a", "shift"]

        def moveTo(self, *args, **kwargs):
            calls.append(("move", *args))

        def click(self, **kwargs):
            calls.append(("click", kwargs["button"]))

        def keyDown(self, key):
            calls.append(("down", key))

        def keyUp(self, key):
            calls.append(("up", key))

    backend = desktop_module.DesktopBackend()
    monkeypatch.setattr(backend, "_pyautogui", lambda: _Gui())
    monkeypatch.setattr(backend, "bounds", lambda: (-1920, 0, 3840, 1080))
    frame = ScreenFrame(b"png", -1920, 0, 3840, 1080, 0, 0)

    backend.perform_computer_action(
        {"type": "click", "x": 100, "y": 80, "button": "left", "keys": ["SHIFT"]},
        frame,
    )

    assert calls == [
        ("down", "shift"),
        ("move", -1820, 80),
        ("click", "left"),
        ("up", "shift"),
    ]


def test_responses_action_rejects_out_of_frame_point(monkeypatch) -> None:
    backend = desktop_module.DesktopBackend()
    monkeypatch.setattr(backend, "_pyautogui", lambda: object())
    frame = ScreenFrame(b"png", 0, 0, 1280, 720, 0, 0)

    with pytest.raises(ValueError, match="outside the screenshot"):
        backend.perform_computer_action({"type": "move", "x": 2000, "y": 1}, frame)
