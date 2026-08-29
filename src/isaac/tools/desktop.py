"""Tools for observing and controlling the user's real desktop.

The read-only view tool is separate from the high-risk control and vision tools
so a model cannot turn a harmless screenshot request into an input action.  All
control primitives are bounded and declarative; this module never accepts code
or shell commands.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ctypes
import io
import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import ImageGrab

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)

DesktopVisualCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ScreenFrame:
    png: bytes
    left: int
    top: int
    width: int
    height: int
    cursor_x: int
    cursor_y: int


class DesktopBackend:
    """Small adapter around Pillow capture and PyAutoGUI input primitives."""

    def _pyautogui(self) -> Any:
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError(
                "Desktop control is not installed. Run: pip install -e '.[desktop]'"
            ) from exc
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.08
        return pyautogui

    def bounds(self) -> tuple[int, int, int, int]:
        if os.name == "nt":
            user32 = ctypes.windll.user32
            return (
                int(user32.GetSystemMetrics(76)),
                int(user32.GetSystemMetrics(77)),
                int(user32.GetSystemMetrics(78)),
                int(user32.GetSystemMetrics(79)),
            )
        gui = self._pyautogui()
        size = gui.size()
        return 0, 0, int(size.width), int(size.height)

    def capture(self) -> ScreenFrame:
        left, top, width, height = self.bounds()
        image = ImageGrab.grab(all_screens=os.name == "nt")
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        try:
            point = self._pyautogui().position()
            cursor_x, cursor_y = int(point.x), int(point.y)
        except Exception:
            cursor_x, cursor_y = left + width // 2, top + height // 2
        return ScreenFrame(
            png=buf.getvalue(),
            left=left,
            top=top,
            width=width,
            height=height,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
        )

    def perform(self, action: str, args: dict[str, Any]) -> None:
        gui = self._pyautogui()
        duration = min(2.0, max(0.0, float(args.get("duration", 0.25))))

        if action in {"move", "click", "double_click"}:
            x, y = self._validated_point(args.get("x"), args.get("y"))
            gui.moveTo(x, y, duration=duration)
            if action == "click":
                gui.click(button=_validated_button(args.get("button")))
            elif action == "double_click":
                gui.doubleClick(button=_validated_button(args.get("button")), interval=0.12)
            return

        if action == "type":
            text = str(args.get("text", ""))
            if not text:
                raise ValueError("No text provided.")
            if len(text) > 4000:
                raise ValueError("Text is limited to 4000 characters per action.")
            gui.write(text, interval=min(0.1, max(0.0, float(args.get("interval", 0.01)))))
            return

        if action == "press":
            key = _validated_key(args.get("key"), gui)
            gui.press(key)
            return

        if action == "hotkey":
            raw_keys = args.get("keys")
            if not isinstance(raw_keys, list) or not 2 <= len(raw_keys) <= 4:
                raise ValueError("hotkey requires a list of 2 to 4 keys.")
            gui.hotkey(*[_validated_key(key, gui) for key in raw_keys])
            return

        if action == "scroll":
            amount = int(args.get("amount", 0))
            if not -20 <= amount <= 20 or amount == 0:
                raise ValueError("Scroll amount must be between -20 and 20, excluding zero.")
            gui.scroll(amount)
            return

        if action == "wait":
            seconds = min(10.0, max(0.0, float(args.get("seconds", 1.0))))
            time.sleep(seconds)
            return

        raise ValueError(f"Unsupported desktop action: {action}")

    def perform_computer_action(self, action: dict[str, Any], frame: ScreenFrame) -> None:
        """Execute one Responses-API computer action against a captured frame.

        Model coordinates are relative to the screenshot. Windows coordinates
        are relative to the virtual desktop and can be negative on a monitor to
        the left of the primary display, so every point is translated through
        ``frame.left``/``frame.top`` before input is sent.
        """
        gui = self._pyautogui()
        kind = str(action.get("type", "")).lower().strip()
        keys = action.get("keys") or []

        if kind == "screenshot":
            return
        if kind == "wait":
            time.sleep(min(10.0, max(0.0, float(action.get("seconds", 2.0)))))
            return
        if kind == "type":
            self._type_text(str(action.get("text", "")), gui)
            return
        if kind == "keypress":
            raw_keys = action.get("keys")
            if not isinstance(raw_keys, list) or not raw_keys:
                raise ValueError("keypress requires a non-empty keys list.")
            for raw_key in raw_keys:
                parts = [part for part in str(raw_key).replace("-", "+").split("+") if part]
                normalized = [_validated_key(part, gui) for part in parts]
                if len(normalized) == 1:
                    gui.press(normalized[0])
                else:
                    gui.hotkey(*normalized)
            return

        if kind in {"move", "click", "double_click", "scroll"}:
            x, y = self._frame_point(action.get("x"), action.get("y"), frame)
            with self._held_modifiers(gui, keys):
                gui.moveTo(x, y, duration=0.2)
                if kind == "click":
                    gui.click(button=_validated_button(action.get("button")))
                elif kind == "double_click":
                    gui.doubleClick(button=_validated_button(action.get("button")), interval=0.12)
                elif kind == "scroll":
                    scroll_y = _bounded_scroll_delta(action.get("scroll_y", 0))
                    scroll_x = _bounded_scroll_delta(action.get("scroll_x", 0))
                    if scroll_y:
                        gui.scroll(-_scroll_steps(scroll_y))
                    if scroll_x and hasattr(gui, "hscroll"):
                        gui.hscroll(_scroll_steps(scroll_x))
            return

        if kind == "drag":
            raw_path = action.get("path")
            if not isinstance(raw_path, list) or not 2 <= len(raw_path) <= 100:
                raise ValueError("drag requires a path with 2 to 100 points.")
            path = [self._frame_point(_point_x(p), _point_y(p), frame) for p in raw_path]
            with self._held_modifiers(gui, keys):
                gui.moveTo(*path[0], duration=0.15)
                gui.mouseDown(button="left")
                try:
                    for point in path[1:]:
                        gui.moveTo(*point, duration=0.05)
                finally:
                    gui.mouseUp(button="left")
            return

        raise ValueError(f"Unsupported computer-use action: {kind}")

    def _frame_point(self, raw_x: Any, raw_y: Any, frame: ScreenFrame) -> tuple[int, int]:
        try:
            relative_x, relative_y = int(raw_x), int(raw_y)
        except (TypeError, ValueError) as exc:
            raise ValueError("Computer action x and y must be integer coordinates.") from exc
        if not (0 <= relative_x < frame.width and 0 <= relative_y < frame.height):
            raise ValueError(
                f"Point ({relative_x}, {relative_y}) is outside the screenshot "
                f"({frame.width}, {frame.height})."
            )
        return self._validated_point(frame.left + relative_x, frame.top + relative_y)

    def _type_text(self, text: str, gui: Any) -> None:
        if not text:
            raise ValueError("No text provided.")
        if len(text) > 4000:
            raise ValueError("Text is limited to 4000 characters per action.")
        try:
            import pyperclip

            previous = pyperclip.paste()
            pyperclip.copy(text)
        except Exception:
            gui.write(text, interval=0.01)
            return
        try:
            gui.hotkey("ctrl", "v")
        finally:
            with contextlib.suppress(Exception):
                pyperclip.copy(previous)

    @contextlib.contextmanager
    def _held_modifiers(self, gui: Any, raw_keys: Any) -> Iterator[None]:
        keys = raw_keys if isinstance(raw_keys, list) else []
        normalized = [_validated_key(key, gui) for key in keys]
        for key in normalized:
            gui.keyDown(key)
        try:
            yield
        finally:
            for key in reversed(normalized):
                gui.keyUp(key)

    def _validated_point(self, raw_x: Any, raw_y: Any) -> tuple[int, int]:
        try:
            x, y = int(raw_x), int(raw_y)
        except (TypeError, ValueError) as exc:
            raise ValueError("x and y must be integer screen coordinates.") from exc
        left, top, width, height = self.bounds()
        if not (left <= x < left + width and top <= y < top + height):
            raise ValueError(
                f"Point ({x}, {y}) is outside the desktop bounds "
                f"({left}, {top}, {width}, {height})."
            )
        return x, y


_backend: DesktopBackend | None = None


def get_desktop_backend() -> DesktopBackend:
    global _backend
    if _backend is None:
        _backend = DesktopBackend()
    return _backend


class _DesktopVisualMixin:
    def __init__(self, *, visual_callback: DesktopVisualCallback | None = None) -> None:
        self._visual_callback = visual_callback

    def set_visual_callback(self, callback: DesktopVisualCallback | None) -> None:
        self._visual_callback = callback

    def _emit_visual(self, kind: str, **data: Any) -> None:
        callback = self._visual_callback
        if callback is None:
            return
        try:
            callback(kind, data)
        except Exception:  # pragma: no cover - UI callbacks are best effort
            logger.debug("Desktop visual callback failed for %s", kind, exc_info=True)

    def _publish_frame(self, frame: ScreenFrame) -> str:
        path = _save_frame(frame.png)
        self._emit_visual(
            "desktop_frame",
            image_base64=base64.b64encode(frame.png).decode("ascii"),
            mime_type="image/png",
            left=frame.left,
            top=frame.top,
            width=frame.width,
            height=frame.height,
            cursor={"x": frame.cursor_x, "y": frame.cursor_y},
        )
        return str(path)


class ComputerViewTool(_DesktopVisualMixin, IsaacTool):
    """Capture the real desktop without sending it to a model provider."""

    name = "computer_view"
    description = (
        "Capture the user's real desktop and report its screenshot path, dimensions, and "
        "cursor position. The image stays on this computer and is shown in the local UI. "
        "Use computer_describe when visual interpretation is needed."
    )
    risk_level = 2
    requires_approval = False
    sandbox_required = False
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            frame = await asyncio.to_thread(get_desktop_backend().capture)
            path = self._publish_frame(frame)
            return ToolResult(
                success=True,
                output=(
                    f"Desktop screenshot saved to {path}. Virtual screen: left={frame.left}, "
                    f"top={frame.top}, width={frame.width}, height={frame.height}. "
                    f"Cursor: ({frame.cursor_x}, {frame.cursor_y})."
                ),
                metadata={
                    "path": path,
                    "left": frame.left,
                    "top": frame.top,
                    "width": frame.width,
                    "height": frame.height,
                    "cursor_x": frame.cursor_x,
                    "cursor_y": frame.cursor_y,
                },
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Could not capture the desktop: {exc}")


class ComputerDescribeTool(_DesktopVisualMixin, IsaacTool):
    """Capture and interpret the desktop with the configured vision model."""

    name = "computer_describe"
    description = (
        "Capture the user's real desktop and ask the configured vision model a question "
        "about it. This may send the screenshot to the selected model provider, so it "
        "always requires explicit human approval."
    )
    risk_level = 4
    requires_approval = True
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to inspect in the current desktop screenshot.",
            }
        },
        "required": ["question"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        question = str(kwargs.get("question", "")).strip()
        if not question:
            return ToolResult(success=False, error="No visual question provided.")
        try:
            frame = await asyncio.to_thread(get_desktop_backend().capture)
            path = self._publish_frame(frame)
            from isaac.multimodal.vision.vision_lm import get_vision_lm

            answer = await asyncio.to_thread(get_vision_lm().ask, question, frame.png)
            return ToolResult(
                success=True,
                output=f"Desktop screenshot: {path}\nVisual analysis:\n{answer}",
                metadata={"path": path},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Could not analyze the desktop: {exc}")


class ComputerControlTool(_DesktopVisualMixin, IsaacTool):
    """Perform one bounded mouse or keyboard action on the real desktop."""

    name = "computer_control"
    description = (
        "Control the user's real mouse and keyboard with one bounded action, then capture "
        "the resulting desktop. Every call requires human approval. The PyAutoGUI failsafe "
        "is enabled: moving the pointer to the top-left corner aborts control."
    )
    risk_level = 4
    requires_approval = True
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "move",
                    "click",
                    "double_click",
                    "type",
                    "press",
                    "hotkey",
                    "scroll",
                    "wait",
                ],
            },
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "button": {"type": "string", "enum": ["left", "middle", "right"]},
            "text": {"type": "string"},
            "key": {"type": "string"},
            "keys": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "amount": {"type": "integer", "minimum": -20, "maximum": 20},
            "duration": {"type": "number", "minimum": 0, "maximum": 2},
            "interval": {"type": "number", "minimum": 0, "maximum": 0.1},
            "seconds": {"type": "number", "minimum": 0, "maximum": 10},
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip()
        if not action:
            return ToolResult(success=False, error="No desktop action provided.")
        try:
            backend = get_desktop_backend()
            await asyncio.to_thread(backend.perform, action, kwargs)
            frame = await asyncio.to_thread(backend.capture)
            path = self._publish_frame(frame)
            self._emit_visual(
                "desktop_cursor",
                x=frame.cursor_x,
                y=frame.cursor_y,
                left=frame.left,
                top=frame.top,
                width=frame.width,
                height=frame.height,
                action=action,
            )
            return ToolResult(
                success=True,
                output=(
                    f"Desktop action '{action}' completed. Updated screenshot: {path}. "
                    f"Cursor: ({frame.cursor_x}, {frame.cursor_y})."
                ),
                metadata={"path": path, "cursor_x": frame.cursor_x, "cursor_y": frame.cursor_y},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Desktop action '{action}' failed: {exc}")


def _validated_button(value: Any) -> str:
    button = str(value or "left").lower()
    if button not in {"left", "middle", "right"}:
        raise ValueError("button must be left, middle, or right.")
    return button


def _validated_key(value: Any, gui: Any) -> str:
    aliases = {"control": "ctrl", "command": "win", "option": "alt", "return": "enter"}
    key = str(value or "").lower().strip()
    key = aliases.get(key, key)
    if not key or key not in set(gui.KEYBOARD_KEYS):
        raise ValueError(f"Unsupported keyboard key: {key!r}")
    return key


def _point_x(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0]
    if isinstance(value, dict):
        return value.get("x")
    raise ValueError("Drag path entries must contain x and y coordinates.")


def _point_y(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[1]
    if isinstance(value, dict):
        return value.get("y")
    raise ValueError("Drag path entries must contain x and y coordinates.")


def _bounded_scroll_delta(value: Any) -> int:
    try:
        return min(5000, max(-5000, int(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Scroll deltas must be integers.") from exc


def _scroll_steps(delta: int) -> int:
    return max(1, abs(round(delta / 100))) if delta else 0


def publish_desktop_frame(
    frame: ScreenFrame,
    callback: DesktopVisualCallback | None,
    *,
    action: str = "screenshot",
) -> str:
    """Persist and publish a desktop frame without adding it to LLM text context."""
    path = _save_frame(frame.png)
    if callback is not None:
        callback(
            "desktop_frame",
            {
                "image_base64": base64.b64encode(frame.png).decode("ascii"),
                "mime_type": "image/png",
                "left": frame.left,
                "top": frame.top,
                "width": frame.width,
                "height": frame.height,
                "cursor": {"x": frame.cursor_x, "y": frame.cursor_y},
                "action": action,
            },
        )
    return str(path)


def _save_frame(png: bytes) -> Path:
    from isaac.config.settings import get_settings

    root = get_settings().isaac_home / "desktop" / "screenshots"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = root / f"desktop-{stamp}-{uuid.uuid4().hex[:8]}.png"
    path.write_bytes(png)
    return path


__all__ = [
    "ComputerControlTool",
    "ComputerDescribeTool",
    "ComputerViewTool",
    "DesktopBackend",
    "ScreenFrame",
    "get_desktop_backend",
    "publish_desktop_frame",
]
