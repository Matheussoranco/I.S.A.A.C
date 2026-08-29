"""Responses computer harness tests with no network or real input."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import isaac.agents.computer_agent as computer_module
from isaac.agents.computer_agent import ComputerAgentRunner
from isaac.tools.desktop import ScreenFrame


class _Backend:
    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] = []

    def capture(self) -> ScreenFrame:
        return ScreenFrame(b"png", 0, 0, 1440, 900, 100, 200)

    def perform_computer_action(self, action: dict[str, Any], frame: ScreenFrame) -> None:
        assert frame.width == 1440
        self.actions.append(action)


class _Responses:
    def __init__(self, values: list[Any]) -> None:
        self.values = values
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.values.pop(0)


def _response(response_id: str, *, actions: list[dict[str, Any]] | None = None, text: str = ""):
    output = []
    if actions is not None:
        output.append(SimpleNamespace(type="computer_call", call_id="call-1", actions=actions))
    return SimpleNamespace(id=response_id, output=output, output_text=text)


def test_computer_runner_repeats_screenshot_action_loop(monkeypatch) -> None:
    responses = _Responses(
        [
            _response("response-1", actions=[{"type": "screenshot"}]),
            _response("response-2", text="Concluído."),
        ]
    )
    backend = _Backend()
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(computer_module, "publish_desktop_frame", lambda *a, **k: "screen.png")
    runner = ComputerAgentRunner(
        client=SimpleNamespace(responses=responses),
        backend=backend,
        on_event=lambda kind, data: events.append((kind, data)),
        approval_callback=lambda *args: True,
    )

    result = runner.run("Observe a tela")

    assert result.success is True
    assert result.output == "Concluído."
    assert backend.actions == [{"type": "screenshot"}]
    follow_up = responses.calls[1]
    assert follow_up["previous_response_id"] == "response-1"
    screenshot = follow_up["input"][0]["output"]
    assert screenshot["detail"] == "original"
    assert screenshot["image_url"].startswith("data:image/png;base64,")
    assert any(kind == "final" for kind, _ in events)


def test_computer_runner_requires_approval_for_actionable_batch(monkeypatch) -> None:
    responses = _Responses([_response("response-1", actions=[{"type": "click", "x": 20, "y": 30}])])
    backend = _Backend()
    monkeypatch.setattr(computer_module, "publish_desktop_frame", lambda *a, **k: "screen.png")
    runner = ComputerAgentRunner(
        client=SimpleNamespace(responses=responses),
        backend=backend,
        approval_callback=lambda name, args, risk: False,
    )

    result = runner.run("Clique")

    assert result.stopped_reason == "approval_denied"
    assert backend.actions == []


def test_computer_runner_honours_cancellation() -> None:
    responses = _Responses([_response("response-1", actions=[{"type": "screenshot"}])])
    runner = ComputerAgentRunner(
        client=SimpleNamespace(responses=responses),
        backend=_Backend(),
        should_stop=lambda: True,
    )

    result = runner.run("Pare")

    assert result.stopped_reason == "cancelled"
