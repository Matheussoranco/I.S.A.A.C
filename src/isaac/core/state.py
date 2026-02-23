"""IsaacState — the strict data contract circulating in the LangGraph.

Every node reads from and writes to fields of this TypedDict.  Sub-schemas
are plain dataclasses to keep serialisation lightweight while remaining
fully typed.

Computer-Use extensions add GUI state, UI actions/results, and a separate
``ui_cycle`` counter so the screenshot→action loop can have its own depth
limit independent of the outer planning iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Sub-schemas — GUI / Computer-Use
# ---------------------------------------------------------------------------


@dataclass
class ScreenElement:
    """A UI element detected on screen via OCR or accessibility tree."""

    label: str
    role: str
    """Semantic role: 'button', 'input', 'link', 'text', 'image', 'checkbox'."""
    bbox: tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))
    """Bounding box (x1, y1, x2, y2) in absolute screen pixels."""
    text: str = ""
    is_focused: bool = False
    confidence: float = 1.0


@dataclass
class GUIState:
    """Snapshot of the graphical desktop state inside the VNC sandbox."""

    screenshot_b64: str = ""
    """Current screen as PNG encoded to base64."""
    active_window_title: str = ""
    active_window_class: str = ""
    cursor_x: int = 0
    cursor_y: int = 0
    screen_width: int = 1280
    screen_height: int = 720
    elements: list[ScreenElement] = field(default_factory=list)
    """Detected UI elements (from vision LLM or AT-SPI tree walk)."""
    accessibility_tree: dict[str, Any] = field(default_factory=dict)
    """Raw AT-SPI / UIA accessibility tree dump (JSON-serialisable)."""
    current_url: str = ""
    """URL of the foreground browser tab, if any."""
    display: str = ":99"


@dataclass
class UIAction:
    """A single atomic UI interaction the agent wants to perform."""

    type: Literal[
        "screenshot",    # capture screen — no side-effects
        "click",
        "double_click",
        "right_click",
        "type",          # inject keyboard text
        "key",           # key or combo, e.g. "ctrl+c", "Return"
        "scroll",
        "move",          # move mouse without pressing
        "drag",          # drag from (x, y) to (target_x, target_y)
        "wait",          # sleep duration_ms
    ] = "screenshot"
    x: int | None = None
    y: int | None = None
    target_x: int | None = None  # drag destination X
    target_y: int | None = None  # drag destination Y
    text: str | None = None      # for "type"
    key: str | None = None       # for "key"
    scroll_direction: Literal["up", "down", "left", "right"] | None = None
    scroll_amount: int = 3
    duration_ms: int = 0
    description: str = ""        # human-readable intent; used by SkillAbstraction


@dataclass
class UIActionResult:
    """Outcome of executing a single UIAction inside the VNC sandbox."""

    action: UIAction = field(default_factory=UIAction)
    success: bool = False
    screenshot_before_b64: str = ""
    screenshot_after_b64: str = ""
    error: str = ""
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Sub-schemas — core cognitive
# ---------------------------------------------------------------------------


@dataclass
class WorldModel:
    """Structured representation of the current environment state."""

    files: dict[str, str] = field(default_factory=dict)
    """path → content hash or short summary."""
    resources: dict[str, Any] = field(default_factory=dict)
    """Snapshot of available resources (cpu, mem, disk, etc.)."""
    constraints: list[str] = field(default_factory=list)
    """Active constraints (e.g., 'no network', 'read-only fs')."""
    observations: list[str] = field(default_factory=list)
    """Recent environment observations produced by the Perception node."""

    # ── Computer-Use extensions ─────────────────────────────────────────────
    gui_state: GUIState | None = None
    """Current desktop GUI state; ``None`` when not in computer-use mode."""
    clipboard: str = ""
    """Last known clipboard content."""
    last_url: str = ""
    """Last navigated URL (populated by computer_use node)."""


@dataclass
class PlanStep:
    """A single step in the agent's dynamic plan (Graph-of-Thought)."""

    id: str
    description: str
    mode: Literal["code", "ui", "hybrid"] = "code"
    """Execution mode: 'code' → Sandbox, 'ui' → ComputerUse, 'hybrid' → both."""
    status: Literal["pending", "active", "done", "failed"] = "pending"
    depends_on: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Raw output captured from a single Docker sandbox run."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_ms: float = 0.0


@dataclass
class SkillCandidate:
    """A code pattern being evaluated for promotion to the Skill Library."""

    name: str = ""
    code: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    task_context: str = ""
    success_count: int = 0
    skill_type: Literal["code", "ui"] = "code"
    """Origin paradigm — 'ui' skills are Playwright macros."""
    tags: list[str] = field(default_factory=list)
    """Semantic tags for retrieval (e.g. ['ui', 'playwright', 'login'])."""


@dataclass
class ErrorEntry:
    """A single failure record for the self-reflection stack."""

    node: str
    message: str
    traceback: str | None = None
    timestamp: str = ""
    attempt: int = 0


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def _append_list(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer that appends new items to an existing list."""
    return left + right


def _replace(left: Any, right: Any) -> Any:
    """Reducer that always takes the newer value."""
    return right


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TaskMode = Literal["code", "computer_use", "hybrid"]


# ---------------------------------------------------------------------------
# IsaacState
# ---------------------------------------------------------------------------


class IsaacState(TypedDict, total=False):
    """Root state schema for the I.S.A.A.C. cognitive graph.

    Each field uses an ``Annotated`` reducer so that LangGraph knows how to
    merge partial updates returned by individual nodes.

    Core fields
    -----------
    * ``messages``       — append-only conversation history (LangGraph built-in).
    * ``world_model``    — latest-wins (Perception node overwrites).
    * ``hypothesis``     — latest-wins (Reflection node overwrites).
    * ``plan``           — latest-wins (Planner node overwrites).
    * ``code_buffer``    — latest-wins (Synthesis node overwrites).
    * ``execution_logs`` — append-only (Sandbox node appends).
    * ``skill_candidate`` — latest-wins (Reflection / Skill Abstraction).
    * ``errors``         — append-only (Reflection node appends).
    * ``iteration``      — latest-wins (incremented by Planner node).
    * ``current_phase``  — latest-wins (set by each node on entry).

    Computer-Use extensions
    -----------------------
    * ``task_mode``  — latest-wins (Perception detects from input modality).
    * ``ui_actions`` — append-only (Synthesis / ComputerUse append pending actions).
    * ``ui_results`` — append-only (ComputerUse appends after each execution).
    * ``ui_cycle``   — latest-wins (screenshot→action loop counter).
    """

    messages:        Annotated[list[BaseMessage], add_messages]
    world_model:     Annotated[WorldModel, _replace]
    hypothesis:      Annotated[str, _replace]
    plan:            Annotated[list[PlanStep], _replace]
    code_buffer:     Annotated[str, _replace]
    execution_logs:  Annotated[list[ExecutionResult], _append_list]
    skill_candidate: Annotated[SkillCandidate | None, _replace]
    errors:          Annotated[list[ErrorEntry], _append_list]
    iteration:       Annotated[int, _replace]
    current_phase:   Annotated[str, _replace]

    # ── Computer-Use ────────────────────────────────────────────────────────
    task_mode:   Annotated[TaskMode, _replace]
    ui_actions:  Annotated[list[UIAction], _append_list]
    ui_results:  Annotated[list[UIActionResult], _append_list]
    ui_cycle:    Annotated[int, _replace]


def make_initial_state() -> IsaacState:
    """Return a fully initialised blank state for a new cognitive cycle."""
    return IsaacState(
        messages=[],
        world_model=WorldModel(),
        hypothesis="",
        plan=[],
        code_buffer="",
        execution_logs=[],
        skill_candidate=None,
        errors=[],
        iteration=0,
        current_phase="init",
        task_mode="code",
        ui_actions=[],
        ui_results=[],
        ui_cycle=0,
    )
