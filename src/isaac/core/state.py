"""IsaacState — the strict data contract circulating in the LangGraph.

Every node reads from and writes to fields of this TypedDict.  Sub-schemas
are plain dataclasses to keep serialisation lightweight while remaining
fully typed.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Sub-schemas
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


@dataclass
class PlanStep:
    """A single step in the agent's dynamic plan (Graph-of-Thought)."""

    id: str
    description: str
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
# IsaacState
# ---------------------------------------------------------------------------

from typing import TypedDict  # noqa: E402


class IsaacState(TypedDict, total=False):
    """Root state schema for the I.S.A.A.C. cognitive graph.

    Each field uses an ``Annotated`` reducer so that LangGraph knows how to
    merge partial updates returned by individual nodes.

    * ``messages`` — append-only conversation history (LangGraph built-in).
    * ``world_model`` — latest-wins (Perception node overwrites).
    * ``hypothesis`` — latest-wins (Reflection node overwrites).
    * ``plan`` — latest-wins (Planner node overwrites).
    * ``code_buffer`` — latest-wins (Synthesis node overwrites).
    * ``execution_logs`` — append-only (Sandbox node appends).
    * ``skill_candidate`` — latest-wins (Reflection / Skill Abstraction).
    * ``errors`` — append-only (Reflection node appends).
    * ``iteration`` — latest-wins (incremented by Planner node).
    * ``current_phase`` — latest-wins (set by each node on entry).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    world_model: Annotated[WorldModel, _replace]
    hypothesis: Annotated[str, _replace]
    plan: Annotated[list[PlanStep], _replace]
    code_buffer: Annotated[str, _replace]
    execution_logs: Annotated[list[ExecutionResult], _append_list]
    skill_candidate: Annotated[SkillCandidate | None, _replace]
    errors: Annotated[list[ErrorEntry], _append_list]
    iteration: Annotated[int, _replace]
    current_phase: Annotated[str, _replace]


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
    )
