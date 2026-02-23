"""Structured prompt templates for each cognitive node.

Each function returns a list of ``BaseMessage`` objects ready to be passed to
the LLM.  Templates are kept as plain f-strings for transparency — no hidden
macro expansion.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from isaac.core.state import ErrorEntry, PlanStep, WorldModel


# ---------------------------------------------------------------------------
# System personas
# ---------------------------------------------------------------------------

_SYSTEM_PERCEPTION = SystemMessage(
    content=(
        "You are the Perception module of I.S.A.A.C., a neuro-symbolic autonomous agent. "
        "Your role is to parse the user's request, extract structured observations about "
        "the environment, and produce an initial hypothesis for solving the task. "
        "Respond ONLY with valid JSON matching the requested schema."
    )
)

_SYSTEM_PLANNER = SystemMessage(
    content=(
        "You are the Planner module of I.S.A.A.C. Given a world model, hypothesis, "
        "past errors, and available skills, decompose the task into an ordered list of "
        "atomic, dependency-aware steps.  Each step must be concrete enough for a code "
        "synthesiser to implement in a single sandbox execution. "
        "Respond ONLY with valid JSON matching the requested schema."
    )
)

_SYSTEM_SYNTHESIS = SystemMessage(
    content=(
        "You are the Synthesis module of I.S.A.A.C. Given a single plan step and the "
        "current world model, generate a self-contained Python script that accomplishes "
        "the step.  The script will run inside an isolated sandbox with NO network.  "
        "Print results to stdout.  Import only from the Python standard library and numpy. "
        "Do NOT use tool-calling JSON — output ONLY a fenced Python code block."
    )
)

_SYSTEM_REFLECTION = SystemMessage(
    content=(
        "You are the Reflection / Critic module of I.S.A.A.C. Analyse the execution logs "
        "from the sandbox run.  Determine whether the step succeeded or failed.  If the step "
        "failed, diagnose the root cause and propose a revised hypothesis.  If it succeeded, "
        "summarise what was achieved and propose a skill candidate for generalisation. "
        "Respond ONLY with valid JSON matching the requested schema."
    )
)

_SYSTEM_SKILL_ABSTRACTION = SystemMessage(
    content=(
        "You are the Skill Abstraction module of I.S.A.A.C. Given a concrete Python script "
        "that successfully solved a task, generalise it into a reusable, parameterised "
        "function.  The function must have clear type hints, a docstring, and handle edge "
        "cases.  It should be importable as a standalone module.  "
        "Respond ONLY with a fenced Python code block."
    )
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def perception_prompt(user_input: str, world_model: WorldModel) -> list[BaseMessage]:
    """Build the prompt for the Perception node."""
    return [
        _SYSTEM_PERCEPTION,
        HumanMessage(
            content=(
                f"## User input\n{user_input}\n\n"
                f"## Current world model\n"
                f"Files: {json.dumps(world_model.files)}\n"
                f"Resources: {json.dumps(world_model.resources)}\n"
                f"Constraints: {json.dumps(world_model.constraints)}\n\n"
                "Respond with JSON: "
                '{"observations": ["..."], "hypothesis": "..."}'
            )
        ),
    ]


def planner_prompt(
    world_model: WorldModel,
    hypothesis: str,
    errors: list[ErrorEntry],
    available_skills: list[str],
) -> list[BaseMessage]:
    """Build the prompt for the Planner node."""
    error_summaries = [
        {"node": e.node, "message": e.message, "attempt": e.attempt}
        for e in errors
    ]
    return [
        _SYSTEM_PLANNER,
        HumanMessage(
            content=(
                f"## Hypothesis\n{hypothesis}\n\n"
                f"## World model observations\n"
                f"{json.dumps(world_model.observations)}\n\n"
                f"## Past errors\n{json.dumps(error_summaries)}\n\n"
                f"## Available skills\n{json.dumps(available_skills)}\n\n"
                "Respond with JSON: "
                '{"steps": [{"id": "s1", "description": "...", "depends_on": []}]}'
            )
        ),
    ]


def synthesis_prompt(
    step: PlanStep,
    world_model: WorldModel,
    hypothesis: str,
    available_skills: list[str],
) -> list[BaseMessage]:
    """Build the prompt for the Synthesis node."""
    return [
        _SYSTEM_SYNTHESIS,
        HumanMessage(
            content=(
                f"## Current step\n"
                f"ID: {step.id}\n"
                f"Description: {step.description}\n\n"
                f"## Hypothesis\n{hypothesis}\n\n"
                f"## World model\n"
                f"Files: {json.dumps(world_model.files)}\n"
                f"Constraints: {json.dumps(world_model.constraints)}\n"
                f"Observations: {json.dumps(world_model.observations)}\n\n"
                f"## Available skills (importable)\n{json.dumps(available_skills)}\n\n"
                "Generate the Python script inside a ```python``` fence."
            )
        ),
    ]


def reflection_prompt(
    code: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    step_description: str,
) -> list[BaseMessage]:
    """Build the prompt for the Reflection node."""
    return [
        _SYSTEM_REFLECTION,
        HumanMessage(
            content=(
                f"## Executed code\n```python\n{code}\n```\n\n"
                f"## Step description\n{step_description}\n\n"
                f"## Execution results\n"
                f"Exit code: {exit_code}\n"
                f"stdout:\n```\n{stdout}\n```\n"
                f"stderr:\n```\n{stderr}\n```\n\n"
                "Respond with JSON:\n"
                "If **failed**: "
                '{"success": false, "diagnosis": "...", "revised_hypothesis": "..."}\n'
                "If **succeeded**: "
                '{"success": true, "summary": "...", '
                '"skill_candidate": {"name": "...", "description": "..."}}'
            )
        ),
    ]


def skill_abstraction_prompt(
    concrete_code: str,
    task_context: str,
) -> list[BaseMessage]:
    """Build the prompt for the Skill Abstraction node."""
    return [
        _SYSTEM_SKILL_ABSTRACTION,
        HumanMessage(
            content=(
                f"## Concrete code that solved the task\n```python\n{concrete_code}\n```\n\n"
                f"## Task context\n{task_context}\n\n"
                "Generalise into a reusable function inside a ```python``` fence. "
                "Include type hints, docstring, and edge-case handling."
            )
        ),
    ]
