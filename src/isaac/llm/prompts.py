"""Structured prompt templates for each cognitive node.

Each function returns a list of ``BaseMessage`` objects ready to be passed to
the LLM.  Templates are kept as plain f-strings for transparency — no hidden
macro expansion.

Computer-Use additions
----------------------
* ``perception_multimodal_prompt`` — handles both text and screenshot input
* ``planner_ui_prompt``            — instructs UI-step generation with mode field
* ``synthesis_ui_prompt``          — produces UIAction JSON from screenshot + step
* ``synthesis_hybrid_prompt``      — produces Playwright Python script
* ``computer_use_prompt``          — vision loop: screenshot + pending actions → next action
* ``reflection_ui_prompt``         — compare before/after screenshots for success/fail
* ``skill_abstraction_ui_prompt``  — generalise UIAction trace into Playwright function
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from isaac.core.state import (
        ErrorEntry,
        GUIState,
        PlanStep,
        UIAction,
        WorldModel,
    )


# ---------------------------------------------------------------------------
# Soul preamble — injected into every system message
# ---------------------------------------------------------------------------


def _soul_preamble() -> str:
    """Return the SOUL identity preamble for system prompts."""
    try:
        from isaac.identity.soul import soul_system_prompt
        return soul_system_prompt() + "\n\n"
    except Exception:
        return ""


def _sys(role_content: str) -> SystemMessage:
    """Build a SystemMessage with the SOUL preamble prepended."""
    return SystemMessage(content=_soul_preamble() + role_content)


# ---------------------------------------------------------------------------
# System personas (role-specific content; soul preamble added lazily via _sys)
# ---------------------------------------------------------------------------

_PERCEPTION_CONTENT = (
    "You are the Perception module of I.S.A.A.C., a neuro-symbolic autonomous agent. "
    "Your role is to parse the user's request, extract structured observations about "
    "the environment, and produce an initial hypothesis for solving the task. "
    "If a screenshot is provided, analyse the GUI state (visible elements, active window, "
    "URL if a browser is visible) and include them in your observations. "
    "Set 'task_mode' to one of:\n"
    "  - 'direct' : simple greetings, casual conversation, knowledge questions, "
    "or anything that can be answered immediately without code, tools, or planning.\n"
    "  - 'code' : tasks that require writing and executing Python code.\n"
    "  - 'computer_use' : tasks that require GUI interaction or when a screenshot is provided.\n"
    "  - 'hybrid' : tasks needing both code execution and GUI interaction.\n"
    "IMPORTANT: Use 'direct' for greetings, chitchat, factual questions, explanations, "
    "and any request that does NOT require running code or interacting with a GUI. "
    "Most conversational messages should be 'direct'. "
    "Respond ONLY with valid JSON matching the requested schema."
)

_PLANNER_CONTENT = (
    "You are the Planner module of I.S.A.A.C. Given a world model, hypothesis, "
    "past errors, and available skills, decompose the task into an ordered list of "
    "atomic, dependency-aware steps.  Each step must be concrete enough for a code "
    "synthesiser or UI automation engine to implement in a single execution cycle. "
    "For each step, set 'mode' to:\n"
    "  - 'code'   → pure Python computation (no GUI required)\n"
    "  - 'ui'     → GUI interaction via mouse/keyboard (no code execution)\n"
    "  - 'hybrid' → Playwright/PyAutoGUI script running inside the virtual desktop\n"
    "Respond ONLY with valid JSON matching the requested schema."
)

_SYNTHESIS_CONTENT = (
    "You are the Synthesis module of I.S.A.A.C. Given a single plan step and the "
    "current world model, generate a self-contained Python script that accomplishes "
    "the step.  The script will run inside an isolated sandbox with NO network.  "
    "Print results to stdout.  Import only from the Python standard library and numpy. "
    "Do NOT use tool-calling JSON — output ONLY a fenced Python code block."
)

_SYNTHESIS_UI_CONTENT = (
    "You are the Synthesis module of I.S.A.A.C. operating in UI mode. "
    "Given a plan step, the current screen screenshot, and detected UI elements, "
    "emit a JSON array of UIActions to accomplish the step. "
    "Each action must have: 'type', 'x', 'y' (when applicable), "
    "'text' or 'key' (when applicable), and a 'description' explaining intent. "
    "Think in absolute screen pixels.  Be precise — off-by-one clicks fail. "
    "Respond ONLY with a valid JSON array: [{\"type\": ..., ...}, ...]"
)

_SYNTHESIS_HYBRID_CONTENT = (
    "You are the Synthesis module of I.S.A.A.C. operating in hybrid mode. "
    "Generate a self-contained Python script using Playwright that automates the "
    "given UI task inside a virtual Chromium browser running on DISPLAY=:99. "
    "The script runs as a standalone program — include the main() call. "
    "Use sync_playwright. The Chromium CDP endpoint is on port 9222 if already open; "
    "otherwise launch a new browser with: "
    "p.chromium.launch(headless=False, args=['--no-sandbox']). "
    "Print a summary to stdout on completion. "
    "Respond ONLY with a fenced Python code block."
)

_COMPUTER_USE_CONTENT = (
    "You are the ComputerUse controller of I.S.A.A.C. "
    "You receive a screenshot of the current desktop and a pending UIAction queue. "
    "Decide ONE next UIAction to execute. "
    "If the screenshot shows the step is already complete, emit: "
    "{\"done\": true, \"summary\": \"...\"}. "
    "Otherwise emit: {\"done\": false, \"action\": {<UIAction fields>}}. "
    "Be conservative — prefer explicit waits after navigation events. "
    "Respond ONLY with valid JSON."
)

_REFLECTION_CONTENT = (
    "You are the Reflection / Critic module of I.S.A.A.C. Analyse the execution logs "
    "from the sandbox run.  Determine whether the step succeeded or failed.  If the step "
    "failed, diagnose the root cause and propose a revised hypothesis.  If it succeeded, "
    "summarise what was achieved and propose a skill candidate for generalisation. "
    "Respond ONLY with valid JSON matching the requested schema."
)

_REFLECTION_UI_CONTENT = (
    "You are the Reflection / Critic module of I.S.A.A.C. operating in Computer-Use mode. "
    "You will receive: the step description, a before screenshot, an after screenshot, "
    "and the UIAction that was executed. "
    "Determine whether the action produced the expected visual change. "
    "If successful, propose a Playwright macro skill for this interaction. "
    "If failed, diagnose what went wrong (wrong coordinates, element not loaded, etc.) "
    "and propose a corrective action. "
    "Respond ONLY with valid JSON matching the requested schema."
)

_SKILL_ABSTRACTION_CONTENT = (
    "You are the Skill Abstraction module of I.S.A.A.C. Given a concrete Python script "
    "that successfully solved a task, generalise it into a reusable, parameterised "
    "function.  The function must have clear type hints, a docstring, and handle edge "
    "cases.  It should be importable as a standalone module.  "
    "Respond ONLY with a fenced Python code block."
)

_SKILL_ABSTRACTION_UI_CONTENT = (
    "You are the Skill Abstraction module of I.S.A.A.C. operating in UI mode. "
    "You will receive a sequence of UIActions that successfully accomplished a task, "
    "plus before/after screenshots. "
    "Generalise this interaction trace into a reusable Python function using Playwright. "
    "The function must accept parameters for variable parts (e.g. credentials, URLs). "
    "Include: sync_playwright context manager, type hints, docstring, error handling. "
    "Respond ONLY with a fenced Python code block."
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def perception_prompt(user_input: str, world_model: WorldModel) -> list[BaseMessage]:
    """Build the prompt for the Perception node (text-only input)."""
    return [
        _sys(_PERCEPTION_CONTENT),
        HumanMessage(
            content=(
                f"## User input\n{user_input}\n\n"
                f"## Current world model\n"
                f"Files: {json.dumps(world_model.files)}\n"
                f"Resources: {json.dumps(world_model.resources)}\n"
                f"Constraints: {json.dumps(world_model.constraints)}\n\n"
                "Respond with JSON:\n"
                '{"observations": ["..."], "hypothesis": "...", '
                '"task_mode": "direct|code|computer_use|hybrid"}'
            )
        ),
    ]


def perception_multimodal_prompt(
    user_text: str,
    world_model: WorldModel,
    screenshot_b64: str,
) -> list[BaseMessage]:
    """Build a multimodal prompt for Perception when a screenshot is provided.

    The LLM receives both the text request and the current screen state,
    enabling it to extract GUI observations and detect ``task_mode``.
    """
    text_block = (
        f"## User request\n{user_text}\n\n"
        f"## Current world model\n"
        f"Files: {json.dumps(world_model.files)}\n"
        f"Observations: {json.dumps(world_model.observations)}\n"
        f"GUI state: {'active' if world_model.gui_state else 'none'}\n\n"
        "Analyse the screenshot and the user request together.\n"
        "Respond with JSON:\n"
        "{\n"
        '  "observations": ["<what you see on screen>", "..."],\n'
        '  "hypothesis": "<how to achieve the task>",\n'
        '  "task_mode": "computer_use",\n'
        '  "active_window_title": "...",\n'
        '  "current_url": "...",\n'
        '  "screen_width": 1280,\n'
        '  "screen_height": 720\n'
        "}"
    )
    content: list[dict] = [
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        },
    ]
    return [_sys(_PERCEPTION_CONTENT), HumanMessage(content=content)]


def planner_prompt(
    world_model: WorldModel,
    hypothesis: str,
    errors: list[ErrorEntry],
    available_skills: list[str],
    episodic_context: str = "",
    completed_descriptions: list[str] | None = None,
) -> list[BaseMessage]:
    """Build the prompt for the Planner node.

    Steps now include a ``mode`` field: 'code', 'ui', or 'hybrid'.
    Episodic context (if available) gives the LLM visibility into recent
    successes and failures so it can avoid repeating mistakes.

    ``completed_descriptions`` surfaces already-finished step descriptions so
    the LLM knows what work has been done and only plans the *remaining* work.
    """
    error_summaries = [
        {"node": e.node, "message": e.message, "attempt": e.attempt}
        for e in errors
    ]
    gui_context = ""
    if world_model.gui_state:
        gui_context = (
            f"  active_window: {world_model.gui_state.active_window_title}\n"
            f"  current_url:   {world_model.gui_state.current_url}\n"
        )
    episodic_section = ""
    if episodic_context and episodic_context != "No prior episodes.":
        episodic_section = (
            f"\n## Recent experience (episodic memory)\n{episodic_context}\n"
        )
    completed_section = ""
    if completed_descriptions:
        done_list = "\n".join(f"  - {d}" for d in completed_descriptions)
        completed_section = (
            f"\n## Already completed steps (do NOT repeat these)\n{done_list}\n"
        )
    return [
        _sys(_PLANNER_CONTENT),
        HumanMessage(
            content=(
                f"## Hypothesis\n{hypothesis}\n\n"
                f"## World model observations\n"
                f"{json.dumps(world_model.observations)}\n"
                + (f"\n## GUI context\n{gui_context}\n" if gui_context else "")
                + f"\n## Past errors\n{json.dumps(error_summaries)}\n\n"
                f"## Available skills\n{json.dumps(available_skills)}\n"
                + episodic_section
                + completed_section
                + "\nRespond with JSON:\n"
                '{"steps": [{"id": "s1", "description": "...", '
                '"mode": "code|ui|hybrid", "depends_on": []}]}'
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
        _sys(_SYNTHESIS_CONTENT),
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
        _sys(_REFLECTION_CONTENT),
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
    """Build the prompt for the Skill Abstraction node (code mode)."""
    return [
        _sys(_SKILL_ABSTRACTION_CONTENT),
        HumanMessage(
            content=(
                f"## Concrete code that solved the task\n```python\n{concrete_code}\n```\n\n"
                f"## Task context\n{task_context}\n\n"
                "Generalise into a reusable function inside a ```python``` fence. "
                "Include type hints, docstring, and edge-case handling."
            )
        ),
    ]


# ---------------------------------------------------------------------------
# Computer-Use prompt builders (new)
# ---------------------------------------------------------------------------


def synthesis_ui_prompt(
    step: PlanStep,
    gui_state: GUIState,
    screenshot_b64: str,
) -> list[BaseMessage]:
    """Build a multimodal prompt for Synthesis in 'ui' mode.

    Returns a UIAction JSON array targeting elements visible in the screenshot.
    """
    elements_summary = [
        {"label": e.label, "role": e.role, "bbox": list(e.bbox), "text": e.text}
        for e in gui_state.elements
    ]
    text_block = (
        f"## Step to accomplish\n{step.description}\n\n"
        f"## Screen dimensions\n{gui_state.screen_width}x{gui_state.screen_height}\n\n"
        f"## Detected UI elements\n{json.dumps(elements_summary, indent=2)}\n\n"
        f"## Active window\n{gui_state.active_window_title}\n"
        f"## Current URL\n{gui_state.current_url or 'n/a'}\n\n"
        "Emit a JSON array of UIActions. "
        "Each action: {\"type\": ..., \"x\": ..., \"y\": ..., \"text\": ..., "
        "\"key\": ..., \"description\": ...}\n"
        "Available types: screenshot, click, double_click, right_click, "
        "type, key, scroll, move, drag, wait"
    )
    content: list[dict] = [
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        },
    ]
    return [_sys(_SYNTHESIS_UI_CONTENT), HumanMessage(content=content)]


def synthesis_hybrid_prompt(
    step: PlanStep,
    gui_state: GUIState,
    screenshot_b64: str,
    available_skills: list[str],
) -> list[BaseMessage]:
    """Build a multimodal prompt for Synthesis in 'hybrid' mode (Playwright script)."""
    text_block = (
        f"## Step to accomplish\n{step.description}\n\n"
        f"## Active window\n{gui_state.active_window_title}\n"
        f"## Current URL\n{gui_state.current_url or 'n/a'}\n\n"
        f"## Available skills\n{json.dumps(available_skills)}\n\n"
        "Generate a self-contained Playwright Python script. "
        "The browser runs on DISPLAY=:99 (headless=False, --no-sandbox). "
        "Print a clear summary to stdout on completion.\n"
        "Respond ONLY with a fenced ```python``` code block."
    )
    content: list[dict] = [
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        },
    ]
    return [_sys(_SYNTHESIS_HYBRID_CONTENT), HumanMessage(content=content)]


def computer_use_prompt(
    step_description: str,
    pending_actions: list[UIAction],
    screenshot_b64: str,
    gui_state: GUIState,
    ui_cycle: int,
) -> list[BaseMessage]:
    """Build the per-cycle prompt for the ComputerUse node.

    The LLM decides whether the step is done or emits the next UIAction.
    """
    pending_summary = [
        {"type": a.type, "description": a.description}
        for a in pending_actions[:5]  # show at most 5 queued
    ]
    text_block = (
        f"## Step goal\n{step_description}\n\n"
        f"## UI cycle\n{ui_cycle}\n\n"
        f"## Pending actions queue\n{json.dumps(pending_summary)}\n\n"
        f"## Screen dimensions\n{gui_state.screen_width}x{gui_state.screen_height}\n"
        f"## Active window\n{gui_state.active_window_title}\n"
        f"## Current URL\n{gui_state.current_url or 'n/a'}\n\n"
        "Look at the screenshot and decide:\n"
        "  - If the step goal is COMPLETE: respond with {\"done\": true, \"summary\": \"...\"}\n"
        "  - Otherwise: respond with {\"done\": false, \"action\": {<UIAction fields>}}\n"
        "Be precise with pixel coordinates. "
        "Prefer 'wait' if a page transition is in progress."
    )
    content: list[dict] = [
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        },
    ]
    return [_sys(_COMPUTER_USE_CONTENT), HumanMessage(content=content)]


def reflection_ui_prompt(
    step_description: str,
    action: UIAction,
    screenshot_before_b64: str,
    screenshot_after_b64: str,
    error: str,
) -> list[BaseMessage]:
    """Build a multimodal reflection prompt for Computer-Use steps.

    The LLM compares before/after screenshots to judge success.
    """
    action_dict = {
        "type": action.type,
        "x": action.x,
        "y": action.y,
        "text": action.text,
        "key": action.key,
        "description": action.description,
    }
    text_block = (
        f"## Step description\n{step_description}\n\n"
        f"## Executed UIAction\n{json.dumps(action_dict, indent=2)}\n\n"
        f"## System error (if any)\n{error or 'none'}\n\n"
        "Two screenshots follow: BEFORE (first) and AFTER (second) the action.\n"
        "Determine if the action achieved its goal.\n\n"
        "If **succeeded**:\n"
        '  {"success": true, "summary": "...", '
        '"skill_candidate": {"name": "...", "description": "..."}}\n\n'
        "If **failed**:\n"
        '  {"success": false, "diagnosis": "...", "revised_hypothesis": "...", '
        '"corrective_action": {<UIAction fields or null>}}'
    )
    content: list[dict] = [
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_before_b64}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_after_b64}"},
        },
    ]
    return [_sys(_REFLECTION_UI_CONTENT), HumanMessage(content=content)]


def skill_abstraction_ui_prompt(
    action_trace: list[UIAction],
    task_context: str,
    screenshot_before_b64: str,
    screenshot_after_b64: str,
) -> list[BaseMessage]:
    """Build a multimodal prompt for UI skill abstraction.

    The LLM converts a UIAction trace + screenshots into a Playwright function.
    """
    trace_dicts = [
        {
            "type": a.type,
            "x": a.x,
            "y": a.y,
            "text": a.text,
            "key": a.key,
            "description": a.description,
        }
        for a in action_trace
    ]
    text_block = (
        f"## Task context\n{task_context}\n\n"
        f"## UIAction trace that solved the task\n{json.dumps(trace_dicts, indent=2)}\n\n"
        "Three screenshots: INITIAL state (first), FINAL state (second), follow.\n"
        "Generalise this interaction into a reusable Python function using Playwright sync API.\n"
        "Requirements:\n"
        "  - Accept parameters for variable data (URLs, credentials, search queries, etc.)\n"
        "  - Use page.get_by_role / get_by_label / locator when possible (avoid raw coords)\n"
        "  - Include type hints, docstring, and error handling\n"
        "  - The function must be self-contained and importable\n"
        "Respond ONLY with a fenced ```python``` code block."
    )
    content: list[dict] = [
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_before_b64}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_after_b64}"},
        },
    ]
    return [_sys(_SKILL_ABSTRACTION_UI_CONTENT), HumanMessage(content=content)]
