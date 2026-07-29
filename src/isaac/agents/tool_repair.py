"""Salvage tool calls that a small model emitted in the wrong shape.

Large models reliably use the provider's native function-calling channel.
Small local models frequently do not: they *intend* a tool call but emit it as
prose — a fenced ``json`` block, a Hermes-style ``<tool_call>`` tag, a bare
object with ``arguments`` spelled ``args``, or a Python-looking call
expression.  The provider reports **no** tool calls for those turns, so
:class:`~isaac.agents.agent_loop.AgentLoop` used to treat the blob as the
model's *final answer* and stop — the single largest source of small-model task
failure in the 1.3.x line.

This module recovers those turns.  :func:`salvage_tool_calls` recognises the
dialects observed in practice and returns normalised call dicts in the same
shape LangChain produces (``{"name", "args", "id"}``), so the caller can splice
them into its existing path.  When nothing is salvageable but the text still
*looks* like an attempted call, :func:`looks_like_attempted_call` says so and
:func:`reflexion_prompt` builds the corrective message that asks the model to
try again — a Reflexion-style retry (Shinn et al., 2023) rather than a silent
failure.

Everything here is pure text processing: no network, no LLM, no provider
imports.  That keeps it fast enough to run on every turn and unit-testable
offline.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RepairOutcome",
    "looks_like_attempted_call",
    "reflexion_prompt",
    "repair_json",
    "salvage_tool_calls",
]

# Keys different model families use for the tool name and its arguments.
_NAME_KEYS = ("name", "tool", "tool_name", "function", "action", "recipient_name")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "tool_input", "action_input")

# ``<tool_call>{...}</tool_call>`` (Hermes / Qwen), also ``<function_call>``.
_TAG_RE = re.compile(
    r"<\s*(tool_call|function_call|tool)\s*>(?P<body>.*?)<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
# Fenced block, optionally tagged ```json / ```tool_code / ```python.
_FENCE_RE = re.compile(
    r"```(?:json|tool_code|tool_call|python|py)?\s*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)
# ``search_web(query="cats", limit=3)`` — a Python-style call expression.
_PYCALL_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>.*?)\)", re.DOTALL)

# Models that were fine-tuned on prose sometimes emit typographic quotes inside
# JSON. Built once at import rather than per call.
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "’": "'"})

# Phrases that signal the model was *trying* to act rather than answer.
_INTENT_RE = re.compile(
    r"\b(tool_call|function_call|i (?:will|'ll|should|need to) (?:call|use|invoke)|"
    r"calling the|use the .{0,24}tool)\b",
    re.IGNORECASE,
)
# Structural markers of a call envelope. Kept separate from _INTENT_RE because
# these are quoted JSON keys: a ``\b`` anchor cannot match against a quote, so
# folding them into the phrase pattern above silently never fires.
_ENVELOPE_KEY_RE = re.compile(
    r"[\"']\s*(arguments|tool_name|tool_call|function_call|parameters)\s*[\"']\s*:",
    re.IGNORECASE,
)


class RepairOutcome:
    """How a turn's text was classified.  Used for metrics, not control flow."""

    NATIVE = "native"  # provider returned real tool calls
    REPAIRED = "repaired"  # we parsed calls out of the text
    REFLEXION = "reflexion"  # unparseable, but a retry was warranted
    FINAL = "final"  # genuinely a final answer


def repair_json(raw: str) -> Any | None:
    """Parse JSON that a small model *nearly* got right.

    Handles, in order of escalation: clean JSON; Python literals (single
    quotes, ``True``/``None``); trailing commas; unquoted keys; and smart
    quotes.  Returns ``None`` if the text cannot be coerced into a value.
    """
    text = (raw or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    # Python dict syntax: single quotes, True/False/None.
    try:
        value = ast.literal_eval(text)
    except Exception:
        pass
    else:
        if isinstance(value, (dict, list)):
            return value

    patched = text
    # Smart quotes → straight quotes.
    patched = patched.translate(_SMART_QUOTES)
    # Trailing commas before a closing brace/bracket.
    patched = re.sub(r",\s*([}\]])", r"\1", patched)
    # Bare keys: {name: "x"} → {"name": "x"}.
    patched = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', patched)
    try:
        return json.loads(patched)
    except Exception:
        return None


def _extract_balanced(text: str) -> list[str]:
    """Return every balanced ``{...}`` region in *text*, outermost first.

    A brace-counting scan rather than a regex, because tool arguments nest and
    regexes cannot match balanced delimiters.  Braces inside string literals
    are skipped so ``{"q": "a } b"}`` stays intact.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    quote = ""
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
            elif depth < 0:
                depth = 0
    return out


def _first_key(obj: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def _normalise(obj: Any, known: set[str] | None) -> dict[str, Any] | None:
    """Turn one candidate object into a ``{"name", "args", "id"}`` call."""
    if not isinstance(obj, dict):
        return None

    name = _first_key(obj, _NAME_KEYS)
    # ``{"function": {"name": ..., "arguments": ...}}`` — OpenAI wire shape.
    if isinstance(name, dict):
        inner = name
        name = _first_key(inner, _NAME_KEYS)
        args = _first_key(inner, _ARG_KEYS)
    else:
        args = _first_key(obj, _ARG_KEYS)

    if not isinstance(name, str) or not name.strip():
        # ``{"search_web": {"query": "cats"}}`` — tool name *as* the only key.
        if known and len(obj) == 1:
            solo_key, solo_val = next(iter(obj.items()))
            if solo_key in known and isinstance(solo_val, dict):
                return {"name": solo_key, "args": solo_val, "id": f"repair_{uuid.uuid4().hex[:8]}"}
        return None

    name = name.strip()
    if known and name not in known:
        return None

    if isinstance(args, str):
        # Arguments double-encoded as a JSON string (common with Ollama shims).
        parsed = repair_json(args)
        args = parsed if isinstance(parsed, dict) else {}
    elif not isinstance(args, dict):
        # No argument key at all: treat any non-name keys as the arguments.
        leftover = {
            k: v
            for k, v in obj.items()
            if str(k).lower() not in _NAME_KEYS and str(k).lower() not in _ARG_KEYS
        }
        args = leftover if leftover else {}

    return {"name": name, "args": args, "id": f"repair_{uuid.uuid4().hex[:8]}"}


def _from_pycall(text: str, known: set[str]) -> list[dict[str, Any]]:
    """Parse ``tool(arg="x", n=3)`` expressions naming a known tool."""
    calls: list[dict[str, Any]] = []
    for match in _PYCALL_RE.finditer(text):
        name = match.group("name")
        if name not in known:
            continue
        try:
            node = ast.parse(match.group(0).strip(), mode="eval").body
        except Exception:
            continue
        if not isinstance(node, ast.Call):
            continue
        args: dict[str, Any] = {}
        ok = True
        for kw in node.keywords:
            if kw.arg is None:
                ok = False
                break
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                ok = False
                break
        if ok and args:
            calls.append({"name": name, "args": args, "id": f"repair_{uuid.uuid4().hex[:8]}"})
    return calls


def salvage_tool_calls(
    text: str,
    known_tools: set[str] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recover tool calls embedded in free-form model output.

    Parameters
    ----------
    text:
        The assistant turn's text content.
    known_tools:
        Names currently bound to the loop.  Strongly recommended: it is what
        separates a real call from prose that merely *mentions* JSON, and it
        gates the riskier dialects (bare Python calls, name-as-key objects)
        that would otherwise fire on ordinary text.

    Returns
    -------
    list of dict
        Normalised ``{"name", "args", "id"}`` calls, in the order found.
        Empty when nothing looked like a call.
    """
    if not text or not text.strip():
        return []
    known = {str(t) for t in known_tools} if known_tools else None

    calls: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(call: dict[str, Any] | None) -> None:
        if not call:
            return
        try:
            sig = f"{call['name']}:{json.dumps(call['args'], sort_keys=True, default=str)}"
        except Exception:
            sig = f"{call['name']}:{call['args']!r}"
        if sig not in seen:
            seen.add(sig)
            calls.append(call)

    # 1. Explicit tags win — they are unambiguous when present.
    for match in _TAG_RE.finditer(text):
        body = match.group("body").strip()
        parsed = repair_json(body)
        if isinstance(parsed, list):
            for item in parsed:
                _add(_normalise(item, known))
        else:
            _add(_normalise(parsed, known))
        if parsed is None:
            for blob in _extract_balanced(body):
                _add(_normalise(repair_json(blob), known))
    if calls:
        return calls

    # 2. Fenced blocks.
    for match in _FENCE_RE.finditer(text):
        body = match.group("body").strip()
        parsed = repair_json(body)
        if isinstance(parsed, list):
            for item in parsed:
                _add(_normalise(item, known))
        elif parsed is not None:
            _add(_normalise(parsed, known))
        else:
            for blob in _extract_balanced(body):
                _add(_normalise(repair_json(blob), known))
        if not calls and known:
            calls.extend(_from_pycall(body, known))
    if calls:
        return calls

    # 3. Bare balanced objects anywhere in the text.
    for blob in _extract_balanced(text):
        _add(_normalise(repair_json(blob), known))
    if calls:
        return calls

    # 4. Last resort: a Python-style call naming a known tool.
    if known:
        for call in _from_pycall(text, known):
            _add(call)
    return calls


def looks_like_attempted_call(text: str, known_tools: set[str] | list[str] | None = None) -> bool:
    """True when *text* reads as a failed tool call rather than an answer.

    Used to decide whether an unparseable turn deserves a Reflexion retry.
    Deliberately conservative: a false positive costs one wasted LLM call,
    while a false negative silently returns a JSON blob to the user.
    """
    if not text or not text.strip():
        return False
    if _INTENT_RE.search(text) or _ENVELOPE_KEY_RE.search(text):
        return True
    known = {str(t) for t in known_tools} if known_tools else set()
    if known and any(re.search(rf"\b{re.escape(n)}\s*\(", text) for n in known):
        return True
    # A brace-heavy turn that mentions a bound tool name.
    return "{" in text and bool(known) and any(n in text for n in known)


def reflexion_prompt(
    text: str,
    known_tools: set[str] | list[str] | None = None,
    error: str = "",
) -> str:
    """Build the corrective turn shown to a model that emitted a bad call.

    Names the specific defect, restates the required shape, and lists the legal
    tool names.  Small models correct far more reliably from a concrete
    contract than from a generic "try again".
    """
    names = sorted({str(t) for t in known_tools}) if known_tools else []
    tool_line = f"Available tools: {', '.join(names)}.\n" if names else ""
    detail = f"Parser error: {error}\n" if error else ""
    excerpt = (text or "").strip()
    if len(excerpt) > 400:
        excerpt = excerpt[:400] + " …"
    return (
        "Your last message looked like a tool call but could not be parsed, so "
        "no tool ran.\n"
        f"{detail}{tool_line}"
        "Do not describe the call in prose and do not wrap it in a code fence. "
        "Emit the tool call using the official tool-calling format.\n"
        "If you instead have the final answer, reply with the answer as plain "
        "text and no JSON.\n\n"
        f"Your unparseable message was:\n{excerpt}"
    )
