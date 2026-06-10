"""Tool-argument validation against the tool's JSON-Schema signature.

Small local models frequently emit tool calls with a missing required field or
a wrong type. Executing those blindly produces confusing stack traces; instead
the loop validates first and feeds a precise correction back to the model so
the *next* attempt is well-formed (ROADMAP-1.0 WS2/WS3).

This is a focused subset of JSON Schema — type checks for the primitive types
tools actually use — not a full validator; tools keep their own runtime guards.
"""

from __future__ import annotations

from typing import Any

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty = valid) for *args* against *schema*.

    Checks: required fields present, declared property types match, and no
    unknown properties (a strong signal of a hallucinated parameter name).
    """
    problems: list[str] = []
    properties: dict[str, Any] = schema.get("properties") or {}
    required = schema.get("required") or []

    for name in required:
        if name not in args or args[name] is None:
            problems.append(f"missing required parameter '{name}'")

    for name, value in args.items():
        spec = properties.get(name)
        if spec is None:
            if properties:  # only flag when the tool declares a signature
                known = ", ".join(sorted(properties))
                problems.append(f"unknown parameter '{name}' (expected one of: {known})")
            continue
        declared = spec.get("type")
        expected = _TYPE_CHECKS.get(str(declared)) if declared else None
        if expected is None or value is None:
            continue
        # bool is an int subclass — don't let True pass as integer/number.
        if isinstance(value, bool) and declared in ("number", "integer"):
            problems.append(f"parameter '{name}' must be {declared}, got boolean")
        elif not isinstance(value, expected):
            problems.append(f"parameter '{name}' must be {declared}, got {type(value).__name__}")
    return problems
