"""Cheap verifiers for :func:`isaac.reasoning.test_time.best_of_n`.

Best-of-N is only worth its compute if checking an answer is much cheaper than
producing one.  Every verifier here is pure local computation — a parse, a
compile, a schema match, a numeric range — with no LLM call and no network.
Each returns a score in ``[0, 1]``; ``1.0`` means "accept and stop sampling".

These check *well-formedness*, not truth.  A syntactically valid program can
still be wrong.  That is the honest limit of a cheap verifier, and it is why
:func:`~isaac.reasoning.test_time.solve_hard_step` falls back to agreement
voting when the verifier never accepts.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "all_of",
    "json_verifier",
    "non_empty_verifier",
    "numeric_range_verifier",
    "python_syntax_verifier",
    "regex_verifier",
    "schema_verifier",
]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    return str(value)


def non_empty_verifier(min_chars: int = 1) -> Callable[[Any], float]:
    """Accept any answer with at least *min_chars* of non-whitespace text.

    The weakest useful check — it only rejects empty or truncated generations,
    which small models produce more often than one would like.
    """

    def verify(value: Any) -> float:
        return 1.0 if len(_text(value).strip()) >= min_chars else 0.0

    return verify


def python_syntax_verifier(require_expr: bool = False) -> Callable[[Any], float]:
    """Accept answers that parse as Python.

    Strips a surrounding markdown fence first, since models fence code even
    when told not to.  With *require_expr*, only a single expression is
    accepted (useful when the step should yield a value, not a program).
    """
    fence = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)

    def verify(value: Any) -> float:
        src = _text(value).strip()
        match = fence.search(src)
        if match:
            src = match.group(1).strip()
        if not src:
            return 0.0
        try:
            ast.parse(src, mode="eval" if require_expr else "exec")
        except SyntaxError:
            return 0.0
        except Exception:  # pragma: no cover - defensive
            return 0.0
        return 1.0

    return verify


def json_verifier(require_keys: Sequence[str] = ()) -> Callable[[Any], float]:
    """Accept answers that are valid JSON containing every key in *require_keys*.

    Scores partially when the JSON parses but keys are missing, so best-of-N
    can still rank a near-miss above garbage.
    """
    fence = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
    keys = list(require_keys)

    def verify(value: Any) -> float:
        raw = _text(value).strip()
        match = fence.search(raw)
        if match:
            raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            return 0.0
        if not keys:
            return 1.0
        if not isinstance(obj, dict):
            return 0.5
        present = sum(1 for k in keys if k in obj)
        return 1.0 if present == len(keys) else 0.5 * (present / len(keys))

    return verify


def schema_verifier(schema: dict[str, Any]) -> Callable[[Any], float]:
    """Accept answers validating against a JSON Schema.

    Uses ``jsonschema`` when installed and degrades to a required-keys check
    otherwise — ISAAC runs without optional dependencies by design, and a
    weaker verifier is better than an import error.
    """
    required = list(schema.get("required") or [])
    fallback = json_verifier(required)

    def verify(value: Any) -> float:
        try:
            import jsonschema
        except ImportError:
            return fallback(value)
        raw = _text(value).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            return 0.0
        try:
            jsonschema.validate(obj, schema)
        except Exception:
            return 0.0
        return 1.0

    return verify


def numeric_range_verifier(
    low: float | None = None,
    high: float | None = None,
) -> Callable[[Any], float]:
    """Accept answers containing a number within ``[low, high]``.

    Reads the *last* number in the text: chain-of-thought output states the
    result after its working, so the final figure is the answer.
    """
    number = re.compile(r"-?\d+(?:\.\d+)?")

    def verify(value: Any) -> float:
        found = number.findall(_text(value).replace(",", ""))
        if not found:
            return 0.0
        try:
            parsed = float(found[-1])
        except ValueError:  # pragma: no cover
            return 0.0
        if low is not None and parsed < low:
            return 0.0
        if high is not None and parsed > high:
            return 0.0
        return 1.0

    return verify


def regex_verifier(pattern: str, flags: int = 0) -> Callable[[Any], float]:
    """Accept answers matching *pattern* anywhere."""
    compiled = re.compile(pattern, flags)

    def verify(value: Any) -> float:
        return 1.0 if compiled.search(_text(value)) else 0.0

    return verify


def all_of(*verifiers: Callable[[Any], float | bool]) -> Callable[[Any], float]:
    """Combine verifiers into their mean score.

    Averaging rather than requiring unanimity keeps the score informative for
    ranking: an answer passing two of three checks should outrank one passing
    none, even though neither is accepted outright.
    """

    def verify(value: Any) -> float:
        if not verifiers:
            return 0.0
        total = 0.0
        for fn in verifiers:
            try:
                total += float(fn(value))
            except Exception:  # pragma: no cover
                logger.debug("verifier raised", exc_info=True)
        return total / len(verifiers)

    return verify
