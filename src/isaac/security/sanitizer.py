"""I/O Sanitization — clean all text crossing the agent boundary.

Provides functions to sanitize:
- **Input**: user messages, tool outputs, external API responses.
- **Output**: LLM-generated text before it reaches tools, emails, files.

Defences
--------
1. Strip ANSI escape sequences.
2. Neutralise Markdown/HTML injection vectors.
3. Enforce maximum length limits.
4. Remove null bytes and other control characters.
5. Validate and sanitize file paths against traversal attacks.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ANSI escape sequence pattern (CSI + OSC + misc)
_ANSI_RE = re.compile(
    r"""
    \x1b          # ESC
    (?:
        \[[\d;]*[A-Za-z]   # CSI sequences
        | \].*?(?:\x07|\x1b\\)  # OSC sequences
        | [()][AB012]       # Charset selection
        | [=>]              # Misc
    )
    """,
    re.VERBOSE,
)

# Control characters to strip (except \n, \r, \t)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Null byte
_NULL_RE = re.compile(r"\x00")

# HTML tag pattern (basic)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Path traversal patterns
_TRAVERSAL_RE = re.compile(r"(?:^|[/\\])\.\.(?:[/\\]|$)")

# Default max lengths
MAX_INPUT_LENGTH = 100_000   # 100 KB
MAX_OUTPUT_LENGTH = 200_000  # 200 KB
MAX_PATH_LENGTH = 500


# ---------------------------------------------------------------------------
# Core sanitizers
# ---------------------------------------------------------------------------


def sanitize_text(
    text: str,
    *,
    max_length: int = MAX_INPUT_LENGTH,
    strip_ansi: bool = True,
    strip_html: bool = False,
    strip_control: bool = True,
) -> str:
    """Sanitize a text string by removing dangerous sequences.

    Parameters
    ----------
    text:
        Raw text to sanitize.
    max_length:
        Truncate to this many characters.
    strip_ansi:
        Remove ANSI escape sequences.
    strip_html:
        Remove HTML tags (set True for user-facing output).
    strip_control:
        Remove control characters (except \\n, \\r, \\t).
    """
    if not text:
        return ""

    # Null bytes first
    text = _NULL_RE.sub("", text)

    if strip_ansi:
        text = _ANSI_RE.sub("", text)

    if strip_control:
        text = _CONTROL_RE.sub("", text)

    if strip_html:
        text = _HTML_TAG_RE.sub("", text)

    # Enforce length
    if len(text) > max_length:
        text = text[:max_length]
        logger.debug("Text truncated to %d chars.", max_length)

    return text


def sanitize_input(text: str) -> str:
    """Sanitize user / external input before processing."""
    return sanitize_text(text, max_length=MAX_INPUT_LENGTH)


def sanitize_output(text: str) -> str:
    """Sanitize LLM output before sending to external systems."""
    return sanitize_text(
        text,
        max_length=MAX_OUTPUT_LENGTH,
        strip_html=True,
    )


# ---------------------------------------------------------------------------
# Path sanitization
# ---------------------------------------------------------------------------


def sanitize_path(
    path_str: str,
    *,
    root: Path | None = None,
    allow_absolute: bool = False,
) -> Path | None:
    """Sanitize a file path string.

    Returns the resolved path if safe, or ``None`` if the path is
    dangerous (traversal, too long, contains null bytes, etc.).

    Parameters
    ----------
    path_str:
        The raw path string.
    root:
        If provided, the path must resolve within this root. 
    allow_absolute:
        If ``False``, reject absolute paths.
    """
    if not path_str:
        return None

    # Null bytes
    if "\x00" in path_str:
        logger.warning("Path contains null byte — rejected.")
        return None

    # Length
    if len(path_str) > MAX_PATH_LENGTH:
        logger.warning("Path exceeds max length (%d) — rejected.", MAX_PATH_LENGTH)
        return None

    # Traversal check on raw string
    if _TRAVERSAL_RE.search(path_str):
        logger.warning("Path contains traversal pattern — rejected: %s", path_str[:50])
        return None

    # Absolute path check
    path = Path(path_str)
    if path.is_absolute() and not allow_absolute:
        logger.warning("Absolute path rejected: %s", path_str[:50])
        return None

    # Root confinement
    if root is not None:
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            logger.warning("Path escapes root boundary — rejected.")
            return None
        return resolved

    return path


# ---------------------------------------------------------------------------
# JSON sanitization
# ---------------------------------------------------------------------------


def sanitize_json_value(value: Any, *, max_depth: int = 10, _depth: int = 0) -> Any:
    """Recursively sanitize values in a JSON-like structure.

    - Strings are sanitized with :func:`sanitize_text`.
    - Dicts and lists are traversed recursively.
    - Depth is limited to prevent stack overflow.
    """
    if _depth > max_depth:
        return "<depth_limit>"

    if isinstance(value, str):
        return sanitize_text(value, max_length=MAX_INPUT_LENGTH)
    elif isinstance(value, dict):
        return {
            sanitize_text(str(k), max_length=200): sanitize_json_value(v, max_depth=max_depth, _depth=_depth + 1)
            for k, v in value.items()
        }
    elif isinstance(value, list):
        return [
            sanitize_json_value(v, max_depth=max_depth, _depth=_depth + 1)
            for v in value
        ]
    else:
        return value
