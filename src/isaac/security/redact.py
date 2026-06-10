"""Secrets redaction — scrub credentials from tool outputs and traces.

Tool outputs flow into the model context, the run trace DB, and the terminal.
A leaked key in any of those is a real exfiltration path (e.g. a prompt-injected
web page asking the model to "repeat everything above"). Redaction is applied
to every tool output before it reaches the model or storage.

Patterns are deliberately high-precision (provider-prefixed tokens, key blocks,
explicit ``key=value`` assignments) — false positives would corrupt legitimate
data the agent is working on.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Provider-prefixed API tokens
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    # PEM/OpenSSH key blocks (multiline)
    (
        "private-key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
            re.DOTALL,
        ),
    ),
    # Explicit assignments: password=..., api_key: ..., secret = "..."
    (
        "credential-assignment",
        re.compile(
            r"(?i)\b(password|passwd|api[_-]?key|access[_-]?token|auth[_-]?token|secret"
            r"|client[_-]?secret)\b(\s*[=:]\s*)(['\"]?)[^\s'\"]{6,}\3"
        ),
    ),
]


def redact_secrets(text: str) -> str:
    """Replace recognised credentials in *text* with ``[REDACTED:<kind>]``."""
    if not text:
        return text
    for kind, pattern in _PATTERNS:
        if kind == "credential-assignment":
            text = pattern.sub(rf"\1\2[REDACTED:{kind}]", text)
        else:
            text = pattern.sub(f"[REDACTED:{kind}]", text)
    return text


def contains_secret(text: str) -> bool:
    """True when *text* matches any known credential pattern."""
    return any(p.search(text) for _, p in _PATTERNS)
