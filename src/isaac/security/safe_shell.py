"""Bounded read-only command dialect. Never launches a host process."""

from __future__ import annotations

import getpass
import os
import platform
import re
import shlex
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from isaac.security.workspace import allowed_roots, resolve_allowed

COMMANDS = frozenset(
    {
        "echo",
        "pwd",
        "whoami",
        "hostname",
        "date",
        "uname",
        "cat",
        "head",
        "tail",
        "wc",
        "ls",
        "dir",
        "grep",
    }
)
_BLOCKED = re.compile(r"[|;&`$><()\n\r\x00]")
_MAX_BYTES = 1_000_000


def _canonical_command_name(raw: str) -> str:
    """First-word allow-list check resistant to path/quote/extension bypass.

    Uses shlex tokenisation (quotes group spaces), then ``basename`` + lower +
    ``.exe`` strip, so ``"git"``, ``/bin/git``, ``C:\\Tools\\git.EXE`` all
    canonicalise to ``git`` before the allow-list test. Windows ``powershell
    -Command "<raw string>"`` would otherwise allow ``;|&$()`` chaining even
    when the first word is allow-listed — blocked here by :data:`_BLOCKED`
    on all platforms (no shell is ever spawned; this dialect never launches
    a host process).
    """
    name = os.path.basename(raw).lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def run_readonly(command: str, cwd: str | None, allowed: frozenset[str]) -> str:
    if len(command) > 16_000 or _BLOCKED.search(command):
        raise ValueError("Blocked metacharacters or oversized command")
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.escape = ""  # Preserve Windows paths; quotes still group spaces.
    parts = list(lexer)
    if not parts:
        raise ValueError("No command provided")
    name, *args = parts
    name = _canonical_command_name(name)
    if name not in COMMANDS or name not in allowed:
        raise ValueError(f"Command '{name}' not in the supported read-only allowlist")
    roots = allowed_roots()
    if not roots:
        raise ValueError("No allowed paths configured")
    base = resolve_allowed(cwd or roots[0])
    if base is None or not base.is_dir():
        raise ValueError("Working directory is outside allowed roots or unavailable")
    if name == "echo":
        return " ".join(args) + "\n"
    if name in {"pwd", "whoami", "hostname", "date", "uname"}:
        if args:
            raise ValueError("This command accepts no arguments")
        return {
            "pwd": str(base),
            "whoami": getpass.getuser(),
            "hostname": platform.node(),
            "date": datetime.now(UTC).isoformat(),
            "uname": platform.system(),
        }[name] + "\n"

    def path(value: str) -> Path:
        raw = Path(value)
        target = resolve_allowed(raw if raw.is_absolute() else base / raw)
        if target is None:
            raise ValueError("Path is outside allowed roots or is a protected location")
        return target

    if name in {"ls", "dir"}:
        if len(args) > 1 or (args and args[0].startswith("-")):
            raise ValueError("Use ls [directory]; recursive traversal and flags are unsupported")
        target = path(args[0] if args else ".")
        if not target.is_dir():
            raise ValueError("Not a directory")
        entries = [p.name for p in islice(target.iterdir(), 2000) if resolve_allowed(p) is not None]
        return "\n".join(sorted(entries))[:10_000]
    count = 10
    if name in {"head", "tail"} and args[:1] == ["-n"]:
        if len(args) != 3:
            raise ValueError("Use head/tail [-n count] file")
        count = int(args[1])
        if not 1 <= count <= 1000:
            raise ValueError("Line count must be between 1 and 1000")
        args = args[2:]
    pattern = ""
    if name == "grep":
        if len(args) != 2:
            raise ValueError("Use grep literal-text file; flags and regex are unsupported")
        pattern, args = args[0], args[1:]
    if len(args) != 1 or args[0].startswith("-"):
        raise ValueError("Exactly one file is required; flags are unsupported")
    target = path(args[0])
    if not target.is_file():
        raise ValueError("Not a regular file")
    with target.open("rb") as stream:
        data = stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError("File exceeds the 1 MB read limit")
    content = data.decode("utf-8", errors="replace")
    lines = content.splitlines(keepends=True)
    if name == "head":
        content = "".join(lines[:count])
    elif name == "tail":
        content = "".join(lines[-count:])
    elif name == "wc":
        content = f"{len(lines)} {len(content.split())} {len(data)}\n"
    elif name == "grep":
        content = "".join(line for line in lines if pattern in line)
    return content[:10_000]
