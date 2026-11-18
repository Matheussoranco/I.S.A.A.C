from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from isaac.config.profiles import read_profile_yaml, save_profile_yaml
from isaac.config.settings import ProfileConfigError


def _default_editor() -> str:
    if sys.platform == "win32":
        import ctypes

        directory = ctypes.create_unicode_buffer(260)
        size = ctypes.windll.kernel32.GetSystemDirectoryW(directory, len(directory))
        if 0 < size < len(directory):
            return str(Path(directory.value) / "notepad.exe")
    elif executable := shutil.which("vi", path=os.defpath):
        return executable
    raise ProfileConfigError("No default editor found. Set VISUAL or EDITOR.")


def _editor_args() -> list[str]:
    command = (os.environ.get("VISUAL") or "").strip() or (os.environ.get("EDITOR") or "").strip()
    if not command:
        return [_default_editor()]
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    if os.name == "nt":
        lexer.escape = ""
    try:
        args = list(lexer)
    except ValueError:
        raise ProfileConfigError("Invalid editor command in VISUAL or EDITOR.") from None
    if not args or not args[0]:
        raise ProfileConfigError("Invalid editor command in VISUAL or EDITOR.")
    return args


def edit_profile_config(profile: str) -> bool:
    """Validate an edited draft before publishing it; return whether it changed."""
    original = read_profile_yaml(profile)
    args = _editor_args()
    with tempfile.TemporaryDirectory(prefix="isaac-config-") as workspace:
        draft = Path(workspace) / "config.yaml"
        draft.write_bytes(original or b"")
        try:
            result = subprocess.run([*args, str(draft)], check=False)
        except (OSError, ValueError):
            raise ProfileConfigError(
                "Could not launch the editor. Check VISUAL or EDITOR."
            ) from None
        if result.returncode:
            raise ProfileConfigError(f"Editor exited with status {result.returncode}.")
        return save_profile_yaml(profile, draft.read_bytes(), expected=original)
