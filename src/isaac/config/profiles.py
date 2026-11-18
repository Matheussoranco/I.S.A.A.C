"""Profile-aware writable configuration (Hermes-mirror, Phase 1.2).

Each profile owns a directory under ``~/.isaac/profiles/<name>/`` containing
a ``config.yaml`` whose keys mirror the Settings schema (nested, e.g.
``llm: {model_name: ...}``).  Profile overrides are layered on top of
environment variables and defaults (precedence: profile > env > default).

The active profile is chosen via the ``ISAAC_PROFILE`` env var
(default ``"default"``).  ``ISAAC_HOME``, when set, relocates the
``~/.isaac`` root — mostly useful for hermetic tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from isaac.config.settings import ProfileConfigError, is_secret_setting, validate_profile_overrides
from isaac.security.file_mutation import process_file_lock, publish_file, write_text

DEFAULT_PROFILE = "default"

_MAX_PROFILE_NAME_LENGTH = 64
_INVALID_PROFILE_CHARS = frozenset('/\\:*?"<>|')


def validate_profile_name(name: str) -> str:
    """Return a filesystem-safe profile name or raise ``ValueError``."""
    name = (name or "").strip() or DEFAULT_PROFILE
    if (
        name in {".", ".."}
        or len(name) > _MAX_PROFILE_NAME_LENGTH
        or name.startswith(".")
        or name.endswith((".", " "))
        or any(char in _INVALID_PROFILE_CHARS or ord(char) < 32 for char in name)
    ):
        raise ValueError(f"Invalid profile name: {name!r}")
    return name


def _isaac_root() -> Path:
    """Root ISAAC data dir; honours ISAAC_HOME for tests/sandboxes."""
    override = (os.environ.get("ISAAC_HOME") or os.environ.get("ISAAC_ISAAC_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".isaac"


def get_active_profile() -> str:
    """Return the active profile name (``ISAAC_PROFILE`` or ``"default"``)."""
    return (os.environ.get("ISAAC_PROFILE") or "").strip() or DEFAULT_PROFILE


def profile_dir(name: str, *, create: bool = True) -> Path:
    """Return the directory for profile *name*, optionally creating it."""
    name = validate_profile_name(name)
    d = _isaac_root() / "profiles" / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def profile_config_path(name: str, *, create: bool = True) -> Path:
    """Return the ``config.yaml`` path inside profile *name*'s directory."""
    return profile_dir(name, create=create) / "config.yaml"


def list_profiles() -> list[str]:
    """List profile names (subdirs of ``~/.isaac/profiles/``)."""
    root = _isaac_root() / "profiles"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


class _ProfileLoader(yaml.SafeLoader):
    def construct_document(self, node: yaml.Node) -> Any:
        # Inspect source mappings before merge expansion can shadow their keys.
        pending = [node]
        seen: set[yaml.Node] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if isinstance(current, yaml.MappingNode):
                keys: set[tuple[str, str]] = set()
                for key, value in current.value:
                    pending.extend((key, value))
                    if not isinstance(key, yaml.ScalarNode):
                        continue
                    tag = key.tag
                    if tag == "tag:yaml.org,2002:value":
                        tag = "tag:yaml.org,2002:str"
                    marker = (tag, key.value)
                    if marker in keys:
                        raise ProfileConfigError("config.yaml contains duplicate mapping keys.")
                    keys.add(marker)
                    if tag == "tag:yaml.org,2002:str" and is_secret_setting(key.value):
                        raise ProfileConfigError(
                            f"{key.value} belongs in .env or the environment, not config.yaml."
                        )
            elif isinstance(current, yaml.SequenceNode):
                pending.extend(current.value)
        return super().construct_document(node)


def parse_profile_yaml(content: bytes) -> dict[str, Any]:
    """Parse and validate profile YAML without exposing invalid input in errors."""
    try:
        data = yaml.load(content.decode("utf-8-sig"), Loader=_ProfileLoader)
    except ProfileConfigError:
        raise
    except (ValueError, yaml.YAMLError, RecursionError, LookupError, AttributeError, OverflowError):
        raise ProfileConfigError("Could not read config.yaml as YAML.") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ProfileConfigError("config.yaml must contain a mapping.")
    validate_profile_overrides(data)
    return data


def load_profile_overrides(name: str, *, strict: bool = False) -> dict[str, Any]:
    """Read a profile's YAML, optionally reporting invalid configuration."""
    try:
        return parse_profile_yaml(profile_config_path(name, create=False).read_bytes())
    except FileNotFoundError:
        return {}
    except ProfileConfigError:
        if strict:
            raise
    except (OSError, ValueError):
        if strict:
            raise ProfileConfigError("Could not read config.yaml for this profile.") from None
    return {}


def _read_profile_file(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def read_profile_yaml(name: str) -> bytes | None:
    """Read the exact bytes of a profile draft, including invalid YAML."""
    return _read_profile_file(profile_config_path(name, create=False))


def save_profile_yaml(name: str, content: bytes, *, expected: bytes | None) -> bool:
    """Save validated YAML if it changed and the on-disk draft still matches."""
    parse_profile_yaml(content)
    if content == (expected or b""):
        return False
    path = profile_config_path(name)
    with process_file_lock(path.with_suffix(".lock")):
        if _read_profile_file(path) != expected:
            raise ProfileConfigError("config.yaml changed while editing. Run config edit again.")
        publish_file(path, lambda stream: stream.write(content), overwrite=expected is not None)
    return True


def save_profile_overrides(name: str, data: dict[str, Any]) -> Path:
    """Validate and atomically replace the profile's YAML."""
    validate_profile_overrides(data)
    path = profile_config_path(name)
    content = yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
    with process_file_lock(path.with_suffix(".lock")):
        write_text(path, content, overwrite=True)
    return path


def sniff_scalar(raw: str) -> Any:
    """Best-effort scalar type-sniff for CLI input: int/float/bool else str."""
    low = raw.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
