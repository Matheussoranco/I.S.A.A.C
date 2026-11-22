"""Validate that a Git release tag matches the package and built wheel version.

Used by ``.github/workflows/release.yml`` to stop a PyPI publication before
upload whenever a pushed tag (e.g. ``refs/tags/v1.6.3``) disagrees with
``pyproject.toml`` or the built ``wheelhouse/*.whl`` distribution metadata.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


def normalize_tag(tag_or_ref: str) -> str:
    """Return the bare tag name from a tag or ``refs/tags/<tag>`` ref.

    Raises ``ValueError`` for empty values, whitespace, or non-tag refs such
    as ``refs/heads/main``.
    """
    if not tag_or_ref or any(ch.isspace() for ch in tag_or_ref):
        raise ValueError(f"Invalid release tag/ref: {tag_or_ref!r}")
    raw = tag_or_ref
    if raw.startswith("refs/"):
        if not raw.startswith("refs/tags/"):
            raise ValueError(f"Expected a refs/tags/ ref, got: {raw!r}")
        tag = raw.removeprefix("refs/tags/")
    else:
        tag = raw
    if not tag or "/" in tag:
        raise ValueError(f"Invalid release tag name: {tag_or_ref!r}")
    return tag


def read_pyproject_version(pyproject_path: Path) -> str:
    """Read ``[project].version`` from ``pyproject.toml``."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"Missing [project] table in {pyproject_path}")
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Missing non-empty [project].version in {pyproject_path}")
    return version.strip()


def read_wheel_version(wheel_path: Path) -> str:
    """Read the ``Version`` field from a wheel's ``*.dist-info/METADATA``."""
    with zipfile.ZipFile(wheel_path) as zf:
        metadata_names = [name for name in zf.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(
                f"Expected exactly 1 .dist-info/METADATA in {wheel_path.name}, "
                f"found {len(metadata_names)}"
            )
        raw_metadata = zf.read(metadata_names[0]).decode("utf-8")
    parsed = Parser().parsestr(raw_metadata)
    version = parsed.get("Version", "").strip()
    if not version:
        raise ValueError(f"Missing Version header in {wheel_path.name}:{metadata_names[0]}")
    return version


def find_single_wheel(wheel_dir: Path) -> Path:
    """Return the unique ``*.whl`` in ``wheel_dir`` or raise ``ValueError``."""
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        names = [w.name for w in wheels]
        raise ValueError(f"Expected exactly 1 wheel in {wheel_dir}, found {len(wheels)}: {names}")
    return wheels[0]


def validate_release_tag(
    *,
    tag_or_ref: str,
    pyproject_path: Path | None = None,
    wheel_dir: Path | None = None,
) -> str:
    """Validate that ``tag_or_ref`` equals ``v<version>`` for all supplied sources.

    Returns the verified package version string on success; raises
    ``ValueError`` on any missing source, version mismatch, or malformed tag.
    """
    if pyproject_path is None and wheel_dir is None:
        raise ValueError("At least one of pyproject_path or wheel_dir must be provided")

    tag = normalize_tag(tag_or_ref)
    versions: dict[str, str] = {}

    if pyproject_path is not None:
        versions["pyproject.toml"] = read_pyproject_version(pyproject_path)

    if wheel_dir is not None:
        wheel = find_single_wheel(wheel_dir)
        versions[wheel.name] = read_wheel_version(wheel)

    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        details = ", ".join(f"{src}={ver}" for src, ver in versions.items())
        raise ValueError(f"Package version mismatch across artifacts: {details}")

    version = next(iter(unique_versions))
    expected_tag = f"v{version}"
    if tag != expected_tag:
        details = ", ".join(f"{src}={ver}" for src, ver in versions.items())
        raise ValueError(
            f"Release tag {tag!r} does not match expected {expected_tag!r} ({details})"
        )
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate release Git tag against pyproject.toml and/or built wheel."
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Git tag or full refs/tags/<tag> ref (defaults to GITHUB_REF or GITHUB_REF_NAME).",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=None,
        help="Path to pyproject.toml to validate against.",
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=None,
        help="Directory containing the built wheel (*.whl) to validate against.",
    )
    args = parser.parse_args(argv)

    tag_or_ref = args.tag or os.environ.get("GITHUB_REF") or os.environ.get("GITHUB_REF_NAME", "")
    try:
        version = validate_release_tag(
            tag_or_ref=tag_or_ref,
            pyproject_path=args.pyproject,
            wheel_dir=args.wheel_dir,
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Release tag validation FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Release tag validation PASSED: {normalize_tag(tag_or_ref)} == v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
