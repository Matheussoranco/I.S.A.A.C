"""Tests for release tag validation and SHA-pinned release workflow (#64)."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_release_tag.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

_PINNED_SHA_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*@[0-9a-f]{40}$")


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_release_tag", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_synthetic_wheel(wheel_dir: Path, version: str, name: str = "isaac") -> Path:
    wheel_path = wheel_dir / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n",
        )
    return wheel_path


def test_matching_synthetic_tag_passes_pyproject_and_wheel(tmp_path: Path) -> None:
    mod = _load_validator()
    pkg_version = mod.read_pyproject_version(PYPROJECT_PATH)
    _make_synthetic_wheel(tmp_path, pkg_version)

    assert (
        mod.validate_release_tag(
            tag_or_ref=f"v{pkg_version}",
            pyproject_path=PYPROJECT_PATH,
            wheel_dir=tmp_path,
        )
        == pkg_version
    )
    assert (
        mod.validate_release_tag(
            tag_or_ref=f"refs/tags/v{pkg_version}",
            pyproject_path=PYPROJECT_PATH,
            wheel_dir=tmp_path,
        )
        == pkg_version
    )


@pytest.mark.parametrize(
    "bad_tag",
    [
        "v0.0.1",
        "v99.99.99",
        "1.6.3",  # missing leading 'v'
        "refs/tags/v0.0.1",
        "refs/heads/v1.6.3",  # branch ref, not tag ref
        "v1.6.3-extra",
        "",
        "v1.6.3 ",
    ],
)
def test_mismatched_or_invalid_synthetic_tags_fail(tmp_path: Path, bad_tag: str) -> None:
    mod = _load_validator()
    pkg_version = mod.read_pyproject_version(PYPROJECT_PATH)
    _make_synthetic_wheel(tmp_path, pkg_version)

    with pytest.raises(ValueError):
        mod.validate_release_tag(
            tag_or_ref=bad_tag,
            pyproject_path=PYPROJECT_PATH,
            wheel_dir=tmp_path,
        )


def test_mismatched_wheel_metadata_against_pyproject_fails(tmp_path: Path) -> None:
    mod = _load_validator()
    pkg_version = mod.read_pyproject_version(PYPROJECT_PATH)
    _make_synthetic_wheel(tmp_path, "9.9.9")

    with pytest.raises(ValueError, match="mismatch"):
        mod.validate_release_tag(
            tag_or_ref=f"v{pkg_version}",
            pyproject_path=PYPROJECT_PATH,
            wheel_dir=tmp_path,
        )


def test_multiple_wheels_in_wheelhouse_fail(tmp_path: Path) -> None:
    mod = _load_validator()
    pkg_version = mod.read_pyproject_version(PYPROJECT_PATH)
    _make_synthetic_wheel(tmp_path, pkg_version, name="isaac")
    _make_synthetic_wheel(tmp_path, "0.0.1", name="other")

    with pytest.raises(ValueError, match="Expected exactly 1 wheel"):
        mod.validate_release_tag(
            tag_or_ref=f"v{pkg_version}",
            pyproject_path=PYPROJECT_PATH,
            wheel_dir=tmp_path,
        )


def test_cli_matching_and_mismatched_synthetic_tags(tmp_path: Path) -> None:
    mod = _load_validator()
    pkg_version = mod.read_pyproject_version(PYPROJECT_PATH)
    _make_synthetic_wheel(tmp_path, pkg_version)

    ok = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--tag",
            f"refs/tags/v{pkg_version}",
            "--pyproject",
            str(PYPROJECT_PATH),
            "--wheel-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert "PASSED" in ok.stdout

    bad = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--tag",
            "refs/tags/v0.0.0",
            "--pyproject",
            str(PYPROJECT_PATH),
            "--wheel-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert "FAILED" in bad.stderr


def test_release_workflow_pins_all_actions_to_full_commit_shas_and_gates_publish() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert {"build-wheel", "test-wheel", "publish-pypi"}.issubset(jobs)

    # Every uses: reference in every job must be pinned to a 40-hex commit SHA.
    seen_uses: list[str] = []
    for job_name, job in jobs.items():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if uses:
                seen_uses.append(uses)
                assert _PINNED_SHA_RE.match(uses), (
                    f"Unpinned action in release.yml job {job_name!r}: {uses!r}"
                )

    assert any(u.startswith("actions/download-artifact@") for u in seen_uses)
    assert any(u.startswith("pypa/gh-action-pypi-publish@") for u in seen_uses)

    # Keep the three-platform installed-wheel matrix intact.
    assert set(jobs["test-wheel"]["strategy"]["matrix"]["os"]) == {
        "ubuntu-latest",
        "macos-latest",
        "windows-latest",
    }

    # Verify tag validation runs before artifact upload in build-wheel and before PyPI publish.
    build_steps = jobs["build-wheel"]["steps"]
    validate_idx = next(
        i for i, s in enumerate(build_steps) if "validate_release_tag.py" in s.get("run", "")
    )
    upload_idx = next(
        i
        for i, s in enumerate(build_steps)
        if s.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert validate_idx < upload_idx

    publish_steps = jobs["publish-pypi"]["steps"]
    gate_idx = next(
        i for i, s in enumerate(publish_steps) if "Pre-publish gate" in s.get("run", "")
    )
    publish_idx = next(
        i
        for i, s in enumerate(publish_steps)
        if s.get("uses", "").startswith("pypa/gh-action-pypi-publish@")
    )
    assert gate_idx < publish_idx
