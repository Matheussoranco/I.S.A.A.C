"""Tests for the ``isaac doctor`` preflight checks."""

from __future__ import annotations

from isaac.doctor import CheckResult, has_failures, run_checks


def test_run_checks_covers_core_and_never_raises() -> None:
    results = run_checks()
    names = {r.name for r in results}
    assert {"python", "settings", "ollama", "docker", "cloud-fallback"} <= names
    assert any(n.startswith("extra:") for n in names)
    assert all(r.status in {"ok", "warn", "fail"} for r in results)


def test_python_check_passes_on_supported_interpreter() -> None:
    py = next(r for r in run_checks() if r.name == "python")
    assert py.status == "ok"


def test_settings_check_reads_nested_llm_config() -> None:
    s = next(r for r in run_checks() if r.name == "settings")
    assert s.status == "ok", s.detail
    assert "provider=" in s.detail


def test_has_failures_only_counts_fail() -> None:
    assert has_failures([CheckResult("x", "fail", "")]) is True
    assert has_failures([CheckResult("x", "warn", ""), CheckResult("y", "ok", "")]) is False
