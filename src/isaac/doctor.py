"""Preflight environment checks — ``isaac doctor``.

Verifies that the pieces I.S.A.A.C. needs (or can optionally use) are present
and reachable, and prints an actionable fix for everything that is not.  The
checks are deliberately dependency-light and never raise: a broken environment
is exactly when this command must still work.

Statuses:
    ok    — works
    warn  — optional capability missing/unreachable; agent degrades gracefully
    fail  — core requirement broken; the agent will not run correctly
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass

#: optional import -> capability it unlocks
_OPTIONAL_DEPS = {
    "playwright": "browser tool (pip install 'isaac[browser]')",
    "faster_whisper": "voice input (pip install 'isaac[voice]')",
    "mss": "screen capture (pip install 'isaac[vision]')",
    "fitz": "PDF extraction (pip install 'isaac[document]')",
    "z3": "theorem prover (pip install 'isaac[reasoning]')",
}


@dataclass
class CheckResult:
    """One preflight check outcome."""

    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str


def _check_python() -> CheckResult:
    v = sys.version_info
    if v >= (3, 10):
        return CheckResult("python", "ok", f"Python {v.major}.{v.minor}.{v.micro}")
    return CheckResult("python", "fail", f"Python {v.major}.{v.minor} found; >= 3.10 is required.")


def _check_settings() -> CheckResult:
    try:
        from isaac.config.settings import get_settings

        s = get_settings()
        return CheckResult(
            "settings",
            "ok",
            f"provider={s.llm.llm_provider} model={s.llm.model_name}",
        )
    except Exception as exc:
        return CheckResult("settings", "fail", f"Settings failed to load: {exc}")


def _check_ollama() -> CheckResult:
    """Check the default (local) backend: daemon reachable + models pulled.

    Escalates to ``fail`` when Ollama *is* the selected provider — in that
    configuration a missing daemon or model means the agent cannot run at all,
    and the fix is a single named command.
    """
    try:
        from isaac.config.settings import get_settings
        from isaac.llm.providers.ollama import health_check, is_model_installed, list_models

        s = get_settings()
        is_primary = s.llm.llm_provider == "ollama"
        severity = "fail" if is_primary else "warn"

        if not health_check(s.ollama_base_url):
            return CheckResult(
                "ollama",
                severity,
                f"Not reachable at {s.ollama_base_url} — run 'ollama serve' "
                "(install: https://ollama.com/download), or point "
                "ISAAC_LLM_PROVIDER at another backend.",
            )

        tags = list_models(s.ollama_base_url)
        if is_primary and tags:
            wanted = {s.llm.model_name, s.ollama_light_model, s.ollama_heavy_model}
            missing = sorted(m for m in wanted if m and not is_model_installed(m, tags))
            if missing:
                cmds = " ; ".join(f"ollama pull {m}" for m in missing)
                return CheckResult(
                    "ollama",
                    "fail",
                    f"Reachable, but configured model(s) not installed: "
                    f"{', '.join(missing)}. Run: {cmds}",
                )
        return CheckResult("ollama", "ok", f"Reachable; {len(tags)} model(s) installed.")
    except Exception as exc:
        return CheckResult("ollama", "warn", f"Check failed: {exc}")


def _check_docker() -> CheckResult:
    try:
        import docker

        docker.from_env().ping()
        return CheckResult("docker", "ok", "Engine reachable (sandboxed code enabled).")
    except Exception:
        return CheckResult(
            "docker",
            "warn",
            "Docker Engine not reachable — the sandboxed code tool is disabled. "
            "Start Docker Desktop / dockerd to enable it.",
        )


def _check_cloud_keys() -> CheckResult:
    have = [
        name
        for name, env in (("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"))
        if os.environ.get(env)
    ]
    if have:
        return CheckResult("cloud-fallback", "ok", f"Keys present: {', '.join(have)}.")
    return CheckResult(
        "cloud-fallback",
        "warn",
        "No cloud API keys set — fully local operation (cloud fallback disabled).",
    )


def _check_optional_deps() -> list[CheckResult]:
    results: list[CheckResult] = []
    for module, capability in _OPTIONAL_DEPS.items():
        present = importlib.util.find_spec(module) is not None
        results.append(
            CheckResult(
                f"extra:{module}",
                "ok" if present else "warn",
                "installed" if present else f"missing — {capability}",
            )
        )
    return results


def run_checks() -> list[CheckResult]:
    """Run every preflight check and return the results (never raises)."""
    results = [
        _check_python(),
        _check_settings(),
        _check_ollama(),
        _check_docker(),
        _check_cloud_keys(),
    ]
    results.extend(_check_optional_deps())
    return results


def has_failures(results: list[CheckResult]) -> bool:
    return any(r.status == "fail" for r in results)
