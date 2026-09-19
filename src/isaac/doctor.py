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
from urllib.parse import urlsplit

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
    severity = "warn"
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
        if is_primary:
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
        return CheckResult("ollama", severity, f"Check failed: {exc}")


def _check_selected_provider() -> CheckResult:
    try:
        from isaac.config.settings import get_settings
        from isaac.llm.provider import _local_endpoint, provider_api_key

        s = get_settings()
        provider = s.llm.llm_provider.strip().lower()
        if provider not in {"ollama", "openai", "anthropic", "llamacpp", "openai_compat"}:
            raise ValueError(f"Unsupported LLM provider: {provider!r}.")
        if not s.llm.model_name.strip():
            raise ValueError("ISAAC_MODEL_NAME must not be empty.")
        api_key = ""
        if provider in {"openai", "anthropic"}:
            api_key = provider_api_key(provider, s)
        if provider == "openai" and not s.llm.base_url:
            models = (s.llm.model_name, s.llm.fast_model, s.llm.strong_model)
            if any(model.endswith(":cloud") for model in models):
                raise ValueError(
                    "Ollama ':cloud' model tags cannot be sent to the default OpenAI endpoint. "
                    "Select ISAAC_LLM_PROVIDER=ollama or configure the intended provider/model."
                )
        endpoint = ""
        if provider == "openai_compat":
            endpoint = s.openai_compat_base_url
            if not endpoint:
                raise ValueError("ISAAC_OPENAI_COMPAT_BASE_URL is required for openai_compat.")
        elif provider == "llamacpp":
            endpoint = s.llamacpp_base_url or "http://localhost:8080"
        elif provider == "openai":
            endpoint = s.llm.base_url
        if endpoint:
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Selected provider endpoint must be an absolute http(s) URL.")
            if parsed.username or parsed.password:
                raise ValueError("Use the API key setting instead of credentials in endpoint URLs.")
            if parsed.port == 0:
                raise ValueError("Selected provider endpoint must use a nonzero port.")
            if _local_endpoint(endpoint):
                import httpx

                base = endpoint.rstrip("/")
                if not base.endswith("/v1"):
                    base += "/v1"
                if provider == "openai_compat":
                    api_key = getattr(s, "openai_compat_api_key", "")
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                response = httpx.get(
                    f"{base}/models", headers=headers, timeout=5, follow_redirects=False
                )
                response.raise_for_status()
                available = {entry["id"] for entry in response.json()["data"]}
                target_models = {s.llm.model_name, s.llm.fast_model, s.llm.strong_model} - {""}
                override = getattr(s, f"{provider}_model", "")
                if provider in {"llamacpp", "openai_compat"} and override:
                    target_models = {override}
                missing = sorted(
                    model
                    for model in target_models
                    if model not in available and f"{model}:latest" not in available
                )
                if missing:
                    raise ValueError(
                        "Selected local endpoint does not advertise configured model(s): "
                        f"{', '.join(missing)}. Select an available model or provision it locally."
                    )
                return CheckResult(
                    "selected-provider",
                    "ok",
                    f"{provider}: local endpoint reachable; configured models advertised. "
                    "Inference not probed.",
                )
        return CheckResult(
            "selected-provider",
            "ok",
            f"{provider}: configuration valid; remote authentication/model access not probed.",
        )
    except Exception as exc:
        return CheckResult("selected-provider", "fail", str(exc))


def _check_docker() -> CheckResult:
    try:
        import docker

        client = docker.from_env(timeout=5)
        try:
            client.ping()
        finally:
            client.close()
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


def _check_connectors() -> list[CheckResult]:
    """Document each connector's ``requires_env`` (presence only, never values)."""
    try:
        from isaac.skills.connectors.registry import get_registry
    except Exception as exc:
        return [CheckResult("connectors", "warn", f"Registry unavailable: {exc}")]
    try:
        reg = get_registry()
    except Exception as exc:
        return [CheckResult("connectors", "warn", f"Discovery failed: {exc}")]
    results: list[CheckResult] = []
    for name in sorted(reg):
        connector = reg[name]
        required = list(getattr(connector, "requires_env", []) or [])
        if not required:
            results.append(CheckResult(f"connector:{name}", "ok", "no env required"))
            continue
        try:
            from isaac.config.settings import env_is_set

            missing = [v for v in required if not env_is_set(v)]
        except Exception:
            missing = [v for v in required if not os.environ.get(v)]
        if not missing:
            results.append(
                CheckResult(f"connector:{name}", "ok", f"env present: {', '.join(required)}")
            )
        else:
            results.append(
                CheckResult(
                    f"connector:{name}",
                    "warn",
                    f"missing env: {', '.join(missing)} — see .env.example",
                )
            )
    return results


def _check_strict_config() -> CheckResult:
    """Strict mode: unknown ``ISAAC_*`` keys are probable typos (fail)."""
    try:
        from isaac.config.settings import find_unknown_env_vars

        unknown = find_unknown_env_vars()
    except Exception as exc:
        return CheckResult("config-strict", "warn", f"Strict check failed: {exc}")
    if unknown:
        return CheckResult(
            "config-strict",
            "fail",
            f"Unknown env var(s): {', '.join(unknown)}. "
            f"See .env.example for canonical names "
            f"(e.g. ISAAC_SANDBOX_TIMEOUT_SECONDS, not TIMEOUT).",
        )
    return CheckResult("config-strict", "ok", "No unknown ISAAC_* env vars.")


def run_checks(*, strict: bool = False) -> list[CheckResult]:
    """Run every preflight check and return the results (never raises)."""
    results = [
        _check_python(),
        _check_settings(),
        _check_selected_provider(),
        _check_ollama(),
        _check_docker(),
        _check_cloud_keys(),
    ]
    results.extend(_check_optional_deps())
    results.extend(_check_connectors())
    if strict:
        results.append(_check_strict_config())
        for result in results:
            if result.name == "docker" and result.status != "ok":
                result.status = "fail"

    return results


def has_failures(results: list[CheckResult]) -> bool:
    return any(r.status == "fail" for r in results)
