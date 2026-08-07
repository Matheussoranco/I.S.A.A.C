"""Skill verification gate — a skill must *run* before it is promoted.

Before 1.5.0 the skill library promoted whatever the abstraction node handed
it: an LLM generalised some code, ``SkillLibrary.commit`` wrote it to disk, and
the library grew with code nobody had ever executed.  Roadmap WS6 calls for
"only promote skills that pass a verification run" — this is that gate.

What is checked
---------------
Every candidate runs through an ordered list of checks; the first hard failure
stops the pipeline and the skill is **rejected** (recorded, not silently
dropped):

``syntax``
    ``ast.parse`` succeeds.
``callable``
    At least one module-level ``def``/``async def``/``class`` is defined —
    a "skill" that is a bare script is not reusable.
``import``
    The module executes top-to-bottom without raising. This is the check that
    catches the common LLM failure mode: a skill referencing a name that only
    existed in the original task's scope.
``doctest``
    If the source carries doctests, they must pass.
``selftest``
    If the source defines ``_selftest()``, it must run and not raise (and must
    not return ``False``).
``example``
    If ``input_schema`` carries an ``example`` mapping, the primary function is
    called with it and must not raise.

The last three are *conditional*: absent, they are reported as ``skipped``,
which is honest — a skill whose only evidence is "it imports" is recorded as
``evidence="import"``, not as if it had been behaviourally tested.

Where it runs
-------------
In a subprocess (``python -I``) with a wall-clock timeout, in a throwaway
temp directory. Docker's :class:`~isaac.sandbox.executor.CodeExecutor` is not
used here: its host-side pre-check blocks ``os``/``sys``/``pathlib``, which
most real skills legitimately import, so it would reject good skills rather
than bad ones. Set ``ISAAC_SKILL_VERIFICATION_REQUIRE_SANDBOX=true`` to refuse
promotion outright when Docker is absent.

Note the threat model: this code was *about to be written into the skill
library and executed later anyway*. Running it once, isolated and time-boxed,
is strictly safer than promoting it unexecuted.
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Marker the harness prints so its verdict can be found in noisy stdout.
_RESULT_MARKER = "__ISAAC_SKILL_VERIFY__"

_HARNESS = """
import doctest, inspect, json, os, sys, traceback

# ``python -I`` implies ``-P`` (3.11+), which keeps the script directory off
# sys.path — so the skill module would not be importable. Put it back
# explicitly; nothing else from the host environment is inherited.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_RESULT = {"checks": [], "ok": True}


def _add(name, status, detail=""):
    _RESULT["checks"].append({"name": name, "status": status, "detail": str(detail)[:400]})
    if status == "failed":
        _RESULT["ok"] = False


import skill_under_test as _mod

_add("import", "passed")

_public = [
    (n, o)
    for n, o in vars(_mod).items()
    if not n.startswith("_")
    and (inspect.isfunction(o) or inspect.isclass(o))
    and getattr(o, "__module__", "") == _mod.__name__
]
_RESULT["callables"] = sorted(n for n, _ in _public)

try:
    _res = doctest.testmod(_mod, verbose=False, report=False)
    if _res.attempted == 0:
        _add("doctest", "skipped", "no doctests in source")
    elif _res.failed:
        _add("doctest", "failed", "%d of %d doctests failed" % (_res.failed, _res.attempted))
    else:
        _add("doctest", "passed", "%d doctests" % _res.attempted)
except Exception as exc:
    _add("doctest", "failed", "doctest runner raised: %r" % (exc,))

_selftest = getattr(_mod, "_selftest", None)
if callable(_selftest):
    try:
        _out = _selftest()
        if _out is False:
            _add("selftest", "failed", "_selftest() returned False")
        else:
            _add("selftest", "passed")
    except Exception:
        _add("selftest", "failed", traceback.format_exc(limit=3))
else:
    _add("selftest", "skipped", "no _selftest() defined")

_example = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
if isinstance(_example, dict) and _example:
    _target = None
    _preferred = _example.pop("__function__", None)
    if _preferred and hasattr(_mod, _preferred):
        _target = getattr(_mod, _preferred)
    elif _public:
        _target = _public[0][1]
    if _target is None:
        _add("example", "failed", "no callable to invoke with the example args")
    else:
        try:
            _target(**_example)
            _add("example", "passed", "called %s(**example)" % getattr(_target, "__name__", "?"))
        except TypeError as exc:
            _add("example", "failed", "signature mismatch: %s" % exc)
        except Exception:
            _add("example", "failed", traceback.format_exc(limit=3))
else:
    _add("example", "skipped", "no example args in input_schema")

print("{marker}" + json.dumps(_RESULT))
"""


@dataclass
class Check:
    """One verification step's verdict."""

    name: str
    status: str  # "passed" | "failed" | "skipped"
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class VerificationOutcome:
    """Result of verifying one skill candidate."""

    skill_name: str
    verified: bool
    reason: str = ""
    checks: list[Check] = field(default_factory=list)
    callables: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    #: Strongest evidence obtained: "behaviour" (a doctest/selftest/example ran
    #: and passed), "import" (it merely loaded), or "static" (UI skills).
    evidence: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "verified": self.verified,
            "reason": self.reason,
            "evidence": self.evidence,
            "callables": list(self.callables),
            "duration_ms": round(self.duration_ms, 1),
            "checks": [c.to_dict() for c in self.checks],
        }


class SkillVerifier:
    """Re-execute a generated skill and decide whether it may be promoted."""

    #: Checks that constitute behavioural (not merely structural) evidence.
    BEHAVIOURAL = ("doctest", "selftest", "example")

    def __init__(
        self,
        *,
        timeout: int | None = None,
        require_sandbox: bool | None = None,
    ) -> None:
        self._timeout = timeout
        self._require_sandbox = require_sandbox

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _settings(self) -> Any | None:
        try:
            from isaac.config.settings import get_settings

            return get_settings()
        except Exception:  # pragma: no cover - defensive
            return None

    @property
    def timeout(self) -> int:
        if self._timeout is not None:
            return int(self._timeout)
        s = self._settings()
        return int(getattr(s, "skill_verification_timeout", 20) or 20)

    @property
    def require_sandbox(self) -> bool:
        if self._require_sandbox is not None:
            return bool(self._require_sandbox)
        s = self._settings()
        return bool(getattr(s, "skill_verification_require_sandbox", False))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, candidate: Any) -> VerificationOutcome:
        """Verify *candidate* and return the outcome (never raises)."""
        start = time.monotonic()
        name = str(getattr(candidate, "name", "") or "").strip() or "<unnamed>"
        code = str(getattr(candidate, "code", "") or "")
        skill_type = str(getattr(candidate, "skill_type", "code") or "code")

        def done(
            verified: bool, reason: str, checks: list[Check], evidence: str = "none", **kw: Any
        ) -> VerificationOutcome:
            return VerificationOutcome(
                skill_name=name,
                verified=verified,
                reason=reason,
                checks=checks,
                duration_ms=(time.monotonic() - start) * 1000,
                evidence=evidence,
                **kw,
            )

        if not code.strip():
            return done(False, "empty skill body", [Check("syntax", "failed", "no code")])

        # 1. Static: does it parse, and does it define anything reusable? ---
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return done(False, f"syntax error: {exc}", [Check("syntax", "failed", str(exc))])
        checks = [Check("syntax", "passed")]

        defined = [
            n.name
            for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]
        if not defined:
            checks.append(Check("callable", "failed", "no module-level def/class"))
            return done(False, "defines no reusable function or class", checks)
        checks.append(Check("callable", "passed", ", ".join(defined)))

        # 2. UI/Playwright skills cannot be replayed offline -----------------
        if skill_type == "ui":
            checks.append(Check("import", "skipped", "UI skill — replay needs a live browser"))
            return done(
                True,
                "static verification only (UI skill)",
                checks,
                evidence="static",
                callables=defined,
            )

        if self.require_sandbox and not _docker_available():
            checks.append(Check("import", "skipped", "Docker sandbox unavailable"))
            return done(
                False,
                "sandbox required but unavailable (ISAAC_SKILL_VERIFICATION_REQUIRE_SANDBOX=true)",
                checks,
            )

        # 3. Dynamic: run it -------------------------------------------------
        example = _example_args(candidate)
        run = self._execute(code, example)
        checks.extend(run.checks)
        if not run.verified:
            return done(
                False,
                run.reason,
                checks,
                callables=run.callables or defined,
            )

        behavioural = [c for c in run.checks if c.name in self.BEHAVIOURAL and c.status == "passed"]
        evidence = "behaviour" if behavioural else "import"
        reason = (
            "executed; " + ", ".join(f"{c.name} passed" for c in behavioural)
            if behavioural
            else "imports and exposes a callable (no self-test in source)"
        )
        return done(True, reason, checks, evidence=evidence, callables=run.callables or defined)

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def _execute(self, code: str, example: dict | None) -> VerificationOutcome:
        """Run the harness against *code* in an isolated subprocess."""
        harness = _HARNESS.replace("{marker}", _RESULT_MARKER)
        with tempfile.TemporaryDirectory(prefix="isaac-skillverify-") as tmp:
            root = Path(tmp)
            (root / "skill_under_test.py").write_text(code, encoding="utf-8")
            harness_path = root / "_verify_harness.py"
            harness_path.write_text(harness, encoding="utf-8")
            argv = [sys.executable, "-I", str(harness_path)]
            if example:
                argv.append(json.dumps(example))
            try:
                proc = subprocess.run(
                    argv,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return VerificationOutcome(
                    skill_name="",
                    verified=False,
                    reason=f"verification run exceeded {self.timeout}s",
                    checks=[Check("import", "failed", f"timeout after {self.timeout}s")],
                )
            except OSError as exc:  # pragma: no cover - defensive
                return VerificationOutcome(
                    skill_name="",
                    verified=False,
                    reason=f"could not start verification subprocess: {exc}",
                    checks=[Check("import", "failed", str(exc))],
                )

        payload = _parse_marker(proc.stdout)
        if payload is None:
            detail = (proc.stderr or proc.stdout or "").strip()[-400:]
            return VerificationOutcome(
                skill_name="",
                verified=False,
                reason="skill failed to import/execute",
                checks=[Check("import", "failed", detail or f"exit code {proc.returncode}")],
            )

        checks = [
            Check(str(c.get("name", "?")), str(c.get("status", "failed")), str(c.get("detail", "")))
            for c in payload.get("checks", [])
        ]
        failed = [c for c in checks if c.status == "failed"]
        return VerificationOutcome(
            skill_name="",
            verified=not failed,
            reason="; ".join(f"{c.name}: {c.detail}" for c in failed)[:400],
            checks=checks,
            callables=[str(x) for x in payload.get("callables", [])],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_marker(stdout: str) -> dict | None:
    """Extract the harness verdict from *stdout*."""
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_RESULT_MARKER):
            try:
                return json.loads(line[len(_RESULT_MARKER) :])
            except ValueError:  # pragma: no cover - defensive
                return None
    return None


def _example_args(candidate: Any) -> dict | None:
    """Pull example call arguments out of a candidate's ``input_schema``.

    Recognised shapes::

        {"example": {"grid": [[1]]}}
        {"example": {"grid": [[1]]}, "__function__": "rotate"}

    Returns ``None`` when no JSON-serialisable example is available.
    """
    schema = getattr(candidate, "input_schema", None)
    if not isinstance(schema, dict):
        return None
    example = schema.get("example")
    if not isinstance(example, dict) or not example:
        return None
    payload = dict(example)
    fn = schema.get("function") or schema.get("__function__")
    if fn:
        payload["__function__"] = str(fn)
    try:
        json.dumps(payload)
    except (TypeError, ValueError):
        return None
    return payload


def _docker_available() -> bool:
    """Best-effort check for a usable Docker daemon."""
    try:
        import docker  # type: ignore[import-not-found]

        docker.from_env().ping()
        return True
    except Exception:
        return False


_verifier: SkillVerifier | None = None


def get_verifier() -> SkillVerifier:
    """Return the process-wide :class:`SkillVerifier`."""
    global _verifier
    if _verifier is None:
        _verifier = SkillVerifier()
    return _verifier


def verification_enabled() -> bool:
    """Whether the promotion gate is switched on (``ISAAC_SKILL_VERIFICATION_ENABLED``)."""
    try:
        from isaac.config.settings import get_settings

        return bool(get_settings().skill_verification_enabled)
    except Exception:  # pragma: no cover - defensive
        return True
