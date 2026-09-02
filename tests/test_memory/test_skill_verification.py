"""Tests for the skill promotion gate (1.5.0, roadmap WS6).

The gate's whole point is that a skill must *run* before it is promoted, so
these tests exercise the real subprocess verifier rather than mocking it — the
failure modes below (undefined names, failing doctests, signature mismatches)
are exactly what an LLM-generalised skill gets wrong, and a mocked verifier
would prove nothing about whether they are caught.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from isaac.core.state import SkillCandidate
from isaac.memory.skill_library import SkillLibrary
from isaac.memory.skill_verification import SkillVerifier

# The subprocess launch dominates each case; keep the budget small so a hung
# skill fails the test fast instead of stalling the suite.
VERIFY_TIMEOUT = 15


@pytest.fixture(autouse=True)
def host_fallback_for_verifier_unit_tests(monkeypatch: pytest.MonkeyPatch):
    """Use the explicitly opted-in host fallback only in this isolated unit module."""
    from isaac.memory import skill_verification

    monkeypatch.setattr(
        skill_verification,
        "_verifier",
        SkillVerifier(timeout=VERIFY_TIMEOUT, require_sandbox=False),
    )


@pytest.fixture()
def verifier() -> SkillVerifier:
    # These tests exercise verifier semantics without requiring Docker. The
    # application default is sandbox-required and is covered separately.
    return SkillVerifier(timeout=VERIFY_TIMEOUT, require_sandbox=False)


def _candidate(name: str, code: str, **kw) -> SkillCandidate:
    return SkillCandidate(name=name, code=code, task_context="test", **kw)


class TestVerifierAccepts:
    def test_application_default_requires_docker(self) -> None:
        assert SkillVerifier().require_sandbox is True

    def test_missing_settings_fail_closed_to_docker(self) -> None:
        verifier = SkillVerifier()
        with patch.object(verifier, "_settings", return_value=None):
            assert verifier.require_sandbox is True

    def test_sandbox_path_is_used_when_required(self) -> None:
        verifier = SkillVerifier(timeout=VERIFY_TIMEOUT, require_sandbox=True)
        payload = {
            "checks": [
                {"name": "import", "status": "passed", "detail": ""},
                {"name": "doctest", "status": "skipped", "detail": ""},
                {"name": "selftest", "status": "skipped", "detail": ""},
                {"name": "example", "status": "skipped", "detail": ""},
            ],
            "callables": ["ok"],
        }
        stdout = "__ISAAC_SKILL_VERIFY__" + __import__("json").dumps(payload)
        with (
            patch("isaac.memory.skill_verification._docker_available", return_value=True),
            patch.object(verifier, "_execute_in_docker", return_value=(stdout, "", 0)) as run,
        ):
            outcome = verifier.verify(_candidate("safe", "def ok():\n    return 1\n"))

        assert outcome.verified
        run.assert_called_once()

    def test_plain_function_verifies_as_import_evidence(self, verifier: SkillVerifier) -> None:
        outcome = verifier.verify(_candidate("adder", "def add(a, b):\n    return a + b\n"))
        assert outcome.verified
        # No self-test in source: the outcome must not overclaim.
        assert outcome.evidence == "import"
        assert "add" in outcome.callables

    def test_passing_doctest_counts_as_behavioural_evidence(self, verifier: SkillVerifier) -> None:
        code = '''
def double(x):
    """Double a number.

    >>> double(3)
    6
    """
    return x * 2
'''
        outcome = verifier.verify(_candidate("doubler", code))
        assert outcome.verified
        assert outcome.evidence == "behaviour"

    def test_passing_selftest_counts_as_behavioural_evidence(self, verifier: SkillVerifier) -> None:
        code = "def inc(x):\n    return x + 1\n\n\ndef _selftest():\n    assert inc(1) == 2\n"
        outcome = verifier.verify(_candidate("inc", code))
        assert outcome.verified
        assert outcome.evidence == "behaviour"

    def test_example_args_are_invoked(self, verifier: SkillVerifier) -> None:
        outcome = verifier.verify(
            _candidate(
                "scale",
                "def scale(v, k):\n    return v * k\n",
                input_schema={"example": {"v": 2, "k": 3}},
            )
        )
        assert outcome.verified
        assert outcome.evidence == "behaviour"

    def test_ui_skill_is_static_only(self, verifier: SkillVerifier) -> None:
        """A Playwright macro cannot be replayed offline — say so, don't fake it."""
        outcome = verifier.verify(
            _candidate("login_macro", "def login(page):\n    page.click('#go')\n", skill_type="ui")
        )
        assert outcome.verified
        assert outcome.evidence == "static"


class TestVerifierRejects:
    @pytest.mark.parametrize(
        ("label", "code"),
        [
            ("syntax_error", "def f(:\n    pass"),
            ("no_callable", "print(1 + 1)"),
            ("missing_import", "import nonexistent_module_xyz\n\n\ndef f():\n    return 1"),
            ("undefined_name", "def f():\n    return 1\n\n\nRESULT = helper()"),
            (
                "failing_doctest",
                'def f(x):\n    """\n    >>> f(1)\n    99\n    """\n    return x + 1',
            ),
            (
                "failing_selftest",
                "def f(x):\n    return x + 1\n\n\ndef _selftest():\n    assert f(1) == 3\n",
            ),
        ],
    )
    def test_broken_candidates_are_rejected(
        self, verifier: SkillVerifier, label: str, code: str
    ) -> None:
        outcome = verifier.verify(_candidate(label, code))
        assert not outcome.verified, f"{label} should not have been promoted"
        assert outcome.reason

    def test_signature_mismatch_against_example(self, verifier: SkillVerifier) -> None:
        outcome = verifier.verify(
            _candidate(
                "scale",
                "def scale(v):\n    return v * 2\n",
                input_schema={"example": {"v": 2, "k": 3}},
            )
        )
        assert not outcome.verified
        assert "signature" in outcome.reason.lower()

    def test_empty_body_is_rejected(self, verifier: SkillVerifier) -> None:
        assert not verifier.verify(_candidate("blank", "   ")).verified

    def test_runaway_skill_is_killed_by_the_timeout(self) -> None:
        outcome = SkillVerifier(timeout=3, require_sandbox=False).verify(
            _candidate("spinner", "def f():\n    return 1\n\n\nwhile True:\n    pass\n")
        )
        assert not outcome.verified
        assert "exceeded" in outcome.reason


class TestLibraryGate:
    def test_rejected_skill_writes_no_file(self, tmp_path: Path) -> None:
        lib = SkillLibrary(tmp_path)
        outcome = lib.commit(_candidate("broken", "def f(:\n    pass"))

        assert not outcome
        assert not outcome.promoted
        assert lib.size == 0
        assert not (tmp_path / "broken.py").exists()

    def test_rejection_is_recorded_and_counted(self, tmp_path: Path) -> None:
        lib = SkillLibrary(tmp_path)
        lib.commit(_candidate("good", "def ok():\n    return 1\n"))
        lib.commit(_candidate("bad", "def f(:\n    pass"))

        stats = lib.promotion_stats()
        assert stats["promoted"] == 1
        assert stats["rejected"] == 1
        assert stats["considered"] == 2
        assert stats["promotion_rate"] == 0.5
        assert lib.rejections[-1]["name"] == "bad"

    def test_rejection_log_survives_reopen(self, tmp_path: Path) -> None:
        SkillLibrary(tmp_path).commit(_candidate("bad", "print('no functions here')"))
        assert SkillLibrary(tmp_path).promotion_stats()["rejected"] == 1

    def test_gate_can_be_disabled(self, tmp_path: Path) -> None:
        """verify=False restores the pre-1.5.0 promote-on-faith behaviour."""
        lib = SkillLibrary(tmp_path)
        outcome = lib.commit(_candidate("unchecked", "def f(:\n    pass"), verify=False)

        assert outcome.promoted
        assert outcome.evidence == "unverified"
        assert lib.size == 1

    def test_promoted_metadata_records_the_evidence(self, tmp_path: Path) -> None:
        lib = SkillLibrary(tmp_path)
        lib.commit(_candidate("inc", "def inc(x):\n    return x + 1\n"))

        meta = lib.get_metadata("inc")
        assert meta is not None
        assert meta["verified"] is True
        assert meta["verification_evidence"] == "import"

    def test_verifier_can_be_injected(self, tmp_path: Path) -> None:
        class _AlwaysReject:
            def verify(self, candidate):
                from isaac.memory.skill_verification import VerificationOutcome

                return VerificationOutcome(candidate.name, False, reason="nope")

        lib = SkillLibrary(tmp_path)
        assert not lib.commit(
            _candidate("fine", "def f():\n    return 1\n"), verifier=_AlwaysReject()
        )
        assert lib.size == 0
