"""Tests for the Skill Library."""

from __future__ import annotations

from pathlib import Path

from isaac.core.state import SkillCandidate
from isaac.memory.skill_library import SkillLibrary
from isaac.memory.skill_verification import SkillVerifier


def _commit_trusted_fixture(lib: SkillLibrary, candidate: SkillCandidate) -> None:
    """Verify deterministic test fixtures with the explicit host-only fallback."""
    lib.commit(candidate, verifier=SkillVerifier(require_sandbox=False))


class TestSkillLibrary:
    def test_commit_and_retrieve(self, tmp_path: Path) -> None:
        lib = SkillLibrary(tmp_path)
        candidate = SkillCandidate(
            name="rotate_grid",
            code="def rotate(grid):\n    return [list(r) for r in zip(*grid[::-1])]",
            input_schema={"grid": "list[list[int]]"},
            output_schema={"result": "list[list[int]]"},
            task_context="ARC rotation task",
            success_count=2,
        )
        _commit_trusted_fixture(lib, candidate)

        assert lib.size == 1
        assert "rotate_grid" in lib.list_names()

        code = lib.get_code("rotate_grid")
        assert code is not None
        assert "def rotate" in code

        meta = lib.get_metadata("rotate_grid")
        assert meta is not None
        assert meta["success_count"] == 2

    def test_search(self, tmp_path: Path) -> None:
        lib = SkillLibrary(tmp_path)
        _commit_trusted_fixture(
            lib,
            SkillCandidate(
                name="flip_horizontal",
                code="def flip(g): return [r[::-1] for r in g]",
                task_context="ARC flip task",
                success_count=1,
            ),
        )
        _commit_trusted_fixture(
            lib,
            SkillCandidate(
                name="fill_color",
                code="def fill(g, c): pass",
                task_context="ARC color fill",
                success_count=1,
            ),
        )

        results = lib.search("flip")
        assert "flip_horizontal" in results

    def test_persistence(self, tmp_path: Path) -> None:
        lib1 = SkillLibrary(tmp_path)
        # ``code="pass"`` used to be enough here; since 1.5.0 the promotion
        # gate rejects a skill that defines nothing reusable, so the fixture
        # is now an actual function.
        _commit_trusted_fixture(
            lib1, SkillCandidate(name="my_skill", code="def go():\n    return 1", success_count=1)
        )

        # Re-open from same directory
        lib2 = SkillLibrary(tmp_path)
        assert lib2.size == 1
        assert "my_skill" in lib2.list_names()

    def test_no_name_skips(self, tmp_path: Path) -> None:
        lib = SkillLibrary(tmp_path)
        lib.commit(SkillCandidate(name="", code="pass"))
        assert lib.size == 0

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        lib = SkillLibrary(tmp_path)
        assert lib.get_code("nope") is None
        assert lib.get_metadata("nope") is None
