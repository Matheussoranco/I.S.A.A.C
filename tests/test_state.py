"""Tests for the IsaacState schema and sub-schemas."""

from __future__ import annotations

from isaac.core.state import (
    ErrorEntry,
    ExecutionResult,
    IsaacState,
    PlanStep,
    SkillCandidate,
    WorldModel,
    make_initial_state,
)


class TestWorldModel:
    def test_defaults(self) -> None:
        wm = WorldModel()
        assert wm.files == {}
        assert wm.resources == {}
        assert wm.constraints == []
        assert wm.observations == []

    def test_custom_values(self) -> None:
        wm = WorldModel(
            files={"a.py": "abc123"},
            resources={"cpu": 4},
            constraints=["no network"],
            observations=["file found"],
        )
        assert wm.files["a.py"] == "abc123"
        assert len(wm.constraints) == 1


class TestPlanStep:
    def test_defaults(self) -> None:
        step = PlanStep(id="s1", description="do something")
        assert step.status == "pending"
        assert step.depends_on == []

    def test_dependency(self) -> None:
        step = PlanStep(id="s2", description="next", depends_on=["s1"])
        assert step.depends_on == ["s1"]


class TestExecutionResult:
    def test_sentinel(self) -> None:
        er = ExecutionResult()
        assert er.exit_code == -1
        assert er.stdout == ""


class TestSkillCandidate:
    def test_empty(self) -> None:
        sc = SkillCandidate()
        assert sc.name == ""
        assert sc.success_count == 0


class TestErrorEntry:
    def test_creation(self) -> None:
        e = ErrorEntry(node="reflection", message="boom", attempt=1)
        assert e.traceback is None


class TestMakeInitialState:
    def test_all_fields_present(self) -> None:
        state = make_initial_state()
        assert "messages" in state
        assert "world_model" in state
        assert "hypothesis" in state
        assert "plan" in state
        assert "code_buffer" in state
        assert "execution_logs" in state
        assert "skill_candidate" in state
        assert "errors" in state
        assert "iteration" in state
        assert "current_phase" in state

    def test_initial_values(self) -> None:
        state = make_initial_state()
        assert state["iteration"] == 0
        assert state["hypothesis"] == ""
        assert state["plan"] == []
        assert state["errors"] == []
        assert state["skill_candidate"] is None
