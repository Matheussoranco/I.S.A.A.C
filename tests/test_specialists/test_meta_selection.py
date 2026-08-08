"""Orchestrator × MetaLearner-guided selection (1.5.0, roadmap WS6).

Offline throughout: the planner, specialist factory, manager LLM, and the
selector's MetaLearner are all injected, so nothing here touches the registry,
a real model, or the user's ``~/.isaac`` database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac.meta.learner import MetaLearner
from isaac.meta.specialist_selector import SPECIALIST_TASK_TYPE, SpecialistSelector
from isaac.specialists.base import SpecialistResult
from isaac.specialists.orchestrator import Orchestrator, SubTask


class _StubSpecialist:
    def __init__(self, name: str, succeed: bool = True) -> None:
        self.name = name
        self._succeed = succeed

    def run(self, task: str, context: str = "") -> SpecialistResult:
        return SpecialistResult(
            specialist=self.name, task=task, output=f"did:{task}", success=self._succeed
        )


class _StubLLM:
    def __init__(self, content: str = "SYNTH") -> None:
        self._content = content

    def invoke(self, messages):
        self.last = messages

        class _Resp:
            content = self._content

        return _Resp()


class _RosterCapturingPlanner:
    """Planner stub that records the roster it was handed."""

    def __init__(self, specialist: str = "coder") -> None:
        self.roster: list[dict] = []
        self._specialist = specialist

    def __call__(self, goal: str, roster: list[dict], context: str) -> list[SubTask]:
        self.roster = roster
        return [SubTask(id="t1", description=goal, specialist=self._specialist)]


@pytest.fixture()
def selector(tmp_path: Path) -> SpecialistSelector:
    return SpecialistSelector(MetaLearner(tmp_path / "meta.db"))


def _factory(succeed: bool = True, known: set[str] | None = None):
    def factory(name: str, **kwargs):
        if known is not None and name not in known:
            raise KeyError(name)
        return _StubSpecialist(name, succeed=succeed)

    return factory


def _seed(selector: SpecialistSelector, name: str, wins: int, losses: int) -> None:
    for _ in range(wins):
        selector.record(name, success=True)
    for _ in range(losses):
        selector.record(name, success=False)


ROSTER = [
    {"name": "coder", "domain": "code"},
    {"name": "researcher", "domain": "research"},
    {"name": "analyst", "domain": "analysis"},
]


class TestToggle:
    def test_off_by_explicit_flag(self, selector: SpecialistSelector) -> None:
        orch = Orchestrator(use_meta_selection=False, selector=selector, manager_llm=_StubLLM())
        assert orch.meta_selection is False

    def test_on_by_explicit_flag(self, selector: SpecialistSelector) -> None:
        orch = Orchestrator(use_meta_selection=True, selector=selector, manager_llm=_StubLLM())
        assert orch.meta_selection is True

    def test_defaults_to_the_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import isaac.meta.specialist_selector as sel_mod

        monkeypatch.setattr(sel_mod, "selection_enabled", lambda: True)
        assert Orchestrator(manager_llm=_StubLLM()).meta_selection is True


class TestRosterBiasing:
    def test_off_leaves_the_roster_untouched(self, selector: SpecialistSelector) -> None:
        _seed(selector, "analyst", wins=10, losses=0)
        planner = _RosterCapturingPlanner()
        orch = Orchestrator(
            planner=planner,
            specialist_factory=_factory(),
            manager_llm=_StubLLM(),
            use_meta_selection=False,
            selector=selector,
        )
        orch._list_roster = staticmethod(lambda: list(ROSTER))  # type: ignore[method-assign]

        orch.run("goal")

        assert [c["name"] for c in planner.roster] == ["coder", "researcher", "analyst"]
        assert "track_record" not in planner.roster[0]

    def test_on_reorders_and_annotates(self, selector: SpecialistSelector) -> None:
        _seed(selector, "analyst", wins=10, losses=0)
        _seed(selector, "coder", wins=0, losses=8)
        planner = _RosterCapturingPlanner()
        orch = Orchestrator(
            planner=planner,
            specialist_factory=_factory(),
            manager_llm=_StubLLM(),
            use_meta_selection=True,
            selector=selector,
        )
        orch._list_roster = staticmethod(lambda: list(ROSTER))  # type: ignore[method-assign]

        orch.run("goal")

        assert [c["name"] for c in planner.roster] == ["analyst", "researcher", "coder"]
        assert planner.roster[0]["track_record"] == "10/10 succeeded (100%)"

    def test_on_with_no_history_is_a_no_op(self, selector: SpecialistSelector) -> None:
        """ON and OFF must be identical until evidence exists — the ablation
        depends on this being true."""
        planner = _RosterCapturingPlanner()
        orch = Orchestrator(
            planner=planner,
            specialist_factory=_factory(),
            manager_llm=_StubLLM(),
            use_meta_selection=True,
            selector=selector,
        )
        orch._list_roster = staticmethod(lambda: list(ROSTER))  # type: ignore[method-assign]

        orch.run("goal")

        assert [c["name"] for c in planner.roster] == ["coder", "researcher", "analyst"]

    def test_ranking_event_is_emitted(self, selector: SpecialistSelector) -> None:
        _seed(selector, "analyst", wins=5, losses=0)
        kinds: list[str] = []
        orch = Orchestrator(
            planner=_RosterCapturingPlanner(),
            specialist_factory=_factory(),
            manager_llm=_StubLLM(),
            use_meta_selection=True,
            selector=selector,
            on_event=lambda k, d: kinds.append(k),
        )
        orch._list_roster = staticmethod(lambda: list(ROSTER))  # type: ignore[method-assign]

        orch.run("goal")

        assert "specialist_ranking" in kinds

    def test_prompt_carries_the_track_record(self, selector: SpecialistSelector) -> None:
        _seed(selector, "analyst", wins=4, losses=0)
        llm = _StubLLM('{"subtasks":[{"id":"t1","description":"d","specialist":"analyst"}]}')
        orch = Orchestrator(manager_llm=llm, use_meta_selection=True, selector=selector)

        orch._plan("goal", selector.annotate_roster(ROSTER), "")

        prompt = " ".join(str(getattr(m, "content", m)) for m in llm.last)
        assert "track record" in prompt
        assert "4/4 succeeded" in prompt


class TestUnknownSpecialistFallback:
    def test_off_falls_back_to_generalist(self, selector: SpecialistSelector) -> None:
        _seed(selector, "analyst", wins=10, losses=0)

        def planner(goal, roster, context):
            return [SubTask(id="t1", description="d", specialist="nonexistent")]

        res = Orchestrator(
            planner=planner,
            specialist_factory=_factory(known={"generalist", "analyst", "coder"}),
            manager_llm=_StubLLM(),
            use_meta_selection=False,
            selector=selector,
        ).run("g")

        assert res.results[0].result.specialist == "generalist"

    def test_on_falls_back_to_the_best_scoring_specialist(
        self, selector: SpecialistSelector, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(selector, "analyst", wins=10, losses=0)
        _seed(selector, "generalist", wins=0, losses=9)
        monkeypatch.setattr(
            "isaac.specialists.registry.specialist_names",
            lambda: ["coder", "analyst", "generalist"],
        )

        def planner(goal, roster, context):
            return [SubTask(id="t1", description="d", specialist="nonexistent")]

        res = Orchestrator(
            planner=planner,
            specialist_factory=_factory(known={"generalist", "analyst", "coder"}),
            manager_llm=_StubLLM(),
            use_meta_selection=True,
            selector=selector,
        ).run("g")

        assert res.results[0].result.specialist == "analyst"


class TestPerSpecialistRecording:
    def test_each_subtask_is_recorded_against_its_specialist(
        self, selector: SpecialistSelector
    ) -> None:
        def planner(goal, roster, context):
            return [
                SubTask(id="t1", description="a", specialist="researcher"),
                SubTask(id="t2", description="b", specialist="coder"),
            ]

        Orchestrator(
            planner=planner,
            specialist_factory=_factory(),
            manager_llm=_StubLLM(),
            use_meta_selection=True,
            selector=selector,
        ).run("g")

        scores = selector.scores(["researcher", "coder"])
        assert scores["researcher"].wins == 1
        assert scores["coder"].wins == 1

    def test_failures_are_recorded_as_losses(self, selector: SpecialistSelector) -> None:
        def planner(goal, roster, context):
            return [SubTask(id="t1", description="a", specialist="coder")]

        Orchestrator(
            planner=planner,
            specialist_factory=_factory(succeed=False),
            manager_llm=_StubLLM(),
            use_meta_selection=True,
            selector=selector,
        ).run("g")

        assert selector.scores(["coder"])["coder"].losses == 1

    def test_recording_happens_even_with_selection_off(self, selector: SpecialistSelector) -> None:
        """Evidence collection is free; only its *use* is ablated."""

        def planner(goal, roster, context):
            return [SubTask(id="t1", description="a", specialist="coder")]

        Orchestrator(
            planner=planner,
            specialist_factory=_factory(),
            manager_llm=_StubLLM(),
            use_meta_selection=False,
            selector=selector,
        ).run("g")

        assert selector.scores(["coder"])["coder"].wins == 1

    def test_orchestration_row_is_still_written(
        self, selector: SpecialistSelector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learner = MetaLearner(tmp_path / "orch.db")
        monkeypatch.setattr("isaac.meta.learner.get_learner", lambda: learner)

        def planner(goal, roster, context):
            return [
                SubTask(id="t1", description="a", specialist="coder"),
                SubTask(id="t2", description="b", specialist="analyst"),
            ]

        Orchestrator(
            planner=planner,
            specialist_factory=_factory(),
            manager_llm=_StubLLM(),
            use_meta_selection=True,
            selector=selector,
        ).run("g")

        rows = learner.get_best_strategy("orchestration")
        assert [r["strategy"] for r in rows] == ["team"]
        # ...and it did not leak into the specialist bucket.
        assert not learner.get_best_strategy(SPECIALIST_TASK_TYPE)


class TestResilience:
    def test_a_broken_selector_never_breaks_a_run(self) -> None:
        class _Exploding:
            def annotate_roster(self, roster):
                raise RuntimeError("boom")

            def record(self, *a, **kw):
                raise RuntimeError("boom")

            def best(self, names, default="generalist"):
                raise RuntimeError("boom")

        planner = _RosterCapturingPlanner()
        orch = Orchestrator(
            planner=planner,
            specialist_factory=_factory(),
            manager_llm=_StubLLM("OK"),
            use_meta_selection=True,
            selector=_Exploding(),
        )
        orch._list_roster = staticmethod(lambda: list(ROSTER))  # type: ignore[method-assign]

        res = orch.run("goal")

        assert res.success is True
        assert [c["name"] for c in planner.roster] == ["coder", "researcher", "analyst"]
