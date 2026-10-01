"""Tests for the Curator subsystem."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from isaac.improvement.curator import Curator, CuratorRecommendation

SECONDS_PER_DAY = 86400


@dataclass
class FakeRecord:
    name: str
    uses: int = 0
    success_rate: float = 1.0
    last_run_at: float | None = None


class FakeProcedural:
    def __init__(self, records: dict[str, FakeRecord]) -> None:
        self._records = records
        self.deprecated: list[str] = []

    def list_active(self) -> list[str]:
        return list(self._records.keys())

    def get_record(self, name: str) -> FakeRecord | None:
        return self._records.get(name)

    def deprecate(self, name: str) -> None:
        self.deprecated.append(name)


@pytest.fixture()
def proc() -> FakeProcedural:
    now = time.time()
    return FakeProcedural(
        {
            "bad-skill": FakeRecord("bad-skill", uses=8, success_rate=0.1, last_run_at=now),
            "meh-skill": FakeRecord("meh-skill", uses=12, success_rate=0.5, last_run_at=now),
            "stale-skill": FakeRecord(
                "stale-skill", uses=1, success_rate=1.0, last_run_at=now - 120 * SECONDS_PER_DAY
            ),
            "good-skill": FakeRecord("good-skill", uses=20, success_rate=0.95, last_run_at=now),
        }
    )


def test_heuristic_rules_trigger(proc: FakeProcedural) -> None:
    curator = Curator(llm=None, procedural=proc)
    recs = {r.skill_name: r for r in curator.audit_all()}
    assert recs["bad-skill"].action == "deprecate"
    assert recs["bad-skill"].confidence == pytest.approx(0.8)
    assert recs["meh-skill"].action == "improve"
    assert recs["meh-skill"].confidence == pytest.approx(0.6)
    assert recs["stale-skill"].action == "deprecate"
    assert recs["stale-skill"].confidence == pytest.approx(0.5)
    assert recs["good-skill"].action == "keep"


def test_apply_dry_run_does_not_deprecate(proc: FakeProcedural) -> None:
    curator = Curator(llm=None, procedural=proc)
    recs = curator.audit_all()
    summary = curator.apply(recs, min_confidence=0.7, dry_run=True)
    assert summary["deprecate"] >= 1
    assert proc.deprecated == []


def test_apply_real_calls_deprecate(proc: FakeProcedural) -> None:
    curator = Curator(llm=None, procedural=proc)
    recs = curator.audit_all()
    curator.apply(recs, min_confidence=0.7, dry_run=False)
    assert "bad-skill" in proc.deprecated


def test_min_confidence_filters(proc: FakeProcedural) -> None:
    curator = Curator(llm=None, procedural=proc)
    recs = curator.audit_all()
    # 0.9 filters out everything (max heuristic confidence is 0.8)
    summary = curator.apply(recs, min_confidence=0.9, dry_run=False)
    assert all(v == 0 for v in summary.values())
    assert proc.deprecated == []
    # 0.5: deprecate(0.8) + stale deprecate(0.5) + improve(0.6) pass; keep (0.4) is filtered
    summary = curator.apply(recs, min_confidence=0.5, dry_run=False)
    assert summary["deprecate"] == 2
    assert summary["improve"] == 1
    assert summary["keep"] == 0
    assert sorted(proc.deprecated) == ["bad-skill", "stale-skill"]


def test_no_llm_still_returns_heuristics(proc: FakeProcedural) -> None:
    curator = Curator(llm=None, procedural=proc)
    recs = curator.audit_all()
    assert len(recs) == 4
    assert all(isinstance(r, CuratorRecommendation) for r in recs)


class FlakyLLM:
    def invoke(self, _messages):
        raise RuntimeError("boom")


class ReasonLLM:
    def invoke(self, messages):
        class R:
            content = "REASON: flaky output from run history"

        # Accept both [HumanMessage] and raw prompt
        assert messages
        return R()


def test_flaky_llm_degrades_to_heuristic(proc: FakeProcedural) -> None:
    curator = Curator(llm=FlakyLLM(), procedural=proc)
    recs = {r.skill_name: r for r in curator.audit_all()}
    assert recs["bad-skill"].action == "deprecate"
    assert "0.3" in recs["bad-skill"].reason  # heuristic reason retained


def test_llm_refinement_updates_reason(proc: FakeProcedural) -> None:
    curator = Curator(llm=ReasonLLM(), procedural=proc)
    recs = {r.skill_name: r for r in curator.audit_all()}
    assert recs["bad-skill"].reason == "flaky output from run history"
    # keep action is never sent to the LLM
    assert recs["good-skill"].reason == "within healthy thresholds"


def test_improve_patch_written_not_applied(proc: FakeProcedural, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("isaac.improvement.curator.patches_dir", lambda: tmp_path)
    curator = Curator(llm=None, procedural=proc)
    rec = CuratorRecommendation(
        skill_name="meh-skill",
        action="improve",
        reason="test",
        suggested_patch="--- a\n+++ b\n@@ fix",
        confidence=0.9,
    )
    summary = curator.apply([rec], min_confidence=0.7, dry_run=False)
    assert summary["improve"] == 1
    patch_file = tmp_path / "meh-skill.diff"
    assert patch_file.exists()
    assert "@@ fix" in patch_file.read_text()
