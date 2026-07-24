"""Tests for DreamCoder-style ARC library learning."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from isaac.arc.dsl import PRIMITIVES, apply_program
from isaac.arc.library_learning import LibraryLearner


def test_records_solutions_and_compresses() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "lib.db"
        learner = LibraryLearner(db, min_support=2, min_len=2, max_len=3)

        # Common fragment: rotate_90 -> flip_horizontal
        common = [{"op": "rotate_90"}, {"op": "flip_horizontal"}]

        for i in range(3):
            learner.record_solution(
                {"id": f"task_{i}"},
                program=[*common, {"op": "identity"}],
                accuracy=1.0,
                strategy="beam",
            )

        promoted = learner.compress()
        assert promoted, "expected at least one abstraction promoted"
        first = promoted[0]
        assert first.support >= 2
        assert first.name in PRIMITIVES

        grid = np.array([[1, 2], [3, 4]])
        new_fn = PRIMITIVES[first.name]
        out = new_fn(grid)
        # The composite should equal applying the fragment manually
        expected = apply_program(first.fragment, grid)
        assert np.array_equal(out, expected)
        learner.close()


def test_low_accuracy_solutions_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "lib.db"
        learner = LibraryLearner(db, min_support=1)
        learner.record_solution(
            {"id": "x"},
            program=[{"op": "rotate_90"}],
            accuracy=0.5,
        )
        stats = learner.stats()
        assert stats["solutions_recorded"] == 0
        learner.close()
