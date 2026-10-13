"""The actual solver boundary must reject executable or incomplete problems."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("z3")

from isaac.reasoning.theorem_prover import TheoremProver


@pytest.mark.parametrize(
    "expression",
    [
        "And.__globals__['__builtins__']",
        "x.__class__.__mro__",
        "(lambda: True)()",
        "[x for x in [1]]",
        "(x := 1)",
        "And(*[True])",
        "If(True, then=True, otherwise=False)",
        "__import__('os')",
        "x < < 0",
        "x + 1",
        "x ** 1000000 == 0",
    ],
)
def test_invalid_problem_is_unknown(expression):
    result = TheoremProver().check_sat(["x > 0", expression], {"x": "Int"})
    assert result["satisfiable"] is None
    assert result["model"] == {}
    assert result["error"]


def test_host_write_is_rejected(tmp_path):
    marker = tmp_path / "not-created.txt"
    expression = (
        "And.__globals__['__builtins__']['__import__']('pathlib')"
        f".Path({str(marker)!r}).write_text('probe') > 0"
    )
    result = TheoremProver().check_sat([expression])
    assert result["satisfiable"] is None
    assert not marker.exists()


def test_symbolic_boolean_chains_and_equations():
    result = TheoremProver().check_sat(
        ["0 < x < 10", "not (x < 3)", "x == 4 or x == 5", "y = x * 2"],
        {"x": "Int", "y": "Int"},
    )
    assert result["satisfiable"] is True
    assert result["model"]["x"] in (4, 5)
    assert result["model"]["y"] == 2 * result["model"]["x"]


def test_valid_property_and_counterexample():
    prover = TheoremProver()
    kwargs = dict(
        preconditions=["n >= 0"],
        code_logic="result = n * n",
        variables={"n": "Int", "result": "Int"},
    )
    assert prover.verify_property(postcondition="result >= 0", **kwargs)["verified"] is True
    assert prover.verify_property(postcondition="result > 0", **kwargs)["verified"] is False


def test_arc_full_size_bounds_remain_supported():
    cells = {f"c_{r}_{c}": (r, c) for r in range(30) for c in range(30)}
    result = TheoremProver().solve_arc_constraints(["c_0_0 == 3"], cells)
    assert result["satisfiable"] is True
    assert all(0 <= value <= 9 for value in result["vars"].values())


@pytest.mark.parametrize("timeout", [0, -1, 60001, True])
def test_timeout_cannot_disable_budget(timeout):
    assert TheoremProver().check_sat(["True"], timeout_ms=timeout)["satisfiable"] is None


@pytest.mark.parametrize("variables", [{"And": "Int"}, {"x": "Unknown"}, {"__x": "Int"}])
def test_invalid_declarations_are_rejected(variables):
    assert TheoremProver().check_sat(["True"], variables)["satisfiable"] is None


def test_audio_file_is_already_created_and_private():
    from isaac.multimodal.audio import _reserve_audio_file

    path = _reserve_audio_file(".wav")
    try:
        assert isinstance(path, Path)
        assert path.exists()
        assert path.stat().st_size == 0
    finally:
        path.unlink()


@pytest.mark.parametrize("spec", [{}, {"constraints": []}, {"variables": {}, "constraints": []}])
def test_llm_formalization_cannot_return_vacuous_success(spec, monkeypatch):
    from isaac.experts.base import ExpertNotApplicable
    from isaac.experts.logic import LogicExpert
    from isaac.llm import provider
    from isaac.reasoning.theorem_prover import verify_code_property

    monkeypatch.setattr(
        provider,
        "get_llm",
        lambda _: SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps(spec))),
    )
    with pytest.raises(ExpertNotApplicable):
        LogicExpert()._answer("prove a constraint", {})
    result = verify_code_property("x = 1", "x is positive")
    assert result["verified"] is None
    assert result["error"]


def test_llm_and_context_constraints_share_the_rejecting_boundary(monkeypatch):
    from isaac.experts.logic import LogicExpert
    from isaac.llm import provider

    spec = {"variables": {"x": "Int"}, "constraints": ["x > 0", "x < < 0"]}
    monkeypatch.setattr(
        provider,
        "get_llm",
        lambda _: SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps(spec))),
    )
    for response in [
        LogicExpert()._answer("prove a constraint", {}),
        LogicExpert()._answer("prove a constraint", spec),
    ]:
        assert response.artifacts["smt_result"]["satisfiable"] is None
        assert response.confidence < 0.5


def test_llm_model_proof_is_not_reported_as_source_code_proof(monkeypatch):
    from isaac.llm import provider
    from isaac.reasoning.theorem_prover import verify_code_property

    spec = {"variables": {}, "preconditions": [], "code_logic": "True", "postcondition": "True"}
    monkeypatch.setattr(
        provider,
        "get_llm",
        lambda _: SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps(spec))),
    )
    result = verify_code_property("def f(n): return -1", "f returns a positive integer")
    assert result["verified"] is None
    assert result["model_verified"] is True
    assert result["formalization"] == spec
    assert result["error"]


def test_inconsistent_model_is_not_a_code_proof():
    result = TheoremProver().verify_property(["x > 0", "x < 0"], "x > 100", "x == x", {"x": "Int"})
    assert result["verified"] is None
    assert "inconsistent" in result["error"]
