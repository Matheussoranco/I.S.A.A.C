"""Neurosymbolic Theorem Prover — Z3-based constraint satisfaction and code verification.

Integrates Z3 SMT solver for:
  - Constraint satisfaction (find values satisfying conditions)
  - Code property verification (pre/post-condition checking)
  - ARC grid constraint solving (exact pixel constraints)
  - Hypothesis falsification (refute candidate solutions symbolically)

Z3 is optional: if not installed, all calls degrade gracefully with a clear message.

Usage
-----
    from isaac.reasoning.theorem_prover import TheoremProver, verify_code_property

    prover = TheoremProver()

    # Check if constraints are satisfiable
    result = prover.check_sat([
        "x > 0", "x < 10", "y == x * 2", "y < 15"
    ], variables={"x": "Int", "y": "Int"})
    # → {"satisfiable": True, "model": {"x": 4, "y": 8}}

    # Verify a simple property
    ok = prover.verify_property(
        preconditions=["n >= 0"],
        postcondition="result >= 0",
        code_logic="result = n * n",
        variables={"n": "Int", "result": "Int"},
    )
"""

from __future__ import annotations

import ast
import logging
from typing import Any

from isaac.reasoning.constraint_parser import (
    FUNCTIONS,
    MAX_CONSTRAINTS,
    MAX_VARIABLES,
    evaluate_constraint,
    parse_constraint,
    validate_name,
)

logger = logging.getLogger(__name__)


def _z3_available() -> bool:
    from importlib.util import find_spec

    return find_spec("z3") is not None


class TheoremProver:
    """Z3-backed symbolic reasoner for constraint satisfaction and verification."""

    def check_sat(
        self,
        constraints: list[str],
        variables: dict[str, str] | None = None,
        timeout_ms: int = 5000,
    ) -> dict[str, Any]:
        """Check if a set of constraints is satisfiable.

        Parameters
        ----------
        constraints:
            List of constraint strings in Python/Z3 syntax
            (e.g. ``"x > 0"``, ``"y == x * 2"``).
        variables:
            Mapping of variable name → Z3 sort (``"Int"``, ``"Real"``, ``"Bool"``).
            If ``None``, variables are auto-detected from constraint strings.
        timeout_ms:
            Solver timeout in milliseconds.

        Returns
        -------
        dict with keys:
            ``satisfiable`` (bool | None) — True/False/None (unknown/timeout)
            ``model`` (dict) — variable assignments if SAT
            ``error`` (str) — error message if failed
        """
        if not _z3_available():
            return {"satisfiable": None, "model": {}, "error": "z3-solver not installed"}

        try:
            return self._solve(constraints, {} if variables is None else variables, timeout_ms)
        except Exception as exc:
            logger.warning("TheoremProver.check_sat failed: %s", exc)
            return {"satisfiable": None, "model": {}, "error": str(exc)}

    def _solve(
        self,
        constraints: list[str],
        var_decls: dict[str, str],
        timeout_ms: int,
    ) -> dict[str, Any]:
        import z3  # type: ignore[import-untyped]

        if not isinstance(constraints, list) or len(constraints) > MAX_CONSTRAINTS:
            raise ValueError("Constraints must be a list of at most 2048 expressions")
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000:
            raise ValueError("Solver timeout must be between 1 and 60000 ms")
        trees = [parse_constraint(c) for c in constraints]
        if not isinstance(var_decls, dict) or len(var_decls) > MAX_VARIABLES:
            raise ValueError("At most 1024 variable declarations are supported")

        solver = z3.Solver()
        solver.set("timeout", timeout_ms)

        # Auto-detect variables if not provided
        if not var_decls:
            var_decls = self._auto_detect_vars(constraints)
        if len(var_decls) > MAX_VARIABLES:
            raise ValueError("At most 1024 variables are supported")

        # Create Z3 variables
        env: dict[str, Any] = {}
        for name, sort in var_decls.items():
            validate_name(name)
            if not isinstance(sort, str):
                raise ValueError("Variable sort must be Int, Real or Bool")
            sort_lower = sort.lower()
            if sort_lower == "int":
                env[name] = z3.Int(name)
            elif sort_lower == "real":
                env[name] = z3.Real(name)
            elif sort_lower == "bool":
                env[name] = z3.Bool(name)
            else:
                raise ValueError(f"Unsupported variable sort: {sort}")

        # Parse every constraint before solving; a rejected expression invalidates the problem.
        for index, tree in enumerate(trees):
            try:
                solver.add(evaluate_constraint(tree, env, z3))
            except Exception as exc:
                raise ValueError(f"Constraint {index + 1} is invalid: {exc}") from exc

        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            assignments = {}
            for var_name, z3_var in env.items():
                try:
                    val = model[z3_var]
                    if val is not None:
                        assignments[var_name] = self._z3_to_python(val)
                except Exception:
                    pass
            return {"satisfiable": True, "model": assignments, "error": ""}
        elif result == z3.unsat:
            return {"satisfiable": False, "model": {}, "error": ""}
        else:
            return {"satisfiable": None, "model": {}, "error": "unknown/timeout"}

    def verify_property(
        self,
        preconditions: list[str],
        postcondition: str,
        code_logic: str,
        variables: dict[str, str] | None = None,
        timeout_ms: int = 5000,
    ) -> dict[str, Any]:
        """Verify that code_logic satisfies postcondition given preconditions.

        Uses negation: checks if (preconditions ∧ code_logic ∧ ¬postcondition) is UNSAT.
        UNSAT → property holds. SAT → counterexample found.
        """
        if not _z3_available():
            return {"verified": None, "counterexample": {}, "error": "z3-solver not installed"}

        # An inconsistent model proves anything vacuously; it is not a code proof.
        premise = self.check_sat([*preconditions, code_logic], variables, timeout_ms)
        if premise["satisfiable"] is not True:
            return {
                "verified": None,
                "counterexample": {},
                "error": premise.get("error") or "Preconditions and code model are inconsistent",
            }
        # Build negated check: pre ∧ code ∧ ¬post
        negated = [*preconditions, code_logic, f"not ({postcondition})"]
        result = self.check_sat(negated, variables, timeout_ms)

        if result["satisfiable"] is False:
            return {"verified": True, "counterexample": {}, "error": ""}
        elif result["satisfiable"] is True:
            return {"verified": False, "counterexample": result["model"], "error": ""}
        else:
            return {"verified": None, "counterexample": {}, "error": result.get("error", "")}

    def solve_arc_constraints(
        self,
        grid_constraints: list[str],
        cell_vars: dict[str, tuple[int, int]],
        color_range: tuple[int, int] = (0, 9),
        timeout_ms: int = 10000,
    ) -> dict[str, Any]:
        """Solve ARC grid constraints symbolically.

        Parameters
        ----------
        grid_constraints:
            Symbolic constraints on cell variables (e.g. ``"c_0_0 == c_1_1"``).
        cell_vars:
            Mapping of variable name → (row, col) for every constrained cell.
        color_range:
            Valid color integer range (inclusive). ARC uses 0-9.

        Returns
        -------
        dict with ``assignments`` mapping var → color int if SAT.
        """
        var_decls = {name: "Int" for name in cell_vars}
        bounds = [
            f"({name} >= {color_range[0]}) and ({name} <= {color_range[1]})" for name in cell_vars
        ]
        all_constraints = bounds + grid_constraints

        result = self.check_sat(all_constraints, var_decls, timeout_ms)
        if result["satisfiable"]:
            grid_assignments = {
                cell_vars[name]: result["model"].get(name)
                for name in cell_vars
                if name in result["model"]
            }
            return {"satisfiable": True, "grid": grid_assignments, "vars": result["model"]}
        return {"satisfiable": result["satisfiable"], "grid": {}, "error": result.get("error", "")}

    def _auto_detect_vars(self, constraints: list[str]) -> dict[str, str]:
        """Detect symbolic names from syntax; function names are not variables."""
        names: set[str] = set()
        for constraint in constraints:
            for node in ast.walk(parse_constraint(constraint)):
                if isinstance(node, ast.Name) and node.id not in FUNCTIONS:
                    validate_name(node.id)
                    names.add(node.id)
        return {name: "Int" for name in sorted(names)}

    @staticmethod
    def _z3_to_python(val: Any) -> Any:
        try:
            import z3

            if z3.is_int_value(val):
                return val.as_long()
            if z3.is_rational_value(val):
                return float(val.numerator_as_long()) / float(val.denominator_as_long())
            if z3.is_true(val):
                return True
            if z3.is_false(val):
                return False
        except Exception:
            pass
        return str(val)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def verify_code_property(
    code: str,
    property_description: str,
    timeout_ms: int = 3000,
) -> dict[str, Any]:
    """High-level wrapper: ask an LLM to generate Z3 constraints then verify.

    Used by the Refinement node to check candidate solutions symbolically.
    """
    try:
        import json

        from langchain_core.messages import HumanMessage

        from isaac.llm.provider import get_llm

        llm = get_llm("fast")
        prompt = (
            f"Given this Python code:\n```python\n{code}\n```\n\n"
            f"Property to verify: {property_description}\n\n"
            "Produce a JSON object with:\n"
            "- 'variables': dict mapping variable names to Z3 sorts ('Int','Real','Bool')\n"
            "- 'preconditions': list of Z3/Python constraint strings\n"
            "- 'postcondition': single constraint string\n"
            "- 'code_logic': one-line equation capturing what the code does\n"
            "Respond only with JSON."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = str(resp.content).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        spec = json.loads(raw)
        if (
            not isinstance(spec, dict)
            or not {"preconditions", "postcondition", "code_logic", "variables"} <= spec.keys()
        ):
            raise ValueError("Formalization is missing required fields")
        if not isinstance(spec["preconditions"], list) or not isinstance(spec["variables"], dict):
            raise ValueError("Invalid formalization field types")
        prover = TheoremProver()
        model_result = prover.verify_property(
            preconditions=spec.get("preconditions", []),
            postcondition=spec.get("postcondition", "True"),
            code_logic=spec.get("code_logic", ""),
            variables=spec.get("variables"),
            timeout_ms=timeout_ms,
        )
        # The solver proves only the generated model. No checked translation
        # currently connects arbitrary Python code to the LLM's formalization.
        return {
            "verified": None,
            "counterexample": model_result["counterexample"],
            "model_verified": model_result["verified"],
            "formalization": spec,
            "error": model_result["error"]
            or "LLM formalization requires review against the source code",
        }
    except Exception as exc:
        logger.warning("verify_code_property failed: %s", exc)
        return {"verified": None, "error": str(exc)}
