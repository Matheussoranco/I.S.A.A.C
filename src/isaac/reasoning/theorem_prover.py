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

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _z3_available() -> bool:
    try:
        import z3  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


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
            return self._solve(constraints, variables or {}, timeout_ms)
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

        solver = z3.Solver()
        solver.set("timeout", timeout_ms)

        # Auto-detect variables if not provided
        if not var_decls:
            var_decls = self._auto_detect_vars(constraints)

        # Create Z3 variables
        env: dict[str, Any] = {}
        for name, sort in var_decls.items():
            sort_lower = sort.lower()
            if sort_lower == "int":
                env[name] = z3.Int(name)
            elif sort_lower == "real":
                env[name] = z3.Real(name)
            elif sort_lower == "bool":
                env[name] = z3.Bool(name)
            else:
                env[name] = z3.Int(name)

        # Add constraints
        local_env = {**env, "And": z3.And, "Or": z3.Or, "Not": z3.Not,
                     "Implies": z3.Implies, "If": z3.If}
        for c in constraints:
            try:
                expr = eval(c, {"__builtins__": {}}, local_env)  # noqa: S307
                solver.add(expr)
            except Exception as exc:
                logger.debug("Could not parse constraint %r: %s", c, exc)

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
        bounds = [f"({name} >= {color_range[0]}) and ({name} <= {color_range[1]})"
                  for name in cell_vars]
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
        """Heuristically detect variable names from constraint strings."""
        ident_re = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
        keywords = {"and", "or", "not", "if", "else", "True", "False",
                    "And", "Or", "Not", "Implies", "If", "in", "is"}
        names: set[str] = set()
        for c in constraints:
            for m in ident_re.finditer(c):
                name = m.group(1)
                if name not in keywords and not name[0].isupper():
                    names.add(name)
        return {n: "Int" for n in names}

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
        from isaac.llm.provider import get_llm
        from langchain_core.messages import HumanMessage
        import json

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
        prover = TheoremProver()
        return prover.verify_property(
            preconditions=spec.get("preconditions", []),
            postcondition=spec.get("postcondition", "True"),
            code_logic=spec.get("code_logic", ""),
            variables=spec.get("variables"),
            timeout_ms=timeout_ms,
        )
    except Exception as exc:
        logger.warning("verify_code_property failed: %s", exc)
        return {"verified": None, "error": str(exc)}
