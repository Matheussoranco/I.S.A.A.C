"""MathExpert — symbolic algebra & calculus via SymPy with LLM fallback.

Symbolic-first: SymPy parses & solves. If SymPy can't, we hand off to the
language expert with the symbolic context attached. This is the canonical
neuro-symbolic pattern: prefer exact symbolic answers, fall back to neural
when the symbolic engine fails.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

from isaac.experts.base import Expert, ExpertNotApplicable, ExpertResponse

logger = logging.getLogger(__name__)

_MATH_KEYWORDS = (
    "solve", "integrate", "differentiate", "derivative", "factor",
    "simplify", "expand", "limit", "matrix", "determinant", "eigen",
    "polynomial", "equation", "compute", "evaluate", "calculate",
)
_MATH_SYMBOL_RE = re.compile(
    r"(?:[\^\=]|\bdx\b|\bsin\b|\bcos\b|\btan\b|\blog\b|\bln\b|\bsqrt\b|"
    r"\\int|\\sum|\\frac|\d+\s*[\+\-\*\/]\s*\d+)"
)


class MathExpert(Expert):
    """SymPy-backed symbolic mathematics expert."""

    name: ClassVar[str] = "math"
    domains: ClassVar[tuple[str, ...]] = ("math", "algebra", "calculus", "symbolic")
    description: ClassVar[str] = "SymPy symbolic algebra, calculus, equation solving."
    cost: ClassVar[float] = 0.2

    def __init__(self) -> None:
        try:
            import sympy  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            logger.info("MathExpert: sympy not installed — symbolic fallback disabled.")

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        q = query.lower()
        score = 0.0
        for kw in _MATH_KEYWORDS:
            if kw in q:
                score = max(score, 0.7)
        if _MATH_SYMBOL_RE.search(query):
            score = max(score, 0.55)
        # Pure arithmetic
        if re.fullmatch(r"\s*[\d\.\+\-\*\/\^\(\)\s]+\s*", query):
            score = max(score, 0.95)
        return score if self._available else min(score, 0.3)

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        if not self._available:
            raise ExpertNotApplicable("sympy not installed")

        result = self._symbolic(query)
        if result is None:
            raise ExpertNotApplicable("sympy could not parse")

        text, evidence, artifact = result
        return ExpertResponse(
            expert=self.name,
            answer=text,
            confidence=0.9,
            evidence=evidence,
            artifacts={"sympy": artifact},
        )

    # ------------------------------------------------------------------
    # Symbolic kernel
    # ------------------------------------------------------------------

    def _symbolic(self, query: str) -> tuple[str, list[str], str] | None:
        import sympy as sp

        q = query.strip().rstrip("?.")
        q_lower = q.lower()

        # Pattern 1: 'solve <eq>' or 'solve <eq> for <var>'
        m = re.match(r"\s*solve\s+(.+?)(?:\s+for\s+([a-zA-Z]\w*))?\s*$", q, re.I)
        if m:
            eq_str, var_str = m.group(1), m.group(2)
            try:
                eq = self._parse_equation(eq_str)
                sym = sp.Symbol(var_str) if var_str else self._pick_symbol(eq)
                if sym is None:
                    return None
                sols = sp.solve(eq, sym)
                if not sols:
                    return f"No solutions for {sym}.", [str(eq)], str(sols)
                return (
                    f"{sym} = " + ", ".join(str(s) for s in sols),
                    [f"equation: {eq}", f"variable: {sym}"],
                    str(sols),
                )
            except Exception as exc:
                logger.debug("solve failed: %s", exc)

        # Pattern 2: integration
        m = re.match(r"\s*integrate\s+(.+?)(?:\s+(?:dx|with\s+respect\s+to\s+([a-zA-Z]\w*)))?\s*$", q, re.I)
        if m:
            try:
                expr = sp.sympify(m.group(1))
                var_name = m.group(2) or "x"
                var = sp.Symbol(var_name)
                result = sp.integrate(expr, var)
                return (
                    f"∫ {expr} d{var} = {result} + C",
                    [f"expr: {expr}"],
                    str(result),
                )
            except Exception as exc:
                logger.debug("integrate failed: %s", exc)

        # Pattern 3: differentiation
        m = re.match(r"\s*(?:differentiate|derivative\s+of)\s+(.+?)(?:\s+(?:with\s+respect\s+to\s+([a-zA-Z]\w*)|wrt\s+([a-zA-Z]\w*)))?\s*$", q, re.I)
        if m:
            try:
                expr = sp.sympify(m.group(1))
                var_name = m.group(2) or m.group(3) or "x"
                var = sp.Symbol(var_name)
                result = sp.diff(expr, var)
                return (
                    f"d/d{var}({expr}) = {result}",
                    [f"expr: {expr}"],
                    str(result),
                )
            except Exception as exc:
                logger.debug("diff failed: %s", exc)

        # Pattern 4: simplify / factor / expand
        for op in ("simplify", "factor", "expand"):
            m = re.match(rf"\s*{op}\s+(.+?)\s*$", q, re.I)
            if m:
                try:
                    expr = sp.sympify(m.group(1))
                    fn = getattr(sp, op)
                    result = fn(expr)
                    return (
                        f"{op}({expr}) = {result}",
                        [f"input: {expr}"],
                        str(result),
                    )
                except Exception as exc:
                    logger.debug("%s failed: %s", op, exc)

        # Pattern 5: pure arithmetic / numeric eval
        try:
            expr = sp.sympify(q.replace("^", "**"))
            simplified = sp.simplify(expr)
            if simplified.free_symbols:
                return (
                    str(simplified),
                    [f"expression: {expr}"],
                    str(simplified),
                )
            value = float(simplified) if simplified.is_real else complex(simplified)
            return (
                f"{q} = {simplified} ≈ {value:g}" if simplified.is_real else f"{q} = {simplified}",
                [f"expression: {expr}"],
                str(simplified),
            )
        except Exception:
            pass

        return None

    @staticmethod
    def _parse_equation(s: str) -> Any:
        import sympy as sp
        s = s.replace("^", "**")
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            return sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
        # "x^2 - 4" treated as = 0
        return sp.sympify(s)

    @staticmethod
    def _pick_symbol(eq: Any) -> Any:
        symbols = sorted(eq.free_symbols, key=lambda s: str(s))
        return symbols[0] if symbols else None
