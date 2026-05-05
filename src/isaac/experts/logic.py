"""LogicExpert — Z3 SMT theorem prover wrapper.

Activates on queries that look like constraint-satisfaction or formal
verification. Delegates to :class:`isaac.reasoning.theorem_prover.TheoremProver`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

from isaac.experts.base import Expert, ExpertNotApplicable, ExpertResponse

logger = logging.getLogger(__name__)


class LogicExpert(Expert):
    name: ClassVar[str] = "logic"
    domains: ClassVar[tuple[str, ...]] = ("logic", "constraints", "verification")
    description: ClassVar[str] = "Z3 SMT solver — constraint satisfaction & code property verification."
    cost: ClassVar[float] = 0.3

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        q = query.lower()
        if any(s in q for s in ("satisfiable", "smt ", "z3 ", "constraint",
                                "verify", "prove ", "counterexample")):
            return 0.8
        if re.search(r"find\s+(?:integers?|values?)\s+such\s+that", q):
            return 0.75
        if context and context.get("constraints"):
            return 0.7
        return 0.0

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        try:
            from isaac.reasoning.theorem_prover import TheoremProver
        except ImportError as exc:
            raise ExpertNotApplicable(str(exc)) from exc

        prover = TheoremProver()

        if context and context.get("constraints"):
            constraints = context["constraints"]
            variables = context.get("variables")
            result = prover.check_sat(constraints, variables)
            return ExpertResponse(
                expert=self.name,
                answer=self._format_sat(result),
                confidence=0.9 if result.get("satisfiable") is not None else 0.4,
                evidence=[f"sat={result.get('satisfiable')}"],
                artifacts={"smt_result": result},
            )

        # Best-effort: ask LLM to extract constraints, then solve
        try:
            from isaac.llm.provider import get_llm
            from langchain_core.messages import HumanMessage
            import json

            llm = get_llm("fast")
            prompt = (
                f"Extract a JSON object with keys 'variables' (name->Z3 sort) "
                f"and 'constraints' (list of Python boolean expressions) "
                f"from this question. Respond only with JSON.\n\nQuestion: {query}"
            )
            raw = str(llm.invoke([HumanMessage(content=prompt)]).content).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            spec = json.loads(raw)
            result = prover.check_sat(
                spec.get("constraints", []),
                spec.get("variables"),
            )
            return ExpertResponse(
                expert=self.name,
                answer=self._format_sat(result),
                confidence=0.75 if result.get("satisfiable") is not None else 0.3,
                evidence=[f"extracted: {spec}"],
                artifacts={"smt_result": result, "spec": spec},
            )
        except Exception as exc:
            raise ExpertNotApplicable(f"could not formalise: {exc}") from exc

    @staticmethod
    def _format_sat(result: dict[str, Any]) -> str:
        if result.get("satisfiable") is True:
            model = result.get("model", {})
            return "SAT — model: " + ", ".join(f"{k}={v}" for k, v in model.items())
        if result.get("satisfiable") is False:
            return "UNSAT — constraints have no solution."
        return f"UNKNOWN — {result.get('error', 'solver gave up')}"
