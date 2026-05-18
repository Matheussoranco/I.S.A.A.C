"""CodeExpert — code understanding, snippet retrieval, and synthesis.

Symbolic side: looks up matching skills in the persistent SkillLibrary.
Neural side: when no skill matches, falls back to the LLM ``"strong"`` tier
with a code-focused prompt.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

from isaac.experts.base import Expert, ExpertResponse

logger = logging.getLogger(__name__)

_CODE_KEYWORDS = (
    "function",
    "class",
    "method",
    "loop",
    "regex",
    "implement",
    "refactor",
    "debug",
    "stack trace",
    "exception",
    "import",
    "compile",
    "lint",
    "type check",
    "snippet",
    "algorithm",
)
_CODE_LANG_HINTS = (
    "python",
    "javascript",
    "typescript",
    "rust",
    "go ",
    "c++",
    "c#",
    "java",
    "bash",
    "shell",
)


class CodeExpert(Expert):
    """Code-focused expert: skill retrieval + LLM synthesis."""

    name: ClassVar[str] = "code"
    domains: ClassVar[tuple[str, ...]] = ("code", "programming", "engineering")
    description: ClassVar[str] = "Code synthesis with skill-library retrieval and LLM fallback."
    cost: ClassVar[float] = 1.5

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        q = query.lower()
        score = 0.0
        if "```" in query or re.search(r"\bdef\s+\w+\s*\(", query):
            score = 0.85
        for kw in _CODE_KEYWORDS:
            if kw in q:
                score = max(score, 0.7)
        for lang in _CODE_LANG_HINTS:
            if lang in q:
                score = max(score, 0.65)
        # Caller hint
        if context and context.get("task_mode") == "code":
            score = max(score, 0.6)
        return score

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        evidence: list[str] = []
        artifacts: dict[str, Any] = {}
        confidence = 0.6

        # Symbolic: try to retrieve a matching skill
        skill_hit = self._retrieve_skill(query)
        if skill_hit:
            evidence.append(f"skill_match: {skill_hit['name']}")
            artifacts["skill"] = skill_hit
            confidence = 0.85

        # Neural: ask the LLM (strong tier) for a code-focused answer
        from langchain_core.messages import HumanMessage, SystemMessage

        from isaac.llm.provider import get_llm

        llm = get_llm("strong")
        system = (
            "You are I.S.A.A.C.'s code expert. Produce correct, idiomatic, "
            "well-typed code. If the query asks for an explanation, be terse "
            "and concrete. Cite line numbers when reading existing code."
        )
        if skill_hit:
            system += f"\n\nA matching skill was found:\n{skill_hit.get('code', '')[:1500]}"

        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
        text = str(resp.content).strip()

        return ExpertResponse(
            expert=self.name,
            answer=text,
            confidence=confidence,
            evidence=evidence,
            artifacts=artifacts,
        )

    @staticmethod
    def _retrieve_skill(query: str) -> dict[str, Any] | None:
        """Look up a matching skill in the persistent library."""
        try:
            from isaac.memory.skill_library import get_skill_library

            lib = get_skill_library()
            results = lib.search(query, top_k=1)
            if results:
                top = results[0]
                if isinstance(top, dict):
                    return top
                # Could be a dataclass — best-effort dict conversion
                try:
                    return {
                        "name": getattr(top, "name", "skill"),
                        "code": getattr(top, "code", str(top)),
                        "score": getattr(top, "score", 0.0),
                    }
                except Exception:
                    return {"name": "skill", "code": str(top)}
        except Exception as exc:
            logger.debug("Skill retrieval failed: %s", exc)
        return None
