"""LanguageExpert — wraps the local LLM as the default fallback expert.

This is the *language expert* the user requested: a local-first LLM (Ollama
by default) that handles general-knowledge questions, conversation, and any
query that no specialised expert claims with high confidence.

It is always applicable (low base confidence, ~0.4) so it acts as a safety
net in the routing layer.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from isaac.experts.base import Expert, ExpertResponse

logger = logging.getLogger(__name__)


class LanguageExpert(Expert):
    """General-purpose LLM-backed expert.

    Uses the ``"default"`` tier of :func:`isaac.llm.provider.get_llm` so it
    benefits from the project's local-first routing (Ollama → OpenAI/Anthropic).
    """

    name: ClassVar[str] = "language"
    domains: ClassVar[tuple[str, ...]] = ("general", "language", "conversation")
    description: ClassVar[str] = "Local LLM — general knowledge, conversation, reasoning."
    cost: ClassVar[float] = 1.0

    BASE_CONFIDENCE: ClassVar[float] = 0.4

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        # Always applicable. Slightly higher when the query looks conversational.
        q = query.strip().lower()
        if not q:
            return 0.0
        score = self.BASE_CONFIDENCE
        for kw in ("hello", "hi ", "thanks", "explain", "summarise", "summarize",
                   "what is", "who is", "describe", "tell me", "why ", "how do",
                   "translate"):
            if kw in q:
                score = max(score, 0.7)
                break
        return score

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        from isaac.llm.provider import get_llm

        llm = get_llm("default")

        system = (
            "You are I.S.A.A.C.'s language expert. Answer the user's query "
            "accurately and concisely. If you are uncertain, say so. "
            "Use plain text — no markdown headings."
        )
        if context.get("evidence"):
            system += "\nContext from other experts:\n" + str(context["evidence"])[:2000]

        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
        text = str(resp.content).strip()

        # Heuristic confidence: lower if the model hedged
        confidence = 0.7
        lowered = text.lower()
        if any(s in lowered for s in ("i'm not sure", "i am not sure", "i don't know",
                                      "uncertain", "as an ai")):
            confidence = 0.45

        return ExpertResponse(
            expert=self.name,
            answer=text,
            confidence=confidence,
        )
