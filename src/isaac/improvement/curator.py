"""Curator — periodic auditor for the skill library.

Mirrors the Hermes curator: surfaces stale and failing skills and proposes
improvements.  Heuristic rules always run; when an LLM is provided it is asked
once per flagged skill for a one-line reason (and, for ``improve``, a
unified-diff patch).  A flaky LLM degrades gracefully to the heuristic output.

The Curator only *recommends* — deprecations are opt-in via ``apply()`` and
improve patches are written to disk for human review, never auto-applied.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400
STALE_DAYS = 90

IMPROVE_PROMPT = """You are auditing a skill library. Skill: {name}
It has been used {uses} times with a success rate of {success_rate:.2f}.
Recommended action: {action}.

Reply with exactly one line: REASON: <one-sentence reason>
{patch_instruction}"""

PATCH_INSTRUCTION = (
    "Then reply with a unified-diff patch that would improve the skill, "
    "introduced by a line containing exactly: PATCH:"
)


@dataclass
class CuratorRecommendation:
    skill_name: str
    action: str  # improve | rename | merge | deprecate | keep
    reason: str
    suggested_patch: str | None = None
    confidence: float = 0.0  # in [0, 1]

    def __post_init__(self) -> None:
        if self.action not in {"improve", "rename", "merge", "deprecate", "keep"}:
            msg = f"invalid curator action: {self.action!r}"
            raise ValueError(msg)
        self.confidence = max(0.0, min(1.0, self.confidence))


def patches_dir() -> Path:
    """Directory where suggested improvement patches are written."""
    path = Path.home() / ".isaac" / "curator" / "patches"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Curator:
    """Audit skills in ProceduralMemory and propose actions."""

    def __init__(self, llm: Any = None, procedural: Any = None) -> None:
        self.llm = llm
        if procedural is not None:
            self._procedural = procedural
        else:
            from isaac.memory.manager import get_memory_manager

            self._procedural = get_memory_manager().procedural

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _skill_stats(self, name: str) -> dict:
        proc = self._procedural
        uses = 0
        success_rate = 0.0
        last_run_at: float | None = None

        record = None
        if hasattr(proc, "get_record"):
            record = proc.get_record(name)
        elif isinstance(proc, dict):
            record = proc.get(name)

        if record is not None:
            uses = int(_get(record, "total_invocations", _get(record, "uses", 0)) or 0)
            versions = _get(record, "versions", []) or []
            if versions:
                latest = versions[-1]
                success_rate = float(_get(latest, "success_rate", 0.0) or 0.0)
                ts = _get(latest, "timestamp", None)
                last_run_at = _parse_ts(ts)
            else:
                success_rate = float(_get(record, "success_rate", 0.0) or 0.0)
            if last_run_at is None:
                last_run_at = _parse_ts(_get(record, "updated_at", None))
            # Explicit stat overrides (synthetic/monkeypatched records).
            override_uses = _get(record, "uses", None)
            if override_uses is not None:
                uses = int(override_uses)
            override_sr = _get(record, "success_rate", None)
            if override_sr is not None:
                success_rate = float(override_sr)
            override_last = _get(record, "last_run_at", None)
            if override_last is not None:
                last_run_at = _parse_ts(override_last)

        return {"uses": uses, "success_rate": success_rate, "last_run_at": last_run_at}

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic(
        name: str, uses: int, success_rate: float, last_run_at: float | None, now: float
    ) -> CuratorRecommendation:
        if success_rate < 0.3 and uses >= 5:
            return CuratorRecommendation(
                skill_name=name,
                action="deprecate",
                reason=f"success rate {success_rate:.2f} < 0.3 over {uses} uses",
                confidence=0.8,
            )
        if success_rate < 0.6 and uses >= 10:
            return CuratorRecommendation(
                skill_name=name,
                action="improve",
                reason=f"mediocre success rate {success_rate:.2f} over {uses} uses",
                confidence=0.6,
            )
        if (
            last_run_at is not None
            and (now - last_run_at) > STALE_DAYS * SECONDS_PER_DAY
            and uses < 3
        ):
            assert last_run_at is not None
            days = int((now - last_run_at) / SECONDS_PER_DAY)
            return CuratorRecommendation(
                skill_name=name,
                action="deprecate",
                reason=f"stale: not run in {days} days, {uses} uses",
                confidence=0.5,
            )
        return CuratorRecommendation(
            skill_name=name,
            action="keep",
            reason="within healthy thresholds",
            confidence=0.4,
        )

    # ------------------------------------------------------------------
    # LLM refinement
    # ------------------------------------------------------------------

    def _refine_with_llm(
        self, rec: CuratorRecommendation, uses: int, success_rate: float
    ) -> CuratorRecommendation:
        if self.llm is None:
            return rec
        try:
            prompt = IMPROVE_PROMPT.format(
                name=rec.skill_name,
                uses=uses,
                success_rate=success_rate,
                action=rec.action,
                patch_instruction=PATCH_INSTRUCTION if rec.action == "improve" else "",
            )
            try:
                from langchain_core.messages import HumanMessage

                response = self.llm.invoke([HumanMessage(content=prompt)])
            except Exception:
                response = self.llm.invoke(prompt)
            content = getattr(response, "content", response) or ""
            content = str(content)
            reason = rec.reason
            patch: str | None = rec.suggested_patch
            for line in content.splitlines():
                if line.strip().upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip() or reason
            if "PATCH:" in content:
                patch = content.split("PATCH:", 1)[1].strip() or None
            return CuratorRecommendation(
                skill_name=rec.skill_name,
                action=rec.action,
                reason=reason,
                suggested_patch=patch,
                confidence=rec.confidence,
            )
        except Exception as exc:
            logger.warning(
                "Curator: LLM refinement failed for %s (%s) — using heuristic.",
                rec.skill_name,
                exc,
            )
            return rec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit_all(self) -> list[CuratorRecommendation]:
        """Audit every active skill and return recommendations."""
        proc = self._procedural
        if hasattr(proc, "list_active"):
            names = list(proc.list_active())
        elif isinstance(proc, dict):
            names = list(proc.keys())
        else:
            logger.warning("Curator: procedural memory has no list_active(); nothing to audit.")
            return []

        now = time.time()
        recommendations: list[CuratorRecommendation] = []
        for name in names:
            try:
                stats = self._skill_stats(name)
                rec = self._heuristic(
                    name,
                    stats["uses"],
                    stats["success_rate"],
                    stats["last_run_at"],
                    now,
                )
                if rec.action != "keep":
                    rec = self._refine_with_llm(rec, stats["uses"], stats["success_rate"])
                recommendations.append(rec)
            except Exception as exc:  # pragma: no cover
                logger.warning("Curator: failed to audit %s: %s", name, exc)
        return recommendations

    def apply(
        self,
        recommendations: list[CuratorRecommendation],
        min_confidence: float = 0.7,
        dry_run: bool = True,
    ) -> dict[str, int]:
        """Apply safe actions.  ``improve`` patches are always written to disk
        for human review and never applied automatically."""
        summary: dict[str, int] = {"improve": 0, "rename": 0, "merge": 0, "deprecate": 0, "keep": 0}
        for rec in recommendations:
            if rec.confidence < min_confidence:
                if rec.action == "improve" and rec.suggested_patch:
                    self._write_patch(rec)
                continue
            summary[rec.action] += 1
            if rec.action == "improve" and rec.suggested_patch:
                self._write_patch(rec)
            if rec.action == "deprecate" and not dry_run:
                try:
                    self._procedural.deprecate(rec.skill_name)
                except Exception as exc:
                    logger.warning("Curator: deprecate(%s) failed: %s", rec.skill_name, exc)
        return summary

    @staticmethod
    def _write_patch(rec: CuratorRecommendation) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in rec.skill_name)
        path = patches_dir() / f"{safe}.diff"
        path.write_text(
            f"# Suggested improvement for skill '{rec.skill_name}'\n"
            f"# Reason: {rec.reason}\n"
            f"# Confidence: {rec.confidence:.2f}\n\n"
            f"{rec.suggested_patch or ''}",
            encoding="utf-8",
        )
        logger.info("Curator: wrote suggested patch %s", path)
        return path


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _parse_ts(value: Any) -> float | None:
    """Best-effort conversion of a timestamp-ish value to epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except (ValueError, OSError):
            return None
    return None
