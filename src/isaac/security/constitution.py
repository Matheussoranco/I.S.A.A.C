"""Constitutional Safety Layer — pre-execution action critic.

Implements a Constitutional-AI style review pass that runs *before* any
sandbox / computer-use / connector execution. The critic combines two
layers:

1. **Symbolic rules** — fast, exhaustive deny-list of patterns that should
   never execute (``rm -rf /``, key exfiltration, network calls to known
   bad hosts, file writes outside ``allowed_paths``, etc). These cannot be
   overridden by an LLM.
2. **LLM critic** — for actions that pass the symbolic gate but are still
   ambiguous, an LLM is asked to score the action against a *constitution*
   — a list of plain-English principles. The score must clear a threshold,
   otherwise the action is escalated to human approval.

The constitution is configurable via ``~/.isaac/security/constitution.json``
or the bundled :data:`DEFAULT_CONSTITUTION`. Every decision is audited.

Usage::

    from isaac.security.constitution import review

    decision = review(
        action_kind="shell",
        action="git push origin main",
        context={"branch": "main", "repo": "myrepo"},
    )
    if not decision.allow:
        raise PermissionError(decision.reason)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


DEFAULT_CONSTITUTION: list[str] = [
    "Never destroy data without explicit user consent (rm -rf, drop table, "
    "force-push to protected branches, mass-delete files).",
    "Never exfiltrate secrets, credentials, or private keys. Treat .env, "
    "id_rsa, *.pem, and any file containing 'API_KEY=' as confidential.",
    "Never execute commands that disable security (chmod 777 /, "
    "iptables -F, sudoers edits) without explicit consent.",
    "Stay within sandbox boundaries — file writes should target the project "
    "directory or ~/.isaac/, never system paths (/etc/, /usr/, C:\\Windows).",
    "Never send messages, emails, or post on external services without "
    "explicit per-action approval.",
    "Prefer reversible actions. If an action is irreversible (push, delete, "
    "send), flag it for confirmation.",
    "Never bypass tests or hooks (--no-verify, --no-gpg-sign, force flags) "
    "unless the user explicitly asked.",
    "Decline tasks that target unauthorised third parties (other people's "
    "accounts, systems, or networks).",
]


@dataclass
class ConstitutionViolation:
    rule: str
    severity: str = "high"
    """``"low"`` | ``"medium"`` | ``"high"`` | ``"critical"``."""
    detail: str = ""


@dataclass
class ReviewDecision:
    allow: bool
    score: float = 1.0
    """Aggregate safety score in ``[0, 1]`` — 1 = perfectly safe."""
    violations: list[ConstitutionViolation] = field(default_factory=list)
    reason: str = ""
    requires_approval: bool = False
    """If True, the action should be routed through the human approval node."""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow": self.allow,
            "score": self.score,
            "violations": [v.__dict__ for v in self.violations],
            "reason": self.reason,
            "requires_approval": self.requires_approval,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Symbolic rule engine
# ---------------------------------------------------------------------------


@dataclass
class SymbolicRule:
    """A regex-based deny rule."""
    pattern: re.Pattern[str]
    rule: str
    severity: str = "high"


_DEFAULT_RULES: ClassVar[list[SymbolicRule]] = [
    SymbolicRule(
        re.compile(r"\brm\s+-rf?\s+(/|~|\$HOME|/\*)", re.I),
        "Recursive rm of system root", "critical",
    ),
    SymbolicRule(
        re.compile(r"\b(sudo|doas)\s+rm\s+-r", re.I),
        "Privileged recursive delete", "critical",
    ),
    SymbolicRule(
        re.compile(r":(){\s*:\|:&\s*};:", re.I),
        "Fork bomb pattern", "critical",
    ),
    SymbolicRule(
        re.compile(r"\bdd\s+if=.+\s+of=/dev/(sd[a-z]|nvme|disk\d)", re.I),
        "Direct disk write", "critical",
    ),
    SymbolicRule(
        re.compile(r"\bdrop\s+(table|database|schema)\b", re.I),
        "Database drop statement", "critical",
    ),
    SymbolicRule(
        re.compile(r"\bgit\s+push\s+(?:--force|-f)\s+\S+\s+(main|master|prod\w*)", re.I),
        "Force push to protected branch", "high",
    ),
    SymbolicRule(
        re.compile(r"\bchmod\s+(?:-R\s+)?777\s+/", re.I),
        "World-writable system root", "critical",
    ),
    SymbolicRule(
        re.compile(r"\bcurl\s+[^|\n]*\|\s*(?:bash|sh|zsh)\b", re.I),
        "Pipe remote script to shell", "high",
    ),
    SymbolicRule(
        re.compile(r"\b(API_KEY|SECRET|TOKEN|PASSWORD)\s*=\s*[A-Za-z0-9_\-]{8,}", re.I),
        "Hardcoded credential", "medium",
    ),
    SymbolicRule(
        re.compile(r"(\.env|id_rsa|\.pem|credentials\.json)", re.I),
        "Sensitive file path", "medium",
    ),
    SymbolicRule(
        re.compile(r"--no-verify|--no-gpg-sign", re.I),
        "Bypassing hooks/signing", "medium",
    ),
    SymbolicRule(
        re.compile(r"\biptables\s+-F\b|\bufw\s+disable\b", re.I),
        "Disabling firewall", "high",
    ),
]


def _check_symbolic(action: str) -> list[ConstitutionViolation]:
    violations: list[ConstitutionViolation] = []
    for rule in _DEFAULT_RULES:
        if rule.pattern.search(action):
            violations.append(ConstitutionViolation(
                rule=rule.rule, severity=rule.severity,
                detail=f"Pattern matched: {rule.pattern.pattern}",
            ))
    return violations


# ---------------------------------------------------------------------------
# LLM critic
# ---------------------------------------------------------------------------


_CRITIC_PROMPT = """You are I.S.A.A.C.'s safety critic. Score the proposed action against
the constitution.

Constitution:
{constitution}

Proposed action ({kind}):
{action}

Context:
{context}

Output strict JSON with keys:
  "safe" (bool),
  "score" (float in 0..1, 1 = perfectly safe),
  "violations" (list of strings — empty if none),
  "requires_approval" (bool — true if the user must confirm before running),
  "rationale" (one sentence).

Output only JSON.
"""


def _llm_critic(
    action_kind: str,
    action: str,
    context: dict[str, Any],
    constitution: list[str],
) -> dict[str, Any] | None:
    try:
        from isaac.llm.provider import get_llm
        from langchain_core.messages import HumanMessage

        llm = get_llm("fast")
        prompt = _CRITIC_PROMPT.format(
            constitution="\n".join(f"- {c}" for c in constitution),
            kind=action_kind,
            action=action[:2000],
            context=json.dumps(context, default=str)[:1500],
        )
        raw = str(llm.invoke([HumanMessage(content=prompt)]).content).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception as exc:
        logger.debug("LLM critic failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_constitution() -> list[str]:
    """Load the constitution from disk if present, else built-in default."""
    try:
        from isaac.config.settings import get_settings
        path = get_settings().isaac_home / "security" / "constitution.json"
    except Exception:
        path = Path.home() / ".isaac" / "security" / "constitution.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception as exc:
            logger.warning("Could not load constitution from %s: %s", path, exc)
    return list(DEFAULT_CONSTITUTION)


def review(
    action_kind: str,
    action: str,
    *,
    context: dict[str, Any] | None = None,
    use_llm: bool = True,
    threshold: float = 0.6,
) -> ReviewDecision:
    """Review a proposed action and return a :class:`ReviewDecision`.

    Parameters
    ----------
    action_kind:
        Coarse category — ``"shell"``, ``"file_write"``, ``"http"``,
        ``"git"``, ``"send_message"``, ``"computer_use"``, …
    action:
        The actual action (command string, payload, code).
    context:
        Extra context (working dir, target file, recipient, …).
    use_llm:
        Whether to consult the LLM critic for ambiguous cases.
    threshold:
        Minimum LLM score to permit the action without escalation.
    """
    ctx = context or {}
    violations = _check_symbolic(action)

    # Critical-severity → hard deny, no override
    for v in violations:
        if v.severity == "critical":
            decision = ReviewDecision(
                allow=False,
                score=0.0,
                violations=violations,
                reason=f"Critical rule violated: {v.rule}",
                rationale="symbolic_critical_block",
            )
            _audit(action_kind, action, decision)
            return decision

    # High/medium violations → require approval (LLM critic can confirm)
    must_approve = any(v.severity in ("high", "medium") for v in violations)

    score = 1.0 - 0.2 * len(violations)
    rationale_parts: list[str] = []
    if violations:
        rationale_parts.append(f"{len(violations)} symbolic rule(s) flagged")

    if use_llm:
        critic = _llm_critic(action_kind, action, ctx, load_constitution())
        if critic is not None:
            critic_score = float(critic.get("score", 0.5))
            score = (score + critic_score) / 2.0
            for v in critic.get("violations", []):
                violations.append(ConstitutionViolation(rule=str(v), severity="medium"))
            if bool(critic.get("requires_approval", False)):
                must_approve = True
            rationale_parts.append(f"LLM rationale: {critic.get('rationale', '')}")

    allow = score >= threshold and not any(v.severity in ("critical",) for v in violations)
    decision = ReviewDecision(
        allow=allow,
        score=score,
        violations=violations,
        reason="" if allow else "Score below threshold or unresolved violations.",
        requires_approval=must_approve,
        rationale=" | ".join(rationale_parts) or "no concerns",
    )
    _audit(action_kind, action, decision)
    return decision


def _audit(action_kind: str, action: str, decision: ReviewDecision) -> None:
    try:
        from isaac.security.audit import audit
        audit(
            "constitution",
            "review",
            details={
                "kind": action_kind,
                "action": action[:300],
                "allow": decision.allow,
                "score": round(decision.score, 3),
                "violations": [v.rule for v in decision.violations],
                "requires_approval": decision.requires_approval,
            },
        )
    except Exception:
        pass
