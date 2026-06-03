"""I.S.A.A.C. Specialists — a team of domain-focused, tool-using agents.

This package turns I.S.A.A.C. into a *multi-specialist* system: a top-level
:class:`~isaac.specialists.orchestrator.Orchestrator` decomposes a goal and
dispatches it to focused :class:`~isaac.specialists.base.Specialist` agents
(coder, file-organizer, researcher, designer, OS-operator, analyst, critic,
planner, generalist), each running its own local-first tool-use loop.

Quick start::

    from isaac.specialists import orchestrate
    result = orchestrate("Research X, then write a report to ~/report.md")
    print(result.final_output)

    # …or drive a single specialist directly:
    from isaac.specialists import get_specialist
    coder = get_specialist("coder", auto_approve=True)
    print(coder.run("Write a prime sieve and run it for n<100").output)

Importing this package is cheap — the heavy machinery (LLM clients, tools) is
imported lazily, and the concrete roster is only loaded on first registry use.
"""

from __future__ import annotations

from isaac.specialists.base import EventCallback, Specialist, SpecialistResult
from isaac.specialists.orchestrator import (
    Orchestrator,
    OrchestrationResult,
    SubTask,
    SubTaskResult,
    orchestrate,
)
from isaac.specialists.registry import (
    SPECIALIST_CLASSES,
    all_specialists,
    get_specialist,
    get_specialist_class,
    list_specialists,
    register_specialist,
    specialist_names,
)

__all__ = [
    "SPECIALIST_CLASSES",
    "EventCallback",
    "OrchestrationResult",
    "Orchestrator",
    "Specialist",
    "SpecialistResult",
    "SubTask",
    "SubTaskResult",
    "all_specialists",
    "get_specialist",
    "get_specialist_class",
    "list_specialists",
    "orchestrate",
    "register_specialist",
    "specialist_names",
]
