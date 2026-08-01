"""Measure how reliably a model emits well-formed tool calls.

The claim behind 1.4.0 is that ISAAC is usable on small local models.  The
failure that most often makes it *un*usable is not bad reasoning — it is a
model that decides on the right tool and then emits the call as prose, which
the provider reports as "no tool calls" and the agent loop used to accept as a
final answer.  This module measures that failure directly.

Each case in :data:`SUITE` is a prompt whose only sensible next action is one
named tool call.  A case is scored by *how the call arrived*, not by whether
the task was solved:

``native``
    The provider returned a structured tool call for the expected tool.  The
    only outcome that needed no recovery.
``repaired``
    No native call, but :func:`~isaac.agents.tool_repair.salvage_tool_calls`
    recovered the intended call from the text.
``reflexion``
    Unparseable, but recovered after one corrective retry.
``unrecovered``
    Intended a call; nothing could recover it.
``wrong_tool``
    A well-formed call to the wrong tool — a reasoning error, not a formatting
    one, and reported separately so it does not flatter the formatting number.
``no_attempt``
    Answered from parametric knowledge instead of calling anything.

The headline metric is **malformed rate**: malformed attempts over all
attempts, where an attempt is any case in which the model tried to call a tool.
It is computed the same way before and after the fix, because the fix changes
what ISAAC *recovers*, not what the model *emits*.  Reporting only the post-fix
success rate would hide that distinction.

Run it with ``isaac eval-toolcalls --model nemotron-3-nano:4b``.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "SUITE",
    "CaseResult",
    "ToolCallReport",
    "run_suite",
    "stub_tool_schemas",
]


@dataclass(frozen=True)
class ToolCallCase:
    """One prompt that should provoke exactly one known tool call."""

    id: str
    prompt: str
    expect_tool: str


# Stub tools: realistic schemas, no side effects. Names and parameters mirror
# the real ISAAC tools so the model sees the shapes it would see in production.
STUB_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Search the web and return result snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "limit": {"type": "integer", "description": "Max results (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "code",
        "description": "Execute a Python snippet and return stdout.",
        "parameters": {
            "type": "object",
            "properties": {"source": {"type": "string", "description": "Python source to run."}},
            "required": ["source"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "browser",
        "description": "Fetch a URL and return its readable text.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Absolute URL."}},
            "required": ["url"],
        },
    },
]

#: 20 cases, 4 per tool. Small enough to run on a laptop GPU in minutes,
#: large enough that one flaky case moves the rate by 5 points, not 25.
SUITE: list[ToolCallCase] = [
    # web_search
    ToolCallCase("ws1", "Who won the 2026 Formula 1 constructors' championship?", "web_search"),
    ToolCallCase("ws2", "Find the current population of Belo Horizonte.", "web_search"),
    ToolCallCase("ws3", "What did the ECB decide at its most recent rate meeting?", "web_search"),
    ToolCallCase("ws4", "Look up recent reviews of the Fujifilm X-T5.", "web_search"),
    # code
    ToolCallCase("cd1", "Compute the 40th Fibonacci number.", "code"),
    ToolCallCase("cd2", "Calculate the standard deviation of 4, 8, 15, 16, 23, 42.", "code"),
    ToolCallCase("cd3", "How many days are between 1999-12-31 and 2026-07-29?", "code"),
    ToolCallCase("cd4", "Find every prime below 200 and give me their sum.", "code"),
    # read_file
    ToolCallCase("rf1", "Show me what's in config/settings.yaml.", "read_file"),
    ToolCallCase("rf2", "Read notes/meeting-2026-07-01.md and summarise it.", "read_file"),
    ToolCallCase("rf3", "Open data/results.csv so we can look at the columns.", "read_file"),
    ToolCallCase("rf4", "What does the README.md in the workspace say?", "read_file"),
    # write_file
    ToolCallCase("wf1", "Save the text 'hello world' to output/greeting.txt.", "write_file"),
    ToolCallCase("wf2", "Create todo.md containing a single line: 'buy milk'.", "write_file"),
    ToolCallCase("wf3", 'Write the JSON {"ok": true} into status.json.', "write_file"),
    ToolCallCase("wf4", "Put a one-line Python hello-world into scripts/hi.py.", "write_file"),
    # browser
    ToolCallCase("br1", "Fetch https://example.com and tell me the heading.", "browser"),
    ToolCallCase("br2", "Read the page at https://docs.python.org/3/library/json.html.", "browser"),
    ToolCallCase("br3", "Open https://news.ycombinator.com and list the top story.", "browser"),
    ToolCallCase("br4", "Get the content of https://en.wikipedia.org/wiki/Chollet.", "browser"),
]

SYSTEM_PROMPT = (
    "You are an agent with access to tools. For the user's request, call the "
    "single most appropriate tool. Do not answer from memory when a tool would "
    "give a better answer. Call exactly one tool."
)


def stub_tool_schemas() -> list[dict[str, Any]]:
    """Return the stub tools in OpenAI function-schema form."""
    return [{"type": "function", "function": dict(tool)} for tool in STUB_TOOLS]


@dataclass
class CaseResult:
    """Outcome of a single case."""

    case_id: str
    expect_tool: str
    outcome: str  # native | repaired | reflexion | unrecovered | wrong_tool | no_attempt | error
    got_tool: str = ""
    latency_s: float = 0.0
    raw_text: str = ""
    error: str = ""
    #: The text held a recoverable call, whether or not this mode recovered it.
    #: Lets a ``native``-mode baseline report what repair *would* have caught
    #: without crediting itself for recovery it did not perform.
    salvageable: bool = False

    @property
    def attempted(self) -> bool:
        """Did the model try to call a tool at all?"""
        return self.outcome in {"native", "repaired", "reflexion", "unrecovered", "wrong_tool"}

    @property
    def malformed(self) -> bool:
        """Did the attempt arrive outside the provider's native channel?"""
        return self.outcome in {"repaired", "reflexion", "unrecovered"}


@dataclass
class ToolCallReport:
    """Aggregate result of a suite run."""

    model: str
    mode: str
    cases: list[CaseResult] = field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0

    def _count(self, outcome: str) -> int:
        return sum(1 for c in self.cases if c.outcome == outcome)

    @property
    def attempts(self) -> int:
        return sum(1 for c in self.cases if c.attempted)

    @property
    def malformed(self) -> int:
        return sum(1 for c in self.cases if c.malformed)

    @property
    def malformed_rate(self) -> float:
        """Headline metric: malformed attempts / all attempts."""
        return self.malformed / self.attempts if self.attempts else 0.0

    @property
    def usable(self) -> int:
        """Cases that yielded a correct, executable call after recovery."""
        return self._count("native") + self._count("repaired") + self._count("reflexion")

    @property
    def usable_rate(self) -> float:
        return self.usable / len(self.cases) if self.cases else 0.0

    @property
    def unrecovered_rate(self) -> float:
        """Share of all cases still broken after recovery — the user-visible failure."""
        total = len(self.cases)
        return self._count("unrecovered") / total if total else 0.0

    @property
    def baseline_usable(self) -> int:
        """Cases 1.3.x would have executed: native calls only.

        Derived from the *same* model turns as :attr:`usable`, so the
        before/after delta is attributable to the recovery policy alone with no
        sampling variance between two separate runs.
        """
        return self._count("native")

    @property
    def baseline_usable_rate(self) -> float:
        return self.baseline_usable / len(self.cases) if self.cases else 0.0

    def summary(self) -> dict[str, Any]:
        latencies = [c.latency_s for c in self.cases if c.latency_s > 0]
        return {
            "model": self.model,
            "mode": self.mode,
            "cases": len(self.cases),
            "native": self._count("native"),
            "repaired": self._count("repaired"),
            "reflexion": self._count("reflexion"),
            "unrecovered": self._count("unrecovered"),
            "wrong_tool": self._count("wrong_tool"),
            "no_attempt": self._count("no_attempt"),
            "error": self._count("error"),
            "salvageable": sum(1 for c in self.cases if c.salvageable),
            "attempts": self.attempts,
            "malformed": self.malformed,
            "malformed_rate": round(self.malformed_rate, 4),
            "usable": self.usable,
            "usable_rate": round(self.usable_rate, 4),
            "baseline_usable": self.baseline_usable,
            "baseline_usable_rate": round(self.baseline_usable_rate, 4),
            "unrecovered_rate": round(self.unrecovered_rate, 4),
            "median_latency_s": round(statistics.median(latencies), 2) if latencies else 0.0,
            "duration_s": round(self.duration_s, 1),
        }

    def to_json(self) -> str:
        return json.dumps(
            {"summary": self.summary(), "cases": [asdict(c) for c in self.cases]},
            indent=2,
        )

    def render(self) -> str:
        s = self.summary()
        lines = [
            f"Tool-call reliability — {s['model']}  (mode: {s['mode']})",
            "=" * 64,
            f"  cases                {s['cases']}",
            f"  native calls         {s['native']}",
            f"  repaired             {s['repaired']}",
            f"  reflexion-recovered  {s['reflexion']}",
            f"  unrecovered          {s['unrecovered']}",
            f"  wrong tool           {s['wrong_tool']}",
            f"  no attempt           {s['no_attempt']}",
            f"  errors               {s['error']}",
            "-" * 64,
            # With no accepted attempt there is nothing to be malformed about.
            # Printing "0.0%" there reads as a perfect score for a model that
            # never got to answer, so say so instead.
            f"  MALFORMED RATE       {s['malformed_rate']:.1%}  "
            f"({s['malformed']}/{s['attempts']} attempts)"
            if s["attempts"]
            else "  MALFORMED RATE       —  (no request was accepted; nothing measured)",
        ]

        if self.mode == "constrained":
            # There is no 1.3.x baseline to compare against here: the envelope
            # replaces the native channel rather than recovering from it, and
            # 1.3.x had no envelope mode at all. Printing a "BEFORE" figure
            # would compare this run against itself.
            lines += [
                f"  well-formed calls    {s['usable_rate']:.1%}  "
                f"({s['usable']}/{s['cases']} cases, grammar-enforced)",
                "  (no 1.3.x baseline: constrained mode replaces native tool calling — "
                "run --mode native for the comparison)",
            ]
        else:
            lines += [
                f"  usable BEFORE (1.3)  {s['baseline_usable_rate']:.1%}  "
                f"({s['baseline_usable']}/{s['cases']} cases, native only)",
                f"  usable AFTER  (1.4)  {s['usable_rate']:.1%}  "
                f"({s['usable']}/{s['cases']} cases, + repair/reflexion)",
            ]

        lines += [
            f"  still broken         {s['unrecovered_rate']:.1%}",
            f"  median latency       {s['median_latency_s']}s   total {s['duration_s']}s",
        ]
        return "\n".join(lines)


def _build_llm(model: str, base_url: str, temperature: float, constrained: bool) -> Any:
    from isaac.llm.providers import ollama as ollama_provider

    llm = ollama_provider.build(model=model, base_url=base_url, temperature=temperature)
    if constrained:
        from isaac.llm.constrained import apply_constraint

        return apply_constraint(llm, stub_tool_schemas())
    try:
        return llm.bind_tools(stub_tool_schemas())
    except Exception as exc:  # pragma: no cover - provider-specific
        logger.warning("bind_tools failed for %s: %s", model, exc)
        return llm


def _classify(
    ai: Any,
    case: ToolCallCase,
    known: set[str],
    mode: str,
) -> tuple[str, str, str, bool]:
    """Return ``(outcome, got_tool, text, salvageable)`` for one model turn.

    In ``native`` mode a salvageable call is still scored ``unrecovered``:
    that is what 1.3.x did with it — returned the blob to the user as a final
    answer. Scoring it as recovered would credit the baseline with the very
    fix being measured. ``salvageable`` records it separately.
    """
    from isaac.agents.tool_repair import looks_like_attempted_call, salvage_tool_calls

    text = ai if isinstance(ai, str) else _content_text(ai)
    native = list(getattr(ai, "tool_calls", None) or [])

    if native:
        got = str(native[0].get("name", ""))
        return ("native" if got == case.expect_tool else "wrong_tool", got, text, False)

    if mode == "constrained":
        from isaac.llm.constrained import parse_envelope

        calls, _answer = parse_envelope(text, known)
        if calls:
            got = calls[0]["name"]
            # Under constraint the envelope *is* the native channel.
            return ("native" if got == case.expect_tool else "wrong_tool", got, text, False)

    salvaged = salvage_tool_calls(text, known)
    if salvaged:
        got = salvaged[0]["name"]
        if mode == "native":
            return ("unrecovered", got, text, True)
        return ("repaired" if got == case.expect_tool else "wrong_tool", got, text, True)

    if looks_like_attempted_call(text, known):
        return ("unrecovered", "", text, False)
    return ("no_attempt", "", text, False)


def _content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content or "")


def run_suite(
    model: str,
    base_url: str = "http://localhost:11434",
    mode: str = "repair",
    temperature: float = 0.2,
    cases: Sequence[ToolCallCase] | None = None,
    reflexion: bool = True,
    progress: bool = False,
    llm: Any | None = None,
) -> ToolCallReport:
    """Run the tool-call suite against *model*.

    Parameters
    ----------
    mode:
        ``"native"``      — bind tools, no repair. The 1.3.x baseline.
        ``"repair"``      — bind tools, repair + Reflexion on malformed output.
        ``"constrained"`` — grammar-constrained envelope instead of native calls.
    reflexion:
        In ``repair`` mode, issue one corrective retry for unparseable output.
        Disable to isolate how much the parser alone recovers.
    llm:
        Pre-built chat client, bypassing *model*/*base_url*. Used by the tests
        to score the suite offline against scripted turns.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    suite = list(cases if cases is not None else SUITE)
    known = {t["name"] for t in STUB_TOOLS}
    constrained = mode == "constrained"
    if llm is None:
        llm = _build_llm(model, base_url, temperature, constrained)

    system = SYSTEM_PROMPT
    if constrained:
        from isaac.llm.constrained import CONSTRAINED_SYSTEM_SUFFIX

        catalogue = "\n".join(f"- {t['name']}: {t['description']}" for t in STUB_TOOLS)
        system = f"{system}\n\nTools:\n{catalogue}\n{CONSTRAINED_SYSTEM_SUFFIX}"

    report = ToolCallReport(model=model, mode=mode)
    report.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_start = time.perf_counter()

    for i, case in enumerate(suite, 1):
        messages: list[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=case.prompt),
        ]
        t0 = time.perf_counter()
        try:
            ai = llm.invoke(messages)
        except Exception as exc:
            logger.warning("case %s failed: %s", case.id, exc)
            report.cases.append(
                CaseResult(case.id, case.expect_tool, "error", error=str(exc)[:300])
            )
            continue

        outcome, got, text, salvageable = _classify(ai, case, known, mode)

        # Reflexion: one corrective retry for output nothing could parse.
        if outcome == "unrecovered" and mode == "repair" and reflexion:
            from isaac.agents.tool_repair import reflexion_prompt

            messages.append(ai)
            messages.append(HumanMessage(content=reflexion_prompt(text, known)))
            try:
                retry = llm.invoke(messages)
            except Exception as exc:  # pragma: no cover
                logger.debug("reflexion retry failed for %s: %s", case.id, exc)
            else:
                r_outcome, r_got, _, _ = _classify(retry, case, known, mode)
                if r_outcome in {"native", "repaired"}:
                    outcome, got = "reflexion", r_got
                elif r_outcome == "wrong_tool":
                    outcome, got = "wrong_tool", r_got

        latency = time.perf_counter() - t0
        report.cases.append(
            CaseResult(
                case_id=case.id,
                expect_tool=case.expect_tool,
                outcome=outcome,
                got_tool=got,
                latency_s=latency,
                raw_text=text[:600],
                salvageable=salvageable,
            )
        )
        if progress:
            print(
                f"  [{i:>2}/{len(suite)}] {case.id:<4} {case.expect_tool:<11} "
                f"→ {outcome:<12} {latency:5.1f}s",
                flush=True,
            )

    report.duration_s = time.perf_counter() - run_start
    return report
