"""Grammar- and schema-constrained decoding for local tool calling.

Repairing a malformed tool call after the fact (:mod:`isaac.agents.tool_repair`)
is a safety net.  Preventing it is better: if the decoder is only *permitted* to
emit tokens that keep the output on a valid parse, a 4B model cannot produce a
broken call at all.  Both local backends support this:

* **Ollama** — the ``format`` request field accepts a full JSON Schema
  (structured outputs, Ollama ≥ 0.5.0).  The server compiles it to a grammar
  and masks logits during sampling.
* **llama.cpp** — the ``grammar`` field accepts GBNF directly, which is the
  same machinery Ollama uses underneath.

Both are exposed here behind one shape: an *envelope* the model fills in.

    {"tool": "search_web", "arguments": {"query": "…"}}
    {"tool": "none", "final_answer": "…"}

The envelope branches per tool (``oneOf``, one arm per tool pinning its own
argument schema) by default.  A flat object with a tool-name enum and a generic
``arguments`` field is also available via ``per_tool=False``, and was the
initial default on the assumption that small models cope badly with branched
schemas.  Measurement said otherwise: on ``gemma3:1b`` / Ollama 0.32.5 the two
picked the right tool equally often (8/20), but the flat schema produced
*executable* calls only 3 times against 8 for the branched one — with nothing
constraining the argument keys, the model invented them.  Branching costs
nothing here because the server compiles the schema to a grammar either way.

Constrained decoding is *not* the default: native function calling is better
when a model supports it well, and constraint costs the model its free-form
reasoning channel.  It is the fallback for models that lack the ``tools``
capability or that measurably fail native calling.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTRAINED_SYSTEM_SUFFIX",
    "NO_TOOL",
    "apply_constraint",
    "gbnf_for_tools",
    "parse_envelope",
    "supports_constrained_decoding",
    "tool_envelope_schema",
]

#: Sentinel the model emits in ``tool`` when it wants to answer instead of act.
NO_TOOL = "none"

CONSTRAINED_SYSTEM_SUFFIX = """
You must reply with a single JSON object and nothing else.

To use a tool:
  {"tool": "<tool name>", "arguments": {<arguments for that tool>}}

To give your final answer:
  {"tool": "none", "final_answer": "<your answer>"}

Never wrap the JSON in a code fence. Never emit more than one object.
"""


def tool_envelope_schema(
    tools: list[dict[str, Any]],
    per_tool: bool = True,
) -> dict[str, Any]:
    """Build the JSON Schema the model's output must satisfy.

    Parameters
    ----------
    tools:
        Function schemas as produced by
        :meth:`isaac.tools.base.IsaacTool.to_function_schema`.  Plain
        ``{"name", "parameters"}`` dicts are accepted too.
    per_tool:
        Emit a ``oneOf`` branch per tool, each pinning ``tool`` to a const and
        carrying that tool's real parameter schema.  On by default: it is what
        constrains the *argument* keys, without which a small model picks the
        right tool and then invents its parameters.  Pass ``False`` for a flat
        object with a tool-name enum if a backend rejects ``oneOf``.
    """
    names = [n for n in (_tool_name(t) for t in tools) if n]

    if not per_tool:
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [*names, NO_TOOL]},
                "arguments": {"type": "object"},
                "final_answer": {"type": "string"},
            },
            "required": ["tool"],
        }

    branches: list[dict[str, Any]] = []
    for tool in tools:
        name = _tool_name(tool)
        if not name:
            continue
        branches.append(
            {
                "type": "object",
                "properties": {
                    "tool": {"const": name},
                    "arguments": _tool_params(tool),
                },
                "required": ["tool", "arguments"],
            }
        )
    branches.append(
        {
            "type": "object",
            "properties": {
                "tool": {"const": NO_TOOL},
                "final_answer": {"type": "string"},
            },
            "required": ["tool", "final_answer"],
        }
    )
    return {"oneOf": branches}


def _tool_name(tool: dict[str, Any]) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(tool.get("name") or "")


def _tool_params(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function") if isinstance(tool, dict) else None
    src = fn if isinstance(fn, dict) else tool
    params = src.get("parameters") if isinstance(src, dict) else None
    if isinstance(params, dict) and params:
        return params
    return {"type": "object"}


# --------------------------------------------------------------------------
# GBNF (llama.cpp)
# --------------------------------------------------------------------------

_GBNF_PRELUDE = r"""
ws      ::= [ \t\n]*
string  ::= "\"" char* "\"" ws
char    ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" hex hex hex hex
hex     ::= [0-9a-fA-F]
number  ::= "-"? int frac? exp? ws
int     ::= "0" | [1-9] [0-9]*
frac    ::= "." [0-9]+
exp     ::= [eE] [-+]? [0-9]+
boolean ::= ("true" | "false") ws
null    ::= "null" ws
value   ::= string | number | object | array | boolean | null
array   ::= "[" ws (value ("," ws value)*)? "]" ws
object  ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
""".strip()


def _gbnf_literal(text: str) -> str:
    """Quote a string for use as a GBNF terminal."""
    return '"\\"' + text.replace("\\", "\\\\").replace('"', '\\"') + '\\""'


def gbnf_for_tools(tools: list[dict[str, Any]], per_tool: bool = True) -> str:
    """Generate a GBNF grammar constraining output to the tool envelope.

    llama.cpp accepts this via the ``grammar`` request field.  Unlike the JSON
    Schema path this alternates per tool by default — GBNF is a real grammar,
    so branching costs the sampler nothing.
    """
    names = [n for n in (_tool_name(t) for t in tools) if n]
    if not names:
        return _GBNF_PRELUDE + '\nroot ::= "{" ws string ":" ws value "}" ws\n'

    lines = [_GBNF_PRELUDE, ""]
    if per_tool:
        alts = []
        for i, name in enumerate(names):
            rule = f"call{i}"
            lines.append(
                f'{rule} ::= "{{" ws "\\"tool\\"" ws ":" ws {_gbnf_literal(name)} ws '
                f'"," ws "\\"arguments\\"" ws ":" ws object "}}" ws'
            )
            alts.append(rule)
        lines.append(
            'answer ::= "{" ws "\\"tool\\"" ws ":" ws '
            f'{_gbnf_literal(NO_TOOL)} ws "," ws "\\"final_answer\\"" ws ":" ws string "}}" ws'
        )
        lines.append("root ::= " + " | ".join([*alts, "answer"]))
    else:
        enum = " | ".join(_gbnf_literal(n) for n in [*names, NO_TOOL])
        lines.append(f"toolname ::= {enum}")
        lines.append(
            'root ::= "{" ws "\\"tool\\"" ws ":" ws toolname ws '
            '("," ws string ":" ws value)* "}" ws'
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Applying the constraint to a provider
# --------------------------------------------------------------------------


def supports_constrained_decoding(llm: Any) -> str:
    """Detect which constraint channel *llm* offers.

    Returns ``"ollama"`` (JSON-Schema ``format``), ``"grammar"`` (llama.cpp
    GBNF), ``"json"`` (generic JSON mode only) or ``""`` when the provider
    exposes no constraint mechanism.
    """
    cls = type(llm).__name__.lower()
    if "ollama" in cls:
        return "ollama"
    base_url = str(
        getattr(llm, "openai_api_base", "") or getattr(llm, "base_url", "") or ""
    ).lower()
    # "ollama" must be tested before "llama": the former *contains* the latter,
    # so an Ollama host reached by name rather than on the default port (e.g.
    # ``https://ollama.example.com``) would otherwise be handed a llama.cpp
    # GBNF grammar it cannot honour.
    if "11434" in base_url or "ollama" in base_url:
        return "ollama"
    if "8080" in base_url or "llama" in base_url:
        return "grammar"
    if "openai" in cls or "chatopenai" in cls:
        return "json"
    return ""


def apply_constraint(
    llm: Any,
    tools: list[dict[str, Any]],
    channel: str = "",
    per_tool: bool = True,
) -> Any:
    """Return a copy of *llm* whose decoder is constrained to the envelope.

    Falls back to the unmodified client whenever the provider cannot be bound
    — a constraint is an optimisation, never a hard requirement, and a local
    stack that cannot honour it must still run.
    """
    channel = channel or supports_constrained_decoding(llm)
    if not channel:
        return llm

    try:
        if channel == "ollama":
            schema = tool_envelope_schema(tools, per_tool=per_tool)
            return llm.bind(format=schema)
        if channel == "grammar":
            grammar = gbnf_for_tools(tools, per_tool=per_tool)
            return llm.bind(extra_body={"grammar": grammar})
        if channel == "json":
            return llm.bind(response_format={"type": "json_object"})
    except Exception as exc:  # pragma: no cover - provider-specific
        logger.debug("constrained decoding unavailable (%s): %s", channel, exc)
    return llm


# --------------------------------------------------------------------------
# Parsing the envelope back out
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def parse_envelope(
    text: str,
    known_tools: set[str] | list[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Decode an envelope turn into ``(tool_calls, final_answer)``.

    Tolerates a stray code fence even under constraint, since a provider that
    silently ignored the ``format`` field still has to be handled.  Falls back
    to :func:`isaac.agents.tool_repair.salvage_tool_calls` when the text is not
    a well-formed envelope, so this is safe to use as the sole parser.
    """
    from isaac.agents.tool_repair import repair_json, salvage_tool_calls

    raw = (text or "").strip()
    if not raw:
        return [], ""

    fence = _FENCE.search(raw)
    if fence:
        raw = fence.group(1).strip()

    obj = repair_json(raw)
    if isinstance(obj, dict):
        tool = obj.get("tool") or obj.get("name")
        if isinstance(tool, str):
            if tool.strip().lower() == NO_TOOL:
                answer = obj.get("final_answer") or obj.get("answer") or ""
                return [], str(answer)
            known = {str(t) for t in known_tools} if known_tools else None
            if known is None or tool.strip() in known:
                args = obj.get("arguments") or obj.get("args") or {}
                if isinstance(args, str):
                    parsed = repair_json(args)
                    args = parsed if isinstance(parsed, dict) else {}
                if not isinstance(args, dict):
                    args = {}
                import uuid

                return [
                    {
                        "name": tool.strip(),
                        "args": args,
                        "id": f"envelope_{uuid.uuid4().hex[:8]}",
                    }
                ], ""

    salvaged = salvage_tool_calls(text, known_tools)
    if salvaged:
        return salvaged, ""
    return [], text or ""
