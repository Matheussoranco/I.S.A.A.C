"""Fast local task classifier — zero LLM calls.

Classifies user input using pure regex + heuristics.  Called BEFORE the
Perception LLM so that obvious conversational messages never touch the LLM
at all.  Only ambiguous inputs fall through to the full Perception node.

Confidence levels
-----------------
* ≥ 0.90 → route immediately, skip Perception LLM entirely
* 0.60–0.89 → route but still populate observations from classifier
* < 0.60 → fall back to Perception LLM for full analysis

Typical latency: < 1 ms.
"""

from __future__ import annotations

import re
from typing import Literal

TaskMode = Literal["direct", "code", "computer_use", "hybrid"]

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for zero init cost after first import)
# ---------------------------------------------------------------------------

# --- DIRECT patterns (conversational, no execution needed) -----------------

_GREETINGS = re.compile(
    r"""^\s*(
        # Portuguese
        ol[aá]|oi|e aí|e ai|bom\s+dia|boa\s+tarde|boa\s+noite|
        tudo\s+(bem|bom|certo)|como\s+(vai|você\s+está|vc\s+tá)|
        # English
        h(ello|i|ey)|good\s+(morning|afternoon|evening|day)|
        what('s|\s+is)\s+up|howdy|greetings|sup\b|yo\b|
        # French / Spanish / German
        bonjour|salut|hola|buenos\s+(dias|tardes)|guten\s+(tag|morgen)|
        # Generic
        hey\s+there|hi\s+there|hello\s+there
    )\s*[!?.]*\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_IDENTITY_QUERY = re.compile(
    r"""(
        who\s+are\s+you |
        what\s+are\s+you |
        what(?:'s|\s+is)\s+your\s+name |
        introduce\s+yourself |
        tell\s+me\s+about\s+yourself |
        what\s+can\s+you\s+do |
        o\s+que\s+você\s+é |
        quem\s+é\s+você |
        qual\s+é\s+o\s+seu\s+nome |
        como\s+você\s+funciona
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_SIMPLE_QUESTION = re.compile(
    r"""^\s*(
        what\s+is\s+|
        what\s+are\s+|
        who\s+is\s+|
        when\s+(is|was|did)\s+|
        where\s+is\s+|
        how\s+(does|do|is|are)\s+|
        why\s+(is|are|does)\s+|
        explain\s+|
        tell\s+me\s+(about|what|how|why)\s+|
        define\s+|
        what\s+does\s+.*\s+mean |
        # Portuguese
        o\s+que\s+[eé]\s+|
        como\s+funciona\s+|
        me\s+explica?\s+|
        o\s+que\s+significa?\s+|
        qual\s+[eé]\s+a?\s+
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_CHITCHAT = re.compile(
    r"""^\s*(
        thanks?|thank\s+you|obrigad[oa]|valeu|
        ok|okay|alright|got\s+it|understood|
        cool|nice|great|awesome|interesting|
        yes|no|nope|yep|yeah|sim|não|nao|
        bye|goodbye|see\s+you|até\s+(logo|mais)|tchau|
        please\s+help|help\s+me|help\b|ajuda?\b|
        can\s+you\s+help|can\s+you\s+assist|
        i\s+need\s+(help|assistance)
    )\s*[!?.]*\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# --- CODE patterns (execution required) ------------------------------------

_CODE_KEYWORDS = re.compile(
    r"""\b(
        def\s+\w|class\s+\w|import\s+\w|from\s+\w+\s+import|
        write\s+(a\s+)?(script|program|code|function|class|module)|
        create\s+(a\s+)?(script|program|file|function|module|api|app|server|bot|cli|tool|endpoint)|
        build\s+(a\s+|me\s+a\s+)?(tool|cli|api|rest|app|server|bot|endpoint|website|pipeline)|
        implement\s+|make\s+(a\s+)?(api|rest|server|cli|script|bot|tool|app)|
        fix\s+(the\s+)?(bug|error|code|script)|
        refactor\s+|optimize\s+(the\s+)?(code|function|script)|
        run\s+(this|the|a)\s+(script|code|file)|
        execute\s+(this|the)\s+(code|script)|
        calculate\s+|compute\s+|solve\s+(this|the)\s+(problem|equation)|
        parse\s+(the\s+)?(file|data|json|csv|xml)|
        sort\s+|filter\s+|transform\s+|convert\s+(the\s+)?(data|file)|
        generate\s+(a\s+)?(report|csv|json|chart|plot|graph)|
        scrape\s+|fetch\s+(data|the\s+url|from)|
        automate\s+|deploy\s+|install\s+(the\s+)?package|
        rest\s+api|graphql|rest(ful)?\s+(api|endpoint)|api\s+(endpoint|server)|
        with\s+(fastapi|flask|django|express|spring|rails|sqlalchemy|pandas|numpy)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

_CODE_SNIPPET = re.compile(
    r"(```|~~~|def |class |import |#!\/|<script|<html|SELECT\s+\w|INSERT\s+INTO)",
    re.IGNORECASE,
)

_FILE_PATH = re.compile(r"([a-zA-Z]:\\|/[a-z]+/|\.py\b|\.js\b|\.ts\b|\.sh\b|\.json\b)")

# --- GUI / COMPUTER-USE patterns -------------------------------------------

_GUI_KEYWORDS = re.compile(
    r"""\b(
        click\s+(on|the)|open\s+(the\s+)?(browser|app|window|tab)|
        screenshot|take\s+a\s+picture\s+of\s+(the\s+)?screen|
        type\s+(in|into)\s+the|navigate\s+to\s+|go\s+to\s+(the\s+)?website|
        scroll\s+(down|up)|drag\s+(and\s+drop|the)|
        right.click|double.click|press\s+(enter|tab|esc|ctrl|alt)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fast_classify(text: str) -> tuple[TaskMode | None, float]:
    """Classify user text without any LLM call.

    Returns
    -------
    (task_mode, confidence)
        ``task_mode`` is ``None`` when the classifier is uncertain.
        ``confidence`` is 0.0–1.0.

    Examples
    --------
    >>> fast_classify("Olá!")
    ('direct', 1.0)
    >>> fast_classify("write a python script to sort a list")
    ('code', 0.95)
    >>> fast_classify("what is the capital of France?")
    ('direct', 0.9)
    >>> fast_classify("build me a REST API")
    ('code', 0.95)
    """
    stripped = text.strip()
    if not stripped:
        return None, 0.0

    # Screenshot / image content → computer_use
    if stripped.startswith("data:image") or "[screenshot]" in stripped.lower():
        return "computer_use", 1.0

    # Strong code signals → code (check before direct to avoid false positives)
    if _CODE_SNIPPET.search(stripped):
        return "code", 0.98
    if _CODE_KEYWORDS.search(stripped):
        return "code", 0.95

    # GUI signals
    if _GUI_KEYWORDS.search(stripped):
        return "computer_use", 0.95

    # File paths in request → likely code
    if _FILE_PATH.search(stripped):
        return "code", 0.85

    # Strong direct signals
    if _GREETINGS.match(stripped):
        return "direct", 1.0
    if _CHITCHAT.match(stripped):
        return "direct", 0.98
    if _IDENTITY_QUERY.search(stripped):
        return "direct", 0.97

    # Simple knowledge questions (short, no code indicators)
    if _SIMPLE_QUESTION.match(stripped) and len(stripped) < 120:
        return "direct", 0.90

    # Very short input with no code markers → direct (chitchat/question)
    _BUILD_WORDS = re.compile(
        r"\b(build|create|make|write|implement|develop|code|program|script|api|server|bot)\b",
        re.IGNORECASE,
    )
    if len(stripped) <= 30 and not any(c in stripped for c in "{}[]()<>") and not _BUILD_WORDS.search(stripped):
        return "direct", 0.82

    # Ambiguous — let the LLM decide
    return None, 0.0


def classify_hypothesis(text: str, task_mode: TaskMode) -> str:
    """Build a lightweight hypothesis string without LLM for direct tasks."""
    stripped = text.strip()

    if _GREETINGS.match(stripped) or _CHITCHAT.match(stripped):
        return "User is initiating a casual conversation. Respond with a warm, brief reply."

    if _IDENTITY_QUERY.search(stripped):
        return "User wants to know who I.S.A.A.C. is. Introduce myself and my capabilities."

    if _SIMPLE_QUESTION.match(stripped):
        return f"User asks a knowledge question: '{stripped[:80]}'. Answer directly and concisely."

    return f"User sent a short message: '{stripped[:80]}'. Respond naturally."
