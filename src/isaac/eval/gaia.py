"""GAIA benchmark adapter — load GAIA tasks as :class:`EvalTask` suites.

GAIA (Mialon et al., 2023 — https://huggingface.co/datasets/gaia-benchmark/GAIA)
is a public benchmark of real-world assistant questions requiring reasoning,
web browsing, and file handling. Scoring follows the **official leaderboard
quasi-exact-match rules** so a number produced here is directly comparable to
published systems (the ROADMAP-1.0 §4 "SOTA" gate requires exactly that).

Usage::

    isaac eval <dir-with-metadata.jsonl> --format gaia --level 1

The dataset is *gated*: accept the terms at the dataset page, authenticate
with ``hf auth login`` (or ``HF_TOKEN``), then ``download_gaia()`` /
``isaac eval --format gaia --download`` fetches the validation split.
Answers for the test split are hidden; the validation split is what local
scoring uses.
"""

from __future__ import annotations

import json
import logging
import re
import string
from pathlib import Path

from isaac.eval.suite import EvalTask

logger = logging.getLogger(__name__)

GAIA_REPO = "gaia-benchmark/GAIA"

# The official GAIA system-prompt answer contract — verbatim semantics so the
# official scorer applies.
GAIA_INSTRUCTION = (
    "Finish your answer with the following template: FINAL ANSWER: [YOUR FINAL "
    "ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible "
    "OR a comma separated list of numbers and/or strings. If you are asked for "
    "a number, don't use comma to write your number neither use units such as $ "
    "or percent sign unless specified otherwise. If you are asked for a string, "
    "don't use articles, neither abbreviations (e.g. for cities), and write the "
    "digits in plain text unless specified otherwise. If you are asked for a "
    "comma separated list, apply the above rules depending of whether the "
    "element to be put in the list is a number or a string."
)


# ---------------------------------------------------------------------------
# Official scoring (mirrors the GAIA leaderboard scorer)
# ---------------------------------------------------------------------------


def _normalize_number_str(s: str) -> float | None:
    cleaned = s.replace("$", "").replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_float(s: str) -> bool:
    # No comma stripping here — the official scorer's is_float() routes
    # comma-formatted ground truths ("3,000") to the *list* branch, and
    # stripping would silently accept answers the leaderboard rejects.
    try:
        float(s)
        return True
    except ValueError:
        return False


def _split_list(s: str) -> list[str]:
    return [e.strip() for e in re.split(r"[,;]", s)]


def _normalize_str(s: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s+", "", s)
    if remove_punct:
        no_spaces = no_spaces.translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def extract_final_answer(text: str) -> str:
    """Pull the answer after the *last* ``FINAL ANSWER:`` marker (the GAIA
    answer contract); fall back to the whole text when the marker is absent."""
    matches = re.findall(r"FINAL ANSWER\s*:\s*(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip().strip("[]").strip()
    return text.strip()


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    """Official GAIA quasi-exact match.

    - numeric ground truth -> compare as floats (units/commas stripped from
      the model answer; comma-formatted ground truths take the list branch,
      as in the official scorer)
    - list ground truth (contains ``,``/``;``) -> element-wise with the same
      rules (numbers as numbers, strings normalized keeping punctuation)
    - otherwise -> normalized string equality
    """
    if _is_float(ground_truth):
        normalized = _normalize_number_str(model_answer)
        return normalized is not None and normalized == float(ground_truth)

    if any(ch in ground_truth for ch in (",", ";")):
        gt_elems = _split_list(ground_truth)
        ma_elems = _split_list(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        for ma, gt in zip(ma_elems, gt_elems, strict=True):
            if _is_float(gt):
                normalized = _normalize_number_str(ma)
                if normalized is None or normalized != float(gt):
                    return False
            elif _normalize_str(ma, remove_punct=False) != _normalize_str(gt, remove_punct=False):
                return False
        return True

    return _normalize_str(model_answer) == _normalize_str(ground_truth)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_gaia_tasks(
    metadata_dir: str | Path,
    level: int | None = 1,
    *,
    max_iterations: int = 20,
    timeout_seconds: float = 300.0,
) -> list[EvalTask]:
    """Load GAIA tasks from a split directory containing ``metadata.jsonl``.

    Attachments referenced by ``file_name`` are seeded into the run workspace
    as binary copies; the prompt tells the agent where to find them.
    """
    split_dir = Path(metadata_dir)
    meta = split_dir / "metadata.jsonl"
    if not meta.is_file():
        raise FileNotFoundError(
            f"{meta} not found. Download the GAIA validation split first "
            "(isaac eval --format gaia --download, after accepting the dataset "
            f"terms at https://huggingface.co/datasets/{GAIA_REPO})."
        )

    tasks: list[EvalTask] = []
    for lineno, raw in enumerate(meta.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{meta.name}:{lineno}: invalid JSON — {exc}") from exc

        task_level = int(obj.get("Level", 0))
        if level is not None and task_level != level:
            continue
        answer = str(obj.get("Final answer", "")).strip()
        if not answer or answer == "?":  # test split hides answers — not scoreable
            continue

        prompt = str(obj["Question"]).strip()
        file_paths: dict[str, str] = {}
        file_name = str(obj.get("file_name", "") or "")
        if file_name:
            src = split_dir / file_name
            if src.is_file():
                dest = f"gaia/{file_name}"
                file_paths[dest] = str(src)
                prompt += (
                    f"\n\nThe file mentioned in the question is at '{dest}' in your workspace."
                )
            else:
                logger.warning(
                    "GAIA %s: attachment %s missing; skipping task", obj["task_id"], file_name
                )
                continue

        tasks.append(
            EvalTask(
                id=str(obj["task_id"]),
                prompt=f"{prompt}\n\n{GAIA_INSTRUCTION}",
                checks=[{"type": "gaia", "value": answer}],
                category=f"gaia-l{task_level}",
                file_paths=file_paths,
                max_iterations=max_iterations,
                timeout_seconds=timeout_seconds,
            )
        )
    if not tasks:
        raise ValueError(f"No scoreable GAIA tasks found in {split_dir} (level={level}).")
    return tasks


def download_gaia(dest: str | Path | None = None, token: str | None = None) -> Path:
    """Download the GAIA 2023 validation split via huggingface_hub.

    Requires accepting the dataset terms on the Hugging Face page and an
    authenticated token (``hf auth login`` or ``HF_TOKEN``).
    Returns the directory containing ``metadata.jsonl``.
    """
    from huggingface_hub import snapshot_download  # optional dep, guarded import

    if dest is None:
        from isaac.config.settings import get_settings

        dest = get_settings().isaac_home / "datasets" / "gaia"
    root = snapshot_download(
        GAIA_REPO,
        repo_type="dataset",
        allow_patterns=["2023/validation/*"],
        local_dir=str(dest),
        token=token,
    )
    return Path(root) / "2023" / "validation"
