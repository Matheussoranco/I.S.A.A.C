"""Long-term Skill Library — persistent store of reusable Programs.

Each skill is a parameterised Python function that was successfully
generalised from a concrete task solution.  Skills are stored as:

* ``skills/{name}.py``  — executable Python source.
* ``skills/_index.json`` — manifest with metadata & embeddings.

Retrieval is performed via cosine similarity on embeddings (ChromaDB).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from isaac.core.state import SkillCandidate

logger = logging.getLogger(__name__)


class SkillLibrary:
    """CRUD + semantic-search interface over the skill directory.

    Parameters
    ----------
    skills_dir:
        Root path where ``.py`` skill files and ``_index.json`` live.
    """

    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "_index.json"
        self._index: dict[str, dict[str, Any]] = self._load_index()

    # -- persistence --------------------------------------------------------

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self._index_path.exists():
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            return raw.get("skills", {})
        return {}

    def _save_index(self) -> None:
        payload = {"version": "0.1.0", "skills": self._index}
        self._index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- write --------------------------------------------------------------

    def commit(self, candidate: SkillCandidate) -> None:
        """Promote a *SkillCandidate* into the persistent library.

        Writes the Python source to ``skills/{name}.py`` and updates the
        manifest index.
        """
        name = candidate.name.strip().replace(" ", "_").lower()
        if not name:
            logger.warning("Skill candidate has no name — skipping commit.")
            return

        py_path = self._dir / f"{name}.py"
        py_path.write_text(candidate.code, encoding="utf-8")

        self._index[name] = {
            "name": name,
            "input_schema": candidate.input_schema,
            "output_schema": candidate.output_schema,
            "task_context": candidate.task_context,
            "success_count": candidate.success_count,
            "file": str(py_path.name),
        }
        self._save_index()
        logger.info("Skill '%s' committed to library at %s", name, py_path)

    # -- read ---------------------------------------------------------------

    def list_names(self) -> list[str]:
        """Return all registered skill names."""
        return list(self._index.keys())

    def get_code(self, name: str) -> str | None:
        """Return the Python source of a skill, or ``None``."""
        entry = self._index.get(name)
        if entry is None:
            return None
        py_path = self._dir / entry["file"]
        if py_path.exists():
            return py_path.read_text(encoding="utf-8")
        return None

    def get_metadata(self, name: str) -> dict[str, Any] | None:
        """Return the index entry for a skill."""
        return self._index.get(name)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Naïve keyword search over skill names and task contexts.

        TODO: Replace with ChromaDB embedding similarity once the embedding
        pipeline is wired up.
        """
        q = query.lower()
        scored: list[tuple[int, str]] = []
        for name, meta in self._index.items():
            score = 0
            if q in name:
                score += 2
            if q in meta.get("task_context", "").lower():
                score += 1
            if score:
                scored.append((score, name))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [name for _, name in scored[:top_k]]

    @property
    def size(self) -> int:
        return len(self._index)
