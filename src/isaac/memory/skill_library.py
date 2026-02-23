"""Long-term Skill Library — persistent store of reusable Programs.

Each skill is a parameterised Python function that was successfully
generalised from a concrete task solution.  Skills are stored as:

* ``skills/{name}.py``  — executable Python source.
* ``skills/_index.json`` — manifest with metadata & embeddings.

Retrieval is performed via cosine similarity on embeddings (ChromaDB)
with a keyword fallback when ChromaDB is unavailable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from isaac.core.state import SkillCandidate

logger = logging.getLogger(__name__)


def _get_chroma_client() -> Any | None:
    """Lazy-load a persistent ChromaDB client, returning None on failure."""
    try:
        import chromadb  # noqa: PLC0415

        return chromadb
    except ImportError:
        logger.warning(
            "ChromaDB not installed — falling back to keyword skill search."
        )
        return None


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
        self._collection: Any | None = None
        self._chroma_client: Any | None = None

    # -- ChromaDB lazy init -------------------------------------------------

    def _ensure_collection(self) -> Any | None:
        """Return the ChromaDB collection, creating it on first access."""
        if self._collection is not None:
            return self._collection

        chromadb = _get_chroma_client()
        if chromadb is None:
            return None

        try:
            chroma_dir = self._dir / ".chromadb"
            chroma_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(chroma_dir),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="skills",
                metadata={"hnsw:space": "cosine"},
            )
            # Sync any index entries that aren't yet in the collection
            self._sync_index_to_collection()
            return self._collection
        except Exception:
            logger.warning("ChromaDB initialisation failed — using keyword fallback.", exc_info=True)
            return None

    def _sync_index_to_collection(self) -> None:
        """Ensure every indexed skill has a ChromaDB document."""
        if self._collection is None:
            return
        existing = set(self._collection.get()["ids"])
        for name, meta in self._index.items():
            if name not in existing:
                doc = self._build_document(name, meta)
                self._collection.add(
                    ids=[name],
                    documents=[doc],
                    metadatas=[{"name": name, "task_context": meta.get("task_context", "")}],
                )

    @staticmethod
    def _build_document(name: str, meta: dict[str, Any]) -> str:
        """Build a searchable document string from skill metadata."""
        parts = [
            f"skill: {name}",
            f"task: {meta.get('task_context', '')}",
        ]
        tags = meta.get("tags", [])
        if tags:
            parts.append(f"tags: {', '.join(tags)}")
        return " | ".join(parts)

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
        manifest index.  Also upserts into ChromaDB for semantic retrieval.
        """
        name = candidate.name.strip().replace(" ", "_").lower()
        if not name:
            logger.warning("Skill candidate has no name — skipping commit.")
            return

        py_path = self._dir / f"{name}.py"
        py_path.write_text(candidate.code, encoding="utf-8")

        meta = {
            "name": name,
            "input_schema": candidate.input_schema,
            "output_schema": candidate.output_schema,
            "task_context": candidate.task_context,
            "success_count": candidate.success_count,
            "skill_type": getattr(candidate, "skill_type", "code"),
            "tags": list(getattr(candidate, "tags", [])),
            "file": str(py_path.name),
        }
        self._index[name] = meta
        self._save_index()

        # Upsert into ChromaDB
        collection = self._ensure_collection()
        if collection is not None:
            doc = self._build_document(name, meta)
            try:
                collection.upsert(
                    ids=[name],
                    documents=[doc],
                    metadatas=[{"name": name, "task_context": candidate.task_context}],
                )
            except Exception:
                logger.warning("ChromaDB upsert failed for skill '%s'.", name, exc_info=True)

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
        """Search for skills relevant to *query*.

        Uses ChromaDB embedding similarity when available, falling back to
        keyword matching otherwise.
        """
        collection = self._ensure_collection()
        if collection is not None and collection.count() > 0:
            return self._search_chromadb(collection, query, top_k)
        return self._search_keyword(query, top_k)

    def _search_chromadb(
        self, collection: Any, query: str, top_k: int,
    ) -> list[str]:
        """Semantic search via ChromaDB embedding similarity."""
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
            )
            ids = results.get("ids", [[]])[0]
            logger.debug("ChromaDB search for '%s' returned %d results.", query, len(ids))
            return ids
        except Exception:
            logger.warning("ChromaDB query failed — falling back to keyword.", exc_info=True)
            return self._search_keyword(query, top_k)

    def _search_keyword(self, query: str, top_k: int) -> list[str]:
        """Naïve keyword search over skill names and task contexts."""
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
