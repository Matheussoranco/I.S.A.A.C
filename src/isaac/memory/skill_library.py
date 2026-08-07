"""Long-term Skill Library — persistent store of *verified* reusable Programs.

Each skill is a parameterised Python function that was successfully
generalised from a concrete task solution.  Skills are stored as:

* ``skills/{name}.py``  — executable Python source.
* ``skills/_index.json`` — manifest with metadata, verification evidence, and
  the rejection log.

Retrieval is performed via cosine similarity on embeddings (ChromaDB)
with a keyword fallback when ChromaDB is unavailable.

Promotion gate (1.5.0)
----------------------
:meth:`SkillLibrary.commit` no longer promotes on faith.  Unless verification
is switched off (``ISAAC_SKILL_VERIFICATION_ENABLED=false``) or the caller
passes ``verify=False``, every candidate is re-executed by
:class:`~isaac.memory.skill_verification.SkillVerifier` first; a candidate that
does not parse, does not define a callable, or blows up on import is
**rejected** — its source is never written to the library — and the rejection
is recorded under ``rejected`` in the manifest so promote/reject counts are
reportable via :meth:`SkillLibrary.promotion_stats`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaac.core.state import SkillCandidate

logger = logging.getLogger(__name__)

#: How many rejection records to keep in the manifest.
MAX_REJECTION_LOG = 200


@dataclass
class PromotionOutcome:
    """Whether a candidate made it into the library, and why (not)."""

    skill_name: str
    promoted: bool
    reason: str = ""
    evidence: str = "none"
    verification: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.promoted

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "promoted": self.promoted,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def _get_chroma_client() -> Any | None:
    """Lazy-load a persistent ChromaDB client, returning None on failure."""
    try:
        import chromadb

        return chromadb
    except ImportError:
        logger.warning("ChromaDB not installed — falling back to keyword skill search.")
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
        self._rejected: list[dict[str, Any]] = self._load_rejections()
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
            logger.warning(
                "ChromaDB initialisation failed — using keyword fallback.", exc_info=True
            )
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

    def _raw_manifest(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {}
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except ValueError:  # pragma: no cover - corrupt manifest
            logger.warning("Skill manifest %s is not valid JSON.", self._index_path)
            return {}
        return raw if isinstance(raw, dict) else {}

    def _load_index(self) -> dict[str, dict[str, Any]]:
        return self._raw_manifest().get("skills", {})

    def _load_rejections(self) -> list[dict[str, Any]]:
        rejected = self._raw_manifest().get("rejected", [])
        return list(rejected) if isinstance(rejected, list) else []

    def _save_index(self) -> None:
        payload = {
            "version": "0.2.0",
            "skills": self._index,
            "rejected": self._rejected[-MAX_REJECTION_LOG:],
        }
        self._index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- write --------------------------------------------------------------

    def commit(
        self,
        candidate: SkillCandidate,
        *,
        verify: bool | None = None,
        verifier: Any | None = None,
    ) -> PromotionOutcome:
        """Promote a *SkillCandidate* into the persistent library, if it works.

        The candidate is re-executed first (see
        :mod:`isaac.memory.skill_verification`).  Only on success is the Python
        source written to ``skills/{name}.py``, the manifest updated, and the
        skill upserted into ChromaDB.  A rejected candidate leaves no ``.py``
        behind — just a row in the manifest's ``rejected`` log.

        Args:
            candidate: The skill to promote.
            verify: Force the gate on/off. ``None`` (default) reads
                ``ISAAC_SKILL_VERIFICATION_ENABLED``.
            verifier: Optional :class:`~isaac.memory.skill_verification.SkillVerifier`
                override (tests inject a stub).

        Returns:
            A :class:`PromotionOutcome`; truthy when the skill was promoted.
        """
        name = candidate.name.strip().replace(" ", "_").lower()
        if not name:
            logger.warning("Skill candidate has no name — skipping commit.")
            return PromotionOutcome("", False, "candidate has no name")

        outcome = self._verify(candidate, name, verify=verify, verifier=verifier)
        if not outcome.promoted:
            self._record_rejection(outcome, candidate)
            logger.info("Skill '%s' REJECTED by the promotion gate: %s", name, outcome.reason)
            return outcome

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
            "verified": outcome.evidence != "none",
            "verification_evidence": outcome.evidence,
            "verification_reason": outcome.reason,
            "promoted_at": time.time(),
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

        logger.info(
            "Skill '%s' PROMOTED to library at %s (evidence=%s)",
            name,
            py_path,
            outcome.evidence,
        )
        return outcome

    # -- verification gate --------------------------------------------------

    def _verify(
        self,
        candidate: SkillCandidate,
        name: str,
        *,
        verify: bool | None,
        verifier: Any | None,
    ) -> PromotionOutcome:
        """Run the promotion gate and translate it into a :class:`PromotionOutcome`."""
        from isaac.memory.skill_verification import verification_enabled

        enabled = verification_enabled() if verify is None else bool(verify)
        if not enabled:
            return PromotionOutcome(
                name, True, "verification disabled", evidence="unverified"
            )

        try:
            if verifier is None:
                from isaac.memory.skill_verification import get_verifier

                verifier = get_verifier()
            result = verifier.verify(candidate)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Skill verification crashed for '%s'", name)
            return PromotionOutcome(name, False, f"verifier crashed: {exc}")

        return PromotionOutcome(
            skill_name=name,
            promoted=bool(getattr(result, "verified", False)),
            reason=str(getattr(result, "reason", "")),
            evidence=str(getattr(result, "evidence", "none")),
            verification=result.to_dict() if hasattr(result, "to_dict") else None,
        )

    def _record_rejection(self, outcome: PromotionOutcome, candidate: SkillCandidate) -> None:
        """Append a rejected candidate to the manifest's rejection log."""
        self._rejected.append(
            {
                "name": outcome.skill_name,
                "reason": outcome.reason,
                "task_context": candidate.task_context[:300],
                "skill_type": getattr(candidate, "skill_type", "code"),
                "rejected_at": time.time(),
                "verification": outcome.verification,
            }
        )
        self._save_index()

    def promotion_stats(self) -> dict[str, Any]:
        """Return promote/reject counts and the top rejection reasons.

        This is what makes the gate *reportable* rather than merely present.
        """
        reasons: dict[str, int] = {}
        for r in self._rejected:
            key = str(r.get("reason", "unknown")).split(";")[0].split(":")[0].strip() or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
        evidence: dict[str, int] = {}
        for meta in self._index.values():
            key = str(meta.get("verification_evidence", "unverified"))
            evidence[key] = evidence.get(key, 0) + 1
        promoted = len(self._index)
        rejected = len(self._rejected)
        total = promoted + rejected
        return {
            "promoted": promoted,
            "rejected": rejected,
            "considered": total,
            "promotion_rate": round(promoted / total, 4) if total else 0.0,
            "evidence_breakdown": evidence,
            "top_rejection_reasons": sorted(reasons.items(), key=lambda kv: -kv[1])[:5],
        }

    @property
    def rejections(self) -> list[dict[str, Any]]:
        """The recorded rejection log (newest last)."""
        return list(self._rejected)

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
        self,
        collection: Any,
        query: str,
        top_k: int,
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
