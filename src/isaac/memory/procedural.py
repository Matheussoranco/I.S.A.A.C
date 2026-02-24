"""Procedural Memory — enhanced Skill Library with versioning and tracking.

Extends the base SkillLibrary with:
* Version tracking for skills
* Success rate tracking
* Deprecation support
* Semantic search by embedding via ChromaDB
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaac.core.state import SkillCandidate
from isaac.memory.skill_library import SkillLibrary

logger = logging.getLogger(__name__)


@dataclass
class SkillVersion:
    """A versioned snapshot of a skill."""

    version: int
    code: str
    timestamp: str
    success_count: int = 0
    failure_count: int = 0
    deprecated: bool = False

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0–1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class SkillRecord:
    """Extended skill metadata with versioning."""

    name: str
    current_version: int = 1
    versions: list[SkillVersion] = field(default_factory=list)
    total_invocations: int = 0
    tags: list[str] = field(default_factory=list)
    deprecated: bool = False
    created_at: str = ""
    updated_at: str = ""


class ProceduralMemory:
    """Enhanced Skill Library with versioning, tracking, and semantic search.

    Wraps the base ``SkillLibrary`` and adds:
    * Version history for each skill
    * Success/failure rate tracking
    * Deprecation workflow
    * Semantic search by embedding

    Parameters
    ----------
    skills_dir:
        Root path where skill files live.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        from isaac.config.settings import settings

        self._skills_dir = skills_dir or settings.skills_dir
        self._base_lib = SkillLibrary(self._skills_dir)
        self._versions_dir = self._skills_dir / ".versions"
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, SkillRecord] = self._load_records()

    # -- Persistence --------------------------------------------------------

    def _records_path(self) -> Path:
        """Path to the version tracking JSON file."""
        return self._versions_dir / "_records.json"

    def _load_records(self) -> dict[str, SkillRecord]:
        """Load version records from disk."""
        path = self._records_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            records: dict[str, SkillRecord] = {}
            for name, data in raw.items():
                versions = [
                    SkillVersion(
                        version=v["version"],
                        code=v.get("code", ""),
                        timestamp=v.get("timestamp", ""),
                        success_count=v.get("success_count", 0),
                        failure_count=v.get("failure_count", 0),
                        deprecated=v.get("deprecated", False),
                    )
                    for v in data.get("versions", [])
                ]
                records[name] = SkillRecord(
                    name=name,
                    current_version=data.get("current_version", 1),
                    versions=versions,
                    total_invocations=data.get("total_invocations", 0),
                    tags=data.get("tags", []),
                    deprecated=data.get("deprecated", False),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                )
            return records
        except (json.JSONDecodeError, KeyError):
            logger.warning("ProceduralMemory: failed to load records — starting fresh.")
            return {}

    def _save_records(self) -> None:
        """Persist version records to disk."""
        data: dict[str, Any] = {}
        for name, record in self._records.items():
            data[name] = {
                "current_version": record.current_version,
                "versions": [
                    {
                        "version": v.version,
                        "code": v.code[:500],  # Truncate for storage
                        "timestamp": v.timestamp,
                        "success_count": v.success_count,
                        "failure_count": v.failure_count,
                        "deprecated": v.deprecated,
                    }
                    for v in record.versions
                ],
                "total_invocations": record.total_invocations,
                "tags": record.tags,
                "deprecated": record.deprecated,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
        self._records_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- Write --------------------------------------------------------------

    def commit(self, candidate: SkillCandidate) -> None:
        """Commit a skill candidate with version tracking.

        If the skill already exists, creates a new version.
        """
        name = candidate.name.strip().replace(" ", "_").lower()
        now = datetime.now(tz=timezone.utc).isoformat()

        # Commit to base library
        self._base_lib.commit(candidate)

        # Track version
        if name in self._records:
            record = self._records[name]
            record.current_version += 1
            record.versions.append(SkillVersion(
                version=record.current_version,
                code=candidate.code,
                timestamp=now,
                success_count=candidate.success_count,
            ))
            record.updated_at = now
            record.tags = list(set(record.tags + list(getattr(candidate, "tags", []))))
        else:
            self._records[name] = SkillRecord(
                name=name,
                current_version=1,
                versions=[SkillVersion(
                    version=1,
                    code=candidate.code,
                    timestamp=now,
                    success_count=candidate.success_count,
                )],
                tags=list(getattr(candidate, "tags", [])),
                created_at=now,
                updated_at=now,
            )

        self._save_records()
        logger.info(
            "ProceduralMemory: committed skill '%s' v%d.",
            name,
            self._records[name].current_version,
        )

    def record_invocation(self, name: str, success: bool) -> None:
        """Record a skill invocation and update success rate.

        Parameters
        ----------
        name:
            The skill name.
        success:
            Whether the invocation succeeded.
        """
        if name not in self._records:
            return

        record = self._records[name]
        record.total_invocations += 1

        if record.versions:
            latest = record.versions[-1]
            if success:
                latest.success_count += 1
            else:
                latest.failure_count += 1

        self._save_records()

    def deprecate(self, name: str) -> None:
        """Mark a skill as deprecated."""
        if name in self._records:
            self._records[name].deprecated = True
            if self._records[name].versions:
                self._records[name].versions[-1].deprecated = True
            self._save_records()
            logger.info("ProceduralMemory: deprecated skill '%s'.", name)

    # -- Read ---------------------------------------------------------------

    def get_record(self, name: str) -> SkillRecord | None:
        """Get the full version record for a skill."""
        return self._records.get(name)

    def get_success_rate(self, name: str) -> float:
        """Get the current success rate for a skill."""
        record = self._records.get(name)
        if record is None or not record.versions:
            return 0.0
        return record.versions[-1].success_rate

    def list_active(self) -> list[str]:
        """List all non-deprecated skill names."""
        return [
            name for name, record in self._records.items()
            if not record.deprecated
        ]

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Semantic search via the base library's ChromaDB integration."""
        return self._base_lib.search(query, top_k)

    @property
    def base_library(self) -> SkillLibrary:
        """Access the underlying SkillLibrary."""
        return self._base_lib
