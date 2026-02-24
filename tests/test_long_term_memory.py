"""Tests for the Long-Term Memory module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestLongTermMemory:
    """Tests for ``isaac.memory.long_term.LongTermMemory``."""

    @pytest.fixture(autouse=True)
    def _setup_ltm(self, tmp_path: Path) -> None:
        """Create a fresh LTM for each test."""
        from isaac.memory.long_term import LongTermMemory, reset_long_term_memory

        reset_long_term_memory()
        self.db_path = str(tmp_path / "test_ltm.db")
        self.ltm = LongTermMemory(db_path=self.db_path)

    def test_remember_and_recall(self) -> None:
        self.ltm.remember("Python is a programming language", type="fact", importance=0.8)
        results = self.ltm.recall("Python")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]

    def test_recall_empty_db(self) -> None:
        results = self.ltm.recall("anything")
        assert results == []

    def test_remember_increments_count(self) -> None:
        self.ltm.remember("fact one", type="fact")
        self.ltm.remember("fact two", type="fact")
        results = self.ltm.recall("fact")
        assert len(results) == 2

    def test_forget_removes_entry(self) -> None:
        self.ltm.remember("temporary memory", type="event", importance=0.3)
        results = self.ltm.recall("temporary")
        assert len(results) == 1
        memory_id = results[0]["id"]
        self.ltm.forget(memory_id)
        results = self.ltm.recall("temporary")
        assert len(results) == 0

    def test_to_context_string_empty(self) -> None:
        ctx = self.ltm.to_context_string("anything")
        assert ctx == ""

    def test_to_context_string_with_data(self) -> None:
        self.ltm.remember("User prefers dark mode", type="preference", importance=0.9)
        ctx = self.ltm.to_context_string("dark mode")
        assert "dark mode" in ctx.lower()

    def test_consolidate_removes_low_importance(self) -> None:
        for i in range(5):
            self.ltm.remember(f"low importance fact {i}", type="fact", importance=0.05)
        self.ltm.remember("important fact", type="fact", importance=0.9)
        self.ltm.consolidate()
        results = self.ltm.recall("fact")
        # Important fact should survive
        found_important = any("important fact" in r["content"] for r in results)
        assert found_important

    def test_recall_top_k_limit(self) -> None:
        for i in range(10):
            self.ltm.remember(f"memory number {i}", type="fact", importance=0.5)
        results = self.ltm.recall("memory", top_k=3)
        assert len(results) <= 3


class TestUserProfile:
    """Tests for ``isaac.memory.user_profile.UserProfile``."""

    @pytest.fixture(autouse=True)
    def _setup_profile(self, tmp_path: Path) -> None:
        from isaac.memory.user_profile import UserProfile, reset_user_profile

        reset_user_profile()
        self.path = tmp_path / "profile.json"
        self.profile = UserProfile(path=self.path)

    def test_set_preference(self) -> None:
        self.profile.set_preference("theme", "dark")
        assert self.profile.preferences["theme"] == "dark"

    def test_add_tag(self) -> None:
        self.profile.add_tag("developer")
        assert "developer" in self.profile.tags

    def test_add_tag_no_duplicates(self) -> None:
        self.profile.add_tag("developer")
        self.profile.add_tag("developer")
        assert self.profile.tags.count("developer") == 1

    def test_record_interaction_increments(self) -> None:
        initial = self.profile.interaction_count
        self.profile.record_interaction()
        assert self.profile.interaction_count == initial + 1

    def test_save_and_reload(self, tmp_path: Path) -> None:
        self.profile.set_preference("lang", "pt-BR")
        self.profile.add_tag("coder")
        self.profile.save()

        from isaac.memory.user_profile import UserProfile

        reloaded = UserProfile(path=self.path)
        assert reloaded.preferences["lang"] == "pt-BR"
        assert "coder" in reloaded.tags

    def test_to_context_string(self) -> None:
        self.profile.set_preference("editor", "vscode")
        ctx = self.profile.to_context_string()
        assert "vscode" in ctx.lower() or "editor" in ctx.lower()

    def test_update_after_session(self) -> None:
        self.profile.update_after_session(inferred_tags=["python", "ai"])
        assert "python" in self.profile.tags
        assert "ai" in self.profile.tags
