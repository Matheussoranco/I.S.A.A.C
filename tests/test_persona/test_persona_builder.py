"""Tests for the user-built persona system (filesystem isolated via tmp_path)."""

from __future__ import annotations

import json

import pytest

from isaac.config.settings import get_settings
from isaac.identity import persona_builder as pb


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point isaac_home at a temp dir and clear soul_path for every test."""
    settings = get_settings()
    monkeypatch.setattr(settings, "isaac_home", tmp_path, raising=False)
    monkeypatch.setattr(settings, "soul_path", "", raising=False)
    return tmp_path


def test_build_persona_from_answers_composes_personality() -> None:
    persona = pb.build_persona_from_answers(
        {
            "name": "Nova",
            "role": "research assistant",
            "tone": "warm and concise",
            "values": "rigor, honesty",
            "expertise": ["ML", "statistics"],
            "quirks": "I cite sources.",
        }
    )
    assert persona["name"] == "Nova"
    assert "Nova" in persona["personality"]
    assert "research assistant" in persona["personality"]
    assert persona["values"] == ["rigor", "honesty"]
    assert "ML" in persona["personality"] and "statistics" in persona["personality"]


def test_build_persona_requires_name() -> None:
    with pytest.raises(ValueError):
        pb.build_persona_from_answers({})


def test_save_and_load_round_trip() -> None:
    persona = pb.build_persona_from_answers({"name": "Nova", "role": "assistant"})
    path = pb.save_persona(persona)
    assert path.name == "nova.json"
    assert "nova" in pb.list_personas()
    assert pb.load_persona("nova")["name"] == "Nova"


def test_activate_persona_sets_active_soul() -> None:
    pb.save_persona(pb.build_persona_from_answers({"name": "Nova", "role": "assistant"}))
    pb.activate_persona("nova")

    # The active soul file on disk equals the persona.
    on_disk = json.loads(pb.active_soul_path().read_text(encoding="utf-8"))
    assert on_disk["name"] == "Nova"
    assert pb.active_persona() == "nova"

    # And the live identity now reflects the persona.
    from isaac.identity.soul import get_soul

    assert get_soul()["name"] == "Nova"


def test_delete_persona() -> None:
    pb.save_persona(pb.build_persona_from_answers({"name": "Nova", "role": "x"}))
    assert pb.delete_persona("nova") is True
    assert pb.delete_persona("nova") is False


def test_slugify() -> None:
    assert pb.slugify("My Cool Persona!!") == "my-cool-persona"
    assert pb.slugify("   ") == "persona"


def test_install_examples() -> None:
    saved = pb.install_examples()
    assert "atlas" in saved and "sage" in saved
    listed = pb.list_personas()
    assert "atlas" in listed and "sage" in listed
    # Idempotent — running again saves nothing new.
    assert pb.install_examples() == []
