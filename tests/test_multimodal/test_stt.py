""" "Tests for Speech-to-Text (STT) fallback logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from isaac.multimodal.voice.stt import STTManager


def test_stt_provider_fallback():
    """Test that STTManager falls back to cloud providers if local fails."""
    audio_file = Path("test_audio.wav")

    # We mock the LocalWhisperProvider in the STTManager's providers list
    with patch("pathlib.Path.exists", return_value=True):
        stt = STTManager(cloud_priority=["groq"])

        # Force the first provider (local) to fail if it exists,
        # or ensure the fallback logic works.
        # To properly test the chain, we can manually inject a failing provider.
        mock_fail = MagicMock()
        mock_fail.transcribe.side_effect = RuntimeError("Local failure")

        mock_success = MagicMock()
        mock_success.transcribe.return_value = "Hello from Groq"

        stt.providers = [mock_fail, mock_success]

        result = stt.transcribe(audio_file)

        assert result == "Hello from Groq"
        assert mock_fail.transcribe.called
        assert mock_success.transcribe.called


def test_stt_all_fail():
    """Test that STTManager raises RuntimeError when all providers fail."""
    audio_file = Path("test_audio.wav")

    with patch("pathlib.Path.exists", return_value=True):
        stt = STTManager(cloud_priority=["groq"])

        mock_fail1 = MagicMock()
        mock_fail1.transcribe.side_effect = RuntimeError("Fail 1")
        mock_fail2 = MagicMock()
        mock_fail2.transcribe.side_effect = RuntimeError("Fail 2")

        stt.providers = [mock_fail1, mock_fail2]

        with pytest.raises(RuntimeError, match="All STT providers failed"):
            stt.transcribe(audio_file)


def test_stt_file_not_found():
    """Test that STTManager raises FileNotFoundError for missing files."""
    stt = STTManager()
    with pytest.raises(FileNotFoundError):
        stt.transcribe(Path("non_existent_audio.wav"))


def test_get_stt_honors_voice_settings(monkeypatch):
    from isaac.config.settings import get_settings
    from isaac.multimodal.voice import stt as stt_module

    captured: dict[str, str] = {}

    class _FakeWhisperModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            captured.update(model=model, device=device, compute_type=compute_type)

    settings = get_settings()
    monkeypatch.setattr(settings, "voice_stt_model", "small")
    monkeypatch.setattr(settings, "voice_device", "cpu")
    monkeypatch.setattr(settings, "voice_stt_compute_type", "float32")
    monkeypatch.setattr(settings, "voice_stt_language", "pt")
    monkeypatch.setattr(stt_module, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(stt_module, "_stt_manager", None)

    manager = stt_module.get_stt()

    assert captured == {"model": "small", "device": "cpu", "compute_type": "float32"}
    assert manager.providers[0].language == "pt"  # type: ignore[attr-defined]


def test_get_stt_respects_voice_disabled(monkeypatch):
    from isaac.config.settings import get_settings
    from isaac.multimodal.voice import stt as stt_module

    monkeypatch.setattr(get_settings(), "voice_enabled", False)
    monkeypatch.setattr(stt_module, "_stt_manager", None)

    with pytest.raises(RuntimeError, match="disabled"):
        stt_module.get_stt()
    assert stt_module.is_stt_available() is False
