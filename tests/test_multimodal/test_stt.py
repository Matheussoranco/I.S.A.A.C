""""Tests for Speech-to-Text (STT) fallback logic."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from isaac.multimodal.voice.stt import STTManager, LocalWhisperProvider, GroqSTTProvider

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
