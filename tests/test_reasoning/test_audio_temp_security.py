import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isaac.multimodal import audio


def test_failed_owned_output_removed_and_explicit_output_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(RuntimeError), audio._audio_destination(None, ".wav") as owned:
        assert owned.exists()
        owned.write_bytes(b"partial")
        raise RuntimeError("failed synthesis")
    assert not owned.exists()
    explicit = tmp_path / "user.wav"
    explicit.write_bytes(b"original")
    with pytest.raises(RuntimeError), audio._audio_destination(explicit, ".wav") as destination:
        assert destination == explicit
        raise RuntimeError("failed synthesis")
    assert explicit.read_bytes() == b"original"


class _FakeOpenAIClient:
    def __init__(self, response):
        self.audio = SimpleNamespace(
            speech=SimpleNamespace(create=lambda **kwargs: response),
            transcriptions=SimpleNamespace(create=lambda **kwargs: "transcript"),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _stub_openai(monkeypatch, response) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **kwargs: _FakeOpenAIClient(response)),
    )


def test_openai_writer_receives_existing_file_and_success_survives(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.tempfile, "tempdir", str(tmp_path))

    def write(path):
        assert Path(path).is_file()
        Path(path).write_bytes(b"audio")

    _stub_openai(monkeypatch, SimpleNamespace(write_to_file=write))
    result = audio.speak("hello", engine="openai")
    assert result.read_bytes() == b"audio"


def test_auto_fallback_does_not_leave_failed_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.tempfile, "tempdir", str(tmp_path))

    def unavailable(*args):
        raise ImportError("unavailable")

    def fail(*args, **kwargs):
        raise RuntimeError("failed engine")

    monkeypatch.setattr(audio, "_speak_pyttsx3", unavailable)
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=lambda **kwargs: fail))
    response = SimpleNamespace(write_to_file=lambda path: Path(path).write_bytes(b"fallback"))
    _stub_openai(monkeypatch, response)
    result = audio.speak("hello")
    assert result.read_bytes() == b"fallback"
    assert list(tmp_path.iterdir()) == [result]


def test_openai_engine_requires_credential(tmp_path, monkeypatch):
    from isaac.config.settings import get_settings

    monkeypatch.setattr(audio.tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(get_settings(), "openai_api_key", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        audio.speak("hello", engine="openai")


def test_transcribe_requires_an_available_backend(tmp_path, monkeypatch):
    from isaac.config.settings import get_settings
    from isaac.multimodal import audio as audio_module

    def _unavailable(*args):
        raise ImportError("faster-whisper is not installed")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(get_settings(), "openai_api_key", "")
    monkeypatch.setattr(audio_module, "_transcribe_faster_whisper", _unavailable)
    target = tmp_path / "voice.wav"
    target.write_bytes(b"RIFF")

    with pytest.raises(RuntimeError, match="No speech-to-text backend available"):
        audio.transcribe(target)


def test_microphone_mock_writes_a_readable_wav(tmp_path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")

    monkeypatch.setattr(audio.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            rec=lambda count, **kwargs: np.zeros((count, 1), dtype=np.float32),
            wait=lambda: None,
        ),
    )
    result = audio.record_microphone(duration_seconds=0.05, sample_rate=16000)
    samples, rate = soundfile.read(result)
    assert rate == 16000 and len(samples) == 800
