import sys
import types

import pytest

pytest.importorskip("faster_whisper")

from transcription_pipeline.audio import load_audio
from transcription_pipeline.errors import TranscriptionError
from transcription_pipeline.transcribe import WhisperTranscriber


def _install_fake_faster_whisper(monkeypatch, model_cls):
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = model_cls
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


def test_whisper_transcriber_wraps_backend_errors(monkeypatch, sample_wav):
    class FailingModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path):
            raise RuntimeError("backend exploded")

    _install_fake_faster_whisper(monkeypatch, FailingModel)

    transcriber = WhisperTranscriber()
    audio = load_audio(sample_wav)

    with pytest.raises(TranscriptionError, match="backend exploded"):
        transcriber.transcribe(audio)


def test_whisper_transcriber_missing_dependency(monkeypatch, sample_wav):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # forces ImportError on import

    transcriber = WhisperTranscriber()
    audio = load_audio(sample_wav)

    with pytest.raises(TranscriptionError, match="not installed"):
        transcriber.transcribe(audio)


def test_whisper_transcriber_loads_model_once(monkeypatch, sample_wav):
    init_calls = []

    class OKModel:
        def __init__(self, *args, **kwargs):
            init_calls.append((args, kwargs))

        def transcribe(self, path):
            info = types.SimpleNamespace(language="en", duration=2.0)
            return iter([]), info

    _install_fake_faster_whisper(monkeypatch, OKModel)

    transcriber = WhisperTranscriber()
    audio = load_audio(sample_wav)

    transcriber.transcribe(audio)
    transcriber.transcribe(audio)

    assert len(init_calls) == 1  # model is lazily loaded once, then reused


@pytest.mark.slow
def test_whisper_transcriber_runs_on_real_audio(sample_wav):
    transcriber = WhisperTranscriber(model_size="tiny")
    audio = load_audio(sample_wav)

    transcript = transcriber.transcribe(audio)

    assert transcript.language is not None
    assert transcript.duration_s is not None
