import pytest

pytest.importorskip("faster_whisper")

from transcription_pipeline.audio import load_audio
from transcription_pipeline.transcribe import WhisperTranscriber


@pytest.mark.slow
def test_whisper_transcriber_runs_on_real_audio(sample_wav):
    transcriber = WhisperTranscriber(model_size="tiny")
    audio = load_audio(sample_wav)

    transcript = transcriber.transcribe(audio)

    assert transcript.language is not None
    assert transcript.duration_s is not None
