import pytest

from transcription_pipeline.errors import ExtractionError, InvalidAudioError, TranscriptionError
from transcription_pipeline.extract import MockExtractor
from transcription_pipeline.pipeline import run_pipeline
from transcription_pipeline.transcribe import MockTranscriber, Transcript


def test_pipeline_ok(sample_wav):
    transcriber = MockTranscriber()
    extractor = MockExtractor()

    result = run_pipeline(sample_wav, transcriber=transcriber, extractor=extractor)

    assert result.status == "ok"
    assert result.structured is not None
    assert result.transcript.text == transcriber._transcript.text


def test_pipeline_empty_transcript_skips_extraction(sample_wav):
    empty_transcript = Transcript(text="", segments=[], language="en", duration_s=2.0)
    transcriber = MockTranscriber(transcript=empty_transcript)
    extractor = MockExtractor()

    result = run_pipeline(sample_wav, transcriber=transcriber, extractor=extractor)

    assert result.status == "empty"
    assert result.structured is None


def test_pipeline_extraction_failure_preserves_transcript(sample_wav):
    transcriber = MockTranscriber()
    extractor = MockExtractor(raise_error=ExtractionError("simulated failure"))

    result = run_pipeline(sample_wav, transcriber=transcriber, extractor=extractor)

    assert result.status == "partial_failure"
    assert result.structured is None
    assert result.transcript.text  # already-obtained transcript is not lost
    assert "simulated failure" in result.error


def test_pipeline_invalid_audio_raises(tmp_path):
    missing = tmp_path / "does_not_exist.wav"

    with pytest.raises(InvalidAudioError):
        run_pipeline(missing)


def test_pipeline_transcription_failure_propagates(sample_wav):
    class FailingTranscriber:
        def transcribe(self, audio):
            raise TranscriptionError("model crashed")

    # No transcript at all -> nothing to salvage, error must propagate rather than
    # being swallowed into a status field.
    with pytest.raises(TranscriptionError, match="model crashed"):
        run_pipeline(sample_wav, transcriber=FailingTranscriber(), extractor=MockExtractor())
