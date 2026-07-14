from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .audio import load_audio
from .errors import ExtractionError
from .extract import Extractor, GeminiExtractor, StructuredResult
from .transcribe import Transcriber, Transcript, WhisperTranscriber

Status = Literal["ok", "empty", "partial_failure"]


@dataclass(frozen=True)
class PipelineResult:
    status: Status
    audio_file: str
    transcript: Transcript
    structured: StructuredResult | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "audio_file": self.audio_file,
            "transcript": {
                "text": self.transcript.text,
                "segments": [s.__dict__ for s in self.transcript.segments],
                "language": self.transcript.language,
                "duration_s": self.transcript.duration_s,
            },
            "structured": self.structured.model_dump() if self.structured else None,
            "error": self.error,
        }


def run_pipeline(
    audio_path: str | Path,
    transcriber: Transcriber | None = None,
    extractor: Extractor | None = None,
) -> PipelineResult:
    transcriber = transcriber or WhisperTranscriber()
    extractor = extractor or GeminiExtractor()

    audio = load_audio(audio_path)  # raises InvalidAudioError - no partial output on bad input

    # STT failure propagates: without a transcript, there's nothing to salvage downstream.
    transcript = transcriber.transcribe(audio)

    if transcript.is_empty:
        return PipelineResult(
            status="empty",
            audio_file=str(audio.path),
            transcript=transcript,
            structured=None,
        )

    try:
        structured = extractor.extract(transcript)
    except ExtractionError as exc:
        # Extraction failed after retries: keep the transcript we already have.
        return PipelineResult(
            status="partial_failure",
            audio_file=str(audio.path),
            transcript=transcript,
            structured=None,
            error=str(exc),
        )

    return PipelineResult(
        status="ok",
        audio_file=str(audio.path),
        transcript=transcript,
        structured=structured,
    )
