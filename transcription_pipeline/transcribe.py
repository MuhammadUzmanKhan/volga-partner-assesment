from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .audio import AudioInput
from .errors import TranscriptionError


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: list[Segment]
    language: str | None
    duration_s: float | None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class Transcriber(Protocol):
    def transcribe(self, audio: AudioInput) -> Transcript: ...


class WhisperTranscriber:
    """Local speech-to-text using faster-whisper."""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None  # lazy-loaded: model weights are large, don't pay the cost until used

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "faster-whisper is not installed. Run `pip install faster-whisper`."
                ) from exc
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, audio: AudioInput) -> Transcript:
        model = self._get_model()
        try:
            segments_iter, info = model.transcribe(str(audio.path))
            segments = [
                Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments_iter
            ]
        except Exception as exc:  # faster-whisper/ctranslate2 raise varied backend errors
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

        text = " ".join(s.text for s in segments).strip()
        return Transcript(
            text=text,
            segments=segments,
            language=getattr(info, "language", None),
            duration_s=getattr(info, "duration", audio.duration_s),
        )


class MockTranscriber:
    """Returns a canned transcript. Used in fast/deterministic tests."""

    def __init__(self, transcript: Transcript | None = None):
        self._transcript = transcript or Transcript(
            text="This is a mock transcript for testing purposes.",
            segments=[
                Segment(start=0.0, end=3.0, text="This is a mock transcript for testing purposes.")
            ],
            language="en",
            duration_s=3.0,
        )

    def transcribe(self, audio: AudioInput) -> Transcript:
        return self._transcript
