class PipelineError(Exception):
    """Base class for all pipeline errors."""


class InvalidAudioError(PipelineError):
    """Raised when the input audio file is missing or unsupported."""


class TranscriptionError(PipelineError):
    """Raised when the STT stage fails unrecoverably."""


class ExtractionError(PipelineError):
    """Raised when the LLM extraction stage fails unrecoverably (after retries)."""
