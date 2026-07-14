from __future__ import annotations

import json
import os
import time
from typing import Callable, Protocol

from pydantic import BaseModel, ValidationError

from .errors import ExtractionError
from .transcribe import Transcript

EXTRACTION_SYSTEM_PROMPT = """You are an information-extraction assistant. Given a meeting/call \
transcript, produce a concise summary, a list of concrete action items, and a list of named \
entities (people, organizations, dates, products). Respond with JSON only, matching this shape:
{"summary": str, "action_items": [str, ...], "entities": [{"text": str, "type": str}, ...]}"""


class Entity(BaseModel):
    text: str
    type: str


class StructuredResult(BaseModel):
    summary: str
    action_items: list[str]
    entities: list[Entity]


class Extractor(Protocol):
    def extract(self, transcript: Transcript) -> StructuredResult: ...


def _run_with_retries(
    get_raw_text: Callable[[str], str], max_retries: int, backoff_base_s: float
) -> StructuredResult:
    """Shared retry policy: one corrective re-prompt on malformed JSON, exponential
    backoff on transient errors. `get_raw_text(correction)` must return the raw model
    response text; `correction` is prepended to the prompt on the retry after bad JSON."""
    last_error: Exception | None = None
    correction = ""

    for attempt in range(max_retries):
        try:
            raw_text = get_raw_text(correction)
            data = json.loads(raw_text)
            return StructuredResult.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            correction = (
                f"Your previous response was invalid ({exc}). "
                "Respond with JSON only, matching the required schema.\n\n"
            )
            continue

        except Exception as exc:  # transient API errors: rate limit, timeout, connection
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base_s * (2**attempt))
            continue

    raise ExtractionError(f"Extraction failed after {max_retries} attempts: {last_error}") from last_error


class GeminiExtractor:
    """Structured extraction over a transcript via the Google Gemini API."""

    def __init__(
        self,
        client=None,
        model: str = "gemini-2.0-flash",
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
    ):
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise ExtractionError(
                    "google-generativeai package is not installed. "
                    "Run `pip install google-generativeai`."
                ) from exc
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ExtractionError("GEMINI_API_KEY environment variable is not set.")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(
                self._model, system_instruction=EXTRACTION_SYSTEM_PROMPT
            )
        return self._client

    def extract(self, transcript: Transcript) -> StructuredResult:
        model = self._get_client()

        def get_raw_text(correction: str) -> str:
            response = model.generate_content(
                f"{correction}Transcript:\n\n{transcript.text}",
                generation_config={"response_mime_type": "application/json"},
            )
            return response.text

        return _run_with_retries(get_raw_text, self._max_retries, self._backoff_base_s)


class MockExtractor:
    """Returns a canned structured result. Used in fast/deterministic tests."""

    def __init__(self, result: StructuredResult | None = None, raise_error: Exception | None = None):
        self._result = result or StructuredResult(
            summary="Mock summary of the conversation.",
            action_items=["Follow up with the client.", "Send the proposal by Friday."],
            entities=[Entity(text="Friday", type="DATE")],
        )
        self._raise_error = raise_error

    def extract(self, transcript: Transcript) -> StructuredResult:
        if self._raise_error is not None:
            raise self._raise_error
        return self._result
