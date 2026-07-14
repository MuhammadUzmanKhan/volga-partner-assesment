from __future__ import annotations

import json
import time
from typing import Protocol

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


class LLMExtractor:
    """Structured extraction over a transcript via the Anthropic API."""

    def __init__(
        self,
        client=None,
        model: str = "claude-sonnet-5",
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
                import anthropic
            except ImportError as exc:
                raise ExtractionError(
                    "anthropic package is not installed. Run `pip install anthropic`."
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def extract(self, transcript: Transcript) -> StructuredResult:
        client = self._get_client()
        last_error: Exception | None = None
        correction = ""

        for attempt in range(self._max_retries):
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=EXTRACTION_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": f"{correction}Transcript:\n\n{transcript.text}"}
                    ],
                )
                data = json.loads(response.content[0].text)
                return StructuredResult.model_validate(data)

            except (json.JSONDecodeError, ValidationError) as exc:
                # malformed response: one corrective re-prompt, no backoff sleep needed
                last_error = exc
                correction = (
                    f"Your previous response was invalid ({exc}). "
                    "Respond with JSON only, matching the required schema.\n\n"
                )
                continue

            except Exception as exc:  # transient API errors: rate limit, timeout, connection
                last_error = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_base_s * (2**attempt))
                continue

        raise ExtractionError(
            f"Extraction failed after {self._max_retries} attempts: {last_error}"
        ) from last_error


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
