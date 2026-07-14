import builtins
import json
from types import SimpleNamespace

import pytest

from transcription_pipeline.errors import ExtractionError
from transcription_pipeline.extract import GeminiExtractor
from transcription_pipeline.transcribe import Transcript

TRANSCRIPT = Transcript(
    text="Let's ship the feature by Friday.", segments=[], language="en", duration_s=5.0
)


def _response(payload: dict):
    return SimpleNamespace(text=json.dumps(payload))


def _bad_response(text: str):
    return SimpleNamespace(text=text)


class FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate_content(self, *args, **kwargs):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeModel:
    """Stands in for genai.Client."""

    def __init__(self, responses):
        self.models = FakeModels(responses)


def test_extract_success_first_try():
    valid_payload = {
        "summary": "Team agreed to ship by Friday.",
        "action_items": ["Ship the feature by Friday."],
        "entities": [{"text": "Friday", "type": "DATE"}],
    }
    client = FakeModel([_response(valid_payload)])
    extractor = GeminiExtractor(client=client)

    result = extractor.extract(TRANSCRIPT)

    assert result.summary == valid_payload["summary"]
    assert result.entities[0].text == "Friday"


def test_extract_recovers_after_malformed_json():
    valid_payload = {"summary": "ok", "action_items": [], "entities": []}
    client = FakeModel([_bad_response("not json"), _response(valid_payload)])
    extractor = GeminiExtractor(client=client)

    result = extractor.extract(TRANSCRIPT)

    assert result.summary == "ok"


def test_extract_raises_after_exhausting_retries():
    client = FakeModel([_bad_response("not json")] * 3)
    extractor = GeminiExtractor(client=client, max_retries=3)

    with pytest.raises(ExtractionError):
        extractor.extract(TRANSCRIPT)


def test_extract_retries_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("transcription_pipeline.extract.time.sleep", lambda s: None)
    valid_payload = {"summary": "ok", "action_items": [], "entities": []}
    client = FakeModel([TimeoutError("timed out"), _response(valid_payload)])
    extractor = GeminiExtractor(client=client, max_retries=3)

    result = extractor.extract(TRANSCRIPT)

    assert result.summary == "ok"


def test_extract_recovers_after_schema_validation_error():
    # valid JSON, but entities don't match the schema (list of strings, not objects)
    invalid_schema_payload = {"summary": "ok", "action_items": [], "entities": ["not-a-dict"]}
    valid_payload = {"summary": "ok", "action_items": [], "entities": []}
    client = FakeModel([_response(invalid_schema_payload), _response(valid_payload)])
    extractor = GeminiExtractor(client=client)

    result = extractor.extract(TRANSCRIPT)

    assert result.summary == "ok"


def test_extract_missing_api_key_raises_without_network_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    extractor = GeminiExtractor()  # no client injected -> real _get_client() path

    with pytest.raises(ExtractionError, match="GEMINI_API_KEY"):
        extractor.extract(TRANSCRIPT)


def test_extract_missing_dependency(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    real_import = builtins.__import__

    # sys.modules-based faking doesn't reliably force ImportError here: "google" is a
    # namespace package, and once `genai` has resolved as an attribute on it (from any
    # earlier import in the process), `from google import genai` stops consulting
    # sys.modules at all. Intercepting __import__ itself is deterministic regardless
    # of caching state.
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" and fromlist and "genai" in fromlist:
            raise ImportError("simulated: google-genai not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    extractor = GeminiExtractor()

    with pytest.raises(ExtractionError, match="not installed"):
        extractor.extract(TRANSCRIPT)
