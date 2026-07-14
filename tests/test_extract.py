import json
from types import SimpleNamespace

import pytest

from transcription_pipeline.errors import ExtractionError
from transcription_pipeline.extract import LLMExtractor
from transcription_pipeline.transcribe import Transcript

TRANSCRIPT = Transcript(
    text="Let's ship the feature by Friday.", segments=[], language="en", duration_s=5.0
)


def _response(payload: dict):
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


def _bad_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def test_extract_success_first_try():
    valid_payload = {
        "summary": "Team agreed to ship by Friday.",
        "action_items": ["Ship the feature by Friday."],
        "entities": [{"text": "Friday", "type": "DATE"}],
    }
    client = FakeClient([_response(valid_payload)])
    extractor = LLMExtractor(client=client)

    result = extractor.extract(TRANSCRIPT)

    assert result.summary == valid_payload["summary"]
    assert result.entities[0].text == "Friday"


def test_extract_recovers_after_malformed_json():
    valid_payload = {"summary": "ok", "action_items": [], "entities": []}
    client = FakeClient([_bad_response("not json"), _response(valid_payload)])
    extractor = LLMExtractor(client=client)

    result = extractor.extract(TRANSCRIPT)

    assert result.summary == "ok"


def test_extract_raises_after_exhausting_retries():
    client = FakeClient([_bad_response("not json")] * 3)
    extractor = LLMExtractor(client=client, max_retries=3)

    with pytest.raises(ExtractionError):
        extractor.extract(TRANSCRIPT)


def test_extract_retries_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("transcription_pipeline.extract.time.sleep", lambda s: None)
    valid_payload = {"summary": "ok", "action_items": [], "entities": []}
    client = FakeClient([TimeoutError("timed out"), _response(valid_payload)])
    extractor = LLMExtractor(client=client, max_retries=3)

    result = extractor.extract(TRANSCRIPT)

    assert result.summary == "ok"
