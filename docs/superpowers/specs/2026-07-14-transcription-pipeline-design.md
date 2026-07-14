# Transcription Pipeline — Design

## Purpose

Take-home exercise: convert audio to text and process the result for downstream use (structured extraction — summary, action items, entities). Goal is to demonstrate engineering decisions on a batch pipeline, not to train or fine-tune a model.

## Requirements (from stakeholder Q&A)

- **Downstream use**: structured extraction — summary + action items + named entities, for later search/analytics use.
- **STT engine**: local, open-source — Whisper via `faster-whisper`. No API key / network dependency for the transcription stage itself.
- **Processing mode**: batch. Point at an audio file, run end-to-end, emit a result. No streaming/live audio.
- **Extraction engine**: LLM API call (Anthropic), with a structured JSON schema for the response.
- **Language/runtime**: Python.
- **Rigor level**: moderate — modular pipeline stages, real error handling and stage isolation, unit tests with mocks, no queue/infra layer.
- **Mock data strategy**: a small real (or generated) audio sample for an end-to-end demo run; unit tests mock the STT and LLM calls with canned responses for speed and determinism.

## Architecture

Three approaches considered:

1. **Single script, linear functions** — fastest to write, but stages can't be tested or swapped independently.
2. **Stage-interface + orchestrator (chosen)** — each stage (`Transcriber`, `Extractor`) is a `Protocol` with a real implementation and a mock implementation; an orchestrator wires them together via dependency injection. Matches the "moderate rigor" requirement: testable, swappable, no infra overhead.
3. **Queue/worker (Celery-style)** — appropriate for scaling to many concurrent jobs, but there's no stated scaling requirement here. Overkill for this exercise.

Chosen: **#2**.

### Data flow

```
audio file
  → AudioLoader (validate: exists, supported extension, readable)
  → Transcriber (Whisper, local)          → Transcript(text, segments[], language, duration)
  → Extractor (LLM, structured JSON)      → StructuredResult(summary, action_items[], entities[])
  → ResultAssembler                       → PipelineResult(transcript, structured, status)
  → written to output JSON
```

### Components

| Module | Responsibility |
|---|---|
| `audio.py` | `AudioInput` loader/validator — file exists, extension supported, basic metadata (duration via the audio lib). |
| `transcribe.py` | `Transcriber` protocol. `WhisperTranscriber` (faster-whisper, local inference). `MockTranscriber` (returns a canned `Transcript` — used in fast tests). |
| `extract.py` | `Extractor` protocol. `LLMExtractor` (Anthropic API call with a JSON schema / tool-use for structured output, pydantic validation of the response). `MockExtractor` (canned `StructuredResult`). |
| `pipeline.py` | Orchestrator — `run(audio_path) -> PipelineResult`. Wires `Transcriber` + `Extractor` (injected, defaults to real impls), tracks per-stage status, assembles final result. |
| `cli.py` | Entrypoint: `python -m transcription_pipeline run --audio path.wav --out result.json`. |

Each component is understandable and testable in isolation: given its protocol, a caller doesn't need to know whether it's talking to Whisper or a mock.

## Error handling & partial-failure model

Core idea: a failure in a later stage must not destroy data already obtained from an earlier stage.

- **Input validation**: missing file or unsupported format → fail fast with a clear error; no partial output is written.
- **STT failure** (corrupt audio, model crash): caught, wrapped as `TranscriptionError`; the run aborts — without a transcript there's nothing for downstream stages to do.
- **Empty/silent audio**: valid input, but empty transcript → output is still written with `status="empty"`; the extraction stage is skipped (nothing to extract).
- **LLM extraction — transient errors** (timeout, rate limit): retried up to 3x with exponential backoff. If retries are exhausted, the transcript is still saved; `status="partial_failure"`, `structured=null`. The caller can re-run just the extraction stage later against the saved transcript.
- **LLM extraction — malformed JSON**: response is validated against a pydantic schema. One corrective retry is attempted (re-prompt including the parse error). If it still fails, same `partial_failure` path — transcript is preserved regardless.

`PipelineResult.status` is always one of `ok | empty | partial_failure`. Nothing silently drops data that was already successfully produced.

## Output format

A single JSON file per run:

```json
{
  "status": "ok",
  "audio_file": "path/to/input.wav",
  "transcript": {
    "text": "...",
    "segments": [{"start": 0.0, "end": 4.2, "text": "..."}],
    "language": "en",
    "duration_s": 42.1
  },
  "structured": {
    "summary": "...",
    "action_items": ["..."],
    "entities": [{"text": "...", "type": "PERSON"}]
  }
}
```

## Testing strategy

- `test_pipeline.py` — orchestrator wiring using `MockTranscriber` + `MockExtractor`. Asserts status transitions (`ok`/`empty`/`partial_failure`) and that a downstream failure doesn't erase the transcript already produced.
- `test_extract.py` — prompt/schema construction, retry-on-malformed-JSON logic, mocked Anthropic client returning both valid and invalid responses.
- `test_transcribe.py` — marked `@slow`; runs real `faster-whisper` inference against a small bundled sample (`tests/fixtures/sample.wav`, a few seconds of speech). One true end-to-end sanity check; skipped in the default fast test run.
- Default test run: fully mocked, no network or model calls — fast enough for routine use.

## Repo layout

```
transcription_pipeline/
  __init__.py
  audio.py
  transcribe.py
  extract.py
  pipeline.py
  cli.py
tests/
  fixtures/sample.wav
  test_pipeline.py
  test_extract.py
  test_transcribe.py
requirements.txt
README.md
```

## Out of scope

- Streaming/live transcription.
- Speaker diarization.
- Queue/worker infrastructure, horizontal scaling, observability/metrics stack.
- Fine-tuning or training any model.
