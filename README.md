# Transcription Pipeline

Batch pipeline: audio file → local Whisper transcription → Gemini structured extraction
(summary, action items, entities) → JSON output.

Design rationale: [docs/superpowers/specs/2026-07-14-transcription-pipeline-design.md](docs/superpowers/specs/2026-07-14-transcription-pipeline-design.md).

## Install

```bash
pip install -r requirements-dev.txt
export GEMINI_API_KEY=...   # only needed for real extraction runs
```

## Run

```bash
python -m transcription_pipeline run --audio path/to/audio.wav --out result.json
```

Output (`result.json`):

```json
{
  "status": "ok",
  "audio_file": "path/to/audio.wav",
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

`status` is one of:

- `ok` — both stages succeeded.
- `empty` — transcription succeeded but produced no speech (extraction is skipped, not attempted).
- `partial_failure` — transcription succeeded, extraction failed after retries. Transcript is
  still written; `error` explains why extraction failed.

Invalid input (missing file / unsupported format) or an unrecoverable transcription failure
raises before anything is written — there's nothing to salvage in that case.

## Test

```bash
pytest                 # fast suite: fully mocked, no model download or network calls
pytest -m slow         # also runs real faster-whisper inference on a generated sample
```

## Architecture

```
audio file
  → AudioLoader (validate)
  → Transcriber (Whisper, local)      → Transcript
  → Extractor (Gemini, structured JSON) → StructuredResult
  → ResultAssembler                   → PipelineResult → JSON
```

`Transcriber` and `Extractor` are protocols with a real and a mock implementation each
(`transcription_pipeline/transcribe.py`, `transcription_pipeline/extract.py`), injected into
`run_pipeline()`. This is what makes the pipeline testable without a model download or an API
key, and what would make either stage swappable later (e.g. a cloud STT API instead of local
Whisper) without touching orchestration code.
