# Transcription Pipeline

Batch pipeline: audio file → local Whisper transcription → Gemini structured extraction
(summary, action items, entities) → JSON output.

See "Design decisions" below for the rationale behind the major choices.

## Install

```bash
pip install -r requirements-dev.txt
cp .env.example .env   # then fill in GEMINI_API_KEY (only needed for real extraction runs)
```

`.env` is gitignored — the CLI loads it automatically via `python-dotenv`. Never commit a real key;
`.env.example` holds the placeholder that is safe to commit.

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

## Design decisions

**Why local Whisper (`faster-whisper`) over a cloud STT API.** No per-request cost, no network
dependency, no vendor rate limits on the transcription stage. The tradeoff is local compute —
acceptable for a batch pipeline where latency-to-first-result isn't the priority.

**Why protocol + orchestrator over a single script or a queue/worker system.** A single script
is fastest to write but stages can't be tested or swapped independently. A queue/worker system
(Celery-style) is the right call once you need to scale to many concurrent jobs — nothing here
asked for that, so it would have been pure overhead. The protocol/DI approach sits in the
middle: `Transcriber` and `Extractor` are structural interfaces (`typing.Protocol`), each with a
real and a mock implementation, injected into `run_pipeline()`. That's what makes the fast test
suite run in under a second with no model download or API key, and what would make either stage
swappable later without touching orchestration code.

**Why the `ok` / `empty` / `partial_failure` status model.** Transcription is the expensive part
of this pipeline; extraction (one LLM call) is cheap by comparison. If extraction fails after
retries, the already-computed transcript must not be thrown away — `partial_failure` preserves it
so a caller can retry just the extraction stage later. If transcription itself fails, there's
nothing to salvage, so that error propagates instead of being swallowed into a status field.
Malformed LLM JSON gets one corrective re-prompt (with the parse error fed back in); transient
errors (timeouts, rate limits) get exponential backoff — both paths share one retry helper so the
policy can't silently drift between call sites.

**Why the transcript isn't fully streamed today.** `faster-whisper` returns a generator of
segments internally (backed by bounded ~30s decode windows with voice-activity detection, so
memory doesn't scale linearly with file length) — but the current code consumes that generator
eagerly to build one `Transcript` before returning. For very long files the honest next step
would be exposing that generator end-to-end (through `Transcriber`, `pipeline.py`, and the CLI)
so callers get segments incrementally rather than waiting on the whole file. Not built because
nothing in scope needed it yet.

**Different audio formats.** An extension whitelist (`audio.py`) rejects unsupported input early
with a clear error. Actual decoding is delegated to Whisper's own pipeline (`ffmpeg` under the
hood) rather than reimplemented — a solved problem, not worth the risk of a custom decoder.

## Scaling beyond this exercise

This ships as a CLI/library, not a service. If it needed to become one:

- **Concurrent uploads** — move transcription off the request path onto a queue (Celery/RQ);
  each worker keeps one Whisper model loaded and reuses it across jobs (already how
  `WhisperTranscriber` behaves — the model is lazily loaded once per instance, not per call).
- **Storing audio and transcripts** — audio into object storage (S3/GCS) keyed by job ID;
  transcripts/structured results into a `jobs` table (`job_id, audio_uri, status,
  transcript_json, structured_json, error`) — this maps almost directly onto the existing
  `PipelineResult` shape, since transcript/structured/status/error are already separated.
- **Retry/recovery** — `partial_failure` already covers extraction-only retries against a saved
  transcript. A full service would extend the same idea to transcription failures: mark the job
  `failed`, store the error, re-queue with backoff up to a retry cap, rather than propagating and
  losing the job.
- **Exposing as an API** — e.g. FastAPI: `POST /transcriptions` (upload, returns `202` + `job_id`
  immediately, processed async), `GET /transcriptions/{job_id}` (status + result),
  `POST /transcriptions/{job_id}/retry-extraction`. `run_pipeline()` barely changes — the API
  layer's job is upload/queue/persistence orchestration, not reimplementing pipeline logic.
