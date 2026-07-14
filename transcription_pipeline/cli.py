from __future__ import annotations

import argparse
import json
import sys

from .errors import InvalidAudioError, TranscriptionError
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcription_pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Transcribe an audio file and extract structured output."
    )
    run_parser.add_argument("--audio", required=True, help="Path to the input audio file.")
    run_parser.add_argument("--out", required=True, help="Path to write the output JSON file.")
    run_parser.add_argument(
        "--model-size", default="base", help="Whisper model size (default: base)."
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        from .transcribe import WhisperTranscriber

        try:
            result = run_pipeline(
                args.audio, transcriber=WhisperTranscriber(model_size=args.model_size)
            )
        except (InvalidAudioError, TranscriptionError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        with open(args.out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Status: {result.status}. Output written to {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
