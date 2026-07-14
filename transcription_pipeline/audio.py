from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from .errors import InvalidAudioError

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass(frozen=True)
class AudioInput:
    path: Path
    duration_s: float | None


def load_audio(path: str | Path) -> AudioInput:
    audio_path = Path(path)

    if not audio_path.exists():
        raise InvalidAudioError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InvalidAudioError(
            f"Unsupported audio format {audio_path.suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return AudioInput(path=audio_path, duration_s=_try_get_wav_duration(audio_path))


def _try_get_wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate) if rate else None
    except (wave.Error, EOFError):
        return None
