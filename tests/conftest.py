import math
import struct
import wave
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory) -> Path:
    """A short synthetic tone, standing in for a real speech sample.

    Enough for the real Whisper integration test to exercise the full
    decode/inference path without needing to ship recorded audio.
    """
    path = tmp_path_factory.mktemp("fixtures") / "sample.wav"
    framerate = 16000
    duration_s = 2
    n_frames = framerate * duration_s

    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        frames = bytearray()
        for i in range(n_frames):
            value = int(3000 * math.sin(2 * math.pi * 440 * (i / framerate)))
            frames += struct.pack("<h", value)
        wav_file.writeframes(bytes(frames))

    return path
