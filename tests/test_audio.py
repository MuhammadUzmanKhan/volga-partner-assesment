import pytest

from transcription_pipeline.audio import load_audio
from transcription_pipeline.errors import InvalidAudioError


def test_load_audio_missing_file(tmp_path):
    with pytest.raises(InvalidAudioError, match="not found"):
        load_audio(tmp_path / "nope.wav")


def test_load_audio_unsupported_extension(tmp_path):
    bad = tmp_path / "clip.xyz"
    bad.write_bytes(b"not real audio")

    with pytest.raises(InvalidAudioError, match="Unsupported"):
        load_audio(bad)


def test_load_audio_wav_duration(sample_wav):
    audio = load_audio(sample_wav)

    assert audio.duration_s == pytest.approx(2.0, abs=0.05)


def test_load_audio_non_wav_duration_is_none(tmp_path):
    mp3 = tmp_path / "clip.mp3"
    mp3.write_bytes(b"fake mp3 bytes, never parsed")

    audio = load_audio(mp3)

    assert audio.duration_s is None
