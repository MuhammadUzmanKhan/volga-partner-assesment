import json

from transcription_pipeline.cli import main
from transcription_pipeline.transcribe import Transcript


class FakeTranscriber:
    """Stands in for WhisperTranscriber so CLI tests never touch a real model."""

    def __init__(self, model_size):
        self.model_size = model_size

    def transcribe(self, audio):
        return Transcript(text="hello world", segments=[], language="en", duration_s=1.0)


def _isolate_from_local_dotenv(monkeypatch):
    # Tests must not depend on (or leak) whatever real key is in the developer's
    # local .env - CLI tests always run with an empty, controlled environment.
    monkeypatch.setattr("transcription_pipeline.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_cli_run_success_writes_json(tmp_path, monkeypatch, sample_wav):
    _isolate_from_local_dotenv(monkeypatch)
    monkeypatch.setattr("transcription_pipeline.transcribe.WhisperTranscriber", FakeTranscriber)
    out_path = tmp_path / "result.json"

    exit_code = main(["run", "--audio", str(sample_wav), "--out", str(out_path)])

    assert exit_code == 0
    data = json.loads(out_path.read_text())
    assert data["transcript"]["text"] == "hello world"
    # no GEMINI_API_KEY in this isolated environment -> extraction can't succeed,
    # but the transcript must still have been written.
    assert data["status"] == "partial_failure"
    assert data["structured"] is None


def test_cli_run_invalid_audio_exits_nonzero(tmp_path, monkeypatch, capsys):
    _isolate_from_local_dotenv(monkeypatch)
    missing = tmp_path / "nope.wav"
    out_path = tmp_path / "result.json"

    exit_code = main(["run", "--audio", str(missing), "--out", str(out_path)])

    assert exit_code == 1
    assert "Audio file not found" in capsys.readouterr().err
    assert not out_path.exists()  # nothing written on a hard failure


def test_cli_requires_command():
    try:
        main([])
        assert False, "expected argparse to reject a missing subcommand"
    except SystemExit as exc:
        assert exc.code != 0
