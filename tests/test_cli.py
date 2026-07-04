"""CLI-level exit-code tests driven through ``typer.testing.CliRunner``.

These are network-free: unknown model names short-circuit before any
download, and the format probe runs before model resolution.
"""

from pathlib import Path

import torch
from click.testing import Result
from torchcodec.encoders import AudioEncoder
from typer.testing import CliRunner

from demucs.cli import build_app

runner = CliRunner()


def _invoke(args: list[str]) -> Result:
    """
    Invoke the CLI app with ``args`` and return the result.

    :param args: CLI arguments to pass to the app
    :return: The runner result for the invocation
    """
    return runner.invoke(build_app(), args)


def test_version_exits_zero() -> None:
    """
    ``demucs version`` succeeds and prints the version.
    """
    result = _invoke(["version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_models_list_exits_zero() -> None:
    """
    ``demucs models list`` succeeds and lists the shipped models.
    """
    result = _invoke(["models", "list"])
    assert result.exit_code == 0
    assert "htdemucs" in result.output


def test_models_download_unknown_model_fails() -> None:
    """
    An unknown model name makes ``models download`` exit nonzero.
    """
    result = _invoke(["models", "download", "not_a_real_model"])
    assert result.exit_code == 1
    assert "not_a_real_model" in result.output


def test_models_remove_unknown_model_fails() -> None:
    """
    An unknown model name makes ``models remove`` exit nonzero.
    """
    result = _invoke(["models", "remove", "not_a_real_model"])
    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_separate_nonexistent_input_fails() -> None:
    """
    A nonexistent input path makes ``separate`` exit nonzero.
    """
    result = _invoke(["separate", "does_not_exist.mp3"])
    assert result.exit_code == 1


def test_separate_unsupported_format_fails_before_separation(tmp_path: Path) -> None:
    """
    An unencodable --format fails fast, before any model work.

    :param tmp_path: pytest temporary directory fixture
    """
    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    result = _invoke(["separate", str(wav_path), "-f", "definitelynotaformat"])
    assert result.exit_code == 1
    assert "Unsupported output format" in result.output


def test_models_remove_all_empty_cache_exits_zero(
    tmp_path: Path, monkeypatch: object
) -> None:
    """
    ``models remove --all`` on an empty cache prints a friendly message and
    exits 0 rather than treating "nothing to remove" as an error.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setattr("demucs.repo.get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("demucs.cli.models.get_cache_dir", lambda: tmp_path)

    result = _invoke(["models", "remove", "--all"])
    assert result.exit_code == 0
    assert "No models" in result.output or "no models" in result.output.lower()


def test_export_onnx_unknown_model_fails() -> None:
    """
    ``export-onnx`` exits nonzero when the requested model name doesn't exist
    in the registry — same fail-fast contract as the user-facing commands.
    """
    result = _invoke(["export-onnx", "--model", "not_a_real_model"])
    assert result.exit_code == 1


def test_models_download_all_with_names_rejected() -> None:
    """
    ``--all`` and positional model names together is ambiguous; the CLI
    refuses rather than silently ignoring the names.
    """
    result = _invoke(["models", "download", "--all", "htdemucs"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_models_remove_all_with_names_rejected() -> None:
    """
    Same combinatorial guard for ``models remove --all <name>``.
    """
    result = _invoke(["models", "remove", "--all", "htdemucs"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output
