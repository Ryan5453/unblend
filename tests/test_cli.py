"""CLI-level exit-code tests driven through ``typer.testing.CliRunner``.

These are network-free: unknown model names short-circuit before any
download, and the format probe runs before model resolution.
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from click.testing import Result
from torchcodec.encoders import AudioEncoder
from typer.testing import CliRunner

from unblend.cli import build_app
from unblend.cli.separate import (
    _estimate_compile_chunks,
    _maybe_enable_auto_compile,
)
from unblend.repo import STAGING_PREFIX, STAGING_STALE_SECONDS, ModelRepository

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
    ``unblend version`` succeeds and prints the version.
    """
    result = _invoke(["version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_models_list_exits_zero() -> None:
    """
    ``unblend models list`` succeeds and lists the shipped models.
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


def test_models_download_accepts_roformer_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Batch download progress treats each RoFormer checkpoint as one layer
    instead of requiring the Demucs-only ``models`` metadata field.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    import unblend.cli.models as models_cli

    class DownloadedModel:
        """Minimal model returned by the network-free repository stub."""

        sources = ["vocals", "other"]

        def eval(self) -> "DownloadedModel":
            """
            Mirror ``nn.Module.eval`` for the download command.

            :return: This stub model.
            """
            return self

    roformers = {
        "roformer_a": {"backend": "roformer"},
        "roformer_b": {"backend": "roformer"},
    }
    monkeypatch.setattr(models_cli, "get_models", lambda: roformers)
    monkeypatch.setattr(ModelRepository, "get_cache_info", lambda self: {})
    monkeypatch.setattr(
        ModelRepository,
        "get_model",
        lambda self, **kwargs: DownloadedModel(),
    )

    result = _invoke(["models", "download", "roformer_a", "roformer_b"])
    assert result.exit_code == 0
    assert "2 total layers" in result.output


def test_ensure_model_available_downloads_uncached_roformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The benchmark/CLI availability preflight can download a RoFormer whose
    metadata has one ``checkpoint`` instead of a Demucs ``models`` list.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    import unblend.cli.models as models_cli

    class DownloadedModel:
        """Minimal model returned by the network-free repository stub."""

        sources = ["vocals", "other"]

        def eval(self) -> "DownloadedModel":
            """
            Mirror ``nn.Module.eval`` for the download command.

            :return: This stub model.
            """
            return self

    roformer = {"roformer": {"backend": "roformer"}}
    monkeypatch.setattr(models_cli, "get_models", lambda: roformer)
    monkeypatch.setattr(ModelRepository, "list_models", lambda self: roformer)
    monkeypatch.setattr(ModelRepository, "get_cache_info", lambda self: {})
    monkeypatch.setattr(
        ModelRepository,
        "get_model",
        lambda self, **kwargs: DownloadedModel(),
    )

    assert models_cli.ensure_model_available("roformer") is True


def test_auto_compile_chunk_estimate_uses_duration_shifts_and_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CLI auto-compile estimates model work from metadata duration rather than
    encoded file size, including shift rounds and overlap-derived stride.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    separator = SimpleNamespace(
        model=SimpleNamespace(samplerate=100, max_allowed_segment=10.0)
    )
    durations = iter([12.0, 3.0])
    monkeypatch.setattr(
        "unblend.cli.separate._audio_duration_seconds",
        lambda path: next(durations),
    )

    chunks, duration, unknown = _estimate_compile_chunks(
        separator,
        [Path("a.flac"), Path("b.wav")],
        shifts=2,
        split_overlap=0.5,
    )

    assert chunks == 8
    assert duration == 15.0
    assert unknown == 0


@pytest.mark.parametrize(
    "chunks, expected",
    [(449, False), (450, True)],
)
def test_auto_compile_uses_predicted_eager_seconds(
    monkeypatch: pytest.MonkeyPatch,
    chunks: int,
    expected: bool,
) -> None:
    """
    Auto mode enables compilation exactly when chunks × runtime probe reaches
    the architecture/dtype GPU-seconds threshold.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param chunks: Estimated workload chunks.
    :param expected: Whether compilation should be enabled.
    """
    enable_compile = Mock()
    separator = SimpleNamespace(
        _eager_probe_seconds=1.0,
        enable_compile=enable_compile,
    )
    monkeypatch.setattr(
        "unblend.cli.separate._compile_profile_key",
        lambda separator: ("bs_roformer", "fp16"),
    )
    monkeypatch.setattr(
        "unblend.cli.separate._estimate_compile_chunks",
        lambda separator, audio_files, *, shifts, split_overlap: (
            chunks,
            60.0,
            0,
        ),
    )

    enabled = _maybe_enable_auto_compile(
        separator,
        [Path("track.wav")],
        shifts=1,
        split_overlap=0.25,
    )

    assert enabled is expected
    assert enable_compile.call_count == int(expected)


def test_models_remove_unknown_model_fails() -> None:
    """
    An unknown model name makes ``models remove`` exit nonzero.
    """
    result = _invoke(["models", "remove", "not_a_real_model"])
    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_models_remove_markup_model_name_renders_literally() -> None:
    """
    A model name containing Rich markup must not raise ``MarkupError``.
    """
    result = _invoke(["models", "remove", "[/red]evil"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
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


class _StubSeparator:
    """Stands in for Separator so the path pre-checks run without a model."""

    class _Model:
        sources = ["drums", "bass", "other", "vocals"]

    def __init__(self, **kwargs: object) -> None:
        self.model = self._Model()


def _stub_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make ``separate`` reach its output-path pre-checks network-free.

    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setattr(
        "unblend.cli.separate.ensure_model_available", lambda *a, **k: True
    )
    monkeypatch.setattr("unblend.cli.separate.Separator", _StubSeparator)


def test_separate_collision_check_uses_written_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Planned paths that differ only by the container appended at write time
    (``mix.wav.flac`` vs ``mix.flac``) are caught as collisions.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    _stub_model_loading(monkeypatch)
    for name in ("mix.wav.flac", "mix.flac"):
        AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(
            tmp_path / name
        )

    result = _invoke(
        [
            "separate",
            str(tmp_path / "mix.wav.flac"),
            str(tmp_path / "mix.flac"),
            "-o",
            str(tmp_path / "out" / "{stem}" / "{track}"),
        ]
    )
    assert result.exit_code == 1
    assert "colliding paths" in result.output


def test_separate_dotted_track_container_fails_before_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A dot leaked from the track name into the written path's suffix is
    validated as a container before separation, not after.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    _stub_model_loading(monkeypatch)
    track = tmp_path / "Song feat. Artist.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(track)

    result = _invoke(["separate", str(track), "-o", str(tmp_path / "{track}_{stem}")])
    assert result.exit_code == 1
    assert "Unsupported output format" in result.output


def test_separate_case_aliasing_paths_fail_before_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Planned paths differing only by letter case are refused conservatively
    before inference because they alias on common filesystems.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    _stub_model_loading(monkeypatch)
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    for rel in ("d1/MIX.wav", "d2/mix.wav"):
        AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(
            tmp_path / rel
        )

    result = _invoke(
        [
            "separate",
            str(tmp_path / "d1" / "MIX.wav"),
            str(tmp_path / "d2" / "mix.wav"),
            "-o",
            str(tmp_path / "out" / "{track}" / "{stem}"),
        ]
    )
    assert result.exit_code == 1
    assert "filesystem aliases" in result.output
    assert "will be stored using template" not in result.output


def test_separate_unicode_aliasing_paths_fail_before_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NFC-equivalent output names are rejected before inference."""
    _stub_model_loading(monkeypatch)
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    for directory, name in (("d1", "Café.wav"), ("d2", "Cafe\u0301.wav")):
        AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(
            tmp_path / directory / name
        )

    result = _invoke(
        [
            "separate",
            str(tmp_path / "d1" / "Café.wav"),
            str(tmp_path / "d2" / "Cafe\u0301.wav"),
            "-o",
            str(tmp_path / "out" / "{track}" / "{stem}"),
        ]
    )
    assert result.exit_code == 1
    assert "filesystem aliases" in result.output


def test_separate_symlink_loop_output_fails_cleanly_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path-resolution failures are reported as CLI errors, not tracebacks."""
    _stub_model_loading(monkeypatch)
    audio = tmp_path / "song.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(audio)
    loop = tmp_path / "loop"
    try:
        loop.symlink_to(loop)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    result = _invoke(
        [
            "separate",
            str(audio),
            "-o",
            str(loop / "{track}" / "{stem}"),
        ]
    )
    assert result.exit_code == 1
    assert "Could not resolve planned output path" in result.output
    assert "Traceback" not in result.output


def test_separate_markup_in_paths_renders_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Rich markup sequences spanning path components (``[/…]``) must render
    literally instead of raising an uncaught ``MarkupError``.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    _stub_model_loading(monkeypatch)
    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    result = _invoke(
        [
            "separate",
            str(wav_path),
            "-o",
            str(tmp_path / "out[" / "{track}]" / "{stem}"),
        ]
    )
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "out[" in result.output


@pytest.mark.parametrize("template", ["", ".", "/"])
def test_separate_empty_name_output_template_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, template: str
) -> None:
    """
    A template resolving to an empty filename exits 1 with a clean message
    instead of an uncaught ``ValueError`` from ``with_suffix``.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param template: Output template that resolves to an empty pathlib name
    """
    _stub_model_loading(monkeypatch)
    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    result = _invoke(["separate", str(wav_path), "-o", template])
    assert result.exit_code == 1
    assert "empty filename" in result.output
    assert not isinstance(result.exception, ValueError)


def test_separate_format_with_path_separator_fails_cleanly(
    tmp_path: Path,
) -> None:
    """
    A --format containing a path separator exits 1 with the full format in a
    clean message instead of an uncaught ``ValueError`` (and the pre-flight
    probe must not validate a truncated version of it).

    :param tmp_path: pytest temporary directory fixture
    """
    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    result = _invoke(["separate", str(wav_path), "-f", "x/wav"])
    assert result.exit_code == 1
    assert "Unsupported output format" in result.output
    assert "x/wav" in result.output
    assert not isinstance(result.exception, ValueError)


@pytest.mark.parametrize("bad_format", ["./wav", "wav/", ""])
def test_separate_path_normalized_format_fails_before_download(
    tmp_path: Path, bad_format: str
) -> None:
    """
    Formats that ``Path()`` would normalize ("./wav" → "wav") must be
    rejected up front with the format as typed, not validated in their
    normalized form and failed (or silently altered) after model download.

    :param tmp_path: pytest temporary directory fixture
    :param bad_format: Format string that Path-normalizes to something else
    """
    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    result = _invoke(["separate", str(wav_path), "-f", bad_format])
    assert result.exit_code == 1
    assert "Unsupported output format" in result.output
    assert f"'{bad_format}'" in result.output
    # Rejected before model selection — no download work.
    assert "Auto-selected model" not in result.output


def test_separate_leading_dot_format_treated_as_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``-f .wav`` means "wav" — the leading dot must not produce doubled-dot
    output names like ``drums..wav``.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    _stub_model_loading(monkeypatch)
    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    result = _invoke(["separate", str(wav_path), "-f", ".wav", "-o", "o/{stem}.{ext}"])
    assert "o/{stem}.wav'" in result.output
    assert "..wav" not in result.output


def test_export_onnx_markup_model_name_renders_literally() -> None:
    """
    Markup in the ``export-onnx`` model name must not raise ``MarkupError``.
    """
    result = _invoke(["export-onnx", "--model", "[/red]evil"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_models_remove_all_empty_cache_exits_zero(
    tmp_path: Path, monkeypatch: object
) -> None:
    """
    ``models remove --all`` on an empty cache prints a friendly message and
    exits 0 rather than treating "nothing to remove" as an error.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: tmp_path)

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


def test_models_remove_all_sweeps_partial_and_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``models remove --all`` removes partially-cached models (e.g. an
    interrupted multi-layer download) and leftover download temp files,
    not just fully-cached models.
    """
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path))
    # One layer of the multi-layer ensemble = a genuinely partial cache.
    checksum = ModelRepository().list_models()["htdemucs_ft"]["models"][0]["checksum"]
    partial_layer = tmp_path / f"{checksum}.safetensors"
    stale_tmp = tmp_path / f"{STAGING_PREFIX}q1w2e3.tmp"
    partial_layer.write_bytes(b"x")
    stale_tmp.write_bytes(b"y")
    old = time.time() - STAGING_STALE_SECONDS - 1
    os.utime(stale_tmp, (old, old))

    result = _invoke(["models", "remove", "--all"])

    assert result.exit_code == 0
    assert not partial_layer.exists()
    assert not stale_tmp.exists()
