"""CLI-level exit-code tests driven through ``typer.testing.CliRunner``.

These are network-free: unknown model names short-circuit before any
download, and the format probe runs before model resolution.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import typer
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


def test_models_info_shows_license_notes() -> None:
    """
    ``models info`` surfaces the registry's license notes, which the list
    table has no room for.
    """
    result = _invoke(["models", "info", "scnet_small"])
    assert result.exit_code == 0
    assert "License notes:" in result.output
    assert "MUSDB18" in result.output


def test_models_info_unknown_model_fails() -> None:
    """
    An unknown name exits nonzero.
    """
    assert _invoke(["models", "info", "nope"]).exit_code == 1


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
    Batch download progress treats each RoFormer checkpoint as one file
    instead of requiring the Demucs-only ``models`` metadata field.

    :param monkeypatch: Pytest monkeypatch fixture.
    """

    class DownloadedModel:
        """
        Minimal model returned by the network-free repository stub.
        """

        sources = ["vocals", "other"]

        def eval(self) -> "DownloadedModel":
            """
            Mirror ``nn.Module.eval`` for the download command.

            :return: This stub model.
            """
            return self

    monkeypatch.setattr(ModelRepository, "get_cache_info", lambda self: {})
    monkeypatch.setattr(
        ModelRepository,
        "get_model",
        lambda self, **kwargs: DownloadedModel(),
    )

    result = _invoke(
        ["models", "download", "bs_roformer_anvuew", "melband_roformer_kim"]
    )
    assert result.exit_code == 0
    assert "(2 files)" in result.output


@pytest.mark.parametrize("architecture", ["bs_roformer", "mel_band_roformer", "scnet"])
def test_ensure_model_available_downloads_uncached_single_checkpoint_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """
    The availability preflight downloads an uncached remote model whatever
    architecture it uses, going through the real registry rather than a
    hand-shaped metadata stub.

    :param tmp_path: Pytest temporary directory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param architecture: Registered single-checkpoint architecture under test.
    """
    import unblend.cli.models as models_cli

    class DownloadedModel:
        """
        Minimal model returned by the network-free repository stub.
        """

        sources = ["vocals", "other"]

        def eval(self) -> "DownloadedModel":
            """
            Mirror ``nn.Module.eval`` for the download command.

            :return: This stub model.
            """
            return self

    extra = tmp_path / "extra-models.json"
    extra.write_text(
        json.dumps(
            {
                "models": {
                    "custom": {
                        "architecture": architecture,
                        "sources": ["vocals", "other"],
                        "samplerate": 44100,
                        "segment_samples": 44100,
                        "config": {"dim": 16},
                        "checkpoint": {
                            "format": "safetensors",
                            "url": "https://example.invalid/custom.safetensors",
                            "sha256": "a" * 64,
                            "size_bytes": 16,
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(extra))
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        ModelRepository,
        "get_model",
        lambda self, **kwargs: DownloadedModel(),
    )

    assert models_cli.ensure_model_available("custom") is True


@pytest.mark.parametrize(
    "entry_factory",
    [
        lambda path: {
            "architecture": "scnet",
            "sources": ["vocals", "other"],
            "samplerate": 44100,
            "segment_samples": 44100,
            "config": {"dims": [4, 8]},
            "checkpoint": {"format": "safetensors", "path": str(path)},
        },
        lambda path: {
            "architecture": "htdemucs",
            "sources": ["vocals", "other"],
            "config": {"sources": ["vocals", "other"]},
            "checkpoint": {"format": "safetensors", "path": str(path)},
        },
    ],
    ids=["single-checkpoint", "demucs-layer"],
)
def test_ensure_model_available_accepts_local_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_factory: object
) -> None:
    """
    Local weights are available without entering download code, whichever
    backend loads them — a Demucs bag of layers included.
    """
    import unblend.cli.models as models_cli

    checkpoint = tmp_path / "custom.safetensors"
    checkpoint.write_bytes(b"weights")
    extra = tmp_path / "extra-models.json"
    extra.write_text(
        json.dumps({"models": {"custom": entry_factory(checkpoint)}})  # type: ignore[operator]
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(extra))
    monkeypatch.setattr(
        models_cli,
        "_download_model_with_progress",
        lambda *args, **kwargs: pytest.fail("local model attempted a download"),
    )

    assert models_cli.ensure_model_available("custom") is True

    # A declared file that is not there is reported, not silently downloaded.
    checkpoint.unlink()
    assert models_cli.ensure_model_available("custom") is False


def test_models_download_reports_missing_local_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``models download`` fails for a local model whose file is gone instead of
    reporting that there is nothing to do.
    """
    extra = tmp_path / "extra-models.json"
    extra.write_text(
        json.dumps(
            {
                "models": {
                    "custom": {
                        "architecture": "htdemucs",
                        "sources": ["vocals", "other"],
                        "config": {"sources": ["vocals", "other"]},
                        "checkpoint": {
                            "format": "safetensors",
                            "path": str(tmp_path / "gone.safetensors"),
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(extra))
    result = _invoke(["models", "download", "custom"])
    assert result.exit_code == 1
    assert "missing" in result.output


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


def test_auto_compile_times_a_forward_when_batch_size_was_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An explicit chunk batch size skips the sizing probe, so auto mode runs
    the timing probe itself instead of always keeping eager.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    enable_compile = Mock()
    separator = SimpleNamespace(
        _eager_probe_seconds=None, enable_compile=enable_compile
    )

    def probe() -> int:
        separator._eager_probe_seconds = 1.0
        return 1

    separator._measure_per_chunk_steady_bytes = probe
    monkeypatch.setattr(
        "unblend.cli.separate._compile_profile_key",
        lambda separator: ("bs_roformer", "fp16"),
    )
    monkeypatch.setattr(
        "unblend.cli.separate._estimate_compile_chunks",
        lambda separator, audio_files, *, shifts, split_overlap: (10_000, 60.0, 0),
    )

    assert _maybe_enable_auto_compile(
        separator, [Path("track.wav")], shifts=1, split_overlap=0.25
    )
    enable_compile.assert_called_once()


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
    """
    Stands in for Separator so the path pre-checks run without a model.
    """

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
    # The auto-compile policy runs on CUDA machines and needs a real model.
    monkeypatch.setattr(
        "unblend.cli.separate._maybe_enable_auto_compile", lambda *a, **k: False
    )


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


def test_separate_dotted_track_name_is_not_a_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A dot in the track name doesn't become the output container; the
    --format extension is appended instead.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    _stub_model_loading(monkeypatch)
    track = tmp_path / "Song feat. Artist.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(track)

    result = _invoke(["separate", str(track), "-o", str(tmp_path / "{track}_{stem}")])
    assert "Unsupported output format" not in result.output
    assert "Separated track will be stored" in result.output


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
    """
    NFC-equivalent output names are rejected before inference.
    """
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
    """
    Path-resolution failures are reported as CLI errors, not tracebacks.
    """
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
    # Rich wraps long paths, so compare with whitespace collapsed.
    output = " ".join(result.output.split())
    assert "Could not resolve planned output path" in output or (
        "isn't a writable folder" in output
    )
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
    layer = ModelRepository().list_models()["htdemucs_ft"]["members"][0]
    partial_layer = tmp_path / f"{layer['sha256'][:16]}.safetensors"
    stale_tmp = tmp_path / f"{STAGING_PREFIX}q1w2e3.tmp"
    partial_layer.write_bytes(b"x")
    stale_tmp.write_bytes(b"y")
    old = time.time() - STAGING_STALE_SECONDS - 1
    os.utime(stale_tmp, (old, old))

    result = _invoke(["models", "remove", "--all"])

    assert result.exit_code == 0
    assert not partial_layer.exists()
    assert not stale_tmp.exists()


def test_unknown_combine_mode_is_rejected_before_any_work() -> None:
    """
    ``--combine`` is checked against the implemented modes, listing them, so a
    typo fails at parse time instead of after a model download.
    """
    result = _invoke(["separate", "--combine", "telepathy", "track.wav"])

    assert result.exit_code != 0
    assert "not a known combine mode" in result.output
    assert "min_fft" in result.output


def test_model_names_include_locally_added_models() -> None:
    """
    ``--model`` accepts and completes every registered name, plus ``auto``.
    """
    from unblend.cli.utils import complete_model_name, validate_model_name

    assert validate_model_name("auto") == "auto"
    assert validate_model_name("scnet_xl_wide_v5") == "scnet_xl_wide_v5"
    assert "roformer_vocals_ensemble" in complete_model_name("roformer")

    with pytest.raises(typer.BadParameter, match="not a known model"):
        validate_model_name("no_such_model")


def test_chunk_batch_size_reaches_the_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``--chunk-batch-size`` is forwarded to ``Separator``, so the value
    ``unblend tune`` recommends is the value inference actually runs at.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    captured: dict[str, object] = {}

    class _RecordingSeparator(_StubSeparator):
        """
        Records the kwargs the CLI constructed it with.
        """

        def __init__(self, **kwargs: object) -> None:
            """
            Capture construction kwargs, then behave like the stub.

            :param kwargs: Keyword arguments the CLI passed.
            """
            captured.update(kwargs)
            super().__init__(**kwargs)

        def separate(self, **kwargs: object) -> object:
            """
            Return a result whose stems the export loop can iterate.

            :param kwargs: Ignored separation arguments.
            :return: An object exposing an empty ``sources`` mapping.
            """
            return SimpleNamespace(sources={})

    _stub_model_loading(monkeypatch)
    monkeypatch.setattr("unblend.cli.separate.Separator", _RecordingSeparator)

    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    result = _invoke(
        [
            "separate",
            str(wav_path),
            "--chunk-batch-size",
            "7",
            "-o",
            str(tmp_path / "out" / "{stem}.{ext}"),
        ]
    )

    assert result.exit_code == 0, result.output
    assert captured["chunk_batch_size"] == 7


def test_chunk_batch_size_defaults_to_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Omitting ``--chunk-batch-size`` passes ``None``, which is what selects
    ``Separator``'s memory-based auto sizing.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    captured: dict[str, object] = {}

    class _RecordingSeparator(_StubSeparator):
        """
        Records the kwargs the CLI constructed it with.
        """

        def __init__(self, **kwargs: object) -> None:
            """
            Capture construction kwargs, then behave like the stub.

            :param kwargs: Keyword arguments the CLI passed.
            """
            captured.update(kwargs)
            super().__init__(**kwargs)

        def separate(self, **kwargs: object) -> object:
            """
            Return a result whose stems the export loop can iterate.

            :param kwargs: Ignored separation arguments.
            :return: An object exposing an empty ``sources`` mapping.
            """
            return SimpleNamespace(sources={})

    _stub_model_loading(monkeypatch)
    monkeypatch.setattr("unblend.cli.separate.Separator", _RecordingSeparator)

    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    result = _invoke(
        ["separate", str(wav_path), "-o", str(tmp_path / "out" / "{stem}.{ext}")]
    )

    assert result.exit_code == 0, result.output
    assert captured["chunk_batch_size"] is None


def test_chunk_batch_size_rejects_out_of_range_values(tmp_path: Path) -> None:
    """
    The CLI bounds match ``Separator``'s own validation (1..1024), so a typo
    fails at parse time instead of after a model download.

    :param tmp_path: pytest temporary directory fixture
    """
    wav_path = tmp_path / "clip.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(wav_path)

    for bad in ("0", "2048"):
        result = _invoke(["separate", str(wav_path), "--chunk-batch-size", bad])
        assert result.exit_code != 0, f"{bad} should be rejected"


def test_separate_refuses_to_overwrite_an_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A template that resolves to one of the inputs is refused before anything
    is written, instead of overwriting that input and then separating it.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    _stub_model_loading(monkeypatch)
    a = tmp_path / "a.wav"
    stem = tmp_path / "a_vocals.wav"
    for path in (a, stem):
        AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(path)
    before = stem.read_bytes()
    result = _invoke(
        ["separate", str(a), str(stem), "-o", str(tmp_path / "{track}_{stem}.wav")]
    )
    assert result.exit_code == 1
    assert "overwrite input" in result.output
    assert stem.read_bytes() == before


def test_output_template_expands_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    ``~`` in ``-o`` is the home directory, not a literal ``./~``.

    :param monkeypatch: pytest monkeypatch fixture
    """
    from unblend.cli.utils import format_output_path

    path = format_output_path("~/x/{stem}.wav", "m", Path("t.wav"), "vocals")
    assert path == Path.home() / "x" / "vocals.wav"


def test_low_precision_on_cpu_fails_before_download(tmp_path: Path) -> None:
    """
    fp16/bf16 on CPU is rejected before any model is fetched.

    :param tmp_path: pytest temporary directory fixture
    """
    track = tmp_path / "a.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(track)
    result = _invoke(["separate", str(track), "-d", "cpu", "--precision", "fp16"])
    assert result.exit_code == 1
    assert "isn't supported on CPU" in result.output


def test_models_unregister_removes_the_entry(
    _isolate_default_models_file: Path,
) -> None:
    """
    ``models unregister`` drops a user entry from the file that defines it, and
    refuses built-in names.

    :param _isolate_default_models_file: the substituted default models file
    """
    import json

    path = _isolate_default_models_file
    entry = {
        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
        "sources": ["drums", "bass", "other", "vocals"],
    }
    path.write_text(json.dumps({"models": {"mine": entry, "other": entry}}))

    result = _invoke(["models", "unregister", "mine"])
    assert result.exit_code == 0, result.output
    import yaml

    assert "mine" not in yaml.safe_load(path.read_text())["models"]
    assert path.with_name(path.name + ".bak").exists()

    assert _invoke(["models", "unregister", "other"]).exit_code == 0
    assert yaml.safe_load(path.read_text())["models"] == {}, "an emptied file stays"

    assert _invoke(["models", "unregister", "htdemucs"]).exit_code == 1


def test_models_unregister_refuses_a_referenced_model_and_keeps_shared_weights(
    _isolate_default_models_file: Path, tmp_path: Path
) -> None:
    """
    Unregistering an entry another entry uses as a member is refused, and
    ``--delete-weights`` never deletes files another model still needs.

    :param _isolate_default_models_file: the substituted default models file
    :param tmp_path: pytest temporary directory fixture
    """
    import json

    from unblend.repo import ModelRepository

    path = _isolate_default_models_file
    # A single model (a member can't itself be an ensemble).
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    ensemble = {k: v for k, v in base.items() if k != "backend"}
    wrapper = {
        "members": [{"model": "inner"}, {"model": "htdemucs"}],
        "sources": ["drums", "bass", "other", "vocals"],
    }
    path.write_text(json.dumps({"models": {"inner": ensemble, "outer": wrapper}}))
    before = path.read_text()

    result = _invoke(["models", "unregister", "inner"])
    assert result.exit_code == 1
    assert "don't load without it" in result.output
    assert path.read_text() == before

    assert _invoke(["models", "unregister", "outer", "--delete-weights"]).exit_code == 0


def test_mixed_ensemble_with_missing_local_member_is_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An ensemble of a remote model and a local file that's gone is reported
    as missing by ``models list`` and fails ``models download``, instead of
    counting only its remote member.
    """
    import unblend.cli.models as models_cli
    from unblend.repo import ModelRepository

    base = ModelRepository(extra_models=[]).list_models()["htdemucs"]
    extra = tmp_path / "extra-models.json"
    extra.write_text(
        json.dumps(
            {
                "models": {
                    "mixed": {
                        "sources": base["sources"],
                        "members": [
                            {"model": "htdemucs"},
                            {
                                "architecture": "htdemucs",
                                "config": base["config"],
                                "checkpoint": {
                                    "format": "safetensors",
                                    "path": str(tmp_path / "gone.safetensors"),
                                },
                            },
                        ],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(extra))
    monkeypatch.setattr(
        models_cli,
        "_download_model_with_progress",
        lambda *args, **kwargs: pytest.fail("attempted a download"),
    )
    listed = _invoke(["models", "list"])
    assert "Local file missing" in listed.output
    result = _invoke(["models", "download", "mixed"])
    assert result.exit_code == 1
    assert "missing" in result.output


@pytest.mark.parametrize(
    "extra, match",
    [
        (["-d", "cuda"], "CUDA isn't available"),
        (["--combine", "max_fft"], "single member"),
    ],
)
def test_separate_checks_cheap_arguments_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], match: str
) -> None:
    """
    An unavailable device or ``--combine`` on a single model is refused before
    the model is downloaded, not after.
    """
    import unblend.cli.separate as separate_cli

    if extra[:2] == ["-d", "cuda"] and torch.cuda.is_available():
        pytest.skip("CUDA is available here")
    monkeypatch.setattr(
        separate_cli,
        "ensure_model_available",
        lambda *a, **k: pytest.fail("downloaded before checking arguments"),
    )
    track = tmp_path / "t.wav"
    track.write_bytes(b"RIFF")
    result = _invoke(["separate", "-m", "htdemucs", *extra, str(track)])
    assert result.exit_code == 1
    assert match in " ".join(result.output.split())


def test_tune_checks_the_device_before_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``tune -d cuda`` without CUDA stops before downloading any model.
    """
    import unblend.cli.tune as tune_cli

    if torch.cuda.is_available():
        pytest.skip("CUDA is available here")
    monkeypatch.setattr(
        tune_cli,
        "ensure_model_available",
        lambda *a, **k: pytest.fail("downloaded before checking the device"),
    )
    result = _invoke(["tune", "-m", "htdemucs", "-d", "cuda"])
    assert result.exit_code == 1
    assert "CUDA isn't available" in " ".join(result.output.split())


def test_unregister_keeps_weights_a_skipped_default_file_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    ``unregister --delete-weights`` keeps a file that a currently skipped
    (broken) default models file still names: fixing that file would
    otherwise find its weights gone.
    """
    weights = tmp_path / "shared.safetensors"
    weights.write_bytes(b"w")
    entry = {
        "architecture": "htdemucs",
        "sources": ["a", "b"],
        "config": {"sources": ["a", "b"]},
        "checkpoint": {"format": "safetensors", "path": str(weights)},
    }
    _isolate_default_models_file.write_text(
        json.dumps(
            {"models": {"keep": entry, "d2": {**entry, "architecture": "nonsense"}}}
        )
    )
    listed = tmp_path / "l.json"
    listed.write_text(json.dumps({"models": {"imp": entry}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    result = _invoke(["models", "unregister", "imp", "--delete-weights"])
    assert result.exit_code == 0, result.output
    assert weights.exists()


def test_separate_refuses_an_unwritable_output_folder_before_separating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An output template under a read-only folder fails before any separation
    runs, not after every track has been separated.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores permission bits")

    # The stub has no separate(): reaching separation would raise.
    _stub_model_loading(monkeypatch)
    track = tmp_path / "t.wav"
    track.write_bytes(b"RIFF")
    readonly = tmp_path / "ro"
    readonly.mkdir()
    os.chmod(readonly, 0o555)
    try:
        result = _invoke(
            [
                "separate",
                "-m",
                "htdemucs",
                "-d",
                "cpu",
                "-o",
                f"{readonly}/{{track}}/{{stem}}.wav",
                str(track),
            ]
        )
    finally:
        os.chmod(readonly, 0o755)
    assert result.exit_code == 1
    assert "Can't write" in " ".join(result.output.split())


def _local_entry(weights: Path) -> dict:
    """
    A minimal local entry pointing at ``weights``.

    :param weights: Weights file.
    :return: A registry entry.
    """
    return {
        "architecture": "htdemucs",
        "sources": ["a", "b"],
        "config": {"sources": ["a", "b"]},
        "checkpoint": {"format": "safetensors", "path": str(weights)},
    }


def test_delete_weights_keeps_files_an_unreadable_models_file_may_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    With a default models file that doesn't even parse, ``--delete-weights``
    keeps the files (the file may name them) and says why.
    """
    weights = tmp_path / "shared.safetensors"
    weights.write_bytes(b"w")
    _isolate_default_models_file.write_text("models: {keep: [unclosed\n")
    listed = tmp_path / "l.json"
    listed.write_text(json.dumps({"models": {"imp": _local_entry(weights)}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    result = _invoke(["models", "unregister", "imp", "--delete-weights"])
    assert result.exit_code == 0, result.output
    assert weights.exists()
    assert "can't be read" in " ".join(result.output.split())


def test_delete_weights_works_for_a_model_in_a_skipped_default_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    Unregistering the broken entry of a skipped default file with
    ``--delete-weights`` deletes its weights (they came from the entry, not
    from the registry that skipped the file), but never a cache file.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(cache))
    weights = tmp_path / "w3.safetensors"
    weights.write_bytes(b"w")
    broken = {**_local_entry(weights), "config": {"sources": ["x", "y"]}}
    cached = cache / "0123456789abcdef.safetensors"
    cached.write_bytes(b"c")
    _isolate_default_models_file.write_text(
        json.dumps({"models": {"m3": broken, "viacache": _local_entry(cached)}})
    )
    result = _invoke(["models", "unregister", "m3", "--delete-weights"])
    assert result.exit_code == 0, result.output
    assert not weights.exists()
    result = _invoke(["models", "unregister", "viacache", "--delete-weights"])
    assert result.exit_code == 0, result.output
    assert cached.exists()


@pytest.mark.parametrize(
    "args, match",
    [
        (["separate", "--isolate-stem", "foo"], "not found"),
        (["tune", "-m", "htdemucs", "-d", "cpu", "-p", "fp16"], "not supported on CPU"),
    ],
)
def test_cheap_refusals_happen_before_any_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str], match: str
) -> None:
    """
    An unknown ``--isolate-stem`` (auto model) and ``tune`` at fp16 on CPU are
    refused before a model is downloaded.
    """
    import unblend.cli.separate as separate_cli
    import unblend.cli.tune as tune_cli

    for module in (separate_cli, tune_cli):
        monkeypatch.setattr(
            module,
            "ensure_model_available",
            lambda *a, **k: pytest.fail("downloaded before a cheap check"),
        )
    if args[0] == "separate":
        track = tmp_path / "t.wav"
        track.write_bytes(b"RIFF")
        args = [*args, str(track)]
    result = _invoke(args)
    assert result.exit_code == 1
    assert match in " ".join(result.output.split())


@pytest.mark.parametrize(
    "output", ["/dev/null/x.onnx", "afile/../x.onnx", "dangling/x.onnx"]
)
def test_export_onnx_checks_the_output_folder_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    """
    An ``-o`` whose folder can't be written — under a file, or through a
    dangling symlink — is refused before the model is downloaded or traced.
    """
    pytest.importorskip("onnx")
    import unblend.cli.models as models_cli

    monkeypatch.setattr(
        models_cli,
        "ensure_model_available",
        lambda *a, **k: pytest.fail("downloaded before checking the output"),
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "afile").write_bytes(b"")
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    result = _invoke(["export-onnx", "-m", "htdemucs", "-o", output])
    assert result.exit_code == 1
    text = " ".join(result.output.split())
    assert "isn't a writable folder" in text or "not a folder" in text
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_delete_weights_removes_a_symlink_not_its_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An entry whose path is a symlink loses only the link; the file it points
    at (say, the original training checkpoint) stays.
    """
    real = tmp_path / "store" / "real.safetensors"
    real.parent.mkdir()
    real.write_bytes(b"w")
    link = tmp_path / "link.safetensors"
    link.symlink_to(real)
    listed = tmp_path / "l.json"
    listed.write_text(json.dumps({"models": {"imp": _local_entry(link)}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    result = _invoke(["models", "unregister", "imp", "--delete-weights"])
    assert result.exit_code == 0, result.output
    assert not link.is_symlink() and not link.exists()
    assert real.read_bytes() == b"w"


@pytest.mark.parametrize("case", ["duplicate-stems", "unwritable-output"])
def test_models_import_refuses_cheap_mistakes_before_converting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """
    Duplicate ``--stem`` names and an unwritable ``--output`` folder are
    refused before the checkpoint is converted.
    """
    if case == "unwritable-output" and os.geteuid() == 0:
        pytest.skip("root ignores permission bits")

    import unblend.importer as importer

    monkeypatch.setattr(
        importer,
        "import_checkpoint",
        lambda *a, **k: pytest.fail("converted before a cheap check"),
    )
    checkpoint = tmp_path / "c.ckpt"
    checkpoint.write_bytes(b"x")
    args = ["models", "import", str(checkpoint), "-n", "mine"]
    readonly = tmp_path / "ro"
    readonly.mkdir()
    if case == "duplicate-stems":
        args += ["--stem", "a", "--stem", "A"]
    else:
        os.chmod(readonly, 0o555)
        args += ["-o", str(readonly / "x.safetensors")]
    try:
        result = _invoke(args)
    finally:
        os.chmod(readonly, 0o755)
    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "must be unique" in output or "isn't a writable folder" in output


@pytest.mark.parametrize("output", ["", "FOLDER"])
def test_export_onnx_refuses_a_folder_or_empty_output_up_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    """
    ``-o`` naming an existing folder, or empty, is refused before the model is
    downloaded or traced.
    """
    pytest.importorskip("onnx")
    import unblend.cli.models as models_cli

    monkeypatch.setattr(
        models_cli,
        "ensure_model_available",
        lambda *a, **k: pytest.fail("downloaded before checking the output"),
    )
    target = str(tmp_path) if output == "FOLDER" else output
    result = _invoke(["export-onnx", "-m", "htdemucs", "-o", target])
    assert result.exit_code == 1
    output_text = " ".join(result.output.split())
    assert "is a folder" in output_text or "is empty" in output_text


@pytest.mark.parametrize("case", ["unwritable", "folder-at-path"])
def test_separate_output_problems_are_found_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """
    Output-path problems (an unwritable folder, a folder where an output file
    goes) are refused before the model is downloaded.
    """
    if case == "unwritable" and os.geteuid() == 0:
        pytest.skip("root ignores permission bits")

    import unblend.cli.separate as separate_cli

    monkeypatch.setattr(
        separate_cli,
        "ensure_model_available",
        lambda *a, **k: pytest.fail("downloaded before checking outputs"),
    )
    track = tmp_path / "t.wav"
    track.write_bytes(b"RIFF")
    out = tmp_path / "out"
    out.mkdir()
    if case == "unwritable":
        os.chmod(out, 0o555)
        template = f"{out}/{{stem}}.wav"
    else:
        (out / "vocals.wav").mkdir()
        template = f"{out}/{{stem}}.wav"
    try:
        result = _invoke(
            ["separate", "-m", "htdemucs", "-d", "cpu", "-o", template, str(track)]
        )
    finally:
        os.chmod(out, 0o755)
    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "isn't a writable folder" in output or "is a folder" in output


def test_separate_self_looping_output_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An output file that is a symlink to itself fails the strict resolve with
    "Could not resolve planned output path", before any download.
    """
    import unblend.cli.separate as separate_cli

    monkeypatch.setattr(
        separate_cli,
        "ensure_model_available",
        lambda *a, **k: pytest.fail("downloaded before checking outputs"),
    )
    track = tmp_path / "t.wav"
    track.write_bytes(b"RIFF")
    out = tmp_path / "out"
    out.mkdir()
    (out / "drums.wav").symlink_to(out / "drums.wav")
    result = _invoke(
        [
            "separate",
            "-m",
            "htdemucs",
            "-d",
            "cpu",
            "-o",
            f"{out}/{{stem}}.wav",
            str(track),
        ]
    )
    assert result.exit_code == 1
    assert "Could not resolve planned output path" in " ".join(result.output.split())


def test_delete_weights_only_touches_the_files_the_registry_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``--delete-weights`` deletes exactly what the registry reads for the
    entry: a stray ``path`` under ``config`` is not a weights file, and a
    ``link/../x`` path goes through the symlinked folder, as the registry
    reads it, rather than collapsing ``..`` first.
    """
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    elsewhere = tmp_path / "elsewhere" / "deep"
    elsewhere.mkdir(parents=True)
    real = tmp_path / "elsewhere" / "x.safetensors"
    real.write_bytes(b"w")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "linkdir").symlink_to(elsewhere)
    decoy = models_dir / "x.safetensors"
    decoy.write_text("unrelated")
    entry = _local_entry(real)
    entry["checkpoint"]["path"] = "linkdir/../x.safetensors"
    entry["config"] = {**entry["config"], "path": str(victim)}
    listed = models_dir / "l.json"
    listed.write_text(json.dumps({"models": {"vv": entry}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    result = _invoke(["models", "unregister", "vv", "--delete-weights"])
    assert result.exit_code == 0, result.output
    assert victim.read_text() == "keep me"
    assert decoy.read_text() == "unrelated"
    assert not real.exists()


def test_unregister_ignores_a_listed_file_that_does_not_exist_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A listed models file that doesn't exist yet is skipped, as the registry
    skips it, so ``unregister`` still says a built-in model is built in.
    """
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(tmp_path / "notyet.yaml"))
    result = _invoke(["models", "unregister", "htdemucs"])
    assert result.exit_code == 1
    assert "built in" in " ".join(result.output.split())


def test_models_commands_count_a_shared_checkpoint_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An ensemble that uses one checkpoint for two members shows one file of
    one checkpoint's size, and downloads it once.
    """
    extra = tmp_path / "models.json"
    extra.write_text(
        json.dumps(
            {
                "models": {
                    "twin": {
                        "sources": ["drums", "bass", "other", "vocals"],
                        "combine": "weighted_mean",
                        "members": [{"model": "htdemucs"}, {"model": "htdemucs"}],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(extra))
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    # The consoles read COLUMNS once, at import; widen them directly.
    import unblend.cli.utils as cli_utils

    monkeypatch.setattr(cli_utils.console, "width", 200)
    repo = ModelRepository()
    assert repo.required_files("twin") == repo.required_files("htdemucs")
    assert len(repo.weight_files("twin")) == 1

    app = build_app()
    listing = runner.invoke(app, ["models", "list"])
    assert listing.exit_code == 0, listing.output
    sizes = {
        line.split("│")[1].strip(): line.split("│")[4].strip()
        for line in listing.output.splitlines()
        if line.count("│") >= 5
    }
    assert sizes["twin"] == sizes["htdemucs"]
    info = runner.invoke(app, ["models", "info", "twin"])
    assert info.exit_code == 0, info.output
    assert " ".join(info.output.split()).count("Files: 1") == 1


def test_models_commands_survive_a_symlink_loop_in_a_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A local weights path through a symlink loop shows as missing instead of
    crashing ``models list`` and ``models info``.
    """
    (tmp_path / "loop").symlink_to("loop")
    extra = tmp_path / "models.json"
    extra.write_text(
        json.dumps(
            {
                "models": {
                    "loopy": {
                        "architecture": "scnet",
                        "sources": ["vocals", "other"],
                        "samplerate": 44100,
                        "segment_samples": 44100,
                        "config": {"dims": [4, 8]},
                        "checkpoint": {
                            "format": "safetensors",
                            "path": "loop/x.safetensors",
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(extra))
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    # The consoles read COLUMNS once, at import; widen them directly.
    import unblend.cli.utils as cli_utils

    monkeypatch.setattr(cli_utils.console, "width", 200)
    app = build_app()
    listing = runner.invoke(app, ["models", "list"])
    assert listing.exit_code == 0, listing.output
    assert "Missing" in next(
        line for line in listing.output.splitlines() if "loopy" in line
    )
    info = runner.invoke(app, ["models", "info", "loopy"])
    assert info.exit_code == 0, info.output


@pytest.mark.parametrize("suffix", ["/", "/."])
def test_models_import_output_ending_in_a_separator_names_a_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """
    ``-o newdir/`` writes ``newdir/NAME.safetensors`` even before the folder
    exists, instead of a file called ``newdir``.
    """
    import unblend.importer as importer
    from unblend.exceptions import UnblendError

    seen: list[Path] = []

    def record(checkpoint: Path, artifact: Path, **kwargs: object) -> None:
        seen.append(artifact)
        raise UnblendError("stop here")

    monkeypatch.setattr(importer, "import_checkpoint", record)
    monkeypatch.setenv("HOME", str(tmp_path))
    checkpoint = tmp_path / "c.ckpt"
    checkpoint.write_bytes(b"x")
    result = _invoke(
        ["models", "import", str(checkpoint), "-n", "mine"]
        + ["-o", f"{tmp_path}/newdir{suffix}"]
    )
    assert result.exit_code == 1
    assert seen == [tmp_path / "newdir" / "mine.safetensors"]


def test_separate_refuses_stem_names_too_long_for_the_os(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A long input name can make ``{track}_{stem}`` exceed 255 bytes; that's a
    clear error before inference, not a traceback from ``Path.is_dir()``.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    _stub_model_loading(monkeypatch)
    track = tmp_path / ("b" * 245 + ".wav")
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(track)
    result = _invoke(
        ["separate", str(track), "-o", str(tmp_path / "out" / "{track}_{stem}.wav")]
    )
    assert result.exit_code == 1
    assert "too long for a file or folder name" in " ".join(result.output.split())
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_export_onnx_takes_an_unknown_tilde_user_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``-o ~nosuchuser/x.onnx`` is a literal relative path (as in ``separate``),
    not a ``RuntimeError`` from ``Path.expanduser()``.
    """
    pytest.importorskip("onnx")
    import unblend.cli.models as models_cli

    reached: list[str] = []
    monkeypatch.setattr(
        models_cli,
        "ensure_model_available",
        lambda model, *a, **k: reached.append(model) or False,
    )
    monkeypatch.chdir(tmp_path)
    result = _invoke(["export-onnx", "-m", "htdemucs", "-o", "~nosuchuserzz/x.onnx"])
    assert reached == ["htdemucs"]
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_delete_weights_keeps_files_a_models_file_in_a_locked_folder_may_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    A listed models file in a folder you can't enter isn't "missing": it may
    name the weights, so ``--delete-weights`` keeps them.
    """
    if os.geteuid() == 0:
        pytest.skip("root enters any folder")
    weights = tmp_path / "shared.safetensors"
    weights.write_bytes(b"w")
    _isolate_default_models_file.write_text(
        json.dumps({"models": {"mine": _local_entry(weights)}})
    )
    locked = tmp_path / "locked"
    locked.mkdir()
    other = locked / "m.json"
    other.write_text(json.dumps({"models": {"theirs": _local_entry(weights)}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(other))
    os.chmod(locked, 0)
    try:
        result = _invoke(["models", "unregister", "mine", "--delete-weights"])
    finally:
        os.chmod(locked, 0o755)
    assert weights.exists(), result.output


def test_models_list_shows_an_empty_license_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``license:`` with nothing after it is null; ``models list`` and ``models
    info`` show it as unknown rather than crashing.
    """
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"w")
    listed = tmp_path / "l.json"
    listed.write_text(
        json.dumps({"models": {"nolic": {**_local_entry(weights), "license": None}}})
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    for args in (["models", "list"], ["models", "info", "nolic"]):
        result = _invoke(args)
        assert result.exit_code == 0, result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)


def test_an_unknown_tilde_user_output_root_is_still_excluded_from_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``-o ~nosuchuser/{track}/...`` writes to a literal ``~nosuchuser`` folder;
    scanning ``.`` must skip it, or a rerun separates its own stems.
    """
    from unblend.cli.separate import _static_output_root

    monkeypatch.chdir(tmp_path)
    root = _static_output_root("~nosuchuserzz/{track}/{stem}.{ext}")
    assert root == (tmp_path / "~nosuchuserzz").resolve()


def test_a_symlink_loop_in_extra_models_doesnt_crash_unregister(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    ``Path.resolve()`` raises on a symlink loop (Python 3.10–3.12); the
    registry skips such a listed file, and unregister must not crash on it.
    """
    loop = tmp_path / "loop.yaml"
    loop.symlink_to(loop)
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(loop))
    _isolate_default_models_file.write_text(json.dumps({"models": {}}))
    result = _invoke(["models", "unregister", "nothing"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_delete_weights_reports_weights_it_cannot_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Weights in a folder you can't enter aren't silently skipped: the output
    says they weren't deleted.
    """
    if os.geteuid() == 0:
        pytest.skip("root enters any folder")
    locked = tmp_path / "locked"
    locked.mkdir()
    weights = locked / "w.safetensors"
    weights.write_bytes(b"w")
    listed = tmp_path / "l.json"
    listed.write_text(json.dumps({"models": {"ll": _local_entry(weights)}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    os.chmod(locked, 0)
    try:
        result = _invoke(["models", "unregister", "ll", "--delete-weights"])
    finally:
        os.chmod(locked, 0o755)
    assert weights.exists()
    assert "Couldn't delete" in " ".join(result.output.split())


def test_separate_refuses_a_combine_the_weights_forbid_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``--combine median_wave`` on an ensemble with blending weights is refused
    from the registry entry, before its members download.
    """
    import unblend.cli.models as models_cli

    monkeypatch.setattr(
        models_cli,
        "ensure_model_available",
        lambda *a, **k: pytest.fail("downloaded before checking --combine"),
    )
    listed = tmp_path / "m.json"
    listed.write_text(
        json.dumps(
            {
                "models": {
                    "blend": {
                        "sources": ["drums", "bass", "other", "vocals"],
                        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
                        "weights": [[0.5, 1, 1, 1], [0.5, 1, 1, 1]],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    track = tmp_path / "t.wav"
    AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100).to_file(track)
    result = _invoke(
        [
            "separate",
            str(track),
            "-m",
            "blend",
            "--combine",
            "median_wave",
            "-o",
            str(tmp_path / "out" / "{stem}.{ext}"),
        ]
    )
    assert result.exit_code == 1
    assert "participation mask" in " ".join(result.output.split())


def test_certainly_missing_treats_a_nul_path_as_missing() -> None:
    """
    ``os.stat`` raises ``ValueError`` on an embedded NUL; no file can have
    such a name, so it's missing (not a crash after unregistering).
    """
    from unblend.cli.models import _certainly_missing

    assert _certainly_missing(Path("w\0.safetensors"))


def test_unregister_survives_a_nul_path_in_another_models_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    A NUL in another (skipped) models file's paths makes that file
    unreadable for the in-use check: the weights are kept, with no crash.
    """
    weights = tmp_path / "good.safetensors"
    weights.write_bytes(b"w")
    _isolate_default_models_file.write_text(
        json.dumps({"models": {"nulm": _local_entry(Path("x\0y.safetensors"))}})
    )
    listed = tmp_path / "l.json"
    listed.write_text(json.dumps({"models": {"good": _local_entry(weights)}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    result = _invoke(["models", "unregister", "good", "--delete-weights"])
    assert result.exit_code == 0, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert weights.exists()


def test_isolate_stem_matches_across_unicode_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``--isolate-stem`` typed in another Unicode form (NFD ``café`` for an NFC
    ``Café`` stem) matches, as the registry treats the two as one name.
    """
    import unblend.cli.separate as separate_cli

    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"w")
    stems = ["Café", "b"]
    entry = {**_local_entry(weights), "sources": stems, "config": {"sources": stems}}
    listed = tmp_path / "l.json"
    listed.write_text(json.dumps({"models": {"cafe": entry}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    loaded = []

    def stop(*args: object, **kwargs: object) -> None:
        loaded.append(kwargs.get("only_load"))
        raise typer.Exit(3)

    monkeypatch.setattr(separate_cli, "ensure_model_available", stop)
    track = tmp_path / "t.wav"
    track.write_bytes(b"RIFF")
    result = _invoke(["separate", "-m", "cafe", "--isolate-stem", "café", str(track)])
    assert "not found" not in result.output, result.output
    # Mapped to the registry's spelling before the model is fetched and loaded.
    assert loaded == ["Caf\u00e9"], result.output
