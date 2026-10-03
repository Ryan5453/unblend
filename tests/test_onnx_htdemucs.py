"""
End-to-end HTDemucs ONNX export through the public ``export_to_onnx`` path.

A tiny randomly initialised HTDemucs is registered through a temporary
extra-models file, so the registry lookup, precision resolution, opset
selection and weight-storage rewrite all run exactly as they would for the
shipped checkpoints, just on a graph small enough to trace in seconds.
"""

import json
from pathlib import Path

import pytest
import torch

MODEL_NAME = "tiny_htdemucs_onnx_test"
SOURCES = ["drums", "bass", "other", "vocals"]
SAMPLERATE = 8000

# Small enough to trace quickly, but still a full hybrid model: two encoder
# levels per branch and one cross-domain transformer layer.
CONFIG = {
    "sources": SOURCES,
    "audio_channels": 2,
    "samplerate": SAMPLERATE,
    "segment": 0.5,
    "nfft": 512,
    "depth": 2,
    "channels": 8,
    "t_layers": 1,
    "cac": True,
}


@pytest.fixture
def tiny_htdemucs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """
    Serialise a tiny HTDemucs and register it via ``UNBLEND_EXTRA_MODELS``.

    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch fixture.
    :return: The registered model name.
    """
    from safetensors.torch import save_file

    from unblend.htdemucs import HTDemucs

    torch.manual_seed(0)
    model = HTDemucs(**CONFIG)
    weights = tmp_path / f"{MODEL_NAME}.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(weights))
    models_file = tmp_path / "extra.json"
    models_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    MODEL_NAME: {
                        "architecture": "htdemucs",
                        "license": "unknown",
                        "sources": SOURCES,
                        "config": CONFIG,
                        "checkpoint": {"format": "safetensors", "path": str(weights)},
                    }
                },
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(models_file))
    return MODEL_NAME


@pytest.mark.parametrize("precision", ["fp32", "fp16", "fp8_e4m3"])
def test_htdemucs_export_to_onnx(
    tiny_htdemucs: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
) -> None:
    """
    Every storage precision yields a checker-valid graph at a sufficient opset
    that onnxruntime loads; the fp32 graph matches the PyTorch wrapper.

    :param tiny_htdemucs: Registered model name.
    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch fixture.
    :param precision: Export precision under test.
    """
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from unblend.onnx import (
        HTDemucsONNXWrapper,
        compute_stft_for_export,
        export_to_onnx,
    )
    from unblend.repo import ModelRepository

    # The saved opset alone can't tell whether tracing itself ran at the
    # storage type's minimum, so record what the tracer was asked for.
    traced_opsets: list[int] = []
    real_export = torch.onnx.export

    def spy_export(*args, **kwargs):
        traced_opsets.append(kwargs["opset_version"])
        return real_export(*args, **kwargs)

    monkeypatch.setattr(torch.onnx, "export", spy_export)

    path = export_to_onnx(
        tiny_htdemucs,
        output_path=str(tmp_path / f"htdemucs_{precision}.onnx"),
        precision=precision,
    )

    minimum = 19 if precision.startswith("fp8") else 17
    assert traced_opsets and all(v >= minimum for v in traced_opsets)

    exported = onnx.load(path)
    onnx.checker.check_model(exported)
    opset = next(
        o.version for o in exported.opset_import if o.domain in ("", "ai.onnx")
    )
    # fp8 tensor types only exist from opset 19; the exporter must have traced
    # there rather than leaving the default 17.
    assert opset >= minimum
    metadata = {p.key: p.value for p in exported.metadata_props}
    assert metadata["model_family"] == "demucs"
    assert metadata["weight_precision"] == precision
    segment_samples = int(metadata["segment_samples"])
    assert segment_samples == int(CONFIG["segment"] * SAMPLERATE)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    torch.manual_seed(1)
    # Batch 2 exercises the dynamic batch axis.
    audio = torch.randn(2, 2, segment_samples)
    spec_real, spec_imag = compute_stft_for_export(
        audio, CONFIG["nfft"], CONFIG["nfft"] // 4
    )
    feeds = {
        "spec_real": spec_real.numpy(),
        "spec_imag": spec_imag.numpy(),
        "audio": audio.numpy(),
    }
    outputs = session.run(None, feeds)
    assert all(torch.isfinite(torch.from_numpy(out)).all() for out in outputs)

    if precision.startswith("fp8"):
        # Scaled fp8 weights: each Cast is followed by its rescale.
        assert any(node.name.endswith("_rescale") for node in exported.graph.node)

    model = ModelRepository().get_model(tiny_htdemucs).eval()
    wrapper = HTDemucsONNXWrapper(model).eval()
    with torch.no_grad():
        expected = wrapper(spec_real, spec_imag, audio)
    if precision != "fp32":
        wave, reference = torch.from_numpy(outputs[2]), expected[2]
        snr = 10 * torch.log10(reference.pow(2).sum() / (wave - reference).pow(2).sum())
        # Scaled e4m3 gives ~32 dB on this model, fp16 ~77 dB.
        assert snr > (25 if precision.startswith("fp8") else 60), float(snr)
        return
    for name, actual, reference in zip(
        ("out_spec_real", "out_spec_imag", "out_wave"), outputs, expected
    ):
        assert actual.shape == tuple(reference.shape), name
        assert torch.allclose(
            torch.from_numpy(actual), reference, atol=1e-4, rtol=1e-4
        ), name


def test_export_refuses_output_paths_that_cant_name_a_file(
    tiny_htdemucs: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An explicit folder or empty path is refused before the model is traced,
    and so is a folder sitting at the default name, which depends on the
    checkpoint's precision and is known only once it is loaded.

    :param tiny_htdemucs: Registered model name.
    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch fixture.
    """
    pytest.importorskip("onnx")
    from unblend.exceptions import ValidationError
    from unblend.onnx import export_to_onnx

    monkeypatch.setattr(
        torch.onnx, "export", lambda *a, **k: pytest.fail("traced before checking")
    )
    (tmp_path / "afile").write_bytes(b"")
    for bad in (
        "",
        str(tmp_path),
        f"{tmp_path}/",
        f"{tmp_path}/nope/..",
        "nope/.",
        f"{tmp_path}/afile/x.onnx",
    ):
        with pytest.raises(ValidationError, match="doesn't name a file|not a folder"):
            export_to_onnx(tiny_htdemucs, output_path=bad)
    monkeypatch.chdir(tmp_path)
    (tmp_path / f"{tiny_htdemucs}_fp32.onnx").mkdir()
    with pytest.raises(ValidationError, match="doesn't name a file"):
        export_to_onnx(tiny_htdemucs)


def test_staging_follows_a_symlink_before_dotdot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    For ``link/../x.onnx`` the file lands beside the link's target, and so
    does its staging copy, so the final rename stays in one folder.

    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch fixture.
    """
    from unblend.onnx import _atomic_onnx_path

    monkeypatch.chdir(tmp_path)
    (tmp_path / "other" / "inner").mkdir(parents=True)
    (tmp_path / "link").symlink_to("other/inner")
    with _atomic_onnx_path("link/../x.onnx") as staging:
        assert Path(staging).resolve().parent == (tmp_path / "other").resolve()
        Path(staging).write_bytes(b"onnx")
    assert (tmp_path / "other" / "x.onnx").read_bytes() == b"onnx"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link", "other"]


def test_output_names_leave_room_for_the_staging_copy(tmp_path: Path) -> None:
    """
    The staging copy adds 19 to the name, so the longest accepted name is 19
    short of the platform limit (UTF-16 units on macOS, bytes on Linux) — and
    one that long really writes; one unit more is refused up front.

    :param tmp_path: pytest temporary directory.
    """
    from unblend._paths import NAME_MAX, name_length
    from unblend.exceptions import ValidationError
    from unblend.onnx import _atomic_onnx_path, _check_output_path

    for char in ("a", "é"):
        stem = char * ((NAME_MAX - 19 - 5) // name_length(char))
        name = stem + ".onnx"
        _check_output_path(str(tmp_path / name))
        with _atomic_onnx_path(str(tmp_path / name)) as staging:
            Path(staging).write_bytes(b"onnx")
        assert (tmp_path / name).read_bytes() == b"onnx"
    with pytest.raises(ValidationError, match="too long to write"):
        _check_output_path(str(tmp_path / ("a" * (NAME_MAX - 19 - 4) + ".onnx")))


def test_names_too_long_for_the_os_are_validation_errors(tmp_path: Path) -> None:
    """
    Names the OS would refuse as too long, a NUL, or an unencodable name give
    a ``ValidationError``, not an ``OSError``/``ValueError`` from the
    filesystem (``Path.is_dir()`` raises on a too-long name).

    :param tmp_path: pytest temporary directory.
    """
    from unblend.exceptions import ValidationError
    from unblend.onnx import _check_output_path

    for bad in ("a" * 300 + ".onnx", "d" * 300 + "/x.onnx"):
        with pytest.raises(ValidationError, match="too long to write"):
            _check_output_path(str(tmp_path / bad))
    import sys

    unencodable = ["a\0b.onnx", "\ud800.onnx"]
    if sys.platform == "darwin":
        # Invalid UTF-8 from argv (surrogate-escaped): legal on Linux, but APFS
        # refuses it.
        unencodable.append("x\udcff.onnx")
    for bad in unencodable:
        with pytest.raises(ValidationError, match="NUL|encoded"):
            _check_output_path(bad)


def test_a_character_the_filesystem_refuses_names_the_users_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    APFS refuses some valid UTF-8 (unassigned code points) with EILSEQ; the
    error names the requested path, not the hidden staging file.

    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch fixture.
    """
    import errno
    import tempfile

    from unblend.exceptions import ValidationError
    from unblend.onnx import _atomic_onnx_path

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EILSEQ, "Illegal byte sequence")

    monkeypatch.setattr(tempfile, "mkstemp", refuse)
    target = str(tmp_path / "a͸.onnx")
    with pytest.raises(ValidationError, match="can't store") as info:
        with _atomic_onnx_path(target):
            pass
    assert repr(target) in str(info.value)
    assert ".tmp.onnx" not in str(info.value)


def test_a_folder_name_the_filesystem_refuses_names_the_users_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    EILSEQ from creating a new output folder gets the same clear error as
    one from the staging file.

    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch fixture.
    """
    import errno

    from unblend.exceptions import ValidationError
    from unblend.onnx import _atomic_onnx_path

    def refuse(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError(errno.EILSEQ, "Illegal byte sequence")

    monkeypatch.setattr(Path, "mkdir", refuse)
    with pytest.raises(ValidationError, match="can't store"):
        with _atomic_onnx_path(str(tmp_path / "d͸" / "x.onnx")):
            pass
