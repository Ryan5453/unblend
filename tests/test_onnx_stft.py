"""
Guards the ONNX-export path against drifting from the model's own processing.

``compute_stft_for_export`` reimplements ``HTDemucs._spec`` for the traced graph,
and ``HTDemucsONNXWrapper`` reimplements the CaC packing / normalization /
de-interleave around ``forward_core``; if either ever diverges, exported models
would silently produce wrong output. ``_spec`` only reads
``self.hop_length``/``self.nfft`` (no submodules), so the STFT test exercises
the real method via a lightweight stand-in; the end-to-end parity test builds a
tiny random-init ``HTDemucs`` instead.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unblend.htdemucs import HTDemucs
from unblend.onnx import (
    HTDemucsONNXWrapper,
    _atomic_onnx_path,
    compute_stft_for_export,
)


def test_atomic_onnx_output_preserves_destination_on_failure(tmp_path) -> None:
    """A failed export removes staging bytes and leaves prior output intact."""
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="export failed"):
        with _atomic_onnx_path(str(destination)) as staging:
            assert os.path.dirname(staging) == str(tmp_path)
            with open(staging, "wb") as file:
                file.write(b"partial")
            raise RuntimeError("export failed")
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".model.onnx.*.tmp.onnx"))


def test_atomic_onnx_output_replaces_symlink_not_target(tmp_path) -> None:
    """Atomic publication replaces a symlink entry without following it."""
    target = tmp_path / "target.onnx"
    target.write_bytes(b"target")
    destination = tmp_path / "model.onnx"
    destination.symlink_to(target)
    with _atomic_onnx_path(str(destination)) as staging:
        with open(staging, "wb") as file:
            file.write(b"published")
    assert not destination.is_symlink()
    assert destination.read_bytes() == b"published"
    assert target.read_bytes() == b"target"


@pytest.mark.parametrize("filename", ["model.onnx", "model[1].onnx", "model?.onnx"])
def test_atomic_onnx_output_rejects_and_cleans_external_data(
    tmp_path, filename: str
) -> None:
    """Sidecar detection treats legal output metacharacters literally."""
    destination = tmp_path / filename
    destination.write_bytes(b"previous")
    staging_stem = ""

    with pytest.raises(RuntimeError, match="External-data ONNX exports"):
        with _atomic_onnx_path(str(destination)) as staging:
            staging_stem = Path(staging).stem
            Path(staging).write_bytes(b"graph")
            Path(f"{staging}.data").write_bytes(b"weights")
            Path(staging).with_suffix(".data").mkdir()

    assert destination.read_bytes() == b"previous"
    assert not any(
        candidate.name.startswith(staging_stem) for candidate in tmp_path.iterdir()
    )


def test_ht_export_failure_preserves_existing_destination(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTDemucs export path publishes only after checker validation."""
    onnx = pytest.importorskip("onnx")
    model = HTDemucs(
        sources=["a", "b"],
        audio_channels=1,
        channels=8,
        nfft=2048,
        t_layers=2,
        t_heads=2,
        samplerate=8000,
        segment=1,
    ).eval()

    class FakeRepository:
        """Return the tiny local model without cache or network access."""

        def get_model(self, _name: str) -> HTDemucs:
            return model

    def fake_export(*_args: object, **_kwargs: object) -> None:
        path = _args[2]
        value = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
        graph = onnx.helper.make_graph([], "test", [value], [value])
        onnx.save(onnx.helper.make_model(graph), path)

    monkeypatch.setattr("unblend.onnx.ModelRepository", FakeRepository)
    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("invalid graph")),
    )
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous")

    from unblend.onnx import export_to_onnx

    with pytest.raises(RuntimeError, match="invalid graph"):
        export_to_onnx("tiny", str(destination))
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".model.onnx.*.tmp.onnx"))


def test_export_stft_matches_model_spec() -> None:
    """
    The exported STFT equals ``HTDemucs._spec`` for the same nfft/hop.
    """
    nfft = 4096
    hop_length = nfft // 4
    audio = torch.randn(1, 2, 44100)

    real, imag = compute_stft_for_export(audio, nfft, hop_length)

    stand_in = SimpleNamespace(hop_length=hop_length, nfft=nfft)
    z = HTDemucs._spec(stand_in, audio)

    assert z.shape == real.shape
    assert torch.allclose(z.real, real, atol=1e-5)
    assert torch.allclose(z.imag, imag, atol=1e-5)


def test_onnx_wrapper_matches_reference_forward() -> None:
    """
    ``HTDemucsONNXWrapper`` plus runtime pre/post-processing reproduces
    ``HTDemucs.forward``.

    The wrapper hand-reimplements the CaC packing, input normalization, and
    real/imag de-interleave that ``HTDemucs.forward`` does around
    ``forward_core``; this checks the whole pipeline end to end on a tiny
    random-init model (CPU, FP32 — no export needed).
    """
    torch.manual_seed(0)
    model = HTDemucs(
        sources=["drums", "bass"],
        audio_channels=1,
        channels=8,  # smallest that keeps the DConv hidden dim (C/8) nonzero
        # Smallest nfft whose frequency dim survives all 4 stride-4 layers
        # without collapsing (1024 -> 256 -> 64 -> 16), matching the real
        # htdemucs geometry (no time/freq branch merge).
        nfft=2048,
        t_layers=2,
        t_heads=2,
        samplerate=8000,
        segment=1,
    ).eval()
    wrapper = HTDemucsONNXWrapper(model).eval()

    # Exactly the training length, like the traced graph requires.
    samples = int(model.max_allowed_segment * model.samplerate)
    mix = torch.randn(1, model.audio_channels, samples)

    with torch.no_grad():
        ref = model(mix)
        spec_real, spec_imag = compute_stft_for_export(
            mix, model.nfft, model.hop_length
        )
        out_real, out_imag, out_wave = wrapper(spec_real, spec_imag, mix)

    # Runtime postprocessing (see onnx.md): recombine real/imag, iSTFT, and
    # sum the two branches. ``_ispec`` is the model's own inverse transform.
    zout = torch.complex(out_real, out_imag)
    freq_audio = model._ispec(zout, samples)
    out = freq_audio + out_wave

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)
