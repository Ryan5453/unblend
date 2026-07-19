"""
Guards the RoFormer ONNX-export path against drifting from the models' own
processing.

``RoformerONNXWrapper`` reimplements the spec-side pipeline (channel
interleave, mel gather, complex mask multiply in real arithmetic, DC zeroing)
and — for Mel-Band — replaces the overlapping-band scatter-average with a
constant averaging-matrix MatMul. The wrapper parity tests run with plain
torch; the export/onnxruntime tests skip unless the ``onnx`` extra (plus
``onnxruntime``) is installed, mirroring the CI onnx job.
"""

import pytest
import torch

from unblend.onnx import RoformerONNXWrapper, compute_roformer_stft_for_export
from unblend.roformer import BSRoformer, MelBandRoformer

SR = 44100
N_FFT, HOP = 2048, 512


def _bs() -> BSRoformer:
    """
    Build a tiny BS-RoFormer for export tests.

    :return: A ``BSRoformer`` in eval mode.
    """
    return BSRoformer(
        dim=16,
        depth=1,
        stereo=True,
        num_stems=2,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=8,
        heads=2,
        stft_n_fft=N_FFT,
        stft_hop_length=HOP,
    ).eval()


def _mel() -> MelBandRoformer:
    """
    Build a tiny Mel-Band RoFormer for export tests.

    :return: A ``MelBandRoformer`` in eval mode.
    """
    return MelBandRoformer(
        dim=16,
        depth=1,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        num_bands=60,
        dim_head=8,
        heads=2,
        stft_n_fft=N_FFT,
        stft_hop_length=HOP,
    ).eval()


def _roundtrip_through_wrapper(model, audio: torch.Tensor) -> torch.Tensor:
    """
    Run ``audio`` through the export wrapper plus a torch iSTFT — the exact
    client-side pipeline an exported model runs in.

    :param model: RoFormer model to wrap.
    :param audio: Mixture ``[B, C, samples]``.
    :return: Reconstructed stems ``[B, num_stems, C, samples]``.
    """
    stft = model.stft_kwargs
    wrapper = RoformerONNXWrapper(model).eval()
    spec_real, spec_imag = compute_roformer_stft_for_export(
        audio,
        n_fft=stft["n_fft"],
        hop_length=stft["hop_length"],
        win_length=stft["win_length"],
        normalized=stft["normalized"],
    )
    with torch.no_grad():
        out_real, out_imag = wrapper(spec_real, spec_imag)
    batch, stems, channels, n_freq, n_frames = out_real.shape
    z = torch.complex(out_real, out_imag).view(-1, n_freq, n_frames)
    window = torch.hann_window(stft["win_length"])
    recon = torch.istft(
        z,
        n_fft=stft["n_fft"],
        hop_length=stft["hop_length"],
        win_length=stft["win_length"],
        window=window,
        normalized=stft["normalized"],
        length=audio.shape[-1],
    )
    return recon.view(batch, stems, channels, -1)


@pytest.mark.parametrize("builder", [_bs, _mel], ids=["bs", "mel"])
def test_wrapper_matches_model_forward(builder) -> None:
    """
    Wrapper + client-side iSTFT reproduces the model's own forward output —
    the spec-in/spec-out export boundary is lossless. For Mel-Band this also
    proves the averaging-matrix MatMul equals the scatter-average.
    """
    torch.manual_seed(0)
    model = builder()
    # Registry models are always configured for inference (this also makes
    # Mel-Band return input-length output, as every real caller sees it).
    model.configure_inference(
        sources=(["vocals", "other"] if model.num_stems == 1 else ["a", "b"]),
        samplerate=SR,
        segment_samples=SR,
    )
    audio = torch.randn(1, 2, SR)
    recon = _roundtrip_through_wrapper(model, audio)
    with torch.no_grad():
        expected = model(audio)
    # The model output may include the complement stem (not part of the
    # exported graph — clients compute mix - vocals themselves).
    assert torch.allclose(recon, expected[:, : model.num_stems], atol=1e-4)


def test_mel_averaging_matrix_shape_and_normalisation() -> None:
    """
    The Mel averaging matrix maps selected band-bins back to the full bin
    axis, and each bin's row sums to 1 (an average over its covering bands).
    """
    model = _mel()
    wrapper = RoformerONNXWrapper(model)
    matrix = wrapper.mel_averaging_matrix
    n_bins = (N_FFT // 2 + 1) * model.audio_channels
    assert matrix.shape == (n_bins, int(model.freq_indices.numel()))
    assert torch.allclose(matrix.sum(dim=1), torch.ones(n_bins), atol=1e-6)


@pytest.mark.parametrize("builder", [_bs, _mel], ids=["bs", "mel"])
def test_export_and_onnxruntime_parity(builder, tmp_path) -> None:
    """
    The dynamo-exported graph loads under onnxruntime and matches the torch
    wrapper numerically, with a working dynamic batch axis.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from unblend.onnx import _export_roformer_to_onnx

    torch.manual_seed(0)
    model = builder()
    model.configure_inference(
        sources=(["vocals", "other"] if model.num_stems == 1 else ["a", "b"]),
        samplerate=SR,
        segment_samples=SR,  # 1s segments keep the traced graph small
    )
    path = str(tmp_path / "model.onnx")
    _export_roformer_to_onnx(model, path, opset_version=17, fp16=False)

    stft = model.stft_kwargs
    audio = torch.randn(2, 2, SR)
    spec_real, spec_imag = compute_roformer_stft_for_export(
        audio,
        n_fft=stft["n_fft"],
        hop_length=stft["hop_length"],
        win_length=stft["win_length"],
        normalized=stft["normalized"],
    )
    wrapper = RoformerONNXWrapper(model).eval()
    with torch.no_grad():
        torch_real, torch_imag = wrapper(spec_real, spec_imag)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_real, ort_imag = session.run(
        None, {"spec_real": spec_real.numpy(), "spec_imag": spec_imag.numpy()}
    )
    assert ort_real.shape == tuple(torch_real.shape)  # batch=2 flowed through
    assert torch.allclose(torch.from_numpy(ort_real), torch_real, atol=1e-4)
    assert torch.allclose(torch.from_numpy(ort_imag), torch_imag, atol=1e-4)

    # Embedded metadata drives the web pipeline's per-model configuration.
    import onnx

    meta = {p.key: p.value for p in onnx.load(path).metadata_props}
    assert meta["model_family"] == "roformer"
    assert meta["stft_n_fft"] == str(N_FFT)
    assert meta["stft_hop_length"] == str(HOP)
    assert meta["precision"] == "fp32"


def test_fp16_export_halves_weights(tmp_path) -> None:
    """
    Weight-only fp16 export produces a smaller artifact that still loads and
    runs under onnxruntime, stamped ``precision=fp16``.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    import os

    import onnx

    from unblend.onnx import _export_roformer_to_onnx

    torch.manual_seed(0)
    model = _bs()
    model.configure_inference(sources=["a", "b"], samplerate=SR, segment_samples=SR)

    fp32_path = str(tmp_path / "m32.onnx")
    fp16_path = str(tmp_path / "m16.onnx")
    _export_roformer_to_onnx(model, fp32_path, opset_version=18, fp16=False)
    _export_roformer_to_onnx(model, fp16_path, opset_version=18, fp16=True)

    assert os.path.getsize(fp16_path) < 0.75 * os.path.getsize(fp32_path)
    meta = {p.key: p.value for p in onnx.load(fp16_path).metadata_props}
    assert meta["precision"] == "fp16"

    stft = model.stft_kwargs
    audio = torch.randn(1, 2, SR)
    spec_real, spec_imag = compute_roformer_stft_for_export(
        audio,
        n_fft=stft["n_fft"],
        hop_length=stft["hop_length"],
        win_length=stft["win_length"],
        normalized=stft["normalized"],
    )
    session = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])
    out_real, _out_imag = session.run(
        None, {"spec_real": spec_real.numpy(), "spec_imag": spec_imag.numpy()}
    )
    assert out_real.shape[0] == 1
    assert torch.isfinite(torch.from_numpy(out_real)).all()
