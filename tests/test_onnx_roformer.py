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
from unblend.roformer import (
    Attention,
    BSRoformer,
    FeedForward,
    MaskEstimator,
    MelBandRoformer,
    RMSNorm,
    _chunked_scaled_dot_product_attention,
)

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


def test_chunked_attention_matches_full_attention() -> None:
    """
    Query chunking preserves exact all-to-all attention semantics.
    """
    torch.manual_seed(0)
    query = torch.randn(2, 3, 73, 8)
    key = torch.randn(2, 3, 73, 8)
    value = torch.randn(2, 3, 73, 8)
    kwargs = {
        "scale": 8**-0.5,
        "dropout": 0.0,
        "training": False,
    }
    full = _chunked_scaled_dot_product_attention(
        query, key, value, query_chunk_size=None, **kwargs
    )
    chunked = _chunked_scaled_dot_product_attention(
        query, key, value, query_chunk_size=16, **kwargs
    )
    assert torch.allclose(chunked, full, atol=1e-6)


def test_head_chunked_attention_matches_full_module() -> None:
    """
    Head-group projections preserve the trained attention operation.
    """
    torch.manual_seed(0)
    module = Attention(dim=16, heads=4, dim_head=8).eval()
    x = torch.randn(5, 73, 16)
    with torch.no_grad():
        expected = module(x)
        module.onnx_query_chunk_size = 17
        module.onnx_head_chunk_size = 2
        actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_hidden_chunked_feedforward_matches_full_module() -> None:
    """
    Feature-group MLP projections sum to the ordinary second linear.
    """
    torch.manual_seed(0)
    module = FeedForward(dim=16, mult=4).eval()
    x = torch.randn(5, 73, 16)
    with torch.no_grad():
        expected = module(x)
        module.onnx_hidden_chunk_size = 13
        actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_sliced_glu_matches_native_mask_estimator() -> None:
    """
    Explicit GLU slices preserve mask-head output without ONNX Split.
    """
    torch.manual_seed(0)
    module = MaskEstimator(
        dim=16,
        dim_inputs=(4, 6),
        mlp_hidden_layers=1,
    ).eval()
    x = torch.randn(2, 7, 2, 16)
    with torch.no_grad():
        expected = module(x)
        module.onnx_safe_glu = True
        actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_export_wrapper_enables_bounded_attention() -> None:
    """
    Every attention and RMSNorm block receives its browser-safe path.
    """
    model = _bs()
    RoformerONNXWrapper(
        model,
        attention_query_chunk_size=17,
        attention_head_chunk_size=2,
        feedforward_hidden_chunk_size=11,
    )
    attention = [module for module in model.modules() if isinstance(module, Attention)]
    feedforwards = [
        module for module in model.modules() if isinstance(module, FeedForward)
    ]
    mask_estimators = [
        module for module in model.modules() if isinstance(module, MaskEstimator)
    ]
    norms = [module for module in model.modules() if isinstance(module, RMSNorm)]
    assert attention
    assert feedforwards
    assert mask_estimators
    assert norms
    assert all(module.onnx_query_chunk_size == 17 for module in attention)
    assert all(module.onnx_head_chunk_size == 2 for module in attention)
    assert all(module.onnx_hidden_chunk_size == 11 for module in feedforwards)
    assert all(module.onnx_safe_glu for module in mask_estimators)
    assert all(module.onnx_safe for module in norms)


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
    _export_roformer_to_onnx(model, path, opset_version=17, storage=torch.float32)

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

    # Torch's dynamo decomposition of aten.rms_norm currently drops a tiny
    # explicit epsilon. The wrapper's manual formula must keep true silence
    # finite instead of producing Reciprocal(Sqrt(0)) -> NaN throughout.
    silent = torch.zeros_like(spec_real)
    silent_real, silent_imag = session.run(
        None, {"spec_real": silent.numpy(), "spec_imag": silent.numpy()}
    )
    assert torch.isfinite(torch.from_numpy(silent_real)).all()
    assert torch.isfinite(torch.from_numpy(silent_imag)).all()

    # Embedded metadata drives the web pipeline's per-model configuration.
    import onnx

    meta = {p.key: p.value for p in onnx.load(path).metadata_props}
    assert meta["model_family"] == "roformer"
    assert meta["stft_n_fft"] == str(N_FFT)
    assert meta["stft_hop_length"] == str(HOP)
    assert meta["precision"] == "fp32"
    assert meta["external_normalization"] == "false"

    # Band splitting used to become one ~60-output Split, which exceeds the
    # WebGPU storage-buffer binding floor. The only remaining Split is the
    # narrow Q/K/V split (three outputs).
    graph = onnx.load(path).graph
    assert (
        max(
            (len(node.output) for node in graph.node if node.op_type == "Split"),
            default=0,
        )
        <= 3
    )
    assert (
        max(
            (len(node.input) for node in graph.node if node.op_type == "Concat"),
            default=0,
        )
        <= 8
    )


@pytest.mark.parametrize("builder", [_bs, _mel], ids=["bs", "mel"])
def test_static_browser_export_chunks_large_buffers(builder, tmp_path) -> None:
    """
    The static browser graph's head/MLP chunking preserves ORT parity.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from unblend.onnx import _export_roformer_to_onnx

    torch.manual_seed(1)
    model = builder()
    model.configure_inference(
        sources=(["vocals", "other"] if model.num_stems == 1 else ["a", "b"]),
        samplerate=SR,
        segment_samples=SR,
    )
    path = str(tmp_path / "browser.onnx")
    _export_roformer_to_onnx(
        model,
        path,
        opset_version=18,
        storage=torch.float32,
        static_batch=True,
    )

    audio = torch.randn(1, 2, SR)
    spec_real, spec_imag = compute_roformer_stft_for_export(
        audio,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        normalized=False,
    )
    wrapper = RoformerONNXWrapper(model).eval()
    with torch.no_grad():
        expected = wrapper(spec_real, spec_imag)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {"spec_real": spec_real.numpy(), "spec_imag": spec_imag.numpy()},
    )
    assert torch.allclose(torch.from_numpy(actual[0]), expected[0], atol=1e-4)
    assert torch.allclose(torch.from_numpy(actual[1]), expected[1], atol=1e-4)


def test_fp16_export_uses_mixed_precision(tmp_path) -> None:
    """
    Browser mixed-fp16 export is smaller, retains fp32 IO/RMS constants, and
    runs under onnxruntime stamped ``precision=fp16``.
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
    _export_roformer_to_onnx(model, fp32_path, opset_version=18, storage=torch.float32)
    _export_roformer_to_onnx(model, fp16_path, opset_version=18, storage=torch.float16)

    assert os.path.getsize(fp16_path) < 0.75 * os.path.getsize(fp32_path)
    fp16_model = onnx.load(fp16_path)
    meta = {p.key: p.value for p in fp16_model.metadata_props}
    assert meta["precision"] == "fp16"
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for value in fp16_model.graph.input
    )
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for value in fp16_model.graph.output
    )
    assert any(
        init.data_type == onnx.TensorProto.FLOAT16
        for init in fp16_model.graph.initializer
    )
    # The mixed graph deliberately raises the RMS floor so rsqrt remains
    # representable when its fp32 island returns to fp16 on true silence.
    from onnx import numpy_helper

    assert any(
        init.data_type == onnx.TensorProto.FLOAT16
        and numpy_helper.to_array(init).size == 1
        and float(numpy_helper.to_array(init)) == pytest.approx(1e-7, abs=3e-8)
        for init in fp16_model.graph.initializer
    )

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

    silent = torch.zeros_like(spec_real)
    silent_real, silent_imag = session.run(
        None, {"spec_real": silent.numpy(), "spec_imag": silent.numpy()}
    )
    assert torch.isfinite(torch.from_numpy(silent_real)).all()
    assert torch.isfinite(torch.from_numpy(silent_imag)).all()
