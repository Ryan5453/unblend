"""
MPS behaviour for the RoFormer backend.

The parity tests need Apple-silicon hardware and skip elsewhere; CI runs them
on a macOS arm64 runner. RoFormer FP16 is SDR-safe and consistently faster
with the optimized MPS attention/RMSNorm paths (Kim 1.06x, SW 1.07x on an
M2 Max), so ``dtype="auto"`` resolves to FP16 on MPS and modern CUDA.
"""

import pytest
import torch
import torch.nn.functional as F

from unblend.api import Separator, _contains_htdemucs
from unblend.apply import ModelEnsemble
from unblend.htdemucs import HTDemucs
from unblend.roformer import (
    BSRoformer,
    MelBandRoformer,
    _scaled_dot_product_attention,
)

SR = 44100

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)


def _tiny_bs() -> BSRoformer:
    """
    Build a tiny BS-RoFormer configured for inference.

    :return: A configured ``BSRoformer`` in eval mode.
    """
    model = BSRoformer(
        dim=32,
        depth=2,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=16,
        heads=2,
    ).eval()
    model.configure_inference(
        sources=["vocals", "other"], samplerate=SR, segment_samples=SR
    )
    return model


def _tiny_mel() -> MelBandRoformer:
    """
    Build a tiny Mel-Band RoFormer configured for inference.

    :return: A configured ``MelBandRoformer`` in eval mode.
    """
    model = MelBandRoformer(
        dim=32,
        depth=2,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        num_bands=60,
        dim_head=16,
        heads=2,
    ).eval()
    model.configure_inference(
        sources=["vocals", "other"], samplerate=SR, segment_samples=SR
    )
    return model


def _tiny_htdemucs() -> HTDemucs:
    """
    Build a tiny HTDemucs for family-detection tests.

    :return: A small ``HTDemucs`` instance.
    """
    return HTDemucs(
        sources=["drums", "bass", "other", "vocals"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )


def test_contains_htdemucs_family_detection() -> None:
    """
    ``_contains_htdemucs`` distinguishes the HTDemucs family (raw or in an
    ensemble) from RoFormer models — it gates the reduced-precision
    auto-default and the MPS Metal-kernel pass.
    """
    ht = _tiny_htdemucs()
    assert _contains_htdemucs(ht)
    assert _contains_htdemucs(ModelEnsemble([ht]))
    assert not _contains_htdemucs(_tiny_bs())
    assert not _contains_htdemucs(_tiny_mel())


def test_auto_dtype_is_fp16_for_roformer_on_mps() -> None:
    """
    ``dtype="auto"`` resolves to FP16 for RoFormer on MPS after 10-track
    validation found SDR parity and 1.06-1.07x speedups on an M2 Max.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("requires an MPS device")
    separator = Separator(model=_tiny_bs(), device="mps", dtype="auto")
    assert separator.dtype == torch.float16
    assert next(separator.model.parameters()).dtype == torch.float16


def test_auto_dtype_is_fp16_for_roformer_on_cuda() -> None:
    """
    On CUDA with tensor cores, ``dtype="auto"`` resolves to FP16 for RoFormer
    models (measured SDR-equal to FP32; see the Separator init comment).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    major, _minor = torch.cuda.get_device_capability()
    expected = torch.float16 if major >= 7 else None
    separator = Separator(model=_tiny_bs(), device="cuda", dtype="auto")
    assert separator.dtype == expected


@mps_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 2, 87, 16), (87, 2, 4, 16)])
def test_mps_manual_attention_matches_native_sdpa(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    The faster explicit MPS attention path matches native SDPA numerically.

    :param dtype: Attention storage dtype under test.
    :param shape: Query/key/value shape under test.
    """
    torch.manual_seed(3)
    query = torch.randn(*shape, device="mps", dtype=dtype)
    key = torch.randn(*shape, device="mps", dtype=dtype)
    value = torch.randn(*shape, device="mps", dtype=dtype)
    scale = shape[-1] ** -0.5

    expected = F.scaled_dot_product_attention(query, key, value)
    actual = _scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        dropout=0.0,
        training=False,
    )
    tolerance = 2e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@mps_only
@pytest.mark.parametrize("builder", [_tiny_bs, _tiny_mel], ids=["bs", "mel"])
def test_mps_forward_matches_cpu(builder) -> None:
    """
    An FP32 forward on MPS matches the CPU forward within kernel tolerance.
    """
    torch.manual_seed(0)
    model = builder()
    audio = torch.randn(1, 2, SR)
    with torch.no_grad():
        cpu_out = model(audio)
        mps_out = model.to("mps")(audio.to("mps")).cpu()
    assert torch.isfinite(mps_out).all()
    assert torch.allclose(cpu_out, mps_out, atol=2e-3)


@mps_only
@pytest.mark.parametrize("builder", [_tiny_bs, _tiny_mel], ids=["bs", "mel"])
def test_metal_rms_norm_preserves_native_mps_forward(builder) -> None:
    """
    The inference-only Metal RMSNorm path stays within 1e-5 of the native MPS
    autograd path for a fixed-seed end-to-end RoFormer forward.
    """
    torch.manual_seed(7)
    model = builder().to("mps")
    audio = torch.randn(1, 2, SR, device="mps")
    # RMSNorm deliberately retains native PyTorch whenever autograd is on.
    with torch.enable_grad():
        native_out = model(audio).detach()
    with torch.no_grad():
        metal_out = model(audio)
    torch.testing.assert_close(metal_out, native_out, atol=1e-5, rtol=1e-5)


@mps_only
@pytest.mark.parametrize("builder", [_tiny_bs, _tiny_mel], ids=["bs", "mel"])
def test_mps_fp16_forward_is_stable(builder) -> None:
    """
    Explicit FP16 weights on MPS produce finite output close to FP32 (the
    STFT/iSTFT, complex mask math, norms, and rotary rotation all run in
    FP32 internally by design).
    """
    torch.manual_seed(0)
    model = builder().to("mps")
    audio = torch.randn(1, 2, SR, device="mps")
    with torch.no_grad():
        fp32_out = model(audio).cpu()
        half_out = model.to(torch.float16)(audio.to(torch.float16)).float().cpu()
    assert torch.isfinite(half_out).all()
    assert torch.allclose(fp32_out, half_out, atol=5e-2)


@mps_only
def test_separator_end_to_end_on_mps() -> None:
    """
    A tiny RoFormer runs through the full ``Separator`` pipeline on MPS:
    raw-audio normalisation gating, chunked tiling, and stem assembly.
    """
    separator = Separator(model=_tiny_bs(), device="mps")
    audio = torch.randn(2, SR * 2)
    result = separator.separate((audio, SR), shifts=1)
    assert set(result.sources) == {"vocals", "other"}
    for stem in result.sources.values():
        assert stem.shape == (2, SR * 2)
        assert torch.isfinite(stem).all()
