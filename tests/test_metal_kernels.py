"""
Numeric-equivalence tests for the MPS Metal kernels in ``demucs.metal``.

Each fused module has a PyTorch fallback (used on CPU / in FP32) and a
hand-written Metal kernel (used on MPS in FP16/BF16). These tests assert the
kernel output matches the fallback reference within low-precision tolerance, so
a kernel regression (bad indexing, a broken reduction, wrong activation math)
can't silently ship. The fallback is treated as ground truth: we run the same
module on a CPU FP32 copy of the input to get the reference, then on MPS in
FP16/BF16 to exercise the kernel.

These only run on Apple Silicon (MPS); elsewhere they skip.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from demucs.metal import (
    FusedGroupNormGelu,
    FusedGroupNormGlu,
    FusedNormGluLayerScaleResid,
    MetalGroupNorm,
    MetalMyGroupNorm,
)

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal kernels only run on Apple Silicon (MPS)",
)

LP_DTYPES = [torch.float16, torch.bfloat16]


def _tol(dtype: torch.dtype) -> dict:
    """Tolerance appropriate for the low-precision dtype under test.

    :param dtype: The reduced-precision dtype the kernel ran in
    :return: ``atol``/``rtol`` kwargs for ``torch.testing.assert_close``
    """
    # FP16 carries ~3 decimal digits; BF16 has only an 8-bit mantissa, so it
    # needs looser bounds. The kernel computes in FP32 internally and casts the
    # result to ``dtype``, so the gap is dominated by that final cast.
    if dtype == torch.float16:
        return dict(atol=3e-2, rtol=2e-2)
    return dict(atol=8e-2, rtol=5e-2)


def _make_gn(channels: int) -> nn.GroupNorm:
    """Build a ``num_groups=1`` affine GroupNorm with non-trivial affine params.

    :param channels: Number of channels (the affine dimension)
    :return: A randomly-initialized ``nn.GroupNorm(1, channels)``
    """
    gn = nn.GroupNorm(1, channels)
    with torch.no_grad():
        gn.weight.normal_(mean=1.0, std=0.1)
        gn.bias.normal_(mean=0.0, std=0.1)
    return gn


def _make_ls(channels: int) -> torch.Tensor:
    """Build a non-trivial per-channel LayerScale parameter.

    :param channels: Number of output channels the scale applies to
    :return: A random FP32 tensor of shape ``(channels,)`` centered near the
        small init the real LayerScale uses
    """
    return torch.randn(channels) * 0.1 + 0.05


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize("shape", [(2, 48, 100), (4, 64, 8, 16)])
def test_metal_group_norm_matches_fallback(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """``MetalGroupNorm`` single-stage kernel matches the FP32 fallback."""
    channels = shape[1]
    mod = MetalGroupNorm(_make_gn(channels))
    x = torch.randn(*shape)

    ref = mod(x.to(torch.float32))
    out = mod.to("mps")(x.to("mps", dtype))

    assert out.dtype == dtype
    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_metal_group_norm_multi_stage(dtype: torch.dtype) -> None:
    """``MetalGroupNorm`` multi-stage (3-kernel) path matches the fallback.

    Uses a per-batch element count above ``_SINGLE_STAGE_LIMIT`` so the
    partial-reduce / finalize / apply kernels fire instead of the single-stage
    kernel.
    """
    channels, frames = 512, 4096
    assert channels * frames > MetalGroupNorm._SINGLE_STAGE_LIMIT
    mod = MetalGroupNorm(_make_gn(channels))
    x = torch.randn(2, channels, frames)

    ref = mod(x.to(torch.float32))
    out = mod.to("mps")(x.to("mps", dtype))

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 48, 100),
        (4, 64, 8, 16),
        # Edge shapes: odd channel counts (fine with num_groups=1) and
        # spatial sizes that don't divide the threadgroup size.
        (2, 49, 101),
        (3, 97, 1023),
    ],
)
def test_fused_group_norm_gelu_matches_fallback(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """``FusedGroupNormGelu`` kernel matches ``gelu(group_norm(x))``.

    The kernel uses the tanh GELU approximation (no ``erf`` in the MPS shader
    toolchain) while the FP32 fallback reference uses exact erf; their ~1e-3
    gap is below the FP16/BF16 tolerance here, so the comparison still catches
    gross kernel errors (indexing, reduction, affine) without flagging the
    intentional sub-precision activation difference.
    """
    channels = shape[1]
    mod = FusedGroupNormGelu(_make_gn(channels))
    x = torch.randn(*shape)

    ref = mod(x.to(torch.float32))
    out = mod.to("mps")(x.to("mps", dtype))

    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@mps_only
def test_fused_gelu_fallback_is_exact_erf() -> None:
    """The FP32 GELU fallback equals PyTorch's exact-erf ``F.gelu`` (the #6 fix).

    The kernel's per-element erf-vs-tanh gap (~1e-3) is below FP16 precision so
    it can't be distinguished by a kernel-vs-fallback comparison; this checks the
    fallback definition directly, which is what the kernel is verified against.
    """
    channels = 64
    gn = _make_gn(channels)
    mod = FusedGroupNormGelu(gn)
    x = torch.randn(2, channels, 128)

    out = mod(x.to(torch.float32))
    normed = F.group_norm(x, 1, gn.weight, gn.bias, gn.eps)
    expected = F.gelu(normed)  # exact erf (PyTorch default)

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_fused_group_norm_gelu_multi_stage(dtype: torch.dtype) -> None:
    """``FusedGroupNormGelu`` multi-stage (3-kernel) path matches the reference.

    Uses a per-batch element count above ``_SINGLE_STAGE_LIMIT`` so the
    partial-reduce / finalize / apply_norm_gelu kernels fire instead of the
    single-stage kernel. The reference is the explicit PyTorch composition
    ``gelu(group_norm(x))`` in FP32.
    """
    channels, frames = 512, 4096
    assert channels * frames > FusedGroupNormGelu._SINGLE_STAGE_LIMIT
    gn = _make_gn(channels)
    mod = FusedGroupNormGelu(gn)
    x = torch.randn(2, channels, frames)

    ref = F.gelu(F.group_norm(x.to(torch.float32), 1, gn.weight, gn.bias, gn.eps))
    out = mod.to("mps")(x.to("mps", dtype))

    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 96, 100),
        (4, 128, 8, 16),
        # Edge shapes: GLU needs an even input channel count, but the halves
        # can be odd (2*49, 2*97); spatial sizes don't divide the threadgroup
        # size.
        (2, 98, 101),
        (3, 194, 1023),
    ],
)
def test_fused_group_norm_glu_matches_fallback(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """``FusedGroupNormGlu`` kernel matches ``glu(group_norm(x), dim=1)``.

    Input has ``2C`` channels; output has ``C`` (GLU halves the channel dim).
    """
    in_channels = shape[1]  # even == 2C
    mod = FusedGroupNormGlu(_make_gn(in_channels))
    x = torch.randn(*shape)

    ref = mod(x.to(torch.float32))
    out = mod.to("mps")(x.to("mps", dtype))

    assert out.shape[1] == in_channels // 2
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_fused_group_norm_glu_multi_stage(dtype: torch.dtype) -> None:
    """``FusedGroupNormGlu`` multi-stage (3-kernel) path matches the reference.

    Uses a per-batch *input* element count above ``_SINGLE_STAGE_LIMIT`` (the
    GLU gate is on the 2C-channel input) so the partial-reduce / finalize /
    apply_norm_glu kernels fire. The reference is the explicit PyTorch
    composition ``glu(group_norm(x), dim=1)`` in FP32.
    """
    in_channels, frames = 512, 4096  # even == 2C
    assert in_channels * frames > FusedGroupNormGlu._SINGLE_STAGE_LIMIT
    gn = _make_gn(in_channels)
    mod = FusedGroupNormGlu(gn)
    x = torch.randn(2, in_channels, frames)

    ref = F.glu(
        F.group_norm(x.to(torch.float32), 1, gn.weight, gn.bias, gn.eps), dim=1
    )
    out = mod.to("mps")(x.to("mps", dtype))

    assert out.shape[1] == in_channels // 2
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 96, 100),
        # Edge shape: odd half-channel count and a spatial size that doesn't
        # divide the threadgroup size.
        (3, 98, 101),
    ],
)
def test_fused_norm_glu_ls_resid_matches_reference(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """``FusedNormGluLayerScaleResid`` matches the unfused PyTorch ops.

    The DConv envelope kernel computes
    ``residual + layer_scale * glu(group_norm(z), dim=1)`` in one launch.
    ``z`` has ``2C`` channels; ``residual`` and the output have ``C``.
    """
    in_channels = shape[1]  # even == 2C
    half = in_channels // 2
    gn = _make_gn(in_channels)
    ls = _make_ls(half)
    mod = FusedNormGluLayerScaleResid(gn, ls)
    z = torch.randn(*shape)
    residual = torch.randn(shape[0], half, *shape[2:])

    zf = z.to(torch.float32)
    ref = residual + ls[:, None] * F.glu(
        F.group_norm(zf, 1, gn.weight, gn.bias, gn.eps), dim=1
    )
    out = mod.to("mps")(z.to("mps", dtype), residual.to("mps", dtype))

    assert out.dtype == dtype
    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_fused_norm_glu_ls_resid_multi_stage(dtype: torch.dtype) -> None:
    """``FusedNormGluLayerScaleResid`` multi-stage path matches the reference.

    Uses a per-batch *input* element count above ``_SINGLE_STAGE_LIMIT`` (the
    gate is on the 2C-channel GLU input) so the partial-reduce / finalize /
    apply_norm_glu_ls_resid kernels fire instead of the single-stage kernel.
    """
    in_channels, frames = 1024, 2048  # even == 2C
    assert in_channels * frames > FusedNormGluLayerScaleResid._SINGLE_STAGE_LIMIT
    half = in_channels // 2
    gn = _make_gn(in_channels)
    ls = _make_ls(half)
    mod = FusedNormGluLayerScaleResid(gn, ls)
    z = torch.randn(2, in_channels, frames)
    residual = torch.randn(2, half, frames)

    zf = z.to(torch.float32)
    ref = residual + ls[:, None] * F.glu(
        F.group_norm(zf, 1, gn.weight, gn.bias, gn.eps), dim=1
    )
    out = mod.to("mps")(z.to("mps", dtype), residual.to("mps", dtype))

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@mps_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_metal_my_group_norm_matches_fallback(dtype: torch.dtype) -> None:
    """``MetalMyGroupNorm`` (transpose-free ``(B, T, C)`` norm) matches fallback."""
    channels = 384
    mod = MetalMyGroupNorm(_make_gn(channels))
    x = torch.randn(2, 200, channels)  # (B, T, C)

    ref = mod(x.to(torch.float32))
    out = mod.to("mps")(x.to("mps", dtype))

    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))
