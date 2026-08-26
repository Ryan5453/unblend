"""
Numeric-equivalence tests for the native CUDA kernels in ``unblend.cuda``.

Each fused module has a PyTorch fallback (used on CPU / in FP32) and a
hand-written CUDA kernel (used on CUDA in FP16/BF16). RoFormer RMSNorm also
supports explicitly requested FP32. These tests assert the kernel output
matches the fallback reference within tolerance, so
a kernel regression (bad indexing, a broken reduction, wrong activation math)
can't silently ship. The fallback is treated as ground truth: we run the same
module on a CPU FP32 copy of the input to get the reference, then on CUDA in
FP16/BF16 to exercise the kernel.

These only run on machines with an NVIDIA GPU; elsewhere they skip.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unblend.cuda import (
    CUDAGroupNorm,
    CUDAMultiheadAttention,
    CUDAMyGroupNorm,
    FusedGroupNormGelu,
    FusedGroupNormGlu,
    FusedNormGluLayerScaleResid,
    apply_cuda_optimizations,
    cuda_rms_norm,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA kernels only run on NVIDIA GPUs",
)

LP_DTYPES = [torch.float16, torch.bfloat16]


def _inference_call(module: nn.Module, *args: torch.Tensor):
    """
    Run a replacement through its kernel-backed inference dispatch.
    """
    with torch.inference_mode():
        return module(*args)


def _tol(dtype: torch.dtype) -> dict[str, float]:
    """
    Tolerance appropriate for the low-precision dtype under test.

    :param dtype: The reduced-precision dtype the kernel ran in
    :return: ``atol``/``rtol`` kwargs for ``torch.testing.assert_close``
    """
    # FP16 carries ~3 decimal digits; BF16 has only an 8-bit mantissa, so it
    # needs looser bounds. The kernel computes in FP32 internally and casts the
    # result to ``dtype``, so the gap is dominated by that final cast.
    if dtype == torch.float16:
        return dict(atol=3e-2, rtol=2e-2)
    return dict(atol=8e-2, rtol=5e-2)


def _device() -> torch.device:
    """
    Return the CUDA device under test.

    :return: The default CUDA device
    """
    return torch.device("cuda")


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, *LP_DTYPES])
@pytest.mark.parametrize("shape", [(7, 31, 256), (5, 62, 384), (3, 11, 516)])
def test_cuda_rms_norm_matches_fp32_reference(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    Fused last-dimension RMSNorm preserves RoFormer's FP32 arithmetic.

    :param dtype: Storage dtype under test.
    :param shape: Input shape ending in the affine dimension.
    """
    dim = shape[-1]
    scale = dim**0.5
    x = torch.randn(*shape)
    gamma = torch.randn(dim) * 0.1 + 1.0
    dev_x = x.to(_device(), dtype)
    dev_gamma = gamma.to(_device(), dtype)

    # Build the reference from dtype-quantized values so the comparison
    # isolates reduction/affine arithmetic rather than input conversion.
    ref_x = dev_x.cpu().float()
    ref_gamma = dev_gamma.cpu().float()
    ref = F.normalize(ref_x, dim=-1) * scale * ref_gamma
    with torch.inference_mode():
        out = cuda_rms_norm(dev_x, dev_gamma, scale).cpu().float()

    tolerance = dict(atol=2e-5, rtol=2e-5) if dtype == torch.float32 else _tol(dtype)
    torch.testing.assert_close(out, ref, **tolerance)


def _make_gn(channels: int) -> nn.GroupNorm:
    """
    Build a ``num_groups=1`` affine GroupNorm with non-trivial affine params.

    :param channels: Number of channels (the affine dimension)
    :return: A randomly-initialized ``nn.GroupNorm(1, channels)``
    """
    gn = nn.GroupNorm(1, channels)
    with torch.no_grad():
        gn.weight.normal_(mean=1.0, std=0.1)
        gn.bias.normal_(mean=0.0, std=0.1)
    return gn


def _make_ls(channels: int) -> torch.Tensor:
    """
    Build a non-trivial per-channel LayerScale parameter.

    :param channels: Number of output channels the scale applies to
    :return: A random FP32 tensor of shape ``(channels,)`` centered near the
        small init the real LayerScale uses
    """
    return torch.randn(channels) * 0.1 + 0.05


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 48, 100),  # single-stage (small per-batch)
        (4, 64, 8, 16),
        (130, 48, 336),  # single-stage via B >= _SINGLE_STAGE_MIN_BATCH
        (2, 48, 4096),  # multi-stage via small B + medium per-batch
        (2, 49, 101),  # odd N: scalar (non-vectorized) apply path
    ],
)
def test_cuda_group_norm_matches_fallback(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    ``CUDAGroupNorm`` kernel paths match the FP32 fallback.

    Shapes chosen to hit both sides of the dispatch heuristic (single-stage
    for large B or small per-batch, multi-stage for small B with larger
    per-batch) and both the vectorized (``N % 4 == 0``) and scalar apply
    loops.

    :param dtype: dtype under test
    :param shape: tensor shape under test
    """
    channels = shape[1]
    mod = CUDAGroupNorm(_make_gn(channels))
    x = torch.randn(*shape)

    ref = mod(x.to(torch.float32))
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    assert out.dtype == dtype
    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
def test_kernel_modules_fall_back_when_autograd_is_enabled() -> None:
    """
    Raw CUDA kernels are bypassed whenever gradients are being recorded.
    """
    mod = CUDAGroupNorm(_make_gn(16)).to(_device())
    x = torch.randn(
        2, 16, 32, device=_device(), dtype=torch.float16, requires_grad=True
    )

    out = mod(x)
    out.float().sum().backward()

    assert out.requires_grad
    assert x.grad is not None
    assert mod.weight.grad is not None

    rms_x = torch.randn(2, 8, device=_device(), dtype=torch.float16, requires_grad=True)
    gamma = torch.ones(8, device=_device(), requires_grad=True)
    cuda_rms_norm(rms_x, gamma, 8**0.5).float().sum().backward()
    assert rms_x.grad is not None
    assert gamma.grad is not None


@cuda_only
def test_kernel_modules_still_dispatch_during_inference(monkeypatch) -> None:
    """
    The performance path remains active under Separator's inference mode.
    """
    import unblend.cuda as cuda_module

    calls: list[str] = []
    original_get_kernel = cuda_module._get_kernel

    def recording_get_kernel(name: str, dtype: torch.dtype):
        calls.append(name)
        return original_get_kernel(name, dtype)

    monkeypatch.setattr(cuda_module, "_get_kernel", recording_get_kernel)
    mod = CUDAGroupNorm(_make_gn(16)).to(_device()).eval()
    _inference_call(mod, torch.randn(2, 16, 32, device=_device(), dtype=torch.float16))
    assert "group_norm_g1" in calls


@cuda_only
def test_kernel_parameter_caches_refresh_after_optimizer_step() -> None:
    """
    Warm inference caches never hide affine/LayerScale optimizer updates.
    """
    x = torch.randn(2, 16, 32, device=_device(), dtype=torch.float16)
    mod = CUDAGroupNorm(_make_gn(16)).to(_device()).eval()
    _inference_call(mod, x)
    old_affine = mod._aff_cache[(torch.float16, x.device)][0]

    mod.train()
    optimizer = torch.optim.SGD(mod.parameters(), lr=0.1)
    optimizer.zero_grad()
    mod(x).float().square().mean().backward()
    optimizer.step()
    mod.eval()

    actual = _inference_call(mod, x)
    new_affine = mod._aff_cache[(torch.float16, x.device)][0]
    assert new_affine is not old_affine
    expected = F.group_norm(x.float(), 1, mod.weight, mod.bias, mod.eps).half()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), **_tol(torch.float16))

    gn = _make_gn(16)
    fused = (
        FusedNormGluLayerScaleResid(gn, torch.ones(8, dtype=torch.float32))
        .to(_device())
        .eval()
    )
    residual = torch.randn(2, 8, 32, device=_device(), dtype=torch.float16)
    _inference_call(fused, x, residual)
    old_scale = fused._ls_cache[(torch.float16, x.device)]

    fused.train()
    optimizer = torch.optim.SGD(fused.parameters(), lr=0.1)
    optimizer.zero_grad()
    fused(x, residual).float().square().mean().backward()
    optimizer.step()
    fused.eval()

    actual = _inference_call(fused, x, residual)
    new_scale = fused._ls_cache[(torch.float16, x.device)]
    assert new_scale is not old_scale
    normalized = F.group_norm(x.float(), 1, fused.weight, fused.bias, fused.eps)
    expected = (
        residual.float() + fused.layer_scale[:, None] * F.glu(normalized, dim=1)
    ).half()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), **_tol(torch.float16))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_cuda_group_norm_multi_stage(dtype: torch.dtype) -> None:
    """
    ``CUDAGroupNorm`` multi-stage (3-kernel) path matches the fallback.

    Uses a per-batch element count above ``_SINGLE_STAGE_LIMIT`` so the
    partial-reduce / finalize / apply kernels fire instead of the single-stage
    kernel.

    :param dtype: dtype under test
    """
    channels, frames = 512, 4096
    assert channels * frames > CUDAGroupNorm._SINGLE_STAGE_LIMIT
    mod = CUDAGroupNorm(_make_gn(channels))
    x = torch.randn(2, channels, frames)

    ref = mod(x.to(torch.float32))
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 48, 100),
        (2, 512, 4096),  # multi-stage (partial_reduce / finalize / apply)
    ],
)
def test_cuda_group_norm_large_dc_offset(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    A large DC offset exercises the kernels' K-shift cancellation guard.

    Without the shift, the one-pass ``E[x^2] - E[x]^2`` variance loses most of
    its significant digits when ``|mean| >> std``. The input is quantized to
    ``dtype`` up front so both paths see identical values and the comparison
    isolates the kernel's reduction math from input-cast error.

    :param dtype: dtype under test
    :param shape: tensor shape under test
    """
    channels = shape[1]
    mod = CUDAGroupNorm(_make_gn(channels))
    x = (torch.randn(*shape) + 100.0).to(dtype)

    ref = mod(x.to(torch.float32))
    out = _inference_call(mod.to(_device()), x.to(_device()))

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 48, 100),
        (4, 64, 8, 16),
        # Edge shapes: odd channel counts (fine with num_groups=1) and
        # spatial sizes that don't divide the block size.
        (2, 49, 101),
        (3, 97, 1023),
    ],
)
def test_fused_group_norm_gelu_matches_fallback(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    ``FusedGroupNormGelu`` kernel matches ``gelu(group_norm(x))``.

    The kernel uses the tanh GELU approximation (matching the Metal kernels;
    the gap vs exact erf is sub-FP16-precision) while the FP32 fallback
    reference uses exact erf; their ~1e-3 gap is below the FP16/BF16
    tolerance here, so the comparison still catches gross kernel errors
    (indexing, reduction, affine) without flagging the intentional
    sub-precision activation difference.

    :param dtype: dtype under test
    :param shape: tensor shape under test
    """
    channels = shape[1]
    mod = FusedGroupNormGelu(_make_gn(channels))
    x = torch.randn(*shape)

    ref = mod(x.to(torch.float32))
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_fused_group_norm_gelu_inject_matches_reference(dtype: torch.dtype) -> None:
    """
    ``FusedGroupNormGelu``'s fused inject-add matches the explicit ops.

    The HTDemucs encoder computes ``gelu(group_norm(conv(x) + inject))``; the
    module folds the add into the normalization launch. Exercises both the
    single-stage and multi-stage dispatch paths.

    :param dtype: dtype under test
    """
    for channels, frames in ((48, 100), (512, 4096)):
        gn = _make_gn(channels)
        mod = FusedGroupNormGelu(gn)
        x = torch.randn(4, channels, frames)
        inj = torch.randn(4, channels, frames)

        xf = (x + inj).to(torch.float32)
        ref = F.gelu(F.group_norm(xf, 1, gn.weight, gn.bias, gn.eps))
        out = _inference_call(
            mod.to(_device()),
            x.to(_device(), dtype),
            inj.to(_device(), dtype),
        )

        assert out.shape == ref.shape
        torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize("shape", [(4, 48, 100), (3, 97, 1023)])
def test_add_gelu_matches_reference(dtype: torch.dtype, shape: tuple) -> None:
    """
    The fused ``gelu(a + b)`` elementwise kernel matches explicit torch ops.

    Covers both the vectorized (total % 4 == 0) and scalar paths via the
    shape matrix, including a size that does not divide the block grid
    evenly.

    :param dtype: dtype under test
    :param shape: tensor shape under test
    """
    from unblend.cuda import _get_kernel

    a = torch.randn(*shape)
    b = torch.randn(*shape)
    ref = F.gelu(a + b)

    dev_a = a.to(_device(), dtype).contiguous()
    dev_b = b.to(_device(), dtype).contiguous()
    out = torch.empty_like(dev_a)
    with torch.inference_mode():
        _get_kernel("add_gelu", dtype)(out, dev_a, dev_b)

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


def _nhwc(t: torch.Tensor) -> torch.Tensor:
    """
    Convert a rank-4 tensor to channels_last storage on its current device.

    :param t: Input tensor
    :return: The same values in NHWC memory format
    """
    return t.contiguous(memory_format=torch.channels_last)


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 48, 8, 100),  # single-stage chlast, vectorized (C % 4 == 0)
        (2, 512, 8, 4096 // 8),  # multi-stage chlast
        (3, 49, 7, 101),  # odd C: scalar affine path
        (130, 48, 2, 168),  # single-stage via large B
    ],
)
def test_cuda_group_norm_channels_last_matches_fallback(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    ``CUDAGroupNorm`` handles channels_last storage via the chlast kernels.

    A channels_last tensor fed to a channel-first kernel would be silently
    misindexed; this pins the routing to the ``_chlast`` kernel family for
    both the single-stage and multi-stage paths and both affine loop flavours.

    :param dtype: dtype under test
    :param shape: ``(B, C, Fr, T)`` shape under test
    """
    channels = shape[1]
    mod = CUDAGroupNorm(_make_gn(channels))
    x = _nhwc(torch.randn(*shape))

    ref = mod(x.contiguous().to(torch.float32))
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    assert out.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize("shape", [(2, 96, 6, 100), (3, 98, 5, 101)])
def test_fused_group_norm_glu_channels_last(dtype: torch.dtype, shape: tuple) -> None:
    """
    ``FusedGroupNormGlu`` routes channels_last inputs to the chlast kernels.

    Input has ``2C`` channels; output halves them. GLU in channel-last storage
    gates at a constant ``C_half`` offset within each spatial row.

    :param dtype: dtype under test
    :param shape: ``(B, 2C, Fr, T)`` shape under test
    """
    in_channels = shape[1]
    mod = FusedGroupNormGlu(_make_gn(in_channels))
    x = _nhwc(torch.randn(*shape))

    ref = mod(x.contiguous().to(torch.float32))
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    assert out.shape[1] == in_channels // 2
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize("shape", [(2, 96, 6, 100), (3, 98, 5, 101)])
def test_envelope_channels_last(dtype: torch.dtype, shape: tuple) -> None:
    """
    ``FusedNormGluLayerScaleResid`` supports channels_last storage.

    Each vector lane is a different channel in this layout, so the
    per-channel LayerScale must be applied per lane — the exact regression
    the chlast envelope kernels exist to handle.

    :param dtype: dtype under test
    :param shape: ``(B, 2C, Fr, T)`` shape under test
    """
    in_channels = shape[1]
    half = in_channels // 2
    gn = _make_gn(in_channels)
    ls = _make_ls(half)
    mod = FusedNormGluLayerScaleResid(gn, ls)
    z = _nhwc(torch.randn(*shape))
    residual = _nhwc(torch.randn(shape[0], half, *shape[2:]))

    zf = z.contiguous().to(torch.float32)
    rf = residual.contiguous().to(torch.float32)
    ref = rf + ls.view(1, -1, 1, 1) * F.glu(
        F.group_norm(zf, 1, gn.weight, gn.bias, gn.eps), dim=1
    )

    out = _inference_call(
        mod.to(_device()), z.to(_device(), dtype), residual.to(_device(), dtype)
    )

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_fused_group_norm_gelu_inject_channels_last(dtype: torch.dtype) -> None:
    """
    The inject-fused GN+GELU path works in channels_last storage too.

    Exercises the chlast gelu kernels' optional second input through the
    module interface (``gelu(group_norm(x + inject))``).

    :param dtype: dtype under test
    """
    channels, fr, frames = 48, 6, 200
    gn = _make_gn(channels)
    mod = FusedGroupNormGelu(gn)
    x = _nhwc(torch.randn(2, channels, fr, frames))
    inj = _nhwc(torch.randn(2, channels, fr, frames))

    xf = (x + inj).contiguous().to(torch.float32)
    ref = F.gelu(F.group_norm(xf, 1, gn.weight, gn.bias, gn.eps))
    out = _inference_call(
        mod.to(_device()), x.to(_device(), dtype), inj.to(_device(), dtype)
    )

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES + [torch.float32])
def test_fused_roformer_rotary_transposed_heads(dtype: torch.dtype) -> None:
    """
    The rotary kernel reads transposed-head views with no contiguous copy.

    Production q/k arrive as ``[B, S, H, Dh].transpose(1, 2)`` — dense but
    not contiguous, last-dim stride 1. The kernel must honor the strides;
    this pins that against a copy of the eager reference on identical values.

    :param dtype: dtype under test
    """
    from unblend.cuda import fused_roformer_rotary

    torch.manual_seed(9)
    b, s, h, dh = 3, 48, 8, 64
    angles = torch.randn(s, dh // 2)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    packed = torch.randn(b, s, h, dh).to(dtype)

    strided_cpu = packed.transpose(1, 2)  # [B, H, S, Dh] — production layout
    ref_t = strided_cpu.clone()
    x1, x2 = ref_t.unflatten(-1, (-1, 2)).unbind(dim=-1)
    ref = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    strided = strided_cpu.to(_device())  # non-contiguous, last dim stride 1
    assert not strided.is_contiguous()
    out = fused_roformer_rotary(strided, cos.to(_device()), sin.to(_device()))

    tol = (
        dict(atol=2e-2, rtol=2e-2)
        if dtype != torch.float32
        else dict(atol=1e-5, rtol=1e-5)
    )
    torch.testing.assert_close(out.float().cpu(), ref.float(), **tol)


@cuda_only
def test_fused_gelu_fallback_is_exact_erf() -> None:
    """
    The FP32 GELU fallback equals PyTorch's exact-erf ``F.gelu``.

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


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_fused_group_norm_gelu_multi_stage(dtype: torch.dtype) -> None:
    """
    ``FusedGroupNormGelu`` multi-stage (3-kernel) path matches the reference.

    Uses a per-batch element count above ``_SINGLE_STAGE_LIMIT`` so the
    partial-reduce / finalize / apply_norm_gelu kernels fire instead of the
    single-stage kernel. The reference is the explicit PyTorch composition
    ``gelu(group_norm(x))`` in FP32.

    :param dtype: dtype under test
    """
    channels, frames = 512, 4096
    assert channels * frames > FusedGroupNormGelu._SINGLE_STAGE_LIMIT
    gn = _make_gn(channels)
    mod = FusedGroupNormGelu(gn)
    x = torch.randn(2, channels, frames)

    ref = F.gelu(F.group_norm(x.to(torch.float32), 1, gn.weight, gn.bias, gn.eps))
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 96, 100),
        (4, 128, 8, 16),
        # Edge shapes: GLU needs an even input channel count, but the halves
        # can be odd (2*49, 2*97); spatial sizes don't divide the block size.
        (2, 98, 101),
        (3, 194, 1023),
    ],
)
def test_fused_group_norm_glu_matches_fallback(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    ``FusedGroupNormGlu`` kernel matches ``glu(group_norm(x), dim=1)``.

    Input has ``2C`` channels; output has ``C`` (GLU halves the channel dim).

    :param dtype: dtype under test
    :param shape: tensor shape under test
    """
    in_channels = shape[1]  # even == 2C
    mod = FusedGroupNormGlu(_make_gn(in_channels))
    x = torch.randn(*shape)

    ref = mod(x.to(torch.float32))
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    assert out.shape[1] == in_channels // 2
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_fused_group_norm_glu_multi_stage(dtype: torch.dtype) -> None:
    """
    ``FusedGroupNormGlu`` multi-stage (3-kernel) path matches the reference.

    Uses a per-batch *input* element count above ``_SINGLE_STAGE_LIMIT`` (the
    GLU gate is on the 2C-channel input) so the partial-reduce / finalize /
    apply_norm_glu kernels fire. The reference is the explicit PyTorch
    composition ``glu(group_norm(x), dim=1)`` in FP32.

    :param dtype: dtype under test
    """
    in_channels, frames = 512, 4096  # even == 2C
    assert in_channels * frames > FusedGroupNormGlu._SINGLE_STAGE_LIMIT
    gn = _make_gn(in_channels)
    mod = FusedGroupNormGlu(gn)
    x = torch.randn(2, in_channels, frames)

    ref = F.glu(F.group_norm(x.to(torch.float32), 1, gn.weight, gn.bias, gn.eps), dim=1)
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    assert out.shape[1] == in_channels // 2
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 96, 100),
        # Edge shape: odd half-channel count and a spatial size that doesn't
        # divide the block size.
        (3, 98, 101),
    ],
)
def test_fused_norm_glu_ls_resid_matches_reference(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    ``FusedNormGluLayerScaleResid`` matches the unfused PyTorch ops.

    The DConv envelope kernel computes
    ``residual + layer_scale * glu(group_norm(z), dim=1)`` in one launch.
    ``z`` has ``2C`` channels; ``residual`` and the output have ``C``.

    :param dtype: dtype under test
    :param shape: tensor shape under test
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
    out = _inference_call(
        mod.to(_device()), z.to(_device(), dtype), residual.to(_device(), dtype)
    )

    assert out.dtype == dtype
    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_fused_norm_glu_ls_resid_multi_stage(dtype: torch.dtype) -> None:
    """
    ``FusedNormGluLayerScaleResid`` multi-stage path matches the reference.

    Uses a per-batch *input* element count above ``_SINGLE_STAGE_LIMIT`` (the
    gate is on the 2C-channel GLU input) so the partial-reduce / finalize /
    apply_norm_glu_ls_resid kernels fire instead of the single-stage kernel.

    :param dtype: dtype under test
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
    out = _inference_call(
        mod.to(_device()), z.to(_device(), dtype), residual.to(_device(), dtype)
    )

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 64, 384),  # single-stage chlast, vectorized (C % 4 == 0)
        (2, 200, 384),  # multi-stage chlast (small B, per-batch > small limit)
        (2, 100, 383),  # odd C: scalar chlast path
        (2, 2688, 512),  # the real cross-transformer shape
    ],
)
def test_cuda_my_group_norm_matches_fallback(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """
    ``CUDAMyGroupNorm`` (transpose-free ``(B, T, C)`` norm) matches fallback.

    Covers the channel-last kernels' single-stage and multi-stage paths and
    both the vectorized (``C % 4 == 0``) and scalar affine loops.

    :param dtype: dtype under test
    :param shape: ``(B, T, C)`` shape under test
    """
    channels = shape[-1]
    mod = CUDAMyGroupNorm(_make_gn(channels))
    x = torch.randn(*shape)  # (B, T, C)

    ref = mod(x.to(torch.float32))
    out = _inference_call(mod.to(_device()), x.to(_device(), dtype))

    assert out.shape == ref.shape
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
def test_cuda_multihead_attention_matches_reference(dtype: torch.dtype) -> None:
    """
    ``CUDAMultiheadAttention``'s SDPA path matches ``nn.MultiheadAttention``.

    Self-attention, batch_first, ``need_weights=False`` — the configuration the
    wrapper optimises — on CUDA in FP16/BF16, against the FP32 CPU reference.

    :param dtype: dtype under test
    """
    import copy

    torch.manual_seed(0)
    mha = nn.MultiheadAttention(64, 4, batch_first=True).eval()
    x = torch.randn(2, 50, 64)

    with torch.no_grad():
        ref, _ = mha(x, x, x, need_weights=False)

    # Deep-copy before wrapping: the wrapper shares parameter storage with the
    # source MHA, so moving it to CUDA would otherwise also move the reference.
    wrapped = CUDAMultiheadAttention.from_mha(copy.deepcopy(mha))
    wrapped = wrapped.to(device=_device(), dtype=dtype).eval()

    with torch.no_grad():
        out, weights = wrapped(
            x.to(_device(), dtype),
            x.to(_device(), dtype),
            x.to(_device(), dtype),
            need_weights=False,
        )

    assert weights is None
    assert out.dtype == dtype
    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
def test_cuda_multihead_attention_training_preserves_dropout() -> None:
    """
    Training calls use native MHA so attention dropout is not omitted.
    """
    import copy

    mha = nn.MultiheadAttention(16, 4, dropout=1.0, batch_first=True).train()
    wrapped = CUDAMultiheadAttention.from_mha(copy.deepcopy(mha)).to(
        device=_device(), dtype=torch.float16
    )
    assert wrapped.training
    x = torch.randn(2, 7, 16, device=_device(), dtype=torch.float16)

    # Spy on the wrapped module to assert the routing itself: training calls
    # must reach the fallback MHA. We don't assert on the native outcome —
    # backends disagree on dropout=1.0 (some raise NotImplementedError, the
    # flash kernels raise RuntimeError about p_dropout < 1) and that behavior
    # belongs to nn.MultiheadAttention, not this wrapper.
    calls: list[int] = []
    original_forward = wrapped._fallback.forward

    def spying_forward(*args: torch.Tensor, **kwargs: object):
        calls.append(1)
        return original_forward(*args, **kwargs)

    wrapped._fallback.forward = spying_forward
    with torch.no_grad():
        try:
            wrapped(x, x, x, need_weights=False)
        except (RuntimeError, NotImplementedError):
            pass  # native rejection of p=1.0 is fine; routing is the contract
    assert calls, "training call did not route to the wrapped MHA"


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES)
@pytest.mark.parametrize("mask_kind", ["bool", "float", "causal"])
def test_cuda_multihead_attention_masked_matches_reference(
    dtype: torch.dtype, mask_kind: str
) -> None:
    """
    Masked / causal calls route to the wrapped MHA and keep its semantics.

    ``nn.MultiheadAttention``'s mask contract (bool ``True`` = disallowed)
    is the opposite of ``F.scaled_dot_product_attention``'s, so these must
    go through the fallback, not a hand-rolled SDPA call.

    :param dtype: dtype under test
    :param mask_kind: attention-mask flavour under test
    """
    import copy

    torch.manual_seed(0)
    mha = nn.MultiheadAttention(64, 4, batch_first=True).eval()
    x = torch.randn(2, 50, 64)
    if mask_kind == "bool":
        mask = torch.zeros(50, 50, dtype=torch.bool)
        mask[:, ::5] = True  # True = NOT allowed to attend
        kwargs: dict = dict(attn_mask=mask)
    elif mask_kind == "float":
        mask = torch.zeros(50, 50)
        mask[:, ::5] = float("-inf")
        kwargs = dict(attn_mask=mask)
    else:
        mask = torch.triu(torch.ones(50, 50, dtype=torch.bool), diagonal=1)
        kwargs = dict(attn_mask=mask, is_causal=True)

    with torch.no_grad():
        ref, _ = mha(x, x, x, need_weights=False, **kwargs)

    wrapped = CUDAMultiheadAttention.from_mha(copy.deepcopy(mha))
    wrapped = wrapped.to(device=_device(), dtype=dtype).eval()
    dev_kwargs = {
        k: (v.to(_device(), dtype) if k == "attn_mask" and mask_kind == "float" else v)
        for k, v in kwargs.items()
    }
    if "attn_mask" in dev_kwargs and mask_kind != "float":
        dev_kwargs["attn_mask"] = dev_kwargs["attn_mask"].to(_device())

    with torch.no_grad():
        out, _ = wrapped(
            x.to(_device(), dtype),
            x.to(_device(), dtype),
            x.to(_device(), dtype),
            need_weights=False,
            **dev_kwargs,
        )

    torch.testing.assert_close(out.float().cpu(), ref, **_tol(dtype))


@cuda_only
def test_apply_cuda_optimizations_idempotent() -> None:
    """
    A second ``apply_cuda_optimizations`` call swaps nothing.

    Unlike the Metal pass, ``nn.MultiheadAttention`` is deliberately left
    alone on CUDA (its C++ inference fast path beats a Python wrapper), so
    the MHA must remain untouched here.
    """
    from unblend.transformer import MyGroupNorm

    model = nn.Module()
    model.attn = nn.MultiheadAttention(32, 4, batch_first=True)
    model.gn = nn.GroupNorm(1, 16)
    model.mygn = MyGroupNorm(1, 16)
    model.eval()

    first = apply_cuda_optimizations(model)
    assert first["group_norm"] == 1
    assert first["my_group_norm"] == 1
    assert first["multi_head_attention"] == 0
    assert type(model.attn) is nn.MultiheadAttention

    second = apply_cuda_optimizations(model)
    assert all(count == 0 for count in second.values()), second
    assert type(model.attn) is nn.MultiheadAttention
    assert not model.gn.training
    assert not model.mygn.training


@cuda_only
@pytest.mark.parametrize("dtype", LP_DTYPES + [torch.float32])
@pytest.mark.parametrize("shape", [(2, 8, 37, 64), (1, 4, 5, 62), (3, 1300, 64)])
def test_fused_roformer_rotary_matches_reference(
    dtype: torch.dtype, shape: tuple
) -> None:
    """
    The fused interleaved rotary kernel matches the eager reference formula.

    Covers vectorized (dim % 4 == 0) and scalar paths plus multi-dim leading
    batch layouts. The kernel accumulates in FP32 so it is at least as exact
    as the reference; tolerance stays modest for the fp16 storage round-trip.

    :param dtype: dtype under test
    :param shape: ``[..., seq, dim]`` shape under test
    """
    from unblend.cuda import fused_roformer_rotary

    torch.manual_seed(5)
    seq_len = shape[-2]
    dim = shape[-1]
    angles = torch.randn(seq_len, dim // 2)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    t = torch.randn(*shape).to(dtype)

    ref_t = t.clone()
    x1, x2 = ref_t.unflatten(-1, (-1, 2)).unbind(dim=-1)
    ref = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    out = fused_roformer_rotary(t.to(_device()), cos.to(_device()), sin.to(_device()))

    tol = (
        dict(atol=2e-2, rtol=2e-2)
        if dtype != torch.float32
        else dict(atol=1e-5, rtol=1e-5)
    )
    torch.testing.assert_close(out.float().cpu(), ref.float(), **tol)
