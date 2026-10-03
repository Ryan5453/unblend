"""
End-to-end check that the fused-kernel swap leaves an HTDemucs forward intact.

The per-kernel tests in ``test_metal_kernels.py`` / ``test_cuda_kernels.py``
pin each fused module in isolation; this one builds a small randomly
initialised HTDemucs, runs it in FP16 once on the reference path and once
after the module swap, and asserts the separated sources agree. It catches
wiring bugs (a dropped inject, a residual added twice, a GELU applied on the
last decoder layer) that no single-module test can see.
"""

import copy

import pytest
import torch

from unblend.backends import disable_custom_kernels
from unblend.htdemucs import HTDemucs

# norm_starts=0 with norm_groups=1 puts fusable GroupNorms in every encoder and
# decoder layer; the defaults (as in the released checkpoints) leave only the
# DConv norms.
CONFIGS = [
    pytest.param(dict(), id="released-norms"),
    pytest.param(dict(norm_starts=0, norm_groups=1), id="all-layer-norms"),
]


def _tiny_htdemucs(**overrides) -> HTDemucs:
    """
    Build a small randomly initialised HTDemucs.

    :param overrides: Constructor arguments replacing the tiny defaults
    :return: The model in eval mode, on CPU in FP32
    """
    torch.manual_seed(0)
    config = dict(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )
    config.update(overrides)
    return HTDemucs(**config).eval()


def _compare(swap, device: str, overrides: dict) -> dict[str, int]:
    """
    Run the reference and swapped models on the same mix and compare.

    FP16 rounding alone moves the output by up to ~0.2% of its peak, so the
    swapped model is held to the FP16 reference within that, and to the FP32
    reference no worse than the FP16 reference is.

    :param swap: ``apply_metal_optimizations`` or ``apply_cuda_optimizations``
    :param device: Device the models run on
    :param overrides: HTDemucs constructor overrides
    :return: The swap counts, so callers can assert the swap did something
    """
    model = _tiny_htdemucs(**overrides)
    reference32 = copy.deepcopy(model)
    disable_custom_kernels(reference32)
    reference32 = reference32.to(device)
    reference16 = copy.deepcopy(reference32).half()
    fused = model.to(device, torch.float16)
    counts = swap(fused)

    torch.manual_seed(1)
    mix = torch.randn(1, 2, 8000).to(device)
    with torch.inference_mode():
        exact = reference32(mix)
        expected = reference16(mix.half()).float()
        actual = fused(mix.half()).float()

    assert actual.shape == expected.shape == (1, 2, 2, 8000)
    assert torch.isfinite(actual).all()
    peak = exact.abs().max().item()
    torch.testing.assert_close(actual, expected, atol=4e-3 * peak, rtol=0)
    fused_err = (actual - exact).abs().max().item()
    eager_err = (expected - exact).abs().max().item()
    assert fused_err <= 2 * eager_err + 1e-4 * peak, (fused_err, eager_err)
    return counts


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs Apple Silicon (MPS)"
)
@pytest.mark.parametrize("overrides", CONFIGS)
def test_metal_swap_matches_reference_htdemucs(overrides: dict) -> None:
    """
    ``apply_metal_optimizations`` leaves a small HTDemucs's output unchanged.

    :param overrides: HTDemucs constructor overrides
    """
    from unblend.metal import apply_metal_optimizations

    counts = _compare(apply_metal_optimizations, "mps", overrides)
    assert counts["h_enc_layer"] and counts["h_dec_layer"] and counts["fused_dconv"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
@pytest.mark.parametrize("overrides", CONFIGS)
def test_cuda_swap_matches_reference_htdemucs(overrides: dict) -> None:
    """
    ``apply_cuda_optimizations`` leaves a small HTDemucs's output unchanged.

    :param overrides: HTDemucs constructor overrides
    """
    from unblend.cuda import apply_cuda_optimizations

    counts = _compare(apply_cuda_optimizations, "cuda", overrides)
    assert counts["h_enc_layer"] and counts["h_dec_layer"] and counts["fused_dconv"]


def _injected_encoder_call() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """
    Capture a frequency-encoder call that receives an inject.

    HTDemucs injects the time branch into the frequency encoder once the
    time encoder runs out of layers; the tiny config above never reaches
    that, so this uses a deeper geometry.

    :return: ``(encoder layer, x, inject)`` from a real FP32 CPU forward.
    """
    model = _tiny_htdemucs(nfft=1024, depth=4)
    captured: list[tuple[torch.nn.Module, torch.Tensor, torch.Tensor]] = []

    def hook(module: torch.nn.Module, args: tuple) -> None:
        """
        Record the first call that carries an inject.
        """
        if len(args) > 1 and args[1] is not None and not captured:
            captured.append((module, args[0].detach(), args[1].detach()))

    handles = [layer.register_forward_pre_hook(hook) for layer in model.encoder]
    with torch.no_grad():
        model(torch.randn(1, 2, 8000))
    for handle in handles:
        handle.remove()
    assert captured, "no encoder layer received an inject"
    return captured[0]


@pytest.mark.parametrize(
    "device, module",
    [
        pytest.param(
            "mps",
            "unblend.metal",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="needs MPS"
            ),
        ),
        pytest.param(
            "cuda",
            "unblend.cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs an NVIDIA GPU"
            ),
        ),
    ],
)
def test_fused_encoder_applies_the_inject(device: str, module: str) -> None:
    """
    The fused encoder layer adds the inject exactly as the eager layer does.
    The inject is enlarged so that dropping it, or adding it twice, moves the
    output far beyond FP16 rounding.

    :param device: GPU device to run on.
    :param module: Module providing ``FusedHEncLayer``.
    """
    import importlib

    fused_cls = importlib.import_module(module).FusedHEncLayer
    layer, x, inject = _injected_encoder_call()
    inject = inject * 20
    with torch.no_grad():
        expected = layer(x, inject)
        without = layer(x, None)
    eager = copy.deepcopy(layer).to(device, torch.float16)
    disable_custom_kernels(eager)
    fused = fused_cls(copy.deepcopy(layer).to(device, torch.float16))
    args = (x.to(device, torch.float16), inject.to(device, torch.float16))
    with torch.inference_mode():
        eager_out = eager(*args).float().cpu()
        fused_out = fused(*args).float().cpu()
    fp16_error = (eager_out - expected).abs().max()
    fused_error = (fused_out - expected).abs().max()
    assert fused_error <= 2 * fp16_error + 1e-3
    assert (without - expected).abs().max() > 20 * (fused_error + 1e-3)
