"""
Micro-experiment: permute-reduced DualPathRNN vs the reference forward.

The reference does, per path (freq then time):
    GN -> permute/reshape to [S, B*other, C] -> LSTM -> Linear
       -> view/permute back to 4D -> + residual
i.e. two full-tensor layout shuffles per path around the LSTM.

The variant keeps each path's residual in its NATIVE [S, B*other, C] layout:
    GN reads x4d; native_in = reshape(...)          (one copy)
    y = linear(lstm(native_in))
    out_native = y + original_native                 (add in native layout;
        original_native = native_in - gn_out? NO — original_native is just the
        pre-GN tensor reshaped once and kept)
then a single shuffle back to 4D only at the END of both paths.

This script times reference-vs-variant on captured real tensors and checks
bit-level closeness. Measurement only — no integration.
"""

import os
import sys

os.environ.setdefault(
    "UNBLEND_CACHE_DIR", "/projects/fahey.rya/unblend-bench/.model-cache"
)
REPO = "/projects/fahey.rya/unblend-cuda/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from unblend.api import Separator  # noqa: E402
from unblend.cuda import apply_cuda_optimizations  # noqa: E402

MUSDB = "/projects/fahey.rya/datasets/musdb18hq/test"
TRACK = "Al James - Schoolboy Facination/mixture.wav"


def timed(fn, iters: int = 30) -> float:
    """
    CUDA-event time a callable.

    :param fn: Zero-arg callable
    :param iters: Timed iterations after 3 warmups
    :return: Mean milliseconds
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> None:
    """Run the comparison."""
    sep = Separator(model="scnet_xl_wide_v5", device="cuda", dtype=torch.float16)
    model = sep.model
    apply_cuda_optimizations(model)

    trunk = model.separation_net
    dp = trunk.dp_modules[0]

    captured: list[torch.Tensor] = []
    original_forward = dp.forward

    def capturing(x: torch.Tensor):
        if len(captured) < 1:
            captured.append(x.detach().clone())
        return original_forward(x)

    dp.forward = capturing
    # One full forward_core to capture a realistic dp input.
    core_inputs: list[torch.Tensor] = []
    orig_core = model.forward_core

    def capturing_core(x: torch.Tensor):
        if len(core_inputs) < 1:
            core_inputs.append(x.detach().clone())
        return orig_core(x)

    model.forward_core = capturing_core
    sep.separate(f"{MUSDB}/{TRACK}")
    model.forward_core = orig_core
    dp.forward = original_forward

    x = captured[0]
    print(f"dp input {tuple(x.shape)} {x.dtype}")

    ref_ms = timed(lambda: original_forward(x))
    ref = original_forward(x)

    # --- variant ---
    def variant(dp_mod: torch.nn.Module, x4d: torch.Tensor) -> torch.Tensor:
        """
        DualPathRNN with residuals kept in the LSTM-native layout.

        :param dp_mod: The DualPathRNN module
        :param x4d: Input ``[B, C, F, T]``
        :return: Output ``[B, C, F, T]``
        """
        batch, channels, freq, time = x4d.shape
        original = x4d
        y = dp_mod.norm_layers[0](x4d)
        nat = y.permute(2, 0, 3, 1).reshape(freq, batch * time, channels)
        y, _ = dp_mod.lstm_layers[0](nat)
        y = dp_mod.linear_layers[0](y)
        # Residual + second-path prep without materializing 4D in between:
        # fold path-0 output back onto the ORIGINAL native layout of the
        # input. The original's native twin is a reshape of `original`
        # itself, which we compute once and reuse as the path-0 residual.
        orig_nat = original.permute(2, 0, 3, 1).reshape(freq, batch * time, channels)
        acc_nat = y + orig_nat

        # Path 1 (time): GN must see 4D semantics; GroupNorm over C·F·T per
        # batch is layout-independent, but the reference applies GN to the 4D
        # residual-updated tensor. Reproduce: GN(acc as 4D) requires the 4D
        # view — do GN math manually on the native layout instead.
        # NOTE: this variant is numerics-first; we replicate exactly what the
        # reference computes by going through 4D here (kept for parity check).
        acc4 = (
            acc_nat.view(freq, batch, time, channels).permute(1, 3, 0, 2).contiguous()
        )
        original2 = acc4
        y2 = dp_mod.norm_layers[1](acc4)
        nat2 = y2.permute(3, 0, 2, 1).reshape(time, batch * freq, channels)
        y2, _ = dp_mod.lstm_layers[1](nat2)
        y2 = dp_mod.linear_layers[1](y2)
        y2 = y2.view(time, batch, freq, channels).permute(1, 3, 2, 0)
        return y2 + original2

    var_ms = timed(lambda: variant(dp, x))
    out_var = variant(dp, x)

    diff = (out_var.float() - ref.float()).abs().max().item()
    print(
        f"reference {ref_ms:.3f} ms   variant {var_ms:.3f} ms   -> {ref_ms / var_ms:.3f}x"
    )
    print(f"max|variant-ref| = {diff:.2e}")


if __name__ == "__main__":
    main()
