"""
Interleaved A/B for the fused Metal kernels on MPS.

Two separators are built up front -- one with the fused kernels, one forced
onto vanilla PyTorch ops -- and then alternated A/B/A/B within a single
process. Sequential passes cannot measure this: the machine drifts ~15% over a
long run, and the effect being measured is expected to be a few percent, so a
pass-per-variant would report drift rather than kernels. Alternating puts both
variants under the same thermal envelope, and best-of-N keeps the least
heat-contaminated sample from each.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from unblend import Separator

SAMPLE_RATE = 44100


def make_clip(seconds: float) -> torch.Tensor:
    """
    Build a fixed synthetic stereo clip.

    Content does not affect throughput, and a synthetic clip avoids decode
    variance contaminating the timing.

    :param seconds: Clip length in seconds.
    :return: A ``[2, seconds * 44100]`` waveform.
    """
    count = int(seconds * SAMPLE_RATE)
    ramp = torch.linspace(0, seconds, count)
    generator = torch.Generator().manual_seed(0)
    noise = torch.rand(count, generator=generator) * 0.05
    return torch.stack(
        [
            torch.sin(2 * torch.pi * 220 * ramp) * 0.3 + noise,
            torch.sin(2 * torch.pi * 330 * ramp) * 0.3 + noise,
        ]
    )


def time_once(separator: Separator, clip: torch.Tensor) -> float:
    """
    Time one seeded separation.

    :param separator: Separator to drive.
    :param clip: Input waveform.
    :return: Wall seconds for the call.
    """
    started = time.perf_counter()
    separator.separate((clip, SAMPLE_RATE), seed=0)
    return time.perf_counter() - started


def run_model(
    model: str, precision: str, seconds: float, rounds: int
) -> dict[str, object]:
    """
    Interleave fused-kernel and native-op runs for one model.

    :param model: Registry model name.
    :param precision: ``fp16``, ``bf16``, or ``fp32``.
    :param seconds: Clip length per timed call.
    :param rounds: How many A/B rounds to run.
    :return: A result row, or an error row if the model could not be built.
    """
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[precision]
    clip = make_clip(seconds)
    try:
        fused = Separator(model=model, device="mps", dtype=dtype, custom_kernels=True)
        native = Separator(model=model, device="mps", dtype=dtype, custom_kernels=False)
    except Exception as exc:  # noqa: BLE001 - one bad model must not stop the sweep
        return {
            "model": model,
            "precision": precision,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # Warm both so neither pays shader compilation inside a timed round.
    time_once(fused, clip)
    time_once(native, clip)

    fused_times: list[float] = []
    native_times: list[float] = []
    for index in range(rounds):
        # Swap which variant leads each round so any within-round ordering
        # effect (cache state, clock ramp) cancels instead of accumulating.
        if index % 2 == 0:
            fused_times.append(time_once(fused, clip))
            native_times.append(time_once(native, clip))
        else:
            native_times.append(time_once(native, clip))
            fused_times.append(time_once(fused, clip))

    del fused, native
    torch.mps.empty_cache()

    best_fused = min(fused_times)
    best_native = min(native_times)
    return {
        "model": model,
        "precision": precision,
        "clip_seconds": seconds,
        "rounds": rounds,
        "fused_best_s": round(best_fused, 4),
        "native_best_s": round(best_native, 4),
        "fused_median_s": round(statistics.median(fused_times), 4),
        "native_median_s": round(statistics.median(native_times), 4),
        "speedup_best": round(best_native / best_fused, 4),
        "speedup_median": round(
            statistics.median(native_times) / statistics.median(fused_times), 4
        ),
        "fused_spread_pct": round(
            100 * (max(fused_times) - min(fused_times)) / min(fused_times), 1
        ),
        "native_spread_pct": round(
            100 * (max(native_times) - min(native_times)) / min(native_times), 1
        ),
    }


def main() -> None:
    """
    Run the interleaved A/B sweep and write a JSON report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--precision", action="append", default=None)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    precisions = args.precision or ["fp16"]
    rows = []
    for precision in precisions:
        for model in args.model:
            row = run_model(model, precision, args.seconds, args.rounds)
            rows.append(row)
            print(json.dumps(row), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
