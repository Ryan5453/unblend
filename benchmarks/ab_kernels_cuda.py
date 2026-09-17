"""
Interleaved A/B harness for the fused CUDA kernels.

``benchmark.py`` has no ``custom_kernels`` flag, and running the two settings
as two sequential passes would measure clock/thermal drift as much as the
kernels (handoff section 6). This alternates A/B/A/B inside a single process
against pre-built separators and reports best-of-N, so drift affects both arms
equally and the minimum is the least contaminated sample.

Also checks that the two paths agree numerically: a speedup is only meaningful
if the fused path computes the same thing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from unblend.api import Separator

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def load_clip(track: Path, seconds: float) -> tuple[torch.Tensor, int]:
    """
    Load a fixed-length clip from a MUSDB mixture.

    A real clip rather than noise, so the measurement reflects the tensor
    shapes and value distribution the models actually see in serving.

    :param track: Path to a ``mixture.wav``.
    :param seconds: Clip length to take from the start.
    :return: ``(waveform, samplerate)`` suitable for ``Separator.separate``.
    """
    # torchcodec is what unblend itself decodes with, so this matches the
    # harness rather than pulling in a dependency the project does not have.
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(str(track)).get_all_samples()
    wav = samples.data.float()
    if wav.dim() == 1:
        wav = wav[None]
    wanted = int(seconds * samples.sample_rate)
    if wav.shape[-1] < wanted:
        raise SystemExit(f"{track} is shorter than {seconds}s")
    return wav[:, :wanted].contiguous(), samples.sample_rate


def time_once(separator: Separator, clip: tuple[torch.Tensor, int], seed: int) -> float:
    """
    Time a single seeded separation with CUDA events.

    :param separator: Separator to exercise.
    :param clip: ``(waveform, samplerate)`` input.
    :param seed: Seed, so both arms draw identical shift offsets.
    :return: Elapsed milliseconds.
    """
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    separator.separate(clip, seed=seed)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop)


def stack_sources(result) -> torch.Tensor:
    """
    Flatten a separation result into one tensor for comparison.

    :param result: ``SeparatedSources`` from a separation.
    :return: Stacked stems in sorted name order, as float32 on CPU.
    """
    return torch.stack(
        [result.sources[name].float().cpu() for name in sorted(result.sources)]
    )


def run_model(
    name: str, precision: str, clip: tuple[torch.Tensor, int], rounds: int, seed: int
) -> dict:
    """
    Interleave fused and reference kernels for one model/precision.

    :param name: Registry model name.
    :param precision: One of ``fp16``, ``bf16``, ``fp32``.
    :param clip: Fixed input clip.
    :param rounds: Number of A/B rounds after warmup.
    :param seed: Seed shared by both arms.
    :return: Result record for this cell.
    """
    dtype = DTYPES[precision]
    on = Separator(model=name, device="cuda", dtype=dtype, custom_kernels=True)
    off = Separator(model=name, device="cuda", dtype=dtype, custom_kernels=False)

    # Warm both arms: the fused path may JIT-compile its extension on first
    # use, and both populate autotune/allocator caches lazily.
    time_once(on, clip, seed)
    time_once(off, clip, seed)

    max_abs_diff = float(
        (
            stack_sources(on.separate(clip, seed=seed))
            - stack_sources(off.separate(clip, seed=seed))
        )
        .abs()
        .max()
    )

    on_ms: list[float] = []
    off_ms: list[float] = []
    for _ in range(rounds):
        on_ms.append(time_once(on, clip, seed))
        off_ms.append(time_once(off, clip, seed))

    best_on, best_off = min(on_ms), min(off_ms)
    record = {
        "model": name,
        "precision": precision,
        "gpu": torch.cuda.get_device_name(0),
        "rounds": rounds,
        "fused_best_ms": best_on,
        "reference_best_ms": best_off,
        "fused_all_ms": on_ms,
        "reference_all_ms": off_ms,
        "speedup": best_off / best_on,
        "fused_spread_pct": 100.0 * (max(on_ms) - best_on) / best_on,
        "reference_spread_pct": 100.0 * (max(off_ms) - best_off) / best_off,
        "max_abs_diff": max_abs_diff,
    }
    print(
        f"{name:28s} {precision:5s} fused {best_on:8.1f} ms  "
        f"ref {best_off:8.1f} ms  -> {record['speedup']:.3f}x  "
        f"(spread {record['fused_spread_pct']:.1f}%/"
        f"{record['reference_spread_pct']:.1f}%, maxdiff {max_abs_diff:.2e})",
        flush=True,
    )
    del on, off
    torch.cuda.empty_cache()
    return record


def main() -> None:
    """Parse arguments and run every requested model/precision cell."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--precisions", nargs="+", default=["fp16", "bf16"])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available")

    clip = load_clip(Path(args.track), args.seconds)
    records = []
    for name in args.models:
        for precision in args.precisions:
            try:
                records.append(run_model(name, precision, clip, args.rounds, args.seed))
            except Exception as exc:
                print(
                    f"{name} {precision}: FAILED {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                records.append(
                    {
                        "model": name,
                        "precision": precision,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            # Persist after every cell so a walltime kill still leaves data.
            Path(args.out).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
