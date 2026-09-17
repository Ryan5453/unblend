"""
Controlled precision A/B for CUDA: why is fp32 outrunning fp16?

The 50-track headline shows fp32 beating fp16 by 1.3-1.4x on plain HTDemucs,
which inverts the usual expectation. The headline runs let CUDA auto-sizing pick
a different ``chunk_batch_size`` per precision, so this pins the batch size,
alternates the arms inside one process (per BENCHMARK.md section 2 -- sequential
passes measure drift), and reports best-of-N.

A third arm converts the model to ``channels_last``. fp16 convolutions only reach
cuDNN's tensor-core kernels in NHWC, whereas fp32 gets TF32 tensor cores in NCHW
regardless; if that is the mechanism, NHWC should recover most of fp16's deficit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unblend.api import Separator

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_clip(track: Path, seconds: float) -> tuple[torch.Tensor, int]:
    """
    Load a fixed clip via the decoder unblend itself uses.

    :param track: Path to a mixture wav.
    :param seconds: Seconds to take from the start.
    :return: ``(waveform, samplerate)``.
    """
    from torchcodec.decoders import AudioDecoder

    s = AudioDecoder(str(track)).get_all_samples()
    wav = s.data.float()
    if wav.dim() == 1:
        wav = wav[None]
    n = int(seconds * s.sample_rate)
    return wav[:, :n].contiguous(), s.sample_rate


def timed(sep: Separator, clip, cbs: int, seed: int) -> float:
    """
    Time one seeded separation at a fixed batch size.

    :param sep: Separator to run.
    :param clip: ``(waveform, samplerate)``.
    :param cbs: Forced chunk batch size, so precision is the only variable.
    :param seed: Shared seed.
    :return: Milliseconds.
    """
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    sep.separate(clip, seed=seed, chunk_batch_size=cbs)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def build(model: str, prec: str, chlast: bool) -> Separator:
    """
    Construct a separator, optionally in channels_last layout.

    :param model: Registry model name.
    :param prec: Precision key.
    :param chlast: Convert the model to NHWC.
    :return: The separator.
    """
    sep = Separator(model=model, device="cuda", dtype=DTYPES[prec])
    if chlast:
        targets = getattr(sep.model, "models", None) or [sep.model]
        for t in targets:
            t.to(memory_format=torch.channels_last)
    return sep


def main() -> None:
    """Run the interleaved precision comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--models", nargs="+", default=["htdemucs", "htdemucs_6s"])
    ap.add_argument("--cbs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clip = load_clip(Path(args.track), args.seconds)
    arms = [
        ("fp32", False),
        ("fp16", False),
        ("bf16", False),
        ("fp16", True),
        ("fp32", True),
    ]
    out = []

    for model in args.models:
        seps = {}
        for prec, chlast in arms:
            try:
                seps[(prec, chlast)] = build(model, prec, chlast)
            except Exception as exc:
                print(f"{model} {prec} chlast={chlast}: BUILD FAILED {exc}", flush=True)

        for key, sep in seps.items():  # warm every arm before timing any
            try:
                timed(sep, clip, args.cbs, args.seed)
            except Exception as exc:
                print(f"{model} {key}: WARMUP FAILED {exc}", flush=True)

        times: dict = {k: [] for k in seps}
        for _ in range(args.rounds):
            for key, sep in seps.items():
                try:
                    times[key].append(timed(sep, clip, args.cbs, args.seed))
                except Exception as exc:
                    print(f"{model} {key}: RUN FAILED {exc}", flush=True)

        base = None
        print(
            f"\n=== {model} (cbs={args.cbs}, {args.seconds:.0f}s clip, "
            f"best of {args.rounds}) ===",
            flush=True,
        )
        for (prec, chlast), ts in times.items():
            if not ts:
                continue
            best = min(ts)
            if base is None:
                base = best
            label = f"{prec}{' +chlast' if chlast else ''}"
            rt = args.seconds * 1000.0 / best
            print(
                f"  {label:14s} {best:8.1f} ms  {rt:7.2f}x realtime  "
                f"{base / best:5.2f}x vs fp32",
                flush=True,
            )
            out.append(
                {
                    "model": model,
                    "precision": prec,
                    "channels_last": chlast,
                    "cbs": args.cbs,
                    "best_ms": best,
                    "all_ms": ts,
                    "realtime": rt,
                    "vs_fp32": base / best,
                }
            )
        for sep in seps.values():
            del sep
        torch.cuda.empty_cache()
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
