"""
Attribute the RoFormer 'other' bucket via chrome-trace JSON parsing.

Avoids key_averages() host-memory blowup: exports the trace, then aggregates
GPU kernel rows by name, bucketing into known vs unknown ("other") and
listing every kernel family inside the unknown bucket with counts and total
microseconds.
"""

import json
import os
import sys

os.environ.setdefault(
    "UNBLEND_CACHE_DIR", "/projects/fahey.rya/unblend-bench/.model-cache"
)
REPO = "/projects/fahey.rya/unblend-cuda/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from unblend.api import Separator  # noqa: E402

MUSDB = "/projects/fahey.rya/datasets/musdb18hq/test"
TRACK = "Al James - Schoolboy Facination/mixture.wav"

KNOWN = (
    "gemm",
    "cutlass",
    "nvjet",
    "flash",
    "elementwise",
    "vectorized",
    "reduce",
    "norm",
    "fft",
    "cat",
    "copy",
    "rnn",
)


def main() -> None:
    """Profile one separation and dump the unknown-kernel families."""
    sep = Separator(model="melband_roformer_kim", device="cuda", dtype=torch.float16)
    sep.separate(f"{MUSDB}/{TRACK}")  # warmup

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        sep.separate(f"{MUSDB}/{TRACK}")

    path = "/tmp/rof_trace.json"
    prof.export_chrome_trace(path)
    del sep
    with open(path) as fh:
        trace = json.load(fh)

    agg = {}
    total_us = 0
    for ev in trace["traceEvents"]:
        if ev.get("cat") != "kernel":
            continue
        dur = ev.get("dur", 0)
        name = ev.get("name", "")
        total_us += dur
        short = name[:110]
        ms, cnt = agg.get(short, (0.0, 0))
        agg[short] = (ms + dur / 1000.0, cnt + 1)

    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    print(
        f"total kernel time {total_us / 1000:.1f} ms across {sum(v[1] for v in agg.values())} launches"
    )
    print("=== ALL kernel families (top 30) ===")
    for name, (ms, cnt) in rows[:30]:
        known = any(k.lower() in name.lower() for k in KNOWN)
        tag = "" if known else "   <-- OTHER"
        print(f"{ms:9.1f} ms {cnt:6d}x  {name[:110]}{tag}")


if __name__ == "__main__":
    main()
