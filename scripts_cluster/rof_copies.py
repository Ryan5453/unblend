"""
Attribute every aten::copy_ in a RoFormer forward to its input/output
shapes via chrome-trace External-id correlation.
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


def main() -> None:
    """Profile and report copy_ shapes."""
    sep = Separator(model="melband_roformer_kim", device="cuda", dtype=torch.float16)
    sep.separate(f"{MUSDB}/{TRACK}")  # warmup

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        sep.separate(f"{MUSDB}/{TRACK}")

    prof.export_chrome_trace("/tmp/rof_copy_trace.json")
    del sep
    with open("/tmp/rof_copy_trace.json") as fh:
        trace = json.load(fh)

    ext_by_key = {}
    for ev in trace["traceEvents"]:
        args = ev.get("args") or {}
        ext = args.get("External id")
        if ext is None:
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if cat == "kernel":
            dur = ev.get("dur", 0)
            prev = ext_by_key.get(ext)
            if prev is None or prev[0] != "op":
                ext_by_key[ext] = ("kernel", name[:60], dur)
            else:
                pass
        elif cat == "cpu_op" and name.startswith("aten::"):
            shapes = args.get("Input Dims")
            ext_by_key[ext] = ("op", f"{name} {shapes}", 0)

    agg = {}
    total_us = 0
    for kind, info, dur in ext_by_key.values():
        if kind != "kernel":
            continue
        total_us += dur

    # Second pass: for each kernel, find the aten op with the SAME External id
    # (kineto assigns one External id per launching aten op, shared by the
    # kernels it launches).
    ops_by_ext = {}
    for ev in trace["traceEvents"]:
        args = ev.get("args") or {}
        ext = args.get("External id")
        cat = ev.get("cat", "")
        if ext is None:
            continue
        if cat == "cpu_op":
            ops_by_ext.setdefault(ext, (ev.get("name", ""), args.get("Input Dims")))

    for ev in trace["traceEvents"]:
        if ev.get("cat") != "kernel":
            continue
        dur = ev.get("dur", 0)
        ext = (ev.get("args") or {}).get("External id")
        op_name, shapes = ops_by_ext.get(ext, ("?", None))
        if not op_name.startswith("aten::copy_") and not op_name.startswith(
            "aten::contiguous"
        ):
            continue
        sig = f"{op_name.split('(')[0]} {shapes}"
        ms, cnt = agg.get(sig, (0.0, 0))
        agg[sig] = (ms + dur / 1000.0, cnt + 1)

    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    print(f"copy_ kernel total {total_us / 1000:.1f} ms (all kernels)")
    print("=== attributed copy_ families (per forward) ===")
    for sig, (ms, cnt) in rows[:16]:
        print(f"{ms:8.1f} ms {cnt:5d}x  {sig}")


if __name__ == "__main__":
    main()
