"""
MSST-side worker for the upstream comparison.

Runs inside the MSST venv, not unblend's, so the two implementations stay
isolated. Drives upstream the way a user would: build the model with MSST's own
``get_model_from_config``, then separate with MSST's own ``demix`` (their
chunking, overlap-add and normalisation), so the comparison is of pipelines
rather than of bare ``forward`` calls.

Emits one JSON object per track on stdout, so a crash on track N still leaves
tracks 1..N-1 usable -- the same lesson as benchmark.py's incremental flush.

``_compute_sdr`` is deliberately not defined here. The driver substitutes
benchmark.py's own implementation into the injection marker below, exactly as
``_build_upstream_worker_source`` does for the demucs worker, so unblend and
upstream are scored by one function rather than by two copies that can drift
apart (a hand-written copy here already disagreed with it on the silent-
reference and noise-floor edge cases). This file is therefore a template and is
not runnable as-is; the marker is spelled only once, at the injection site, so
that substituting it cannot corrupt this docstring.
"""

from __future__ import annotations

import argparse
import json
import math  # noqa: F401  -- used by the injected _compute_sdr
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# __SHARED_SDR__


def main() -> int:
    """
    Separate every requested track with upstream MSST and report per-track
    timing and SDR.

    :return: Process exit status.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--msst-root", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--musdb-root", required=True)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    sys.path.insert(0, args.msst_root)
    from utils.model_utils import demix
    from utils.settings import get_model_from_config

    device = torch.device(args.device)
    model, config = get_model_from_config(args.model_type, args.config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    tracks = sorted(d for d in Path(args.musdb_root).iterdir() if d.is_dir())[
        : args.limit
    ]
    for index, track in enumerate(tracks, start=1):
        mixture, rate = sf.read(track / "mixture.wav", dtype="float32", always_2d=True)
        mix = mixture.T  # soundfile gives [samples, channels]
        try:
            started = time.perf_counter()
            with torch.inference_mode():
                estimates = demix(config, model, mix, device, args.model_type)
            elapsed = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(
                json.dumps(
                    {
                        "track": track.name,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
            continue

        # demix returns a dict for multi-instrument models; a bare array only
        # in Demucs mode, which this harness never drives.
        if not isinstance(estimates, dict):
            print(
                json.dumps(
                    {
                        "track": track.name,
                        "status": "error",
                        "error": f"expected a dict of stems, got {type(estimates).__name__}",
                    }
                ),
                flush=True,
            )
            continue

        scores = {}
        for stem, estimate in estimates.items():
            reference_path = track / f"{stem}.wav"
            if not reference_path.exists():
                continue
            reference, _ = sf.read(reference_path, dtype="float32", always_2d=True)
            scores[stem] = _compute_sdr(  # noqa: F821  -- injected above
                torch.from_numpy(np.asarray(estimate)),
                torch.from_numpy(reference.T.copy()),
            )

        print(
            json.dumps(
                {
                    "track": track.name,
                    "track_index": index,
                    "status": "ok",
                    "elapsed_sec": elapsed,
                    "audio_sec": mix.shape[-1] / rate,
                    "sdr": scores,
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
