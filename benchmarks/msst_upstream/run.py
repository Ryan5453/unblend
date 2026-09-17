"""
Drive the MSST upstream comparison for the RoFormer and SCNet models.

unblend's ``benchmark.py --include-upstream`` only drives adefossez/demucs, so
it covers the HTDemucs family and nothing else. This runs the same shape of
comparison for the four architectures whose checkpoints come from ZFTurbo's
Music-Source-Separation-Training.

Upstream runs in its own venv as a subprocess. The SDR implementation is taken
from ``benchmark.py`` and substituted into the worker, so both sides are scored
by one function.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import benchmark  # noqa: E402  (needs ROOT on the path first)

#: Registry model -> (MSST model_type, config filename, checkpoint filename).
MODELS: dict[str, tuple[str, str, str]] = {
    "bs_roformer_sw": (
        "bs_roformer",
        "config_bs_roformer_sw.yaml",
        "bs_roformer_sw.ckpt",
    ),
    "bs_roformer_anvuew": (
        "bs_roformer",
        "config_bs_roformer_anvuew.yaml",
        "bs_roformer_anvuew.ckpt",
    ),
    "melband_roformer_kim": (
        "mel_band_roformer",
        "config_melband_roformer_kim.yaml",
        "melband_roformer_kim.ckpt",
    ),
    "scnet_small": (
        "scnet_masked",
        "config_musdb18_scnet_small.yaml",
        "scnet_small.ckpt",
    ),
    "scnet_xl_wide_v5": (
        "scnet",
        "config_musdb18_scnet_xl_more_wide_v5.yaml",
        "scnet_xl_wide_v5.ckpt",
    ),
}


def build_worker_source() -> str:
    """
    Materialise the worker with benchmark.py's SDR substituted in.

    :return: Runnable worker source.
    """
    template = (HERE / "worker.py").read_text()
    return template.replace(
        "# __SHARED_SDR__", inspect.getsource(benchmark._compute_sdr)
    )


def run_model(
    name: str, musdb_root: Path, limit: int, device: str, out_dir: Path
) -> dict:
    """
    Run one model through upstream and collect its per-track rows.

    :param name: Registry model name.
    :param musdb_root: MUSDB test split directory.
    :param limit: Track count.
    :param device: Torch device string.
    :param out_dir: Where to write the per-model NDJSON.
    :return: Summary dict for this model.
    """
    model_type, config_name, ckpt_name = MODELS[name]
    config = ROOT / "benchmarks" / "msst_configs" / config_name
    checkpoint = HERE / "checkpoints" / ckpt_name
    for path in (config, checkpoint):
        if not path.exists():
            return {"model": name, "status": "missing", "missing": str(path)}

    worker = HERE / "_worker_materialised.py"
    worker.write_text(build_worker_source())
    out_path = out_dir / f"{name}.ndjson"

    proc = subprocess.run(
        [
            str(HERE / ".venv" / "bin" / "python"),
            str(worker),
            "--msst-root",
            str(HERE / "msst"),
            "--model-type",
            model_type,
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--musdb-root",
            str(musdb_root),
            "--limit",
            str(limit),
            "--device",
            device,
        ],
        capture_output=True,
        text=True,
    )
    rows = [
        json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")
    ]
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n" if rows else "")
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "model": name,
        "model_type": model_type,
        "status": "ok" if ok else "failed",
        "tracks_ok": len(ok),
        "tracks_error": len(rows) - len(ok),
        "wall_sec": sum(r["elapsed_sec"] for r in ok) or None,
        "audio_sec": sum(r["audio_sec"] for r in ok) or None,
        "returncode": proc.returncode,
    }
    if summary["wall_sec"]:
        summary["realtime"] = summary["audio_sec"] / summary["wall_sec"]
    per_stem: dict[str, list[float]] = {}
    for row in ok:
        for stem, value in row.get("sdr", {}).items():
            per_stem.setdefault(stem, []).append(value)
    summary["sdr"] = {stem: sum(v) / len(v) for stem, v in per_stem.items() if v}
    if not ok:
        summary["stderr_tail"] = proc.stderr.strip().splitlines()[-6:]
    return summary


def main() -> int:
    """
    Run every configured model and write a combined summary.

    :return: Exit status.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb-root", default="/Users/ryan/Music/musdb18hq/test")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.model or list(MODELS)

    summaries = []
    for name in names:
        print(f"==> upstream {name}", flush=True)
        summary = run_model(
            name, Path(args.musdb_root), args.limit, args.device, out_dir
        )
        summaries.append(summary)
        print(f"    {json.dumps(summary)}", flush=True)
        # Written after every model so an interrupt keeps prior results.
        (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
