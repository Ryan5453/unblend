"""
Collect sharded SLURM benchmark results and check them before reporting.

The campaign fans out across ~100 jobs, so the results are only trustworthy if
the shards are verified to have run where they claimed. This does two things:

1. **Asserts host homogeneity.** Every CPU shard must report the same
   ``cpu_model`` and thread count; any shard that differs is quarantined rather
   than averaged in (handoff section 0). The same check applies per GPU type.
2. **Emits the markdown tables** in the column layout BENCHMARK.md already uses.

Reads ``benchmark_summary.csv`` + ``benchmark_metadata.json`` from each result
directory, and falls back to ``benchmark_partial.jsonl`` when a shard was killed
before it could write its CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _f(row: dict, *keys: str) -> float | None:
    """
    First numeric value among ``keys`` present in ``row``.

    :param row: A summary row.
    :param keys: Candidate column names, in preference order.
    :return: The value as a float, or ``None`` if none are usable.
    """
    for key in keys:
        raw = row.get(key)
        if raw in (None, "", "None"):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value == value:  # reject nan
            return value
    return None


def realtime_of(row: dict) -> tuple[float | None, bool]:
    """
    Realtime factor for one summary row, preferring steady state.

    Two corrections matter here. First, the audio duration is the shard's own
    rather than a mean track length, because MUSDB durations span 76-430 s and
    a mean would misreport any ``--limit`` or ``--track-offset`` shard. Second,
    the first track carries model init, cuDNN autotuning and CUDA-graph
    capture; BENCHMARK.md section 2 requires that be discarded, so tracks 2..N
    are used whenever the shard has more than one.

    :param row: A summary row.
    :return: ``(realtime_factor, is_steady_state)``; factor is ``None`` when it
        cannot be computed.
    """
    audio = _f(row, "ok_audio_sec_excl_first")
    wall = _f(row, "remaining_total_sec")
    if audio is not None and wall is not None and wall > 0 and audio > 0:
        return audio / wall, True

    # Single-track shard: no warmup is possible, so this is a cold measurement.
    audio = _f(row, "ok_audio_sec")
    wall = _f(row, "track_total_sec", "dataset_wall_sec", "attempted_track_sec")
    if audio is None or wall is None or wall <= 0:
        return None, False
    return audio / wall, False


def emit_markdown(shards: list, group_precision: bool) -> None:
    """
    Print the results as a markdown table in BENCHMARK.md's column layout.

    Shards whose host disagrees with the majority are excluded, not rendered:
    the point of recording ``cpu_model``/``gpu_name`` is that a shard from the
    wrong hardware must be thrown out rather than averaged in, and a table is
    exactly where such a row would silently do damage.

    :param shards: Loaded shards.
    :param group_precision: Include a precision column.
    """
    report = check_homogeneity(shards)
    quarantined = {
        name
        for info in report.values()
        for names in info["outliers"].values()
        for name in names
    }

    rows = []
    excluded = []
    for directory, meta, summary_rows in shards:
        if directory.name in quarantined:
            excluded.append(
                f"{directory.name} ({meta.get('gpu_name') or meta.get('cpu_model')})"
            )
            continue
        for row in summary_rows:
            rt, steady = realtime_of(row)
            rows.append(
                {
                    "steady": steady,
                    "model": row.get("model", "?"),
                    "precision": row.get("precision", "?"),
                    "wall": _f(row, "track_total_sec", "dataset_wall_sec"),
                    "realtime": rt,
                    "sdr": _f(row, "mean_sdr"),
                    "vram": _f(row, "peak_vram_mb"),
                    "rss": _f(row, "peak_rss_mb"),
                    "status": row.get("status", "?"),
                    "hw": meta.get("gpu_name") or meta.get("cpu_model") or "?",
                }
            )
    rows.sort(key=lambda r: (r["model"], r["precision"]))

    head = "| model |" + (" precision |" if group_precision else "")
    head += " wall | realtime | mean SDR | peak VRAM |"
    sep = "|---|" + ("---|" if group_precision else "") + "---:|---:|---:|---:|"
    print(head)
    print(sep)
    for r in rows:
        wall = f"{r['wall']:.1f} s" if r["wall"] is not None else "—"
        rt = f"{r['realtime']:.2f}x" if r["realtime"] is not None else "—"
        if r["realtime"] is not None and not r["steady"]:
            rt += " (cold)"
        sdr = f"{r['sdr']:.3f}" if r["sdr"] is not None else "—"
        vram = f"{r['vram']:.0f} MB" if r["vram"] is not None else "—"
        line = f"| `{r['model']}` |"
        if group_precision:
            line += f" {r['precision']} |"
        line += f" {wall} | {rt} | {sdr} | {vram} |"
        print(line)

    if any(r["realtime"] is not None and not r["steady"] for r in rows):
        print()
        print(
            "> Rows marked (cold) are single-track shards: no warmup pass is "
            "possible, so they include model init and first-call autotuning "
            "and are **not** steady-state throughput."
        )

    if excluded:
        print()
        print(
            f"> **{len(excluded)} shard(s) excluded** for running on non-majority "
            f"hardware: {', '.join(sorted(excluded))}."
        )


MEAN_TRACK_SEC = 249.4


def load_shard(directory: Path) -> tuple[dict, list[dict]]:
    """
    Load one result directory's metadata and summary rows.

    :param directory: A single shard's output directory.
    :return: ``(metadata, summary_rows)``; rows fall back to the JSONL sidecar.
    """
    meta_path = directory / "benchmark_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    summary_path = directory / "benchmark_summary.csv"
    if summary_path.is_file():
        with open(summary_path, newline="") as handle:
            return meta, list(csv.DictReader(handle))

    # Shard died before the CSV write; recover whatever the sidecar caught.
    partial = directory / "benchmark_partial.jsonl"
    if partial.is_file():
        rows = []
        for line in partial.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a killed job
            if record.get("kind") == "summary":
                rows.append(record)
        return meta, rows
    return meta, []


def check_homogeneity(shards: list[tuple[Path, dict, list[dict]]]) -> dict:
    """
    Group shards by device and verify each group ran on one hardware model.

    :param shards: Loaded shards.
    :return: Report mapping each device group to its hosts and any outliers.
    """
    groups: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for directory, meta, _ in shards:
        groups[meta.get("device", "unknown")].append((directory, meta))

    report: dict[str, dict] = {}
    for device, entries in sorted(groups.items()):
        key = "gpu_name" if device == "cuda" else "cpu_model"
        counts: dict[str, list[str]] = defaultdict(list)
        threads: dict[str, list[str]] = defaultdict(list)
        for directory, meta in entries:
            counts[str(meta.get(key))].append(directory.name)
            if device == "cpu":
                threads[str(meta.get("torch_num_threads"))].append(directory.name)

        majority = max(counts, key=lambda k: len(counts[k])) if counts else None
        outliers = {k: v for k, v in counts.items() if k != majority}
        report[device] = {
            "key": key,
            "majority": majority,
            "majority_shards": len(counts.get(majority, [])),
            "outliers": outliers,
            "thread_counts": {k: len(v) for k, v in threads.items()},
            "thread_outlier_shards": (
                {
                    k: v
                    for k, v in threads.items()
                    if k != max(threads, key=lambda t: len(threads[t]))
                }
                if len(threads) > 1
                else {}
            ),
        }
    return report


def main() -> None:
    """Load every shard, run the checks, and print the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-json")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="emit a BENCHMARK.md-shaped table instead of the report",
    )
    parser.add_argument("--precision-column", action="store_true")
    args = parser.parse_args()

    root = Path(args.results)
    shards = []
    for meta_path in sorted(root.rglob("benchmark_metadata.json")):
        directory = meta_path.parent
        meta, rows = load_shard(directory)
        shards.append((directory, meta, rows))
    # Shards killed before any CSV still have a sidecar worth recovering.
    for partial in sorted(root.rglob("benchmark_partial.jsonl")):
        if not (partial.parent / "benchmark_metadata.json").is_file():
            meta, rows = load_shard(partial.parent)
            shards.append((partial.parent, meta, rows))

    if args.markdown:
        emit_markdown(shards, args.precision_column)
        return

    print(f"loaded {len(shards)} shards from {root}\n")

    report = check_homogeneity(shards)
    print("=== host homogeneity ===")
    for device, info in report.items():
        print(f"\n[{device}] {info['key']}")
        print(f"  majority: {info['majority']}  ({info['majority_shards']} shards)")
        if info["outliers"]:
            print("  !! OUTLIERS — quarantine these, do not average them in:")
            for value, names in info["outliers"].items():
                print(f"     {value}: {', '.join(sorted(names)[:8])}")
        else:
            print("  no outliers")
        if device == "cpu":
            print(f"  torch_num_threads: {info['thread_counts']}")
            if info["thread_outlier_shards"]:
                print(f"  !! thread-count outliers: {info['thread_outlier_shards']}")

    print("\n=== shards ===")
    for directory, meta, rows in shards:
        ok = sum(1 for r in rows if str(r.get("status")) == "ok")
        realtimes = [v for v, _ in (realtime_of(row) for row in rows) if v is not None]
        rt = f"{sum(realtimes) / len(realtimes):.2f}x" if realtimes else "-"
        note = (
            "" if (directory / "benchmark_summary.csv").is_file() else "  [RECOVERED]"
        )
        print(
            f"  {str(directory.relative_to(root)):48s} rows={len(rows):3d} ok={ok:3d} rt={rt}{note}"
        )

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "homogeneity": report,
                    "shards": [
                        {"dir": str(d.relative_to(root)), "meta": m, "rows": r}
                        for d, m, r in shards
                    ],
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
