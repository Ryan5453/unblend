from __future__ import annotations

import csv
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import typer
from filelock import FileLock

from unblend.api import Separator, default_device
from unblend.cli.models import ensure_model_available

REFERENCE_STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
DEFAULT_MODELS = ["htdemucs", "htdemucs_ft"]
DEFAULT_PRECISIONS = ["fp32", "fp16", "bf16"]
DEFAULT_COMPILE_MODES = ["false", "true"]
DEFAULT_SHIFTS = [1, 2, 4]
DEFAULT_SPLIT_OVERLAPS = [0.1, 0.25, 0.5]
# Tracks per batched ``separate([...])`` call in --dataset-throughput mode.
# Feeding the whole 50-track set at once holds every decoded waveform plus
# every output stem in CPU RAM simultaneously (~60+ GB for full-length MUSDB
# tracks → OOM). Processing in small groups bounds RAM to ~group_size tracks
# while still exercising apply_model_multi's cross-input pooling; per-track
# throughput is unchanged (the batched win is intra-call tail pooling).
DATASET_THROUGHPUT_GROUP = 8
DEFAULT_UPSTREAM_VERSION = "main"
DEFAULT_UPSTREAM_PYTHON = "3.11"
UPSTREAM_VENV_ROOT = Path(".upstream-venv")
UPSTREAM_REPO = "https://github.com/adefossez/demucs.git"

# Worker source executed inside the isolated upstream-demucs venv. Piped to the
# venv's interpreter via stdin (`python -`) so we don't need a sibling .py file.
# Communicates with the parent via NDJSON on stdout: one JSON object per line.
# Anything non-JSON on stdout is treated as upstream's own logging and surfaced
# verbatim. The worker only depends on torch/torchaudio + upstream demucs and
# must NOT import anything from the unblend repo (different interpreter).
_UPSTREAM_WORKER_TEMPLATE = r'''
from __future__ import annotations
import argparse, json, math, random, sys, time, traceback
from pathlib import Path
import torch
import torchaudio

def _emit(payload: dict) -> None:
    """
    Write one NDJSON event to stdout for the parent benchmark process.

    :param payload: JSON-serialisable dict describing the event.
    """
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

# __SHARED_SDR__

def main() -> int:
    """
    Worker entry point. Parses CLI args, loads upstream demucs, and runs
    the requested track list, emitting NDJSON events on stdout.

    :return: Process exit code (0 on success, 2 on init failure).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--shifts", type=int, required=True)
    parser.add_argument("--overlap", type=float, required=True)
    parser.add_argument("--tracks-json", required=True)
    parser.add_argument("--no-sdr", action="store_true")
    args = parser.parse_args()

    try:
        from demucs.api import Separator
    except Exception as exc:
        _emit({"event": "init_error", "error_type": type(exc).__name__,
               "error_message": "failed to import upstream demucs: " + str(exc)})
        return 2

    if args.precision == "fp16":
        # Upstream has no end-to-end fp16 plumbing (model.half() leaves inputs
        # float32 and crashes mid-track); refuse cleanly so the parent records it.
        _emit({"event": "init_error", "error_type": "UnsupportedPrecision",
               "error_message": "Upstream demucs does not support fp16 inference end-to-end."})
        return 0

    tracks = json.loads(Path(args.tracks_json).read_text())

    init_t0 = time.perf_counter()
    try:
        # No ``segment=`` on purpose — upstream's default is the model's
        # training length, which is exactly what we use locally too.
        separator = Separator(
            model=args.model, device=args.device, shifts=args.shifts,
            overlap=args.overlap,
        )
    except Exception as exc:
        _emit({"event": "init_error", "error_type": type(exc).__name__,
               "error_message": str(exc), "traceback": traceback.format_exc(limit=3)})
        return 2
    _emit({"event": "init_complete", "init_sec": time.perf_counter() - init_t0})

    for track in tracks:
        track_name = track["name"]
        track_seed = track.get("track_seed")
        stem_paths = track["stem_paths"]
        if track_seed is not None:
            random.seed(track_seed)
            torch.manual_seed(track_seed)
            if args.device == "cuda":
                torch.cuda.manual_seed_all(track_seed)
        t0 = time.perf_counter()
        try:
            _, separated = separator.separate_audio_file(Path(track["mixture_path"]))
            elapsed = time.perf_counter() - t0
            stem_scores = {}
            if not args.no_sdr:
                for stem_name, ref_path in stem_paths.items():
                    if stem_name not in separated:
                        continue
                    reference, _ = torchaudio.load(ref_path)
                    stem_scores[stem_name] = _compute_sdr(separated[stem_name], reference)
            _emit({"event": "track_complete", "track_name": track_name,
                   "track_seed": track_seed, "elapsed_sec": elapsed,
                   "stem_scores": stem_scores})
        except Exception as exc:
            _emit({"event": "track_error", "track_name": track_name,
                   "track_seed": track_seed,
                   "elapsed_sec": time.perf_counter() - t0,
                   "error_type": type(exc).__name__, "error_message": str(exc),
                   "traceback": traceback.format_exc(limit=3)})

    _emit({"event": "done"})
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


def _build_upstream_worker_source() -> str:
    """
    Build the runnable upstream worker program.

    ``_compute_sdr`` is a real function in this module; its source is injected
    into the worker template so the parent process and the isolated upstream
    subprocess share one SDR implementation rather than two copies that can drift.

    :return: Complete worker source to feed to ``python -P -`` over stdin.
    """
    return _UPSTREAM_WORKER_TEMPLATE.replace(
        "# __SHARED_SDR__", inspect.getsource(_compute_sdr)
    )


app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass(frozen=True)
class BenchmarkTrack:
    name: str
    directory: Path
    mixture_path: Path
    reference_stems: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    config_id: str
    model: str
    precision: str
    compile: bool
    shifts: int
    split_overlap: float
    variant: str = "local"
    upstream_version: str = ""


def _format_float(value: float | None) -> str:
    """
    Render a float for the CSV output. ``None`` becomes empty; NaN/Inf get
    spelled out so the CSV stays parseable.

    :param value: Float to format, or ``None``.
    :return: String representation suitable for the benchmark CSV.
    """
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def _build_track_seed(seed: int, track_name: str) -> int:
    """
    Derive a deterministic per-track seed from the base seed and track name.

    :param seed: Base benchmark seed.
    :param track_name: Track name used as the keyspace.
    :return: 31-bit positive integer seed.
    """
    digest = hashlib.blake2b(
        f"{seed}:{track_name}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % (2**31)


def _compute_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Compute SDR (in dB) for one estimated stem against its reference.

    This is the single SDR implementation for the whole benchmark: it scores
    unblend in-process, and ``_build_upstream_worker_source`` injects its
    source into the isolated upstream subprocess so both sides score identically.

    :param estimate: Separator output for the stem.
    :param reference: Ground-truth stem.
    :return: SDR in dB, ``nan`` if the reference is silent, ``inf`` if the
        residual is below the numeric floor.
    """
    estimate = estimate.to(dtype=torch.float64, device="cpu")
    reference = reference.to(dtype=torch.float64, device="cpu")
    length = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :length]
    reference = reference[..., :length]
    noise = estimate - reference
    reference_energy = torch.sum(reference * reference).item()
    noise_energy = torch.sum(noise * noise).item()
    if reference_energy <= 0.0:
        return float("nan")
    if noise_energy <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(reference_energy / noise_energy)


def _score_stems(
    separator: Separator,
    track: BenchmarkTrack,
    separated: Any,
    only_stem: str | None = None,
) -> dict[str, float]:
    """
    Compute per-stem SDR for one separated track against its references.

    :param separator: The separator (used for its ``_to_tensor`` loader).
    :param track: The track whose reference stems to score against.
    :param separated: The ``SeparatedSources`` returned for this track.
    :param only_stem: If set, score only this stem (the others are
        deliberately degraded under ``use_only_stem``).
    :return: Mapping of stem name to SDR in dB.
    """
    stem_scores: dict[str, float] = {}
    scored = (
        tuple(s for s in track.reference_stems if s == only_stem)
        if only_stem is not None
        else track.reference_stems
    )
    for stem_name in scored:
        if stem_name not in separated.sources:
            continue
        reference = separator._to_tensor(track.directory / f"{stem_name}.wav")
        stem_scores[stem_name] = _compute_sdr(separated.sources[stem_name], reference)
    return stem_scores


def _discover_tracks(musdb_root: Path) -> list[BenchmarkTrack]:
    """
    Walk a MUSDB18-HQ split directory and return the per-track records.

    :param musdb_root: Path to a MUSDB split (e.g. the ``test`` directory).
    :return: Sorted list of ``BenchmarkTrack`` entries with ``mixture.wav``
        and at least one reference stem.
    :raises typer.BadParameter: If ``musdb_root`` is missing or contains no usable tracks.
    """
    if not musdb_root.is_dir():
        raise typer.BadParameter(f"MUSDB root does not exist: {musdb_root}")

    tracks: list[BenchmarkTrack] = []
    for track_dir in sorted(d for d in musdb_root.iterdir() if d.is_dir()):
        mixture_path = track_dir / "mixture.wav"
        reference_stems = tuple(
            stem for stem in REFERENCE_STEMS if (track_dir / f"{stem}.wav").exists()
        )

        if not mixture_path.exists() or not reference_stems:
            continue

        tracks.append(
            BenchmarkTrack(
                name=track_dir.name,
                directory=track_dir,
                mixture_path=mixture_path,
                reference_stems=reference_stems,
            )
        )

    if not tracks:
        raise typer.BadParameter(
            f"No MUSDB tracks with mixture.wav and reference stems found under {musdb_root}"
        )
    return tracks


def _build_configs(
    models: list[str],
    precisions: list[str],
    compile_modes: list[bool],
    shifts_values: list[int],
    split_overlaps: list[float],
    include_upstream: bool = False,
    upstream_version: str = DEFAULT_UPSTREAM_VERSION,
) -> list[BenchmarkConfig]:
    """
    Cartesian-product the input axes into a list of benchmark configs,
    optionally appending an upstream-comparison config alongside each
    eligible local config (no compile, fp32 only).

    :param models: Model names to benchmark.
    :param precisions: Precision modes (``"fp32"``, ``"fp16"``).
    :param compile_modes: Whether ``torch.compile`` is enabled for the run.
    :param shifts_values: Shift counts for equivariant stabilization.
    :param split_overlaps: Overlap fractions between consecutive segments.
    :param include_upstream: If True, also emit upstream-variant configs.
    :param upstream_version: Git ref of upstream demucs to install when included.
    :return: All configs in evaluation order.
    """
    configs: list[BenchmarkConfig] = []
    config_index = 1

    for model in models:
        for precision in precisions:
            for compile_mode in compile_modes:
                for shifts in shifts_values:
                    for split_overlap in split_overlaps:
                        configs.append(
                            BenchmarkConfig(
                                config_id=f"cfg_{config_index:04d}",
                                model=model,
                                precision=precision,
                                compile=compile_mode,
                                shifts=shifts,
                                split_overlap=split_overlap,
                                variant="local",
                            )
                        )
                        config_index += 1
                        if (
                            include_upstream
                            and compile_mode is False
                            and precision == "fp32"
                        ):
                            # Upstream demucs has no torch.compile path and
                            # no end-to-end FP16 plumbing; only emit upstream
                            # rows for FP32, no-compile combinations.
                            configs.append(
                                BenchmarkConfig(
                                    config_id=f"cfg_{config_index:04d}",
                                    model=model,
                                    precision=precision,
                                    compile=False,
                                    shifts=shifts,
                                    split_overlap=split_overlap,
                                    variant="upstream",
                                    upstream_version=upstream_version,
                                )
                            )
                            config_index += 1
    return configs


def _upstream_venv_spec(version: str, python_version: str) -> tuple[Path, str]:
    """
    Derive a contained cache path and identity marker for an upstream venv.

    Git refs are untrusted path input (they may contain ``../`` or slashes), so
    only a fixed-size digest is used as the directory name. The marker binds a
    cached environment to both the exact raw ref and requested Python version.

    :param version: Raw upstream Git ref.
    :param python_version: Requested Python interpreter version.
    :return: ``(venv_path, marker_json)``.
    """
    identity = {"version": version, "python_version": python_version}
    marker_text = json.dumps(identity, sort_keys=True) + "\n"
    digest = hashlib.blake2b(marker_text.encode("utf-8"), digest_size=16).hexdigest()
    root = UPSTREAM_VENV_ROOT.resolve()
    venv_dir = root / f"upstream-{digest}"
    if not venv_dir.is_relative_to(root):  # defensive: digest names cannot escape
        raise RuntimeError(f"Unsafe upstream venv path: {venv_dir}")
    return venv_dir, marker_text


def _upstream_venv_ready(venv_dir: Path, marker_text: str) -> bool:
    """Return whether a cached upstream environment matches its full identity."""
    try:
        return (venv_dir / "bin" / "python").exists() and (
            venv_dir / ".demucs-installed"
        ).read_text() == marker_text
    except OSError:
        return False


def _provision_upstream_venv(
    venv_dir: Path, version: str, python_version: str, marker_text: str
) -> None:
    """Create and populate one upstream environment while its lock is held."""
    python_bin = venv_dir / "bin" / "python"
    uv_bin = shutil.which("uv")
    # Install upstream from git rather than PyPI: PyPI's last release (4.0.1)
    # predates upstream's `demucs.api.Separator`, which we depend on.
    install_targets = [
        f"demucs @ git+{UPSTREAM_REPO}@{version}",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "numpy<2",
        "setuptools<80",
        "soundfile",
    ]
    if uv_bin is not None:
        typer.echo(
            f"Creating upstream venv via uv at {venv_dir} (python {python_version})"
        )
        subprocess.run(
            [uv_bin, "venv", "--python", python_version, str(venv_dir)],
            check=True,
        )
        typer.echo(
            f"Installing demucs=={version} into upstream venv (this can take a few minutes)"
        )
        subprocess.run(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                str(python_bin),
                *install_targets,
            ],
            check=True,
        )
    else:
        typer.echo(f"Creating upstream venv via python -m venv at {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        typer.echo(
            f"Installing demucs=={version} into upstream venv (this can take a few minutes)"
        )
        subprocess.run(
            [str(python_bin), "-m", "pip", "install", *install_targets],
            check=True,
        )
    (venv_dir / ".demucs-installed").write_text(marker_text)


def _ensure_upstream_venv(version: str, python_version: str) -> Path:
    """
    Create (or reuse) an isolated venv with ``demucs==<version>`` installed.

    Marker validation, deletion, creation, installation, and publication are
    serialized across processes. The lock is a sibling of the deletable venv,
    so rebuilding the environment cannot remove the active lock itself.

    :param version: Git ref of upstream demucs to install.
    :param python_version: Python interpreter version for the venv.
    :return: Path to the prepared venv directory.
    """
    venv_dir, marker_text = _upstream_venv_spec(version, python_version)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{venv_dir}.lock")
    with FileLock(lock_path, timeout=2 * 60 * 60):
        # Another process may have completed provisioning while this caller
        # waited, so the identity check belongs inside the lock.
        if _upstream_venv_ready(venv_dir, marker_text):
            return venv_dir
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        _provision_upstream_venv(venv_dir, version, python_version, marker_text)
        if not _upstream_venv_ready(venv_dir, marker_text):
            raise RuntimeError(
                f"Upstream venv provisioning did not complete: {venv_dir}"
            )
    return venv_dir


def _build_upstream_tracks_payload(
    tracks: list[BenchmarkTrack], seed: int | None
) -> list[dict[str, Any]]:
    """Build the isolated worker payload with stable per-track seeds."""
    return [
        {
            "name": track.name,
            "track_seed": (
                _build_track_seed(seed, track.name) if seed is not None else None
            ),
            "mixture_path": str(track.mixture_path),
            "reference_stems": list(track.reference_stems),
            "stem_paths": {
                stem: str(track.directory / f"{stem}.wav")
                for stem in track.reference_stems
            },
        }
        for track in tracks
    ]


def _run_upstream_config(
    config: BenchmarkConfig,
    tracks: list[BenchmarkTrack],
    device: str,
    seed: int | None,
    chunk_batch_size: int | None,
    output_dir: Path,
    upstream_python_version: str,
    compute_sdr: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run one upstream-variant config in a subprocess.

    :param config: The upstream-variant benchmark config to run.
    :param tracks: Tracks to separate and score.
    :param device: Device string the subprocess runs on.
    :param seed: Base seed recorded in the detail rows.
    :param chunk_batch_size: Resolved chunk batch size recorded in the rows.
    :param output_dir: Directory for the temporary tracks JSON payload.
    :param upstream_python_version: Python version for the upstream venv.
    :param compute_sdr: Score stems against references in the worker.
    :return: ``(detail_rows, summary_extras)`` where ``summary_extras`` carries
        aggregate fields (model_init_sec, error info) for the summary row.
    """
    venv_dir = _ensure_upstream_venv(config.upstream_version, upstream_python_version)
    venv_python = venv_dir / "bin" / "python"

    tracks_payload = _build_upstream_tracks_payload(tracks, seed)
    track_seed_by_name = {
        str(track["name"]): track.get("track_seed") for track in tracks_payload
    }
    tracks_json_path = output_dir / f"_tmp_upstream_tracks_{config.config_id}.json"
    tracks_json_path.write_text(json.dumps(tracks_payload))

    cmd = [
        # ``-P`` keeps cwd (the unblend repo) off ``sys.path`` so the
        # subprocess imports the upstream ``demucs`` from the venv rather than
        # our local checkout. ``-`` reads the worker source from stdin.
        str(venv_python),
        "-P",
        "-",
        "--model",
        config.model,
        "--device",
        device,
        "--precision",
        config.precision,
        "--shifts",
        str(config.shifts),
        "--overlap",
        str(config.split_overlap),
        "--tracks-json",
        str(tracks_json_path),
    ]
    if not compute_sdr:
        cmd.append("--no-sdr")

    detail_rows: list[dict[str, Any]] = []
    init_sec: float | None = None
    init_error_type = ""
    init_error_message = ""
    track_index_by_name = {track.name: idx for idx, track in enumerate(tracks, start=1)}
    seen_track_names: set[str] = set()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    # Drain stderr concurrently: the worker can emit large volumes there
    # (tqdm download bars, torch warnings), and a full pipe buffer would
    # block the worker's writes — deadlocking the stdout loop below.
    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for err_line in proc.stderr:
            stderr_chunks.append(err_line)
            del stderr_chunks[:-50]

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()
    rss_sampler = _PeakRssSampler(proc.pid)
    rss_sampler.start()
    vram_sampler = None
    if device == "cuda":
        vram_sampler = _PeakVramSampler(proc.pid)
        vram_sampler.start()
    try:
        # Inside the try: a worker that dies before reading stdin raises
        # BrokenPipeError here, and the finally must still stop the sampler.
        assert proc.stdin is not None
        try:
            proc.stdin.write(_build_upstream_worker_source())
            proc.stdin.close()
        except BrokenPipeError:
            # Worker died before reading its stdin. Fall through: stdout
            # EOFs immediately and the unreported tracks become
            # WorkerCrashed rows carrying the stderr tail.
            pass
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # upstream prints non-JSON status lines; surface them but skip parsing.
                typer.echo(f"  upstream> {line}")
                continue

            kind = event.get("event")
            if kind == "init_complete":
                init_sec = float(event.get("init_sec", 0.0))
            elif kind == "init_error":
                init_error_type = str(event.get("error_type", "InitError"))
                init_error_message = str(event.get("error_message", ""))
            elif kind in ("track_complete", "track_error"):
                track_name = event["track_name"]
                seen_track_names.add(track_name)
                track_index = track_index_by_name.get(track_name, 0)
                detail_row = {
                    **_detail_row_base(config, chunk_batch_size, seed, device),
                    "track_index": track_index,
                    "track_name": track_name,
                    "track_seed": event.get(
                        "track_seed", track_seed_by_name.get(track_name)
                    ),
                    "elapsed_sec": float(event.get("elapsed_sec", 0.0)),
                }
                if kind == "track_complete":
                    stem_scores = event.get("stem_scores", {})
                    detail_row["status"] = "ok"
                    detail_row["error_type"] = ""
                    detail_row["error_message"] = ""
                    detail_row["num_scored_stems"] = len(stem_scores)
                    detail_row["mean_sdr"] = (
                        statistics.fmean(stem_scores.values())
                        if stem_scores
                        else float("nan")
                    )
                    for stem_name in REFERENCE_STEMS:
                        detail_row[f"{stem_name}_sdr"] = stem_scores.get(stem_name)
                else:
                    error_type = str(event.get("error_type", ""))
                    error_message = str(event.get("error_message", "")).lower()
                    is_oom = (
                        error_type == "OutOfMemoryError"
                        or "out of memory" in error_message
                    )
                    detail_row["status"] = "oom" if is_oom else "error"
                    detail_row["error_type"] = error_type
                    detail_row["error_message"] = str(event.get("error_message", ""))
                    detail_row["num_scored_stems"] = 0
                    detail_row["mean_sdr"] = float("nan")
                    for stem_name in REFERENCE_STEMS:
                        detail_row[f"{stem_name}_sdr"] = None
                    typer.echo(
                        f"  {config.config_id} (upstream) track {track_name} "
                        f"failed: {error_type}: {detail_row['error_message']}"
                    )
                detail_rows.append(detail_row)
            elif kind == "done":
                break
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        rss_sampler.stop()
        if vram_sampler:
            vram_sampler.stop()
        tracks_json_path.unlink(missing_ok=True)

    stderr_thread.join(timeout=5.0)
    stderr_tail = "".join(stderr_chunks)

    # Tracks the worker never reported (e.g., if it died mid-loop).
    for track in tracks:
        if track.name in seen_track_names:
            continue
        detail_rows.append(
            {
                **_detail_row_base(config, chunk_batch_size, seed, device),
                "track_index": track_index_by_name[track.name],
                "track_name": track.name,
                "track_seed": track_seed_by_name[track.name],
                "elapsed_sec": None,
                "status": "error",
                "error_type": init_error_type or "WorkerCrashed",
                "error_message": init_error_message or stderr_tail[-300:],
                "num_scored_stems": 0,
                "mean_sdr": float("nan"),
                **{f"{stem}_sdr": None for stem in REFERENCE_STEMS},
            }
        )

    summary_extras = {
        "model_init_sec": init_sec,
        "error_type": init_error_type,
        "error_message": init_error_message
        or (stderr_tail[-300:] if proc.returncode != 0 else ""),
        "peak_rss_mb": rss_sampler.stop(),
        "peak_vram_smi_mb": vram_sampler.stop() if vram_sampler else None,
    }
    return detail_rows, summary_extras


def _precision_to_dtype(precision: str) -> torch.dtype | None:
    """
    Map a CLI precision label to the corresponding ``torch.dtype``.

    :param precision: ``"fp32"``, ``"fp16"``, or ``"bf16"``.
    :return: ``torch.float16`` for fp16, ``torch.bfloat16`` for bf16,
        ``None`` (default float32) for fp32.
    :raises ValueError: If ``precision`` is anything else.
    """
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def _detail_row_base(
    config: BenchmarkConfig,
    chunk_batch_size: int | None,
    base_seed: int | None,
    device: str,
    use_only_stem: str | None = None,
) -> dict[str, Any]:
    """
    Build the prefix of identifying columns shared by every per-track row
    in the details CSV.

    :param config: The benchmark config the rows belong to.
    :param chunk_batch_size: Resolved chunk batch size (None if auto).
    :param base_seed: Base seed used to derive per-track seeds.
    :param device: Device string the run executed on.
    :param use_only_stem: Stem restriction the run used, if any.
    :return: Dict of identifying columns.
    """
    return {
        "config_id": config.config_id,
        "variant": config.variant,
        "upstream_version": config.upstream_version,
        "model": config.model,
        "precision": config.precision,
        "device": device,
        "compile": config.compile,
        "shifts": config.shifts,
        "split_overlap": config.split_overlap,
        "chunk_batch_size": chunk_batch_size,
        "base_seed": base_seed,
        "use_only_stem": use_only_stem,
    }


def _summary_row_base(
    config: BenchmarkConfig,
    chunk_batch_size: int | None,
    base_seed: int | None,
    device: str,
    use_only_stem: str | None = None,
) -> dict[str, Any]:
    """
    Build the prefix of identifying columns shared by every config-level
    row in the summary CSV.

    :param config: The benchmark config the row belongs to.
    :param chunk_batch_size: Resolved chunk batch size (None if auto).
    :param base_seed: Base seed used to derive per-track seeds.
    :param device: Device string the run executed on.
    :param use_only_stem: Stem restriction the run used, if any.
    :return: Dict of identifying columns.
    """
    return {
        "config_id": config.config_id,
        "variant": config.variant,
        "upstream_version": config.upstream_version,
        "model": config.model,
        "precision": config.precision,
        "device": device,
        "compile": config.compile,
        "shifts": config.shifts,
        "split_overlap": config.split_overlap,
        "chunk_batch_size": chunk_batch_size,
        "base_seed": base_seed,
        "use_only_stem": use_only_stem,
    }


def _resolve_device(device_flag: str | None) -> str:
    """
    Pick the actual device string for the run.

    :param device_flag: Requested device (``cuda``/``mps``/``cpu``/``auto``/None).
    :return: Resolved device string.
    :raises typer.BadParameter: If the requested device is unavailable or unknown.
    """
    if device_flag in (None, "auto"):
        return default_device()
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise typer.BadParameter("CUDA requested but not available.")
        return "cuda"
    if device_flag == "mps":
        if not torch.backends.mps.is_available():
            raise typer.BadParameter("MPS requested but not available.")
        return "mps"
    if device_flag == "cpu":
        return "cpu"
    raise typer.BadParameter(
        f"Unknown --device {device_flag!r}; expected cuda|mps|cpu|auto."
    )


def _empty_cache(device: str) -> None:
    """
    Empty the framework allocator cache for the given device, if supported.

    :param device: ``"cuda"``, ``"mps"``, or ``"cpu"``. Anything else is a no-op.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _peak_memory_bytes(device: str) -> int | None:
    """
    Best-effort peak GPU memory in bytes for the run.

    CUDA note: ``max_memory_allocated`` cannot see CUDAGraph private pools,
    so compiled configs under-report badly (a compiled htdemucs_ft measured
    7.6 GB here vs 27.7 GB device-truth on a V100); ``peak_vram_smi_mb``
    from ``_PeakVramSampler`` is the honest number.

    :param device: Device string to query.
    :return: Peak allocation in bytes, or ``None`` if the backend does not
        expose a peak-memory API.
    """
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return int(torch.mps.driver_allocated_memory())
    return None


def _reset_peak_memory(device: str) -> None:
    """
    Reset the backend's peak-memory counter so the next ``_peak_memory_bytes``
    reading reflects only this benchmark config.

    :param device: Device string to reset; only ``"cuda"`` exposes a reset API.
    """
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


class _PeakRssSampler:
    """
    Samples a process's resident set size on a background thread via
    ``ps -o rss=`` (KB on both Linux and macOS), tracking the peak.

    Sampling (rather than ``ru_maxrss``) is deliberate: the rusage high-water
    mark is monotonic for the life of the process, so it can't isolate one
    benchmark config from the previous one. A 200 ms sampler can miss
    sub-interval spikes; treat readings as a floor, not an exact peak.

    When sampling the benchmark process itself over a multi-config run,
    allocators rarely return freed pages to the OS, so later configs inherit
    earlier configs' heap floor (bias upward). A SLURM shard bounds this to
    one (model, precision, compile) cell — its shifts/overlap sweep still
    shares a process, so compare RSS across shards, not within one. Upstream
    rows are immune (fresh subprocess per config).
    """

    def __init__(self, pid: int, interval_sec: float = 0.2) -> None:
        """
        :param pid: Process ID to sample (self or a subprocess).
        :param interval_sec: Seconds between samples.
        """
        self.pid = pid
        self.interval_sec = interval_sec
        self._peak_kb = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        """
        Begin sampling on the background thread.
        """
        self._started = True
        self._thread.start()

    def stop(self) -> float | None:
        """
        Stop sampling and return the observed peak.

        Idempotent; safe to call from a ``finally`` after an explicit call.

        :return: Peak RSS in MB, or ``None`` if no sample succeeded.
        """
        if self._started and not self._stop_event.is_set():
            self._stop_event.set()
            self._thread.join()
        return self._peak_kb / 1024 if self._peak_kb > 0 else None

    def _sample_kb(self) -> int:
        """
        Read the process's current RSS.

        :return: RSS in KB, or 0 if the process is gone or ``ps`` failed.
        """
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(self.pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(out.stdout.strip() or 0)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return 0

    def _run(self) -> None:
        """
        Sampler loop: poll until stopped, then take one final sample.
        """
        while not self._stop_event.is_set():
            self._peak_kb = max(self._peak_kb, self._sample_kb())
            self._stop_event.wait(self.interval_sec)
        self._peak_kb = max(self._peak_kb, self._sample_kb())


class _PeakVramSampler(_PeakRssSampler):
    """
    Samples a process's device-level GPU memory via
    ``nvidia-smi --query-compute-apps``, tracking the peak.

    This is NVML ground truth — caching-allocator segments, CUDAGraph private
    pools, CUDA context, and cuDNN workspaces all included — unlike
    ``torch.cuda.max_memory_allocated``, which cannot see CUDAGraph private
    pools and badly under-reports compiled configs. Returns None (via
    ``stop()``) on hosts without ``nvidia-smi``.
    """

    def _sample_kb(self) -> int:
        """
        Read the process's current GPU memory across all devices.

        :return: GPU memory in KB (MiB * 1024), or 0 if unavailable.
        """
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            total_mib = 0
            for line in out.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == str(self.pid):
                    total_mib += int(parts[1])
            return total_mib * 1024
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return 0


@app.command()
def main(
    musdb_root: Path = typer.Option(
        ...,
        "--musdb-root",
        help="Path to the MUSDB18-HQ split directory containing per-track folders",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory to write benchmark CSV/JSON results into",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Limit benchmarking to the first N tracks",
    ),
    seed: int | None = typer.Option(
        1234,
        "--seed",
        help="Base random seed used to derive a deterministic seed per track",
    ),
    chunk_batch_size: int | None = typer.Option(
        None,
        "--chunk-batch-size",
        min=1,
        help="Override how many split chunks are processed per batch",
    ),
    dataset_throughput: bool = typer.Option(
        False,
        "--dataset-throughput",
        help="Measure whole-dataset wall time via the batched separate([...]) "
        "path (apply_model_multi) instead of per-track latency. Local configs "
        "only; upstream stays file-per-file (its total wall is still recorded "
        "for the comparison). Reports tracks/sec + dataset_wall_sec.",
    ),
    compute_sdr: bool = typer.Option(
        True,
        "--sdr/--no-sdr",
        help="Score separated stems against references. Disable for pure "
        "timing/memory shards (SDR is device-independent, so quality only "
        "needs measuring once per model/precision/shifts/overlap combo).",
    ),
    use_only_stem: str | None = typer.Option(
        None,
        "--use-only-stem",
        help="Load only this stem's specialist (only_load) and run separation "
        "with use_only_stem, scoring just this stem. Pairs with a normal run "
        "of the same config to verify the single-specialist path scores "
        "identically to the full ensemble on its stem (htdemucs_ft). Local "
        "configs only — incompatible with --include-upstream.",
    ),
    models: list[str] = typer.Option(
        DEFAULT_MODELS,
        "--model",
        help="Model(s) to benchmark. Repeat to benchmark multiple models.",
    ),
    precisions: list[str] = typer.Option(
        DEFAULT_PRECISIONS,
        "--precision",
        help="Precision mode(s) to benchmark. Repeat to benchmark multiple values.",
    ),
    compile_modes: list[str] = typer.Option(
        DEFAULT_COMPILE_MODES,
        "--compile-mode",
        help="Compilation mode(s) to benchmark: 'true' and/or 'false'. "
        "Repeat to benchmark multiple values (e.g. --compile-mode false).",
    ),
    shifts_values: list[int] = typer.Option(
        DEFAULT_SHIFTS,
        "--shifts",
        min=1,
        help="Shift counts to benchmark. Repeat to benchmark multiple values.",
    ),
    split_overlaps: list[float] = typer.Option(
        DEFAULT_SPLIT_OVERLAPS,
        "--split-overlap",
        help="Split overlaps to benchmark. Repeat to benchmark multiple values.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device: 'cuda', 'mps', 'cpu', or 'auto' (default: auto). On MPS/CPU,"
        " --compile-mode true is silently dropped (compile is CUDA-only here).",
    ),
    include_upstream: bool = typer.Option(
        False,
        "--include-upstream",
        help="Also benchmark the upstream PyPI demucs release in an isolated venv.",
    ),
    upstream_version: str = typer.Option(
        DEFAULT_UPSTREAM_VERSION,
        "--upstream-version",
        help=(
            "Git ref (branch, tag, or commit SHA) of adefossez/demucs to install "
            "for the comparison. Defaults to 'main'."
        ),
    ),
    upstream_python: str = typer.Option(
        DEFAULT_UPSTREAM_PYTHON,
        "--upstream-python",
        help="Python interpreter version to use for the upstream venv.",
    ),
) -> None:
    """
    Benchmark the MUSDB matrix on the chosen device and record SDR + timing.

    :param musdb_root: Path to the MUSDB18-HQ split directory.
    :param output_dir: Directory to write results into (auto-named if None).
    :param limit: Limit benchmarking to the first N tracks.
    :param seed: Base seed used to derive a deterministic per-track seed.
    :param chunk_batch_size: Override for chunks processed per batch.
    :param dataset_throughput: Measure whole-dataset wall via the batched path.
    :param compute_sdr: Score stems against references; disable for timing-only shards.
    :param use_only_stem: Load/run only this stem's specialist and score only it.
    :param models: Model name(s) to benchmark.
    :param precisions: Precision mode(s) to benchmark.
    :param compile_modes: Compilation mode(s) to benchmark (``true``/``false``).
    :param shifts_values: Shift count(s) to benchmark.
    :param split_overlaps: Split overlap fraction(s) to benchmark.
    :param device: Device to run on (``cuda``/``mps``/``cpu``/``auto``).
    :param include_upstream: Also benchmark the upstream demucs release.
    :param upstream_version: Git ref of upstream demucs to install.
    :param upstream_python: Python version for the upstream venv.
    """

    # --compile-mode takes string values ("true"/"false"); a list[bool] Typer
    # option degenerates into a value-less flag (can't express "false only"),
    # so parse the strings to bools here before any truthiness checks below.
    def _parse_compile_mode(value: str) -> bool:
        """
        Parse a ``--compile-mode`` string into a bool.

        :param value: The CLI string (``true``/``false`` and synonyms).
        :return: The parsed boolean.
        :raises typer.BadParameter: If the value is not a recognized boolean.
        """
        v = value.strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
        raise typer.BadParameter(
            f"--compile-mode must be 'true' or 'false', got '{value}'"
        )

    compile_modes = [_parse_compile_mode(c) for c in compile_modes]

    if use_only_stem is not None and not use_only_stem.strip():
        # "" would skip Separator's truthiness-based only_load validation,
        # load the full ensemble, then fail on every track.
        raise typer.BadParameter(
            "--use-only-stem needs a stem name (got an empty string)."
        )
    if use_only_stem is not None and include_upstream:
        raise typer.BadParameter(
            "--use-only-stem is a local-only comparison (upstream has no "
            "only_load equivalent); drop --include-upstream."
        )

    device = _resolve_device(device)
    # ``compile`` is CUDA-only (Inductor codegen errors on MPS for HTDemucs;
    # CPU compile path is not exercised here).
    if device != "cuda" and any(compile_modes):
        typer.echo(
            f"Note: device={device}; dropping compile=True from the matrix "
            f"(torch.compile is unsupported here)."
        )
        compile_modes = [c for c in compile_modes if not c]
        if not compile_modes:
            compile_modes = [False]

    if dataset_throughput and device not in ("cuda", "mps"):
        typer.echo(
            f"Note: device={device}; --dataset-throughput is cuda/mps-only, "
            "falling back to per-track latency mode."
        )

    reduced_for_cpu = {"fp16", "bf16"}
    if device == "cpu":
        dropped = [p for p in precisions if p in reduced_for_cpu]
        if dropped:
            typer.echo(
                f"Note: device=cpu; dropping {dropped} from the matrix "
                f"(reduced precision is not supported on CPU)."
            )
        precisions = [p for p in precisions if p not in reduced_for_cpu]
        if not precisions:
            precisions = ["fp32"]

    tracks = _discover_tracks(musdb_root)
    if limit is not None:
        tracks = tracks[:limit]

    configs = _build_configs(
        models=models,
        precisions=precisions,
        compile_modes=compile_modes,
        shifts_values=shifts_values,
        split_overlaps=split_overlaps,
        include_upstream=include_upstream,
        upstream_version=upstream_version,
    )

    if include_upstream:
        # Provision the upstream venv up-front so that errors surface before any
        # local runs eat into wall-clock time.
        try:
            _ensure_upstream_venv(upstream_version, upstream_python)
        except subprocess.CalledProcessError as exc:
            raise typer.BadParameter(
                f"Failed to set up upstream demucs venv (version={upstream_version}, "
                f"python={upstream_python}): {exc}"
            )

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmarks") / "musdb" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Benchmarking {len(configs)} configs across {len(tracks)} MUSDB tracks")
    typer.echo(f"Results directory: {output_dir}")

    details_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for config_index, config in enumerate(configs, start=1):
        variant_label = (
            f"upstream-{config.upstream_version}"
            if config.variant == "upstream"
            else "local"
        )
        label = (
            f"[{config_index}/{len(configs)}] {config.config_id} variant={variant_label} "
            f"model={config.model} precision={config.precision} "
            f"compile={config.compile} shifts={config.shifts} "
            f"split_overlap={config.split_overlap}"
        )
        typer.echo(label)

        if config.variant == "upstream":
            upstream_details, upstream_extras = _run_upstream_config(
                config=config,
                tracks=tracks,
                device=device,
                seed=seed,
                chunk_batch_size=chunk_batch_size,
                output_dir=output_dir,
                upstream_python_version=upstream_python,
                compute_sdr=compute_sdr,
            )
            details_rows.extend(upstream_details)
            ok_rows = [row for row in upstream_details if row.get("status") == "ok"]
            ok_elapsed = [
                float(row["elapsed_sec"])
                for row in ok_rows
                if row.get("elapsed_sec") is not None
            ]
            remaining_elapsed = ok_elapsed[1:]
            error_count = sum(
                1 for row in upstream_details if row.get("status") == "error"
            )
            oom_count = sum(1 for row in upstream_details if row.get("status") == "oom")
            attempted_elapsed = [
                float(row["elapsed_sec"])
                for row in upstream_details
                if row.get("elapsed_sec") is not None
            ]
            mean_sdr_values = [
                float(row["mean_sdr"])
                for row in ok_rows
                if not math.isnan(float(row["mean_sdr"]))
            ]
            per_stem_summary: dict[str, float] = {}
            for stem_name in REFERENCE_STEMS:
                stem_values = [
                    float(row[f"{stem_name}_sdr"])
                    for row in ok_rows
                    if row.get(f"{stem_name}_sdr") is not None
                    and not math.isnan(float(row[f"{stem_name}_sdr"]))
                ]
                per_stem_summary[stem_name] = (
                    statistics.fmean(stem_values) if stem_values else float("nan")
                )

            summary_rows.append(
                {
                    **_summary_row_base(config, chunk_batch_size, seed, device),
                    "status": "ok" if len(ok_rows) == len(tracks) else "partial",
                    "error_type": upstream_extras.get("error_type", ""),
                    "error_message": upstream_extras.get("error_message", ""),
                    "num_tracks": len(tracks),
                    "ok_tracks": len(ok_rows),
                    "error_tracks": error_count,
                    "oom_tracks": oom_count,
                    "model_init_sec": upstream_extras.get("model_init_sec"),
                    "first_attempt_sec": (
                        float(upstream_details[0]["elapsed_sec"])
                        if upstream_details
                        and upstream_details[0].get("elapsed_sec") is not None
                        else None
                    ),
                    "first_attempt_status": (
                        str(upstream_details[0].get("status", ""))
                        if upstream_details
                        else ""
                    ),
                    "first_track_sec": ok_elapsed[0] if ok_elapsed else None,
                    "remaining_total_sec": sum(remaining_elapsed)
                    if remaining_elapsed
                    else 0.0,
                    "steady_state_mean_sec": (
                        statistics.fmean(remaining_elapsed)
                        if remaining_elapsed
                        else None
                    ),
                    "steady_state_median_sec": (
                        statistics.median(remaining_elapsed)
                        if remaining_elapsed
                        else None
                    ),
                    "attempted_track_sec": sum(attempted_elapsed),
                    "track_total_sec": sum(ok_elapsed),
                    # Upstream is always file-per-file; its whole-dataset wall
                    # is the summed per-track time, directly comparable to a
                    # local --dataset-throughput run's measured batched wall.
                    "dataset_wall_sec": sum(attempted_elapsed),
                    "tracks_per_sec": (
                        len(ok_rows) / sum(attempted_elapsed)
                        if ok_rows and sum(attempted_elapsed) > 0
                        else None
                    ),
                    "mean_sdr": (
                        statistics.fmean(mean_sdr_values)
                        if mean_sdr_values
                        else float("nan")
                    ),
                    "median_sdr": (
                        statistics.median(mean_sdr_values)
                        if mean_sdr_values
                        else float("nan")
                    ),
                    "peak_vram_mb": None,
                    "peak_vram_smi_mb": upstream_extras.get("peak_vram_smi_mb"),
                    "peak_rss_mb": upstream_extras.get("peak_rss_mb"),
                    **{
                        f"{stem}_mean_sdr": value
                        for stem, value in per_stem_summary.items()
                    },
                }
            )
            continue

        if not ensure_model_available(config.model):
            summary_rows.append(
                {
                    **_summary_row_base(
                        config,
                        chunk_batch_size,
                        seed,
                        device,
                        use_only_stem=use_only_stem,
                    ),
                    "status": "model_unavailable",
                    "error_type": "ModelUnavailable",
                    "error_message": f"Could not download or load model '{config.model}'",
                    "num_tracks": len(tracks),
                }
            )
            continue

        # Started before Separator init so model-load RSS counts toward the
        # config's peak.
        rss_sampler = _PeakRssSampler(os.getpid())
        rss_sampler.start()
        # NVML-truth GPU memory (sees CUDAGraph pools that the torch
        # allocator metric misses); None off-CUDA.
        vram_sampler = None
        if device == "cuda":
            vram_sampler = _PeakVramSampler(os.getpid())
            vram_sampler.start()

        init_started_at = perf_counter()
        try:
            # chunk_batch_size goes to init, not per-call: compiled
            # separators capture the batch size at init and reject per-call
            # overrides.
            separator = Separator(
                model=config.model,
                device=device,
                only_load=use_only_stem,
                dtype=_precision_to_dtype(config.precision),
                compile=config.compile,
                chunk_batch_size=chunk_batch_size,
            )
        except Exception as error:
            summary_rows.append(
                {
                    **_summary_row_base(
                        config,
                        chunk_batch_size,
                        seed,
                        device,
                        use_only_stem=use_only_stem,
                    ),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "num_tracks": len(tracks),
                    "model_init_sec": perf_counter() - init_started_at,
                    # Init-time RSS matters most when init fails from memory
                    # pressure.
                    "peak_rss_mb": rss_sampler.stop(),
                    "peak_vram_smi_mb": (vram_sampler.stop() if vram_sampler else None),
                }
            )
            typer.echo(f"Failed to initialize {config.config_id}: {error}")
            continue

        effective_chunk_batch_size = (
            chunk_batch_size
            if chunk_batch_size is not None
            else separator.chunk_batch_size
        )

        model_init_sec = perf_counter() - init_started_at
        _empty_cache(device)
        _reset_peak_memory(device)
        peak_vram_bytes = 0

        successful_track_rows: list[dict[str, Any]] = []
        error_count = 0
        oom_count = 0
        config_error_type = ""
        config_error_message = ""

        # Dataset-throughput mode: run the track list through the batched
        # ``separate([...])`` path (apply_model_multi) in bounded groups and
        # sum the separation wall, instead of timing each track separately.
        # Groups bound CPU RAM (all-at-once holds every decoded input + output
        # stem in memory → OOM on full-length tracks). Only the separate calls
        # are timed; SDR scoring happens between groups, untimed, and each
        # group's audio is dropped before the next. Local configs only —
        # upstream has no batched path. ``dataset_wall_override`` carries the
        # summed separation wall into the summary; the per-track loop below is
        # skipped (it iterates an empty list when ``use_batched``).
        dataset_wall_override: float | None = None
        use_batched = (
            dataset_throughput
            and config.variant == "local"
            and device in ("cuda", "mps")
        )
        if use_batched:
            sep_wall = 0.0
            bi = 0
            batched_failed = False
            for grp_start in range(0, len(tracks), DATASET_THROUGHPUT_GROUP):
                group = tracks[grp_start : grp_start + DATASET_THROUGHPUT_GROUP]
                try:
                    grp_t0 = perf_counter()
                    group_results = separator.separate(
                        [t.mixture_path for t in group],
                        shifts=config.shifts,
                        split_overlap=config.split_overlap,
                        seed=seed,
                        use_only_stem=use_only_stem,
                    )
                    sep_wall += perf_counter() - grp_t0  # times separation only
                except Exception as error:
                    error_type = type(error).__name__
                    is_oom = isinstance(error, torch.OutOfMemoryError) or (
                        "out of memory" in str(error).lower()
                    )
                    config_error_type = error_type
                    config_error_message = str(error)
                    if is_oom:
                        oom_count += 1
                    else:
                        error_count += 1
                    typer.echo(
                        f"{config.config_id} batched group "
                        f"[{grp_start}:{grp_start + len(group)}] failed with "
                        f"{'oom' if is_oom else 'error'}: {error_type}: {error}"
                    )
                    traceback.print_exc()
                    _empty_cache(device)
                    batched_failed = True
                    break

                # Score + record each track in the group, then drop the audio.
                for track, separated in zip(group, group_results):
                    bi += 1
                    stem_scores = (
                        _score_stems(
                            separator, track, separated, only_stem=use_only_stem
                        )
                        if compute_sdr
                        else {}
                    )
                    detail_row = {
                        **_detail_row_base(
                            config,
                            effective_chunk_batch_size,
                            seed,
                            device,
                            use_only_stem=use_only_stem,
                        ),
                        "track_index": bi,
                        "track_name": track.name,
                        # Batched calls run on the base seed once per group,
                        # not per-track derived seeds — record what was used.
                        "track_seed": seed,
                        # Per-track wall isn't individually observable inside a
                        # batched call; attributed evenly below once the total
                        # is known. ``dataset_wall_sec`` is the authoritative
                        # number.
                        "elapsed_sec": None,
                        "status": "ok",
                        "error_type": "",
                        "error_message": "",
                        "num_scored_stems": len(stem_scores),
                        "mean_sdr": statistics.fmean(stem_scores.values())
                        if stem_scores
                        else float("nan"),
                    }
                    for stem_name in REFERENCE_STEMS:
                        detail_row[f"{stem_name}_sdr"] = stem_scores.get(stem_name)
                    successful_track_rows.append(detail_row)
                    details_rows.append(detail_row)
                del group_results

            # Per-track wall time isn't individually observable inside a
            # batched ``separate([...])`` call, so batched rows keep
            # ``elapsed_sec=None``. We deliberately do NOT back-fill it with
            # ``sep_wall / n``: that would make every track report the same
            # fabricated number, which the summary would then present as a
            # real first-track / steady-state latency curve. ``dataset_wall_sec``
            # and ``tracks_per_sec`` (from the measured ``sep_wall`` override)
            # are the authoritative timing numbers in this mode.
            #
            # Only treat the summed wall as the authoritative dataset time when
            # the whole set completed; a partial run falls back downstream.
            if not batched_failed:
                dataset_wall_override = sep_wall

            measured_peak = _peak_memory_bytes(device)
            if measured_peak is not None:
                peak_vram_bytes = max(peak_vram_bytes, measured_peak)

        for track_index, track in enumerate([] if use_batched else tracks, start=1):
            detail_row = {
                **_detail_row_base(
                    config,
                    effective_chunk_batch_size,
                    seed,
                    device,
                    use_only_stem=use_only_stem,
                ),
                "track_index": track_index,
                "track_name": track.name,
                "track_seed": None
                if seed is None
                else _build_track_seed(seed, track.name),
            }

            started_at = perf_counter()
            try:
                separated = separator.separate(
                    audio=track.mixture_path,
                    shifts=config.shifts,
                    split_overlap=config.split_overlap,
                    seed=detail_row["track_seed"],
                    use_only_stem=use_only_stem,
                )
                elapsed_sec = perf_counter() - started_at
                detail_row["elapsed_sec"] = elapsed_sec

                stem_scores: dict[str, float] = {}
                if compute_sdr:
                    # With use_only_stem, other stems are deliberately
                    # degraded (they come from the one specialist) — scoring
                    # them would only pollute the summary means.
                    scored_stems = (
                        tuple(s for s in track.reference_stems if s == use_only_stem)
                        if use_only_stem is not None
                        else track.reference_stems
                    )
                    for stem_name in scored_stems:
                        if stem_name not in separated.sources:
                            continue
                        reference = separator._to_tensor(
                            track.directory / f"{stem_name}.wav"
                        )
                        stem_scores[stem_name] = _compute_sdr(
                            separated.sources[stem_name],
                            reference,
                        )

                detail_row["status"] = "ok"
                detail_row["error_type"] = ""
                detail_row["error_message"] = ""
                detail_row["num_scored_stems"] = len(stem_scores)
                detail_row["mean_sdr"] = (
                    statistics.fmean(stem_scores.values())
                    if stem_scores
                    else float("nan")
                )
                for stem_name in REFERENCE_STEMS:
                    detail_row[f"{stem_name}_sdr"] = stem_scores.get(stem_name)

                successful_track_rows.append(detail_row)
                details_rows.append(detail_row)
            except Exception as error:
                elapsed_sec = perf_counter() - started_at
                error_type = type(error).__name__
                is_oom = isinstance(error, torch.OutOfMemoryError) or (
                    "out of memory" in str(error).lower()
                )

                detail_row["elapsed_sec"] = elapsed_sec
                detail_row["status"] = "oom" if is_oom else "error"
                detail_row["error_type"] = error_type
                detail_row["error_message"] = str(error)
                detail_row["num_scored_stems"] = 0
                detail_row["mean_sdr"] = float("nan")
                for stem_name in REFERENCE_STEMS:
                    detail_row[f"{stem_name}_sdr"] = None

                details_rows.append(detail_row)
                if is_oom:
                    oom_count += 1
                else:
                    error_count += 1

                typer.echo(
                    f"{config.config_id} track {track.name} failed with "
                    f"{detail_row['status']}: {error_type}: {error}"
                )
                traceback.print_exc()
                _empty_cache(device)

            measured_peak = _peak_memory_bytes(device)
            if measured_peak is not None:
                peak_vram_bytes = max(peak_vram_bytes, measured_peak)

        config_detail_rows = [
            row for row in details_rows if row["config_id"] == config.config_id
        ]
        ok_rows = [row for row in successful_track_rows if row["status"] == "ok"]
        # Batched (dataset-throughput) rows have ``elapsed_sec=None`` — per-track
        # time isn't observable there — so filter them out. This leaves
        # ok_elapsed empty in batched mode, which correctly nulls the
        # first-track / steady-state per-track stats below; dataset_wall_sec /
        # tracks_per_sec carry the real timing.
        ok_elapsed = [
            float(row["elapsed_sec"])
            for row in ok_rows
            if row.get("elapsed_sec") is not None
        ]
        remaining_elapsed = ok_elapsed[1:]
        attempted_elapsed = [
            float(row["elapsed_sec"])
            for row in config_detail_rows
            if row.get("elapsed_sec") is not None
        ]
        mean_sdr_values = [
            float(row["mean_sdr"])
            for row in ok_rows
            if not math.isnan(float(row["mean_sdr"]))
        ]

        per_stem_summary: dict[str, float] = {}
        for stem_name in REFERENCE_STEMS:
            stem_values = [
                float(row[f"{stem_name}_sdr"])
                for row in ok_rows
                if row.get(f"{stem_name}_sdr") is not None
                and not math.isnan(float(row[f"{stem_name}_sdr"]))
            ]
            per_stem_summary[stem_name] = (
                statistics.fmean(stem_values) if stem_values else float("nan")
            )

        peak_rss_mb = rss_sampler.stop()
        peak_vram_smi_mb = vram_sampler.stop() if vram_sampler else None

        summary_rows.append(
            {
                **_summary_row_base(
                    config,
                    effective_chunk_batch_size,
                    seed,
                    device,
                    use_only_stem=use_only_stem,
                ),
                "status": "ok" if len(ok_rows) == len(tracks) else "partial",
                "error_type": config_error_type,
                "error_message": config_error_message,
                "num_tracks": len(tracks),
                "ok_tracks": len(ok_rows),
                "error_tracks": error_count,
                "oom_tracks": oom_count,
                "model_init_sec": model_init_sec,
                "first_attempt_sec": (
                    float(config_detail_rows[0]["elapsed_sec"])
                    if config_detail_rows
                    and config_detail_rows[0].get("elapsed_sec") is not None
                    else None
                ),
                "first_attempt_status": (
                    str(config_detail_rows[0]["status"]) if config_detail_rows else ""
                ),
                "first_track_sec": ok_elapsed[0] if ok_elapsed else None,
                "remaining_total_sec": sum(remaining_elapsed)
                if remaining_elapsed
                else 0.0,
                "steady_state_mean_sec": (
                    statistics.fmean(remaining_elapsed) if remaining_elapsed else None
                ),
                "steady_state_median_sec": (
                    statistics.median(remaining_elapsed) if remaining_elapsed else None
                ),
                "attempted_track_sec": sum(attempted_elapsed),
                "track_total_sec": sum(ok_elapsed),
                # Whole-dataset wall: the measured batched total in
                # dataset-throughput mode, else the summed per-track time.
                # ``tracks_per_sec`` is the headline throughput number.
                "dataset_wall_sec": (
                    dataset_wall_override
                    if dataset_wall_override is not None
                    else sum(attempted_elapsed)
                ),
                "tracks_per_sec": (
                    len(ok_rows)
                    / (
                        dataset_wall_override
                        if dataset_wall_override is not None
                        else sum(attempted_elapsed)
                    )
                    if (
                        ok_rows
                        and (
                            dataset_wall_override
                            if dataset_wall_override is not None
                            else sum(attempted_elapsed)
                        )
                        > 0
                    )
                    else None
                ),
                "mean_sdr": (
                    statistics.fmean(mean_sdr_values)
                    if mean_sdr_values
                    else float("nan")
                ),
                "median_sdr": (
                    statistics.median(mean_sdr_values)
                    if mean_sdr_values
                    else float("nan")
                ),
                "peak_vram_mb": peak_vram_bytes / (1024 * 1024)
                if peak_vram_bytes
                else None,
                "peak_vram_smi_mb": peak_vram_smi_mb,
                "peak_rss_mb": peak_rss_mb,
                **{
                    f"{stem}_mean_sdr": value
                    for stem, value in per_stem_summary.items()
                },
            }
        )

        del separator
        gc.collect()
        _empty_cache(device)

    detail_csv = output_dir / "benchmark_details.csv"
    summary_csv = output_dir / "benchmark_summary.csv"
    metadata_json = output_dir / "benchmark_metadata.json"

    detail_fieldnames = [
        "config_id",
        "variant",
        "upstream_version",
        "model",
        "precision",
        "device",
        "compile",
        "shifts",
        "split_overlap",
        "chunk_batch_size",
        "base_seed",
        "use_only_stem",
        "track_index",
        "track_name",
        "track_seed",
        "status",
        "error_type",
        "error_message",
        "elapsed_sec",
        "num_scored_stems",
        "mean_sdr",
        *[f"{stem}_sdr" for stem in REFERENCE_STEMS],
    ]
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        for row in details_rows:
            writer.writerow(
                {
                    key: _format_float(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    summary_fieldnames = [
        "config_id",
        "variant",
        "upstream_version",
        "model",
        "precision",
        "device",
        "compile",
        "shifts",
        "split_overlap",
        "chunk_batch_size",
        "base_seed",
        "use_only_stem",
        "status",
        "error_type",
        "error_message",
        "num_tracks",
        "ok_tracks",
        "error_tracks",
        "oom_tracks",
        "model_init_sec",
        "first_attempt_sec",
        "first_attempt_status",
        "first_track_sec",
        "remaining_total_sec",
        "steady_state_mean_sec",
        "steady_state_median_sec",
        "attempted_track_sec",
        "track_total_sec",
        "dataset_wall_sec",
        "tracks_per_sec",
        "peak_vram_mb",
        "peak_vram_smi_mb",
        "peak_rss_mb",
        "mean_sdr",
        "median_sdr",
        *[f"{stem}_mean_sdr" for stem in REFERENCE_STEMS],
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: _format_float(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    metadata = {
        "musdb_root": str(musdb_root),
        "output_dir": str(output_dir),
        "device": device,
        "models": models,
        "precisions": precisions,
        "compile_modes": compile_modes,
        "shifts": shifts_values,
        "split_overlaps": split_overlaps,
        "seed": seed,
        "limit": limit,
        "dataset_throughput": dataset_throughput,
        "compute_sdr": compute_sdr,
        "use_only_stem": use_only_stem,
        "num_tracks": len(tracks),
        "num_configs": len(configs),
        "include_upstream": include_upstream,
        "upstream_version": upstream_version if include_upstream else None,
        "upstream_python": upstream_python if include_upstream else None,
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))

    typer.echo(f"Wrote details: {detail_csv}")
    typer.echo(f"Wrote summary: {summary_csv}")
    typer.echo(f"Wrote metadata: {metadata_json}")


if __name__ == "__main__":
    app()
