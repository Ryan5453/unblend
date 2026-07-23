# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import time
from typing import Annotated

import torch
import typer
from rich.table import Table

from ..api import Separator, default_device
from ..apply import ModelEnsemble
from ..exceptions import ModelLoadingError, ValidationError
from ..htdemucs import HTDemucs
from ..roformer import _RoformerBase
from .types import DeviceType, ModelName, Precision
from .utils import console

# Batch sizes probed to find the smallest eager batch that already saturates
# throughput (bigger ones only cost VRAM). Bounded low because these models
# saturate a modern GPU at a small batch.
_EAGER_CBS_CANDIDATES: tuple[int, ...] = (4, 8, 16, 32)
# A batch is "as fast" as a bigger one if within this fraction of the best
# measured per-chunk throughput; the smallest such batch is recommended.
_EAGER_CBS_TOLERANCE: float = 1.03
# Dummy-audio length used to measure end-to-end eager/compiled throughput.
# Long enough to amortize tail-padding like a real multi-minute song, short
# enough to keep tuning quick.
_TUNE_AUDIO_SECONDS: int = 180
# Compile speedups within this band of 1.0x are reported as a wash: for models
# already at 100x+ realtime the eager/compile delta is smaller than run-to-run
# clock/noise, so a hard verdict would flip between runs.
_COMPILE_NEGLIGIBLE_BAND: float = 0.05
# Default set: one of each architecture. ``-m`` narrows or widens it.
_DEFAULT_TUNE_MODELS: tuple[str, ...] = (
    "htdemucs",
    "bs_roformer_sw",
    "melband_roformer_kim",
)


def _sync(device: str) -> None:
    """
    Block until queued GPU work finishes, so wall timings are real.

    :param device: Inference device string.
    """
    if device == "cuda":
        torch.cuda.synchronize()


def _free(separator: Separator | None) -> None:
    """
    Drop a separator and release its GPU memory back to the allocator.

    :param separator: Separator to release (``None`` is a no-op).
    """
    del separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reference_model(separator: Separator) -> HTDemucs | _RoformerBase | None:
    """
    The concrete HTDemucs/RoFormer used for per-forward timing (an ensemble's
    first supported member; ``None`` for unsupported models).

    :param separator: Loaded separator.
    :return: A timing reference model or ``None``.
    """
    model = separator.model
    if isinstance(model, ModelEnsemble):
        return next(
            (m for m in model.models if isinstance(m, (HTDemucs, _RoformerBase))), None
        )
    return model if isinstance(model, (HTDemucs, _RoformerBase)) else None


def _recommended_eager_cbs(separator: Separator, ref: HTDemucs | _RoformerBase) -> int:
    """
    Smallest eager batch size whose per-chunk throughput is within tolerance of
    the best measured -- i.e. the memory-frugal batch that still saturates the
    GPU (the auto-sizer's memory-greedy pick is often much larger for no speed).

    :param separator: Eager separator (its ``chunk_batch_size`` is used/mutated).
    :param ref: Timing reference model.
    :return: Recommended eager chunk_batch_size.
    """
    original = separator.chunk_batch_size
    measurements: list[tuple[int, float]] = []
    for cbs in _EAGER_CBS_CANDIDATES:
        separator.chunk_batch_size = cbs
        try:
            per_chunk = separator._time_forward_seconds_per_chunk(ref)
        except RuntimeError:
            break  # ran out of VRAM at this batch; larger won't fit either
        measurements.append((cbs, per_chunk))
        if separator.device == "cuda":
            torch.cuda.empty_cache()
    separator.chunk_batch_size = original
    if not measurements:
        return original
    best = min(per_chunk for _, per_chunk in measurements)
    for cbs, per_chunk in measurements:
        if per_chunk <= best * _EAGER_CBS_TOLERANCE:
            return cbs
    return measurements[-1][0]


def _separation_wall(
    separator: Separator, dummy: torch.Tensor, sample_rate: int, reps: int = 2
) -> float:
    """
    Best-of-N wall time to separate a fixed dummy clip through the real
    ``separate`` path (captures everything: tiling, overlap-add, tail padding).

    :param separator: Separator to time.
    :param dummy: Zero waveform ``[channels, samples]``.
    :param sample_rate: Sample rate of the dummy.
    :param reps: Timed repetitions after one warmup.
    :return: Best wall-clock seconds.
    """
    separator.separate((dummy, sample_rate), shifts=1, split_overlap=0.25)
    _sync(separator.device)
    best = float("inf")
    for _ in range(reps):
        started = time.perf_counter()
        separator.separate((dummy, sample_rate), shifts=1, split_overlap=0.25)
        _sync(separator.device)
        best = min(best, time.perf_counter() - started)
    return best


def _format_compile(
    speedup: float, saved_per_audio_second: float, setup: float, compile_cbs: int
) -> str:
    """
    Render the compile verdict: a wash, a loss, or a break-even amount of audio.

    :param speedup: eager_wall / compile_wall.
    :param saved_per_audio_second: Wall seconds saved per audio-second by compiling.
    :param setup: Measured compile setup cost in seconds.
    :param compile_cbs: Batch size the compile capture settled on.
    :return: Human-readable compile column.
    """
    if abs(speedup - 1.0) < _COMPILE_NEGLIGIBLE_BAND:
        # Within noise for this (usually already very fast) model: report a wash
        # rather than flip between "worth it"/"not worth it" run to run.
        return f"negligible (~{speedup:.2f}x, compile \u2248 eager)"
    if saved_per_audio_second <= 0:
        return f"not worth it ({speedup:.2f}x, slower than eager)"
    breakeven_min = setup / saved_per_audio_second / 60.0
    if breakeven_min >= 120:
        return (
            f"{speedup:.2f}x \u2014 rarely worth it (~{breakeven_min / 60:.0f} h/run) "
            f"(cbs {compile_cbs}, ~{setup:.0f}s setup)"
        )
    return (
        f"{speedup:.2f}x \u2014 worth it above ~{breakeven_min:.0f} min/run "
        f"(cbs {compile_cbs}, ~{setup:.0f}s setup)"
    )


def _tune_one(model_name: str, device: str, dtype: object) -> dict[str, object]:
    """
    Measure one model: recommended eager batch, and (on CUDA) the compiled
    speedup and its break-even audio length.

    :param model_name: Model to tune.
    :param device: Inference device.
    :param dtype: Inference dtype (``"auto"``/``torch.dtype``/``None``).
    :return: Row dict for the results table.
    """
    row: dict[str, object] = {"model": model_name}
    probe = Separator(model=model_name, device=device, dtype=dtype, compile=False)
    ref = _reference_model(probe)
    param = next(probe.model.parameters(), None)
    row["dtype"] = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(
        param.dtype if param is not None else torch.float32, "fp32"
    )
    if ref is None:
        row["eager_cbs"] = probe.chunk_batch_size
        row["compile"] = "unsupported model"
        _free(probe)
        return row

    sample_rate = ref.samplerate
    dummy = torch.zeros(ref.audio_channels, _TUNE_AUDIO_SECONDS * sample_rate)

    # Measure compilation FIRST, while the process is freshest. A compiled
    # capture sizes itself from the free VRAM its batch sweep reads; running the
    # eager pass first (with its GPU-resident overlap-add accumulators) leaves
    # enough allocator residue -- even after empty_cache -- to starve the sweep
    # into a smaller, slower capture than a real Separator(compile=True) picks.
    # ``setup`` is timed around enable_compile alone; model-load cost cancels
    # out of the break-even because eager startup pays it too.
    setup: float | None = None
    compile_wall: float | None = None
    compile_cbs: int | None = None
    if device == "cuda":
        setup_started = time.perf_counter()
        try:
            probe.enable_compile()
        except (ValidationError, ModelLoadingError, RuntimeError) as error:
            row["compile"] = f"unavailable ({type(error).__name__})"
        else:
            setup = time.perf_counter() - setup_started
            compile_wall = _separation_wall(probe, dummy, sample_rate)
            compile_cbs = probe.chunk_batch_size
    _free(probe)

    # Eager measurement second: its memory-greedy sizing is robust to residue.
    eager_separator = Separator(
        model=model_name, device=device, dtype=dtype, compile=False
    )
    eager_ref = _reference_model(eager_separator)
    assert eager_ref is not None  # same model as the supported ``ref`` above
    eager_cbs = (
        _recommended_eager_cbs(eager_separator, eager_ref)
        if device == "cuda"
        else eager_separator.chunk_batch_size
    )
    eager_separator.chunk_batch_size = eager_cbs
    eager_wall = _separation_wall(eager_separator, dummy, sample_rate)
    row["eager_cbs"] = eager_cbs
    row["eager_rt"] = _TUNE_AUDIO_SECONDS / eager_wall
    _free(eager_separator)

    if device != "cuda":
        row["compile"] = "n/a (CUDA only)"
    elif compile_wall is not None and compile_cbs is not None and setup is not None:
        speedup = eager_wall / compile_wall
        saved_per_audio_second = (eager_wall - compile_wall) / _TUNE_AUDIO_SECONDS
        row["compile_cbs"] = compile_cbs
        row["compile"] = _format_compile(
            speedup, saved_per_audio_second, setup, compile_cbs
        )
    # else: an "unavailable (...)" message was already set above.
    return row


def tune_command(
    models: Annotated[
        list[ModelName] | None,
        typer.Option(
            "-m",
            "--model",
            help="Model(s) to tune (repeat to add more). Defaults to one of each architecture.",
        ),
    ] = None,
    precision: Annotated[
        Precision,
        typer.Option("--precision", "-p", help="Inference precision to tune for."),
    ] = Precision.auto,
    device: Annotated[
        DeviceType | None,
        typer.Option(
            "--device", "-d", help="Device to tune on (default: auto-detect)."
        ),
    ] = None,
) -> None:
    """
    Measure the fastest settings for *this* machine and print them \u2014 nothing is
    written to disk. Reports the recommended eager ``--chunk-batch-size`` and,
    on CUDA, whether ``--compile`` pays off (with the break-even amount of audio
    per run). Feed the results back via explicit flags.

    :param models: Models to tune; defaults to one model per architecture.
    :param precision: Inference precision (``auto`` resolves per device).
    :param device: Inference device (``None`` auto-detects cuda > mps > cpu).
    """
    resolved_device = (device or DeviceType(default_device())).value
    if precision is Precision.auto:
        dtype: object = "auto"
    elif precision is Precision.fp32:
        dtype = None
    elif precision is Precision.fp16:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    model_names = [m.value for m in models] if models else list(_DEFAULT_TUNE_MODELS)

    header = f"unblend tune \u00b7 {resolved_device}"
    if resolved_device == "cuda" and torch.cuda.is_available():
        header += f" ({torch.cuda.get_device_name(0)}) \u00b7 torch {torch.__version__}"
    console.print(f"[cyan]{header}[/cyan]")
    console.print(
        "[dim]Measuring \u2014 the compile pass recompiles a few batch sizes, so this "
        "takes a couple of minutes per model. Nothing is saved to disk.[/dim]"
    )

    rows: list[dict[str, object]] = []
    for name in model_names:
        console.print(f"[dim]  tuning {name}\u2026[/dim]")
        try:
            rows.append(_tune_one(name, resolved_device, dtype))
        except (ValidationError, ModelLoadingError) as error:
            rows.append(
                {"model": name, "eager_cbs": "\u2014", "compile": f"error: {error}"}
            )

    table = Table(title=None, show_lines=False)
    table.add_column("model", style="bold")
    table.add_column("dtype")
    table.add_column("eager batch", justify="right")
    table.add_column("eager rt", justify="right")
    table.add_column("compile")
    for row in rows:
        rt = row.get("eager_rt")
        table.add_row(
            str(row.get("model", "")),
            str(row.get("dtype", "")),
            str(row.get("eager_cbs", "")),
            f"{rt:.0f}x" if isinstance(rt, (int, float)) else "",
            str(row.get("compile", "")),
        )
    console.print(table)
    console.print(
        "[dim]Use with: unblend separate <tracks> -m <model> "
        "--chunk-batch-size <eager batch> [--compile][/dim]"
    )
