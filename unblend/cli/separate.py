# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import click
import torch
import typer
from rich.markup import escape

from .._paths import name_encodable, name_fits
from ..api import (
    SeparatedSources,
    Separator,
    _check_combine_override,
    default_device,
    select_model,
)
from ..apply import COMBINE_DEFAULT, ModelEnsemble
from ..audio import AUDIO_SUFFIXES, resolve_output_path
from ..exceptions import ModelLoadingError, ValidationError
from ..htdemucs import HTDemucs
from ..repo import stem_key
from ..roformer import BSRoformer, MelBandRoformer
from .models import ensure_model_available
from .progress import FileProgressTracker
from .types import ClipMode, DeviceType, Precision
from .utils import (
    AUTO_MODEL,
    TEMPLATE_VARIABLES,
    complete_combine_mode,
    complete_model_name,
    complete_stem_name,
    console,
    err_console,
    expand_paths_to_audio_files,
    format_output_path,
    get_models,
    unknown_placeholders,
    validate_combine_mode,
    validate_model_name,
)

# Auto-compile break-even thresholds, in seconds of predicted eager GPU work
# (estimated chunks x a measured batch-1 eager timing) rather than raw chunk
# counts, so one threshold adapts across GPUs. Each is the largest measured
# break-even across the supported datacenter GPUs, rounded up; the fastest
# GPUs set it, since compilation's fixed setup cost does not shrink with
# eager time. FP32/BF16 compilation is marginal on modern GPUs, so those
# thresholds are high enough that auto mode rarely compiles them.
_AUTO_COMPILE_EAGER_SECONDS: dict[tuple[str, str], int] = {
    ("htdemucs", "fp32"): 1100,
    ("htdemucs", "fp16"): 700,
    ("htdemucs", "bf16"): 1100,
    ("bs_roformer", "fp32"): 500,
    ("bs_roformer", "fp16"): 450,
    ("bs_roformer", "bf16"): 500,
    ("mel_band_roformer", "fp32"): 1000,
    ("mel_band_roformer", "fp16"): 350,
    ("mel_band_roformer", "bf16"): 1000,
}


def _compile_profile_key(separator: Separator) -> tuple[str, str] | None:
    """
    Resolve the architecture/dtype key used by the auto-compile policy.

    :param separator: Initialized eager separator.
    :return: ``(architecture, precision)`` or ``None`` when unsupported.
    """
    model = (
        separator.model.models[0]
        if isinstance(separator.model, ModelEnsemble)
        else separator.model
    )
    if isinstance(model, HTDemucs):
        architecture = "htdemucs"
    elif isinstance(model, BSRoformer):
        architecture = "bs_roformer"
    elif isinstance(model, MelBandRoformer):
        architecture = "mel_band_roformer"
    else:
        return None

    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    precision = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(dtype, "fp32")
    return architecture, precision


def _audio_duration_seconds(path: Path) -> float | None:
    """
    Read audio duration from container metadata without decoding samples.

    :param path: Input audio path.
    :return: Positive duration in seconds, or ``None`` when unavailable.
    """
    try:
        from unblend.api import _torchcodec

        duration = _torchcodec("decoder")(str(path)).metadata.duration_seconds
    except Exception:
        return None
    if duration is None or not math.isfinite(duration) or duration <= 0:
        return None
    return float(duration)


def _estimate_compile_chunks(
    separator: Separator,
    audio_files: list[Path],
    *,
    shifts: int,
    split_overlap: float,
) -> tuple[int, float, int]:
    """
    Estimate total model chunks from metadata durations and inference options.

    Shift offsets are random in ``[0, 0.5s]``; using their 0.25-second mean
    makes the estimate unbiased without consuming or perturbing the run's RNG.
    Ensemble members aren't multiplied in here: the eager probe time the
    estimate is scaled by already sums every member's forward.

    :param separator: Eager separator providing segment/sample-rate metadata.
    :param audio_files: Expanded input file list.
    :param shifts: Shift rounds per input.
    :param split_overlap: Fractional chunk overlap.
    :return: ``(estimated_chunks, known_duration_seconds, unknown_file_count)``.
    """
    sample_rate = separator.model.samplerate
    segment_samples = int(round(separator.model.max_allowed_segment * sample_rate))
    # At least 1, so the estimate never divides by zero (apply() itself
    # refuses an overlap that leaves no stride).
    stride = max(1, int((1 - split_overlap) * segment_samples))
    total_chunks = 0
    total_duration = 0.0
    unknown_files = 0
    expected_shift_padding = int(0.25 * sample_rate)
    for path in audio_files:
        duration = _audio_duration_seconds(path)
        if duration is None:
            unknown_files += 1
            continue
        total_duration += duration
        samples = int(math.ceil(duration * sample_rate))
        if shifts:
            chunks = shifts * math.ceil((samples + expected_shift_padding) / stride)
        else:
            chunks = math.ceil(samples / stride)
        total_chunks += chunks
    return total_chunks, total_duration, unknown_files


def _maybe_enable_auto_compile(
    separator: Separator,
    audio_files: list[Path],
    *,
    shifts: int,
    split_overlap: float,
) -> bool:
    """
    Apply the cache-free CUDA auto-compile policy to an eager separator.

    :param separator: Initialized eager CUDA separator.
    :param audio_files: Complete expanded CLI workload.
    :param shifts: Shift rounds per input.
    :param split_overlap: Fractional chunk overlap.
    :return: ``True`` when compilation was enabled successfully.
    """
    profile_key = _compile_profile_key(separator)
    threshold = (
        _AUTO_COMPILE_EAGER_SECONDS.get(profile_key)
        if profile_key is not None
        else None
    )
    estimated_chunks, known_duration, unknown_files = _estimate_compile_chunks(
        separator,
        audio_files,
        shifts=shifts,
        split_overlap=split_overlap,
    )
    probe_seconds = separator._eager_probe_seconds
    if threshold is not None and probe_seconds is None:
        # An explicit --chunk-batch-size skips the sizing probe that also
        # times a forward, so time one here.
        separator._measure_per_chunk_steady_bytes()
        probe_seconds = separator._eager_probe_seconds
    if threshold is None or probe_seconds is None:
        console.print(
            "[cyan]Auto compile:[/cyan] keeping eager execution — "
            "no supported timing profile was available"
        )
        return False

    estimated_eager_seconds = estimated_chunks * probe_seconds
    detail = (
        f"{estimated_chunks:,} estimated chunks, "
        f"{known_duration / 60:.1f} min known audio, "
        f"{estimated_eager_seconds:.1f}s predicted eager GPU work "
        f"(threshold {threshold}s)"
    )
    if unknown_files:
        suffix = "s" if unknown_files != 1 else ""
        detail += f", {unknown_files} duration{suffix} unavailable"
    if estimated_eager_seconds < threshold:
        console.print(f"[cyan]Auto compile:[/cyan] keeping eager execution — {detail}")
        return False

    console.print(f"[cyan]Auto compile:[/cyan] enabling CUDA compilation — {detail}")
    try:
        separator.enable_compile()
    except Exception as error:
        # Auto mode is opportunistic: enable_compile restores the eager
        # callable/batch size before re-raising, so an unsupported
        # compiler/toolchain should not kill the job.
        err_console.print(
            f"[yellow]![/yellow] CUDA compile setup failed; "
            f"continuing eager: {escape(str(error))}"
        )
        return False
    return True


def _static_output_root(template: str) -> Path | None:
    """
    The fixed directory prefix of an output template, e.g. ``separated`` for
    ``separated/{model}/{track}/{stem}.{ext}``.

    :param template: Output path template
    :return: The resolved prefix, or ``None`` if the template has none
    """
    parts: list[str] = []
    for part in Path(template).parts[:-1]:
        if "{" in part:
            break
        parts.append(part)
    if not parts:
        return None
    try:
        # os.path.expanduser, as format_output_path uses: an unknown ~user
        # stays a literal folder, which this must then exclude too.
        return Path(os.path.expanduser(Path(*parts))).resolve()
    except (OSError, RuntimeError):
        # e.g. a symlink loop; the output planning reports it properly.
        return None


def _validate_output_format(format: str) -> None:
    """
    Fail fast on an unsupported --format by encoding a tiny silent clip
    through ``export_stem``, instead of erroring after an expensive separation.

    :param format: Output format/extension to validate
    :raises typer.Exit: If the format is not encodable
    """
    probe = SeparatedSources(
        {"probe": torch.zeros(2, 4410)}, 44100, torch.zeros(2, 4410)
    )
    try:
        probe.export_stem("probe", format=format, clip=None)
    except Exception as error:
        if str(error).startswith("FFmpeg couldn't be loaded"):
            err_console.print(f"[red]✗[/red] {escape(str(error))}")
            raise typer.Exit(1) from error
        err_console.print(
            f"[red]✗[/red] Unsupported output format '{escape(format)}': "
            f"{escape(str(error))}"
        )
        raise typer.Exit(1)


def separate_command(
    # Input/Output
    tracks: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Audio files or directories to separate",
            show_default=False,
        ),
    ] = None,
    # Model Selection
    model: Annotated[
        str,
        typer.Option(
            "-m",
            "--model",
            help="Model name, or 'auto' (an HTDemucs model suited to --isolate-stem)",
            rich_help_panel="Model Selection",
            callback=validate_model_name,
            autocompletion=complete_model_name,
        ),
    ] = AUTO_MODEL,
    combine: Annotated[
        str | None,
        typer.Option(
            "--combine",
            help="Ensemble combine mode (default: the model's own)",
            show_default=False,
            rich_help_panel="Model Selection",
            callback=validate_combine_mode,
            autocompletion=complete_combine_mode,
        ),
    ] = None,
    # Processing Options
    device: Annotated[
        DeviceType | None,
        typer.Option(
            "-d",
            "--device",
            help="Device (default: cuda, then mps, then cpu)",
            show_default=False,
            rich_help_panel="Processing",
        ),
    ] = None,
    shifts: Annotated[
        int,
        typer.Option(
            min=0,
            max=20,
            help="Random shifts to average; slower, slightly better. 0 is deterministic",
            rich_help_panel="Processing",
        ),
    ] = 1,
    split_overlap: Annotated[
        float,
        typer.Option(
            "--split-overlap",
            min=0.0,
            max=0.99,
            help="Overlap between chunks; higher smooths chunk boundaries",
            rich_help_panel="Processing",
        ),
    ] = 0.25,
    seed: Annotated[
        int | None,
        typer.Option(
            help="Seed for the shifts, for reproducible output",
            rich_help_panel="Processing",
        ),
    ] = None,
    compile_model: Annotated[
        bool | None,
        typer.Option(
            "--compile/--no-compile",
            help="Force torch.compile on/off (CUDA; default: decided per workload)",
            rich_help_panel="Processing",
        ),
    ] = None,
    custom_kernels: Annotated[
        bool | None,
        typer.Option(
            "--custom-kernels/--native-ops",
            help="--native-ops disables the fused CUDA/Metal kernels",
            rich_help_panel="Processing",
        ),
    ] = None,
    precision: Annotated[
        Precision,
        typer.Option(
            "--precision",
            help="Compute precision; auto is fp16 on MPS and on CUDA GPUs with tensor cores, else fp32",
            rich_help_panel="Processing",
        ),
    ] = Precision.auto,
    chunk_batch_size: Annotated[
        int | None,
        typer.Option(
            "--chunk-batch-size",
            min=1,
            max=1024,
            help="Chunks per forward pass (default: sized from free memory on CUDA, 1 elsewhere)",
            show_default=False,
            rich_help_panel="Processing",
        ),
    ] = None,
    # Output
    output: Annotated[
        str,
        typer.Option(
            "-o",
            "--output",
            help="Path template: {model} {track} {parent} {stem} {ext} {date} {time} {timestamp}",
            rich_help_panel="Output",
        ),
    ] = "separated/{model}/{track}/{stem}.{ext}",
    isolate_stem: Annotated[
        str | None,
        typer.Option(
            help="Write only STEM and no_STEM",
            rich_help_panel="Output",
            autocompletion=complete_stem_name,
        ),
    ] = None,
    clip_mode: Annotated[
        ClipMode,
        typer.Option(
            help="How to keep output within [-1, 1]",
            rich_help_panel="Output",
        ),
    ] = ClipMode.rescale,
    format: Annotated[
        str,
        typer.Option(
            "-f",
            "--format",
            help="Output format, e.g. wav, flac, mp3",
            rich_help_panel="Output",
        ),
    ] = "wav",
) -> None:
    """
    Separates the given tracks.

    :param tracks: Paths to audio files or directories containing audio files
    :param model: Model to use for separation
    :param combine: For an ensemble, how member outputs are combined
    :param device: Device to process separation on
    :param shifts: Number of random shifts for equivariant stabilization;
        increases separation time but improves quality
    :param split_overlap: Overlap between split chunks; higher values improve
        quality at chunk boundaries
    :param seed: Random seed for reproducible shift-based inference
    :param compile_model: CUDA compile override: ``True`` forces compile,
        ``False`` forces eager, and ``None`` automatically compares estimated
        eager GPU work with the architecture/dtype break-even threshold.
    :param custom_kernels: Fused-kernel override: ``False`` (``--native-ops``)
        forces vanilla PyTorch ops on every device, ``True`` keeps fused
        CUDA/Metal kernels where eligible, and ``None`` defers to the
        ``UNBLEND_CUSTOM_KERNELS`` environment variable (enabled unless it is
        set to a falsy value).
    :param precision: Inference precision; auto picks fp16 on CUDA (with tensor
        cores) and MPS, fp32 on CPU
    :param chunk_batch_size: How many split chunks to run per forward pass.
        ``None`` sizes it from available memory and backs off on OOM; an
        explicit value is used exactly as given, and OOM raises.
    :param output: Output path template; variables are {model}, {track}, {parent}, {stem},
        {ext}, {date}, {time}, {timestamp}
    :param isolate_stem: Only creates a {stem} and no_{stem} stem/file
    :param clip_mode: How to keep output within [-1, 1]
    :param format: Output format, e.g. wav, flac, mp3
    """
    if tracks is None or not tracks:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    unknown = unknown_placeholders(output)
    if unknown:
        err_console.print(
            f"[red]✗[/red] Unknown placeholder(s) {escape(', '.join(unknown))} in "
            f"the output template; use {escape(', '.join('{' + v + '}' for v in TEMPLATE_VARIABLES))}."
        )
        raise typer.Exit(1)

    # Resolved at invocation rather than as the parameter default: a default-
    # argument ternary would probe CUDA/MPS at import time (even for --help).
    if device is None:
        device = DeviceType(default_device())

    audio_files, had_path_errors = expand_paths_to_audio_files(
        tracks, exclude=_static_output_root(output)
    )
    # The same file named twice (or via two paths) is separated once.
    unique: dict[Path, Path] = {}
    for track in audio_files:
        unique.setdefault(track.resolve(), track)
    audio_files = list(unique.values())

    if not audio_files:
        err_console.print("[red]No audio files found to process.[/red]")
        raise typer.Exit(1)

    # A format is used as a file suffix, so anything Path() would normalize
    # away ("./wav", "wav/", "") can pass a probe built from the normalized
    # string yet fail (or silently differ) at write time — reject it up front.
    if "/" in format or "\\" in format or format != str(Path(format)):
        err_console.print(
            f"[red]✗[/red] Unsupported output format '{escape(format)}': "
            "not a valid file extension."
        )
        raise typer.Exit(1)

    # Accept "-f .wav" as "wav" rather than writing "drums..wav". Done after
    # the guard so rejection messages above stay as typed.
    if format.startswith(".") and format.lstrip("."):
        format = format.lstrip(".")
    format = format.lower()

    # Catch an unsupported output container before any model download or
    # separation work. An audio extension in the template (``out/{stem}.flac``)
    # overrides --format, since export keys the container off the path suffix.
    template_suffix = Path(output).suffix
    if "{" in template_suffix:
        template_suffix = str(
            format_output_path(
                template_suffix, "model", Path("track.wav"), "stem", format
            )
        )
    if template_suffix.lower() in AUDIO_SUFFIXES:
        effective_format = template_suffix.lstrip(".")
    else:
        effective_format = format
    _validate_output_format(effective_format)

    if device is DeviceType.cpu and precision in (Precision.fp16, Precision.bf16):
        err_console.print(
            f"[red]✗[/red] --precision {precision.value} isn't supported on CPU; "
            "use fp32 or auto."
        )
        raise typer.Exit(1)

    if model == AUTO_MODEL:
        selected_model_name, only_load_stem = select_model(
            isolate_stem=isolate_stem,
        )
        console.print(
            f"[cyan]Auto-selected model:[/cyan] [bold]{selected_model_name}[/bold]"
        )
    else:
        selected_model_name = model
        if isolate_stem is not None:
            sources = (get_models().get(model) or {}).get("sources", [])
            by_key = {stem_key(source): source for source in sources}
            isolate_stem = by_key.get(stem_key(isolate_stem), isolate_stem)
        only_load_stem = isolate_stem

    # Cheap checks Separator would make anyway, before a possibly large
    # download.
    if isolate_stem is not None:
        sources = (get_models().get(selected_model_name) or {}).get("sources", [])
        if sources and stem_key(isolate_stem) not in {stem_key(s) for s in sources}:
            err_console.print(
                f"[red]✗[/red] Stem {escape(repr(isolate_stem))} not found in "
                f"{escape(selected_model_name)}. Available stems: "
                f"{escape(', '.join(sources))}"
            )
            raise typer.Exit(1)
    if device is DeviceType.cuda and not torch.cuda.is_available():
        err_console.print("[red]✗[/red] --device cuda, but CUDA isn't available.")
        raise typer.Exit(1)
    if device is DeviceType.mps and not torch.backends.mps.is_available():
        err_console.print("[red]✗[/red] --device mps, but MPS isn't available.")
        raise typer.Exit(1)
    if combine is not None:
        # The library's own check, from the registry entry, so a mode the
        # ensemble can't use is refused before its members download.
        info = get_models().get(selected_model_name, {})
        try:
            _check_combine_override(
                combine,
                None,
                label=selected_model_name,
                is_ensemble="members" in info,
                own_combine=info.get("combine", COMBINE_DEFAULT),
                own_params=info.get("combine_params"),
                weights=info.get("weights"),
            )
        except ValidationError as error:
            err_console.print(f"[red]✗[/red] {escape(str(error))}")
            raise typer.Exit(1) from error

    # Detect output-path collisions before any separation: if two (track,
    # stem) pairs resolve to the same path, the second would silently
    # overwrite the first.
    # From the registry, so every path problem is caught before the model is
    # downloaded; the loaded model has the same sources.
    registry_sources = list(
        (get_models().get(selected_model_name) or {}).get("sources", [])
    )
    if isolate_stem is not None:
        by_key = {stem_key(source): source for source in registry_sources}
        stem = by_key.get(stem_key(isolate_stem), isolate_stem)
        planned_stems = [stem, f"no_{stem}"]
    else:
        planned_stems = registry_sources

    # One timestamp for the whole run, so {date}/{time}/{timestamp} resolve
    # identically in the collision check, the displayed template, and the
    # writes.
    now = datetime.now()

    planned_paths: dict[str, list[str]] = {}
    planned_containers: set[str] = set()
    for track in audio_files:
        for stem_name in planned_stems:
            path = format_output_path(
                output, selected_model_name, track, stem_name, format, now=now
            )
            if not path.name:
                err_console.print(
                    "[red]✗[/red] Output template resolves to an empty filename "
                    f"('{escape(str(path))}' for {escape(track.name)} → "
                    f"{escape(stem_name)}). Add a filename component such as "
                    "[bold]{track}[/bold] or [bold]{stem}[/bold]."
                )
                raise typer.Exit(1)
            # Check the path exactly as export_stem will write it.
            path = resolve_output_path(path, format)
            if not name_encodable(str(path)):
                err_console.print(
                    f"[red]✗[/red] {escape(str(path))} can't be encoded as a "
                    "file name here."
                )
                raise typer.Exit(1)
            too_long = [part for part in path.parts if not name_fits(part)]
            if too_long:
                err_console.print(
                    f"[red]✗[/red] {escape(too_long[-1])} is too long for a "
                    "file or folder name; shorten the output template."
                )
                raise typer.Exit(1)
            if os.path.isdir(path):
                err_console.print(
                    f"[red]✗[/red] {escape(str(path))} is a folder; the output "
                    "template must name files."
                )
                raise typer.Exit(1)
            planned_containers.add(path.suffix.lstrip("."))
            # The full path: colliding tracks often share a basename.
            planned_paths.setdefault(str(path), []).append(f"{track} → {stem_name}")

    for container in sorted(planned_containers - {effective_format}):
        _validate_output_format(container)

    collisions = {p: srcs for p, srcs in planned_paths.items() if len(srcs) > 1}
    if collisions:
        err_console.print(
            "[red]✗[/red] Output template produces colliding paths; outputs would "
            "overwrite each other. Add [bold]{track}[/bold], [bold]{stem}[/bold] "
            "or, for same-named files in different folders, [bold]{parent}[/bold] "
            "to the template:"
        )
        for path, srcs in collisions.items():
            err_console.print(
                f"  [bold]{escape(path)}[/bold] ← {escape(', '.join(srcs))}"
            )
        raise typer.Exit(1)

    # Every output folder must be creatable and writable before any track is
    # separated, not discovered failing at the first export.
    for folder in sorted({str(Path(p).parent) for p in planned_paths}):
        existing = Path(folder)
        # lexists: a dangling symlink in the way stops the walk and fails
        # is_dir() below instead of being walked past.
        while not os.path.lexists(existing) and existing != existing.parent:
            existing = existing.parent
        if not os.path.isdir(existing) or not os.access(existing, os.W_OK):
            err_console.print(
                f"[red]✗[/red] Can't write to {escape(folder)} "
                f"({escape(str(existing))} isn't a writable folder)."
            )
            raise typer.Exit(1)

    canonical_groups: dict[str, set[str]] = {}
    for planned in planned_paths:
        # resolve() collapses dot components and symlinked parents; a caseless
        # Unicode match catches aliases on case- or normalization-insensitive
        # filesystems. Non-strict resolve no longer raises on symlink loops
        # (Python 3.13+), so probe strictly first; only a missing file is OK.
        path = Path(planned)
        try:
            path.resolve(strict=True)
        except FileNotFoundError:
            pass
        except (OSError, RuntimeError) as exc:
            err_console.print(
                "[red]✗[/red] Could not resolve planned output path "
                f"'{escape(planned)}': {escape(str(exc))}"
            )
            raise typer.Exit(1) from exc
        try:
            resolved = str(path.resolve(strict=False))
        except (OSError, RuntimeError) as exc:
            err_console.print(
                "[red]✗[/red] Could not resolve planned output path "
                f"'{escape(planned)}': {escape(str(exc))}"
            )
            raise typer.Exit(1) from exc
        key = stem_key(resolved)
        canonical_groups.setdefault(key, set()).add(planned)
    input_keys = {stem_key(str(track.resolve())) for track in audio_files}
    clobbered = sorted(
        alias
        for key, aliases in canonical_groups.items()
        if key in input_keys
        for alias in aliases
    )
    if clobbered:
        err_console.print(
            "[red]✗[/red] Output template would overwrite input files. Choose a "
            "template that writes somewhere else:"
        )
        for path in clobbered:
            err_console.print(f"  {escape(path)}")
        raise typer.Exit(1)

    aliasing_groups = [
        sorted(aliases) for aliases in canonical_groups.values() if len(aliases) > 1
    ]
    if aliasing_groups:
        err_console.print(
            "[red]✗[/red] Output paths resolve to case, Unicode, or filesystem "
            "aliases and may overwrite each other. Choose a template that "
            "produces distinct paths:"
        )
        for aliases in aliasing_groups:
            err_console.print(f"  {escape(', '.join(aliases))}")
        raise typer.Exit(1)

    # only_load keeps the download to the single specialist file when set.
    if not ensure_model_available(selected_model_name, only_load=only_load_stem):
        raise typer.Exit(1)

    # "auto" defers to Separator's own device-based choice.
    if precision is Precision.auto:
        dtype: torch.dtype | str | None = "auto"
    elif precision is Precision.fp32:
        dtype = None
    elif precision is Precision.fp16:
        dtype = torch.float16
    else:  # Precision.bf16
        dtype = torch.bfloat16

    # Separator validates only_load against the model's sources, and can
    # still raise ModelLoadingError after ensure_model_available (a corrupt
    # cache file triggers a re-download, which fails offline).
    try:
        separator = Separator(
            model=selected_model_name,
            device=device.value,
            only_load=only_load_stem,
            dtype=dtype,
            compile=compile_model is True,
            chunk_batch_size=chunk_batch_size,
            custom_kernels=custom_kernels,
            combine=combine,
        )
    except (ValidationError, ModelLoadingError) as error:
        err_console.print(
            f"[red]✗[/red] [bold]{escape(selected_model_name)}[/bold]: error: "
            f"{escape(str(error))}"
        )
        raise typer.Exit(1)

    # Also covers the case where --isolate-stem is set but only_load is not (the
    # auto path can pick a single model where only_load is a no-op): the stem
    # must still exist in the loaded model's sources.
    if isolate_stem is not None and isolate_stem not in separator.model.sources:
        # Stem names are matched ignoring case and Unicode form
        # (``Vocals`` -> ``vocals``).
        matches = [
            source
            for source in separator.model.sources
            if stem_key(source) == stem_key(isolate_stem)
        ]
        if len(matches) == 1:
            isolate_stem = matches[0]
    if isolate_stem is not None and isolate_stem not in separator.model.sources:
        err_console.print(
            f"[red]✗[/red] Stem {escape(repr(isolate_stem))} not found in "
            f"{escape(selected_model_name)}. Available stems: "
            f"{escape(', '.join(separator.model.sources))}"
        )
        raise typer.Exit(1)

    # Reuse format_output_path so the displayed template matches the written
    # paths. Passing "{stem}" (and "{track}" for several tracks) as the values
    # keeps them as literal placeholders.
    if len(audio_files) == 1:
        track = audio_files[0]
        message = "Separated track will be stored using template"
    else:
        # Keeps {parent} and {track} as literal placeholders in the display.
        track = Path("{parent}") / "{track}"
        message = "Separated tracks will be stored using template"
    resolved_template = resolve_output_path(
        format_output_path(
            output, selected_model_name, track, "{stem}", format, now=now
        ),
        format,
    )
    console.print(f"{message} '{escape(str(resolved_template))}'")

    # A literal extension in the template overrides --format (export keys the
    # container off the path suffix). Warn when -f was passed explicitly but
    # the template ignores it.
    template_ext = (
        resolved_template.suffix.lstrip(".")
        if resolved_template.suffix.lower() in AUDIO_SUFFIXES
        else ""
    )
    if template_ext and template_ext.lower() != format:
        ctx = click.get_current_context(silent=True)
        source = ctx.get_parameter_source("format") if ctx else None
        if source is not None and source.name == "COMMANDLINE":
            err_console.print(
                f"[yellow]Warning:[/yellow] output template extension "
                f"'.{escape(template_ext)}' overrides --format '{escape(format)}'"
            )

    # After every cheap check, so a bad template never costs a compile. The
    # auto policy reads container metadata and reuses the batch-1 timing
    # from Separator's VRAM-sizing probe (running one itself if an explicit
    # batch size skipped the probe).
    if compile_model and device is not DeviceType.cuda:
        err_console.print(
            "[yellow]![/yellow] --compile only applies on CUDA; ignoring it."
        )
    if compile_model is None and device is DeviceType.cuda:
        _maybe_enable_auto_compile(
            separator,
            audio_files,
            shifts=shifts,
            split_overlap=split_overlap,
        )

    had_error = False
    with FileProgressTracker() as progress_tracker:
        for track in audio_files:
            # Keyed on the full path so same-named files in different
            # directories don't collide.
            file_key = str(track)
            progress_tracker.start_file(file_key)

            try:
                audio_callback = progress_tracker.create_audio_callback(file_key)

                separated = separator.separate(
                    audio=track,
                    shifts=shifts,
                    split_overlap=split_overlap,
                    seed=seed,
                    progress_callback=audio_callback,
                )

                if isolate_stem is not None:
                    separated = separated.isolate_stem(isolate_stem)

                for stem_name in separated.sources:
                    stem_path = format_output_path(
                        output, selected_model_name, track, stem_name, format, now=now
                    )
                    separated.export_stem(
                        stem_name,
                        stem_path,
                        format=format,
                        clip=None if clip_mode == ClipMode.none else clip_mode.value,
                    )

            except Exception as e:
                had_error = True
                progress_tracker.error_file(file_key)
                err_console.print(
                    f"[red]✗[/red] Error processing {escape(track.name)}: "
                    f"{escape(str(e))}"
                )

    # Exit nonzero if any track or input path failed, after processing the rest.
    if had_error or had_path_errors:
        raise typer.Exit(1)
