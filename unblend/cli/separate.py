# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Annotated

import click
import torch
import typer
from rich.markup import escape
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from ..api import Separator, default_device, select_model
from ..apply import ModelEnsemble
from ..exceptions import ModelLoadingError, ValidationError
from ..htdemucs import HTDemucs
from ..roformer import BSRoformer, MelBandRoformer
from .models import ensure_model_available
from .progress import FileProgressTracker
from .types import ClipMode, DeviceType, ModelName, Precision, StemName
from .utils import console, expand_paths_to_audio_files, format_output_path

# Cache-free auto-compile policy. Values are conservative break-even amounts
# of predicted eager GPU work, not raw chunk counts: a runtime batch-1 timing
# adapts the same architecture/dtype threshold to T4, V100, A100/Hopper, etc.
# RoFormer FP16 values include a margin over the worst measured break-even
# eager time (T4/V100/H200); unmeasured precision/family combinations are more
# conservative until their matrix is filled in.
_AUTO_COMPILE_EAGER_SECONDS: dict[tuple[str, str], int] = {
    ("htdemucs", "fp32"): 240,
    ("htdemucs", "fp16"): 200,
    ("htdemucs", "bf16"): 200,
    ("bs_roformer", "fp32"): 160,
    ("bs_roformer", "fp16"): 120,
    ("bs_roformer", "bf16"): 160,
    ("mel_band_roformer", "fp32"): 180,
    ("mel_band_roformer", "fp16"): 135,
    ("mel_band_roformer", "bf16"): 180,
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
        duration = AudioDecoder(str(path)).metadata.duration_seconds
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
    Ensemble work multiplies by its sub-model count because each member runs
    the complete chunk stream.

    :param separator: Eager separator providing segment/sample-rate metadata.
    :param audio_files: Expanded input file list.
    :param shifts: Shift rounds per input.
    :param split_overlap: Fractional chunk overlap.
    :return: ``(estimated_chunks, known_duration_seconds, unknown_file_count)``.
    """
    sample_rate = separator.model.samplerate
    segment_samples = int(round(separator.model.max_allowed_segment * sample_rate))
    stride = int((1 - split_overlap) * segment_samples)
    model_count = (
        len(separator.model.models) if isinstance(separator.model, ModelEnsemble) else 1
    )
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
        chunks_per_shift = math.ceil((samples + expected_shift_padding) / stride)
        total_chunks += shifts * chunks_per_shift
    return total_chunks * model_count, total_duration, unknown_files


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
        console.print(
            f"[yellow]![/yellow] CUDA compile setup failed; "
            f"continuing eager: {escape(str(error))}"
        )
        return False
    return True


def _validate_output_format(format: str) -> None:
    """
    Fail fast on an unsupported --format by encoding a tiny silent clip the
    same way ``export_stem`` will (torchcodec keys the container off the
    file extension), instead of erroring after an expensive separation.

    :param format: Output format/extension to validate
    :raises typer.Exit: If the format is not encodable
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            encoder = AudioEncoder(samples=torch.zeros(2, 4410), sample_rate=44100)
            encoder.to_file(Path(tmp) / f"probe.{format}")
    except Exception as error:
        console.print(
            f"[red]✗[/red] Unsupported output format '{escape(format)}': "
            f"{escape(str(error))}"
        )
        raise typer.Exit(1)


def separate_command(
    # Input/Output
    tracks: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Path to audio files or directories containing audio files",
            show_default=False,
        ),
    ] = None,
    # Model Selection
    model: Annotated[
        ModelName,
        typer.Option(
            "-m",
            "--model",
            help="Model to use for separation",
            rich_help_panel="Model Selection",
        ),
    ] = ModelName.auto,
    # Processing Options
    device: Annotated[
        DeviceType | None,
        typer.Option(
            "-d",
            "--device",
            help="Device to process separation on (defaults to the best available: cuda > mps > cpu)",
            show_default=False,
            rich_help_panel="Processing",
        ),
    ] = None,
    shifts: Annotated[
        int,
        typer.Option(
            min=1,
            max=20,
            help="Number of random shifts for equivariant stabilization, increases separation time but improves quality",
            rich_help_panel="Processing",
        ),
    ] = 1,
    split_overlap: Annotated[
        float,
        typer.Option(
            "--split-overlap",
            min=0.0,
            max=0.999999,
            help="Overlap between split chunks, higher values improve quality at chunk boundaries",
            rich_help_panel="Processing",
        ),
    ] = 0.25,
    seed: Annotated[
        int | None,
        typer.Option(
            help="Random seed for reproducible shift-based inference",
            rich_help_panel="Processing",
        ),
    ] = None,
    compile_model: Annotated[
        bool | None,
        typer.Option(
            "--compile/--no-compile",
            help="CUDA compilation policy. By default, estimate the complete workload and compile only past the architecture/dtype break-even point; --compile and --no-compile force either choice.",
            rich_help_panel="Processing",
        ),
    ] = None,
    precision: Annotated[
        Precision,
        typer.Option(
            "--precision",
            help="Inference precision; auto picks fp16 on CUDA (with tensor cores) and MPS, fp32 on CPU",
            rich_help_panel="Processing",
        ),
    ] = Precision.auto,
    # Output
    output: Annotated[
        str,
        typer.Option(
            "-o",
            "--output",
            help="Output path template. Variables: {model}, {track}, {stem}, {ext}, {date}, {time}, {timestamp}",
            rich_help_panel="Output",
        ),
    ] = "separated/{model}/{track}/{stem}.{ext}",
    isolate_stem: Annotated[
        StemName | None,
        typer.Option(
            help="Only creates a {stem} and no_{stem} stem/file",
            rich_help_panel="Output",
        ),
    ] = None,
    clip_mode: Annotated[
        ClipMode,
        typer.Option(
            help="Strategy for avoiding clipping",
            rich_help_panel="Output",
        ),
    ] = ClipMode.rescale,
    format: Annotated[
        str,
        typer.Option(
            "-f",
            "--format",
            help="Output audio format, anything supported by FFmpeg",
            rich_help_panel="Output",
        ),
    ] = "wav",
) -> None:
    """
    Separates the given tracks.

    :param tracks: Paths to audio files or directories containing audio files
    :param model: Model to use for separation
    :param device: Device to process separation on
    :param shifts: Number of random shifts for equivariant stabilization;
        increases separation time but improves quality
    :param split_overlap: Overlap between split chunks; higher values improve
        quality at chunk boundaries
    :param seed: Random seed for reproducible shift-based inference
    :param compile_model: CUDA compile override: ``True`` forces compile,
        ``False`` forces eager, and ``None`` automatically compares estimated
        eager GPU work with the architecture/dtype break-even threshold.
    :param precision: Inference precision; auto picks fp16 on CUDA (with tensor
        cores) and MPS, fp32 on CPU
    :param output: Output path template; variables are {model}, {track}, {stem},
        {ext}, {date}, {time}, {timestamp}
    :param isolate_stem: Only creates a {stem} and no_{stem} stem/file
    :param clip_mode: Strategy for avoiding clipping
    :param format: Output audio format, anything supported by FFmpeg
    """
    if tracks is None or not tracks:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    # Resolved at invocation rather than as the parameter default: a default-
    # argument ternary would probe CUDA/MPS at import time (even for --help).
    if device is None:
        device = DeviceType(default_device())

    audio_files, had_path_errors = expand_paths_to_audio_files(tracks)

    if not audio_files:
        console.print("[red]No audio files found to process.[/red]")
        raise typer.Exit(1)

    # A format is used as a file suffix, so anything Path() would normalize
    # away ("./wav", "wav/", "") can pass a probe built from the normalized
    # string yet fail (or silently differ) at write time — reject it up front.
    if "/" in format or "\\" in format or format != str(Path(format)):
        console.print(
            f"[red]✗[/red] Unsupported output format '{escape(format)}': "
            "not a valid file extension."
        )
        raise typer.Exit(1)

    # "-f .wav" is a natural spelling for "wav": the leading dot is extension
    # syntax, not part of the format name — drop it so writes don't produce
    # doubled-dot filenames ("drums..wav"). After the guard, so rejection
    # messages above stay as-typed.
    if format.startswith(".") and format.lstrip("."):
        format = format.lstrip(".")

    # Catch an unsupported output container before any model download or
    # separation work: encode a tiny silent clip the same way export_stem
    # will. A literal extension in the template (e.g. ``out/{stem}.flac``)
    # overrides --format since export keys the container off the path suffix,
    # so validate that extension when present rather than the unused --format.
    template_suffix = Path(output).suffix.lstrip(".")
    if template_suffix and "{" in template_suffix:
        # Resolve placeholders so the probe validates the container export
        # will actually key off: ``.{ext}`` → --format, while e.g.
        # ``.{timestamp}`` resolves to digits and fails here rather than
        # per-track after the model download.
        # str(), not .name: .name would silently truncate a format containing
        # a path separator (e.g. "x/wav" → "wav") and validate the wrong thing.
        template_suffix = str(
            format_output_path(
                template_suffix, "model", Path("track.wav"), "stem", format
            )
        )
    effective_format = template_suffix if template_suffix else format
    _validate_output_format(effective_format)

    if model.value == ModelName.auto.value:
        selected_model_name, only_load_stem = select_model(
            isolate_stem=isolate_stem.value if isolate_stem else None,
        )
        console.print(
            f"[cyan]Auto-selected model:[/cyan] [bold]{selected_model_name}[/bold]"
        )
    else:
        selected_model_name = model.value
        only_load_stem = isolate_stem.value if isolate_stem else None

    # only_load keeps the download to the single specialist layer when set.
    if not ensure_model_available(selected_model_name, only_load=only_load_stem):
        raise typer.Exit(1)

    # Resolve --precision auto → the API's default_dtype for the device:
    # FP16 on CUDA-with-tensor-cores and MPS, FP32 on CPU and older CUDA
    # GPUs. One source of truth with Separator's own ``dtype="auto"``.
    if precision is Precision.auto:
        dtype: torch.dtype | str | None = "auto"
    elif precision is Precision.fp32:
        dtype = None
    elif precision is Precision.fp16:
        dtype = torch.float16
    else:  # Precision.bf16
        dtype = torch.bfloat16

    # only_load (set from --isolate-stem for an explicit model, or by
    # select_model on the auto path) is validated against the model's sources
    # inside Separator.__init__. ModelLoadingError can also surface here even
    # after ensure_model_available: a corrupt cache file triggers a
    # re-download inside get_model, which fails offline. Catch both so they
    # print a clean message instead of an uncaught traceback.
    try:
        separator = Separator(
            model=selected_model_name,
            device=device.value,
            only_load=only_load_stem,
            dtype=dtype,
            compile=compile_model is True,
        )
    except (ValidationError, ModelLoadingError) as error:
        console.print(
            f"[red]✗[/red] [bold]{selected_model_name}[/bold]: error: "
            f"{escape(str(error))}"
        )
        raise typer.Exit(1)

    # Default CLI policy is workload-aware on CUDA. The Separator's mandatory
    # VRAM-sizing probe already recorded a steady eager batch-1 time, so this
    # adds only cheap container-metadata reads — no persistent calibration
    # cache and no GPU-name/PyTorch-version table.
    if compile_model is None and device is DeviceType.cuda:
        _maybe_enable_auto_compile(
            separator,
            audio_files,
            shifts=shifts,
            split_overlap=split_overlap,
        )

    # Also covers the case where --isolate-stem is set but only_load is not (the
    # auto path can pick a single model where only_load is a no-op): the stem
    # must still exist in the loaded model's sources.
    if isolate_stem is not None and isolate_stem.value not in separator.model.sources:
        console.print(
            f'[red]✗[/red] [bold]{selected_model_name}[/bold]: error: stem "{isolate_stem.value}" is not in selected model. STEM must be one of {", ".join(separator.model.sources)}.'
        )
        raise typer.Exit(1)

    # Detect output-path collisions up front, before doing any expensive
    # separation. Compute the actual resolved path for every (track, stem) pair
    # the run will write; if any two pairs map to the same path, the second
    # would silently overwrite the first. Exact-path collisions are a hard
    # error; paths that differ only by case or Unicode form can still alias
    # the same file on case-insensitive filesystems (macOS/Windows), which
    # isn't knowable up front, so those only warn below.
    if isolate_stem is not None:
        planned_stems = [isolate_stem.value, f"no_{isolate_stem.value}"]
    else:
        planned_stems = list(separator.model.sources)

    # Capture a single timestamp for the whole run. The collision pre-check, the
    # displayed template, and the actual writes below all reuse this so that
    # {date}/{time}/{timestamp} resolve identically — otherwise the check could
    # evaluate a different second than the writes and either miss a real
    # collision or report a path that isn't the one written.
    now = datetime.now()

    planned_paths: dict[str, list[str]] = {}
    planned_containers: set[str] = set()
    for track in audio_files:
        for stem_name in planned_stems:
            path = format_output_path(
                output, selected_model_name, track, stem_name, format, now=now
            )
            # Mirror export_stem: a suffix-less path gets ``.{format}`` appended
            # at write time, and an existing suffix (possibly leaked from a dot
            # in the track name) becomes the container. Collision-check and
            # container-validate the paths as they will actually be written.
            # ``with_suffix`` raises on an empty-name path ("", ".", "/");
            # such templates can never name a file, so refuse cleanly.
            if not path.name:
                console.print(
                    "[red]✗[/red] Output template resolves to an empty filename "
                    f"('{escape(str(path))}' for {escape(track.name)} → "
                    f"{stem_name}). Add a filename component such as "
                    "[bold]{track}[/bold] or [bold]{stem}[/bold]."
                )
                raise typer.Exit(1)
            if not path.suffix:
                try:
                    path = path.with_suffix(f".{format}")
                except ValueError:
                    # e.g. a format containing a path separator.
                    console.print(
                        f"[red]✗[/red] Unsupported output format "
                        f"'{escape(format)}': not a valid file extension."
                    )
                    raise typer.Exit(1)
            planned_containers.add(path.suffix.lstrip("."))
            planned_paths.setdefault(str(path), []).append(
                f"{track.name} → {stem_name}"
            )

    for container in sorted(planned_containers - {effective_format}):
        _validate_output_format(container)

    collisions = {p: srcs for p, srcs in planned_paths.items() if len(srcs) > 1}
    if collisions:
        console.print(
            "[red]✗[/red] Output template produces colliding paths; outputs would "
            "overwrite each other. Add [bold]{track}[/bold] and/or "
            "[bold]{stem}[/bold] to the template to make each path unique:"
        )
        for path, srcs in collisions.items():
            console.print(f"  [bold]{escape(path)}[/bold] ← {escape(', '.join(srcs))}")
        raise typer.Exit(1)

    folded_groups: dict[str, set[str]] = {}
    for planned in planned_paths:
        key = unicodedata.normalize("NFC", planned).casefold()
        folded_groups.setdefault(key, set()).add(planned)
    aliasing_groups = [
        sorted(aliases) for aliases in folded_groups.values() if len(aliases) > 1
    ]
    if aliasing_groups:
        console.print(
            "[yellow]![/yellow] Some output paths differ only by letter case or "
            "Unicode form; on filesystems that treat those as the same file "
            "(e.g. macOS/Windows) they may overwrite each other:"
        )
        for aliases in aliasing_groups:
            console.print(f"  {escape(', '.join(aliases))}")

    # Reuse format_output_path so the displayed template can never drift from the
    # paths actually written. Keep {stem} as a literal placeholder (and {track}
    # too when there are multiple tracks) by passing them through as the values:
    # format_output_path applies {track} as track.name minus its extension, so a
    # Path of "{track}" resolves back to "{track}".
    if len(audio_files) == 1:
        track = audio_files[0]
        message = "Separated track will be stored using template"
    else:
        track = Path("{track}")
        message = "Separated tracks will be stored using template"
    resolved_template = format_output_path(
        output, selected_model_name, track, "{stem}", format, now=now
    )
    console.print(f"{message} '{escape(str(resolved_template))}'")

    # A literal extension in the template overrides --format (export keys the
    # container off the path suffix). Warn when -f was passed explicitly but
    # the template ignores it.
    template_ext = resolved_template.suffix.lstrip(".")
    if template_ext and template_ext != format:
        ctx = click.get_current_context(silent=True)
        source = ctx.get_parameter_source("format") if ctx else None
        if source is not None and source.name == "COMMANDLINE":
            console.print(
                f"[yellow]Warning:[/yellow] output template extension "
                f"'.{escape(template_ext)}' overrides --format '{escape(format)}'"
            )

    had_error = False
    with FileProgressTracker(len(audio_files)) as progress_tracker:
        for track in audio_files:
            # Key tracker entries on the full path so two same-named files in
            # different directories don't collide; the tracker shows just the
            # basename for display.
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
                    stem_name = isolate_stem.value
                    separated = separated.isolate_stem(stem_name)

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
                console.print(
                    f"[red]✗[/red] Error processing {escape(track.name)}: "
                    f"{escape(str(e))}"
                )

    # If any track failed — or any input path didn't resolve to audio — exit
    # nonzero so scripts/pipelines can detect it, while still having processed
    # every other track.
    if had_error or had_path_errors:
        raise typer.Exit(1)
