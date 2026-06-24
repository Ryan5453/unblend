# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import click
import torch
import typer
from torchcodec.encoders import AudioEncoder

from ..api import Separator, default_device, select_model
from ..exceptions import ValidationError
from .models import ensure_model_available
from .progress import FileProgressTracker
from .types import ClipMode, DeviceType, ModelName, Precision, StemName
from .utils import console, expand_paths_to_audio_files, format_output_path


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
        console.print(f"[red]✗[/red] Unsupported output format '{format}': {error}")
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
        bool,
        typer.Option(
            "--compile/--no-compile",
            help="Compile the HTDemucs neural network core on CUDA. Improves steady-state throughput for long-running jobs, but adds a heavy warmup cost. Ignored on non-CUDA devices.",
            rich_help_panel="Processing",
        ),
    ] = False,
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
    :param compile_model: Compile the HTDemucs neural network core on CUDA; improves
        steady-state throughput for long-running jobs, but adds a heavy warmup cost
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

    # Catch an unsupported --format before any model download or separation
    # work: encode a tiny silent clip the same way export_stem will.
    _validate_output_format(format)

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
    # inside Separator.__init__. Catch that here so an invalid stem prints a
    # clean message instead of surfacing an uncaught ValidationError traceback.
    try:
        separator = Separator(
            model=selected_model_name,
            device=device.value,
            only_load=only_load_stem,
            dtype=dtype,
            compile=compile_model,
        )
    except ValidationError as error:
        console.print(
            f"[red]✗[/red] [bold]{selected_model_name}[/bold]: error: {error}"
        )
        raise typer.Exit(1)

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
    # would silently overwrite the first. This catches every cause — same
    # basename in different directories, a missing {track} across multiple
    # files, or a missing {stem} collapsing all stems onto one file.
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
    for track in audio_files:
        for stem_name in planned_stems:
            path = str(
                format_output_path(
                    output, selected_model_name, track, stem_name, format, now=now
                )
            )
            planned_paths.setdefault(path, []).append(f"{track.name} → {stem_name}")

    collisions = {p: srcs for p, srcs in planned_paths.items() if len(srcs) > 1}
    if collisions:
        console.print(
            "[red]✗[/red] Output template produces colliding paths; outputs would "
            "overwrite each other. Add [bold]{track}[/bold] and/or "
            "[bold]{stem}[/bold] to the template to make each path unique:"
        )
        for path, srcs in collisions.items():
            console.print(f"  [bold]{path}[/bold] ← {', '.join(srcs)}")
        raise typer.Exit(1)

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
    console.print(f"{message} '{resolved_template}'")

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
                f"'.{template_ext}' overrides --format '{format}'"
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
                console.print(f"[red]✗[/red] Error processing {track.name}: {e}")

    # If any track failed — or any input path didn't resolve to audio — exit
    # nonzero so scripts/pipelines can detect it, while still having processed
    # every other track.
    if had_error or had_path_errors:
        raise typer.Exit(1)
