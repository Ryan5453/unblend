# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
from pathlib import Path

import click
import torch
import typer
from typing_extensions import Annotated

from ..api import Separator, select_model
from .models import ensure_model_available
from .progress import FileProgressTracker
from .types import ClipMode, DeviceType, ModelName, Precision, StemName
from .utils import console, expand_paths_to_audio_files, format_output_path


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
        DeviceType,
        typer.Option(
            "-d",
            "--device",
            help="Device to process separation on",
            rich_help_panel="Processing",
        ),
    ] = (
        DeviceType.cuda
        if torch.cuda.is_available()
        else DeviceType.mps
        if torch.backends.mps.is_available()
        else DeviceType.cpu
    ),
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
    compile: Annotated[
        bool,
        typer.Option(
            "--compile/--no-compile",
            help="Compile the HTDemucs neural network core on CUDA. Improves steady-state throughput for long-running jobs, but adds a heavy warmup cost.",
            rich_help_panel="Processing",
        ),
    ] = False,
    precision: Annotated[
        Precision,
        typer.Option(
            "--precision",
            help="Inference precision; auto picks bf16 on CUDA, fp16 on MPS, fp32 on CPU",
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
    """
    if tracks is None or not tracks:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()
        return

    audio_files = expand_paths_to_audio_files(tracks)

    if not audio_files:
        console.print("[red]No audio files found to process.[/red]")
        return

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

    if not ensure_model_available(selected_model_name):
        return

    # Resolve --precision auto → the best-known dtype for the device:
    # CUDA picks bf16 (native tensor cores at the same throughput as fp16,
    # with FP32 exponent range); MPS picks fp16 (custom Metal kernels in
    # ``demucs.metal``; BF16 on MPS works but is slower on PyTorch 2.11);
    # CPU stays at fp32 (no faster path).
    if precision is Precision.auto:
        if device.value == DeviceType.cuda.value:
            dtype: torch.dtype | None = torch.bfloat16
        elif device.value == DeviceType.mps.value:
            dtype = torch.float16
        else:
            dtype = None
    elif precision is Precision.fp32:
        dtype = None
    elif precision is Precision.fp16:
        dtype = torch.float16
    else:  # Precision.bf16
        dtype = torch.bfloat16

    separator = Separator(
        model=selected_model_name,
        device=device.value,
        only_load=only_load_stem,
        dtype=dtype,
        compile=compile,
    )

    if isolate_stem is not None and isolate_stem.value not in separator.model.sources:
        console.print(
            f'[red]✗[/red] [bold]{selected_model_name}[/bold]: error: stem "{isolate_stem.value}" is not in selected model. STEM must be one of {", ".join(separator.model.sources)}.'
        )
        return

    if audio_files:
        resolved_template = output
        resolved_template = resolved_template.replace("{model}", selected_model_name)
        resolved_template = resolved_template.replace("{ext}", format)

        now = datetime.now()
        resolved_template = resolved_template.replace(
            "{date}", now.strftime("%Y-%m-%d")
        )
        resolved_template = resolved_template.replace(
            "{time}", now.strftime("%H-%M-%S")
        )
        resolved_template = resolved_template.replace(
            "{timestamp}", str(int(now.timestamp()))
        )

        if len(audio_files) == 1:
            track_name = audio_files[0].name.rsplit(".", 1)[0]
            resolved_template = resolved_template.replace("{track}", track_name)
            console.print(
                f"Separated track will be stored using template '{resolved_template}'"
            )
        else:
            console.print(
                f"Separated tracks will be stored using template '{resolved_template}'"
            )

    with FileProgressTracker(len(audio_files)) as progress_tracker:
        for track in audio_files:
            filename = track.name

            progress_tracker.start_file(filename)

            try:
                audio_callback = progress_tracker.create_audio_callback(filename)

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
                        output, selected_model_name, track, stem_name, format
                    )
                    separated.export_stem(
                        stem_name,
                        stem_path,
                        format=format,
                        clip=None if clip_mode == ClipMode.none else clip_mode.value,
                    )

            except Exception as e:
                error_msg = str(e)
                progress_tracker.error_file(filename, error_msg)
                console.print(f"[red]✗[/red] Error processing {filename}: {error_msg}")
