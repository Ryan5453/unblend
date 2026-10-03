# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from ..exceptions import UnblendError
from ..onnx import _check_output_path, export_to_onnx, validate_export_request
from .types import ExportPrecision
from .utils import console, err_console


def export_onnx_command(
    model: Annotated[
        str,
        typer.Option(
            "-m",
            "--model",
            help="Model name to export",
        ),
    ] = "htdemucs",
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output",
            help="Output ONNX file path (defaults to {model}_{precision}.onnx, "
            "with the resolved precision for --precision native, and "
            "{model}_{precision}_static.onnx with --static-batch)",
        ),
    ] = None,
    opset: Annotated[
        int,
        typer.Option(
            help="ONNX opset version (raised to 18 for RoFormer and SCNet, 19 for fp8)",
        ),
    ] = 17,
    precision: Annotated[
        ExportPrecision,
        typer.Option(
            "--precision",
            help="Weight storage precision (compute stays fp32, except RoFormer at "
            "fp16, which runs mixed precision). native keeps the precision the "
            "checkpoint is stored at: fp16 for HTDemucs, fp32 for RoFormer and "
            "SCNet.",
        ),
    ] = ExportPrecision.native,
    static_batch: Annotated[
        bool,
        typer.Option(
            "--static-batch",
            help="Trace with a fixed batch=1 instead of a dynamic batch axis. "
            "RoFormer needs it in the browser (an onnxruntime-web WebGPU "
            "memory-planner bug); HTDemucs and SCNet run there with the "
            "dynamic axis. Leave off for batched ONNX inference.",
        ),
    ] = False,
) -> None:
    """
    Export a model (HTDemucs, RoFormer, or SCNet) to the ONNX format.

    See https://github.com/Ryan5453/unblend/blob/main/onnx.md for the full
    export contract.

    :param model: Model name to export
    :param output: Output ONNX file path (defaults to
        {model}_{precision}.onnx, ``_static`` with --static-batch)
    :param opset: ONNX opset version
    :param precision: Weight storage precision; native follows the checkpoint
    :param static_batch: Trace with a fixed batch=1 instead of a dynamic batch
        axis (see ``export_to_onnx`` for details)
    """
    from .models import ensure_model_available

    try:
        # Everything checkable from the registry, before a possibly large
        # download (which then gets a progress bar instead of silence).
        validate_export_request(model, opset, precision.value)
    except (ValueError, ImportError, UnblendError) as e:
        err_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    # The output must be writable before the download and the trace. The
    # default name ({model}_{precision}.onnx in the current folder) depends
    # on the checkpoint's precision, known only once it's downloaded, so
    # export_to_onnx checks that one itself, before tracing.
    if output is not None:
        if not output.strip():
            err_console.print("[red]Error:[/red] --output is empty")
            raise typer.Exit(1)
        if output.endswith(("/", os.sep)) or os.path.basename(output) in {".", ".."}:
            err_console.print(
                f"[red]Error:[/red] {escape(output)} names a folder; name the .onnx file"
            )
            raise typer.Exit(1)
        # os.path, not Path: Path.expanduser() raises for an unknown ~user.
        output = os.path.expanduser(output)
        if os.path.isdir(output):
            err_console.print(
                f"[red]Error:[/red] {escape(output)} is a folder; name the .onnx file"
            )
            raise typer.Exit(1)
        try:
            # The library's own check, so a path it would refuse after the
            # download is refused before it.
            _check_output_path(output)
        except UnblendError as error:
            err_console.print(f"[red]Error:[/red] {escape(str(error))}")
            raise typer.Exit(1) from error
    # realpath, not abspath: "link/../x.onnx" lands beside the link's target.
    folder = Path(os.path.realpath(Path(output).parent)) if output else Path.cwd()
    while not os.path.lexists(folder) and folder != folder.parent:
        folder = folder.parent
    if not folder.is_dir() or not os.access(folder, os.W_OK):
        err_console.print(
            f"[red]Error:[/red] can't write the export "
            f"({escape(str(folder))} isn't a writable folder)"
        )
        raise typer.Exit(1)
    if not ensure_model_available(model):
        raise typer.Exit(1)
    try:
        with console.status(
            f"Exporting {escape(model)} (tracing can take several minutes)…"
        ):
            written = export_to_onnx(
                model_name=model,
                output_path=output,
                opset_version=opset,
                precision=precision.value,
                static_batch=static_batch,
            )
        console.print(f"Exported [green]{escape(written)}[/green]")
    except (ValueError, UnblendError) as e:
        err_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        err_console.print(f"[red]Error exporting model:[/red] {escape(str(e))}")
        raise typer.Exit(1)
