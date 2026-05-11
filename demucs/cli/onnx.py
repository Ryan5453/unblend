# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typer
from typing_extensions import Annotated

from ..onnx import export_to_onnx
from .utils import console


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
            help="Output ONNX file path (defaults to {model}.onnx)",
        ),
    ] = None,
    opset: Annotated[
        int,
        typer.Option(
            help="ONNX opset version",
        ),
    ] = 17,
    fp16: Annotated[
        bool,
        typer.Option(
            "--fp16",
            help="Convert weights and IO to float16 after export.",
        ),
    ] = False,
) -> None:
    """
    Export a HTDemucs model to the ONNX format.

    This is an internal developer tool for creating ONNX models for deployment.
    """
    if output is not None:
        output_path = output
    else:
        suffix = "_fp16" if fp16 else "_fp32"
        output_path = f"{model}{suffix}.onnx"

    try:
        export_to_onnx(
            model_name=model,
            output_path=output_path,
            opset_version=opset,
            fp16=fp16,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error exporting model:[/red] {e}")
        raise typer.Exit(1)
