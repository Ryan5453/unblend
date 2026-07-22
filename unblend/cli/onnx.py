# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Annotated

import typer
from rich.markup import escape

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
            help="Output ONNX file path (defaults to {model}_fp16.onnx or "
            "{model}_fp32.onnx depending on --fp16)",
        ),
    ] = None,
    opset: Annotated[
        int,
        typer.Option(
            help="ONNX opset version (raised to 18 for RoFormer models)",
        ),
    ] = 17,
    fp16: Annotated[
        bool,
        typer.Option(
            "--fp16",
            help="Store weights as float16 (weight-only; compute and IO stay fp32). "
            "Roughly halves file size; output is near-identical to fp32.",
        ),
    ] = False,
    static_batch: Annotated[
        bool,
        typer.Option(
            "--static-batch",
            help="RoFormer only. Trace with a fixed batch=1 instead of a dynamic "
            "batch axis; works around an onnxruntime-web WebGPU memory-planner bug. "
            "Use for browser deployment. Leave off for server-side/library "
            "consumers that want batched ONNX inference.",
        ),
    ] = False,
) -> None:
    """
    Export a model (HTDemucs or RoFormer) to the ONNX format.

    This is an internal developer tool for creating ONNX models for deployment.

    :param model: Model name to export
    :param output: Output ONNX file path (defaults to {model}_fp16.onnx or
        {model}_fp32.onnx depending on --fp16)
    :param opset: ONNX opset version
    :param fp16: Store weights as float16 (weight-only; compute and IO stay fp32)
    :param static_batch: RoFormer only. Trace with a fixed batch=1 instead of a
        dynamic batch axis (see ``export_to_onnx`` for details)
    """
    if output is not None:
        output_path = output
    else:
        suffix = "_fp16" if fp16 else "_fp32"
        static_suffix = "_static" if static_batch else ""
        output_path = f"{model}{suffix}{static_suffix}.onnx"

    try:
        export_to_onnx(
            model_name=model,
            output_path=output_path,
            opset_version=opset,
            fp16=fp16,
            static_batch=static_batch,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error exporting model:[/red] {escape(str(e))}")
        raise typer.Exit(1)
