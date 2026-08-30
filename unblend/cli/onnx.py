# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Annotated

import typer
from rich.markup import escape

from ..onnx import export_to_onnx
from .types import ExportPrecision
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
            help="Output ONNX file path (defaults to {model}_{precision}.onnx, "
            "with the resolved precision for --precision native)",
        ),
    ] = None,
    opset: Annotated[
        int,
        typer.Option(
            help="ONNX opset version (raised to 18 for RoFormer models)",
        ),
    ] = 17,
    precision: Annotated[
        ExportPrecision,
        typer.Option(
            "--precision",
            help="Weight storage precision, independent of compute precision. "
            "native resolves to the narrowest precision the checkpoint's own "
            "weights survive losslessly: fp16 for HTDemucs (upstream shipped "
            "it that way) and fp32 for RoFormer and SCNet. Narrower settings "
            "shrink the file and keep arithmetic in fp32, except fp16 on "
            "RoFormer, which uses mixed precision. See onnx.md.",
        ),
    ] = ExportPrecision.native,
    static_batch: Annotated[
        bool,
        typer.Option(
            "--static-batch",
            help="Trace with a fixed batch=1 instead of a dynamic batch axis; "
            "works around an onnxruntime-web WebGPU memory-planner bug. "
            "Use for browser deployment. Leave off for server-side/library "
            "consumers that want batched ONNX inference.",
        ),
    ] = False,
) -> None:
    """
    Export a model (HTDemucs, RoFormer, or SCNet) to the ONNX format.

    See onnx.md for the full export contract.

    :param model: Model name to export
    :param output: Output ONNX file path (defaults to
        {model}_{precision}.onnx)
    :param opset: ONNX opset version
    :param precision: Weight storage precision; native follows the checkpoint
    :param static_batch: Trace with a fixed batch=1 instead of a dynamic batch
        axis (see ``export_to_onnx`` for details)
    """
    try:
        written = export_to_onnx(
            model_name=model,
            output_path=output,
            opset_version=opset,
            precision=precision.value,
            static_batch=static_batch,
        )
        console.print(f"Exported [green]{escape(written)}[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error exporting model:[/red] {escape(str(e))}")
        raise typer.Exit(1)
