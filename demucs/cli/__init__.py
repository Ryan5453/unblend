# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import typer

from .. import __version__
from .models import download_models_command, list_models_command, remove_models_command
from .onnx import export_onnx_command
from .separate import separate_command
from .utils import console


def version_command() -> None:
    """
    Show the installed version of Demucs.
    """
    typer.echo(f"Demucs version: {__version__}")


def build_app() -> typer.Typer:
    """
    Build the Typer application (factored out of ``main`` so tests can drive
    the CLI through ``typer.testing.CliRunner``).

    :return: The fully wired Typer app.
    """
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
    )

    models_app = typer.Typer(
        help="Download, list and manage models",
        no_args_is_help=True,
        rich_markup_mode="rich",
    )
    # Explicit ``help=`` strings keep the reST ``:param`` docstrings (required
    # by the repo's code-standards test) out of the rendered ``--help`` output;
    # each option already documents itself via its own ``help=``.
    models_app.command(name="list", help="List available and downloaded models.")(
        list_models_command
    )
    models_app.command(
        name="download", help="Download and cache models for offline use."
    )(download_models_command)
    models_app.command(name="remove", help="Remove downloaded models from the cache.")(
        remove_models_command
    )

    app.command(
        name="separate", help="Separate audio tracks into their component stems."
    )(separate_command)
    app.add_typer(models_app, name="models")
    app.command(name="version", help="Show the installed version of Demucs.")(
        version_command
    )

    app.command(
        name="export-onnx",
        hidden=True,
        help="Export a HTDemucs model to ONNX (internal developer tool).",
    )(export_onnx_command)

    return app


def main() -> None:
    """
    Entry point for the Demucs CLI.
    """
    try:
        build_app()()
    except KeyboardInterrupt:
        # Standard "killed by SIGINT" exit code (128 + 2). Keeps shell pipelines
        # and CI runners correctly distinguishing a Ctrl-C from a hard failure.
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
