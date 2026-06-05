# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typer

from .. import __version__
from .models import download_models_command, list_models_command, remove_models_command
from .onnx import export_onnx_command
from .separate import separate_command


def version_command() -> None:
    """
    Show the installed version of Demucs.
    """
    typer.echo(f"Demucs version: {__version__}")


def main() -> None:
    """
    Entry point for the Demucs CLI.
    """
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
    )

    models_app = typer.Typer(
        help="Download, list and manage models", no_args_is_help=True,
        rich_markup_mode="rich",
    )
    models_app.command(name="list")(list_models_command)
    models_app.command(name="download")(download_models_command)
    models_app.command(name="remove")(remove_models_command)

    app.command(name="separate")(separate_command)
    app.add_typer(models_app, name="models")
    app.command(name="version")(version_command)

    app.command(name="export-onnx", hidden=True)(export_onnx_command)

    app()


if __name__ == "__main__":
    main()
