# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import Annotated, TextIO

import typer
from rich.markup import escape

from .. import __version__
from ..exceptions import ModelLoadingError
from .models import (
    download_models_command,
    import_model_command,
    list_models_command,
    model_info_command,
    remove_models_command,
    unregister_model_command,
)
from .onnx import export_onnx_command
from .separate import separate_command
from .tune import tune_command
from .utils import err_console

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def version_command() -> None:
    """
    Show the installed version of unblend.
    """
    typer.echo(f"unblend version: {__version__}")


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

    def _print_version(value: bool) -> None:
        if value:
            version_command()
            raise typer.Exit()

    @app.callback(help="Music source separation.")
    def _root(
        version: Annotated[
            bool,
            typer.Option(
                "--version",
                help="Show the version and exit.",
                is_eager=True,
                callback=_print_version,
            ),
        ] = False,
    ) -> None:
        """
        Root options shared by every command.

        :param version: Handled eagerly by ``_print_version``.
        """

    models_app = typer.Typer(
        help="Download, list and manage models",
        no_args_is_help=True,
        rich_markup_mode="rich",
    )
    # Explicit ``help=`` strings keep the reST ``:param`` fields of the
    # command docstrings out of the rendered ``--help`` output.
    models_app.command(name="list", help="List available and downloaded models.")(
        list_models_command
    )
    models_app.command(
        name="info", help="Show a model's stems, license terms and provenance."
    )(model_info_command)
    models_app.command(
        name="download", help="Download and cache models for offline use."
    )(download_models_command)
    models_app.command(
        name="import",
        help="Import a checkpoint from elsewhere and register it.",
    )(import_model_command)
    models_app.command(name="remove", help="Remove downloaded models from the cache.")(
        remove_models_command
    )
    models_app.command(
        name="unregister", help="Remove an imported or custom model's entry."
    )(unregister_model_command)

    app.command(
        name="separate", help="Separate audio tracks into their component stems."
    )(separate_command)
    app.command(
        name="tune",
        help="Measure and print the fastest batch-size / compile settings for this machine.",
    )(tune_command)
    app.add_typer(models_app, name="models")
    app.command(name="version", help="Show the installed version of unblend.")(
        version_command
    )

    app.command(
        name="export-onnx",
        help="Export a model (HTDemucs, RoFormer, or SCNet) to the ONNX format. "
        "Details: https://github.com/Ryan5453/unblend/blob/main/onnx.md",
    )(export_onnx_command)

    return app


def main() -> None:
    """
    Entry point for the unblend CLI.
    """
    # Library warnings (a skipped models file, a kernel fallback) are messages
    # for the user, not Python tracebacks.
    fallback_show = warnings.showwarning
    shown: set[str] = set()

    def show(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        if os.path.abspath(filename).startswith(_PACKAGE_DIR):
            text = str(message)
            if text not in shown:
                shown.add(text)
                err_console.print(f"[yellow]![/yellow] {escape(text)}")
        else:
            fallback_show(message, category, filename, lineno, file, line)

    warnings.showwarning = show
    # Typer already turns Ctrl-C into exit code 130. A broken models file (or
    # any other registry error) is a one-line message, not a traceback.
    try:
        build_app()()
    except ModelLoadingError as error:
        err_console.print(f"[red]✗[/red] {escape(str(error))}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
