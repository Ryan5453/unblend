# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Annotated

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..apply import Model, ModelEnsemble
from ..exceptions import ModelLoadingError
from ..repo import ModelRepository
from .progress import create_model_progress_bar, create_progress_callback
from .utils import console, format_file_size, get_models


def list_models_command() -> None:
    """
    List all available models and show which ones are downloaded.
    """
    model_repo = ModelRepository()
    models = get_models()

    cache_info = model_repo.get_cache_info()

    table = Table(title="Available Demucs Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Layers", style="blue")
    table.add_column("Segment", style="yellow")
    table.add_column("Size", style="magenta")
    table.add_column("Status", style="bright_green")

    for name in models.keys():
        info = models[name]

        layer_count = len(info.get("models", []))
        segment = info.get("segment", "N/A")

        is_downloaded = name in cache_info
        model_size = "N/A"
        if is_downloaded:
            model_size = format_file_size(cache_info[name]["size_bytes"])

        table.add_row(
            name,
            str(layer_count) + (" layer" if layer_count == 1 else " layers"),
            str(segment),
            model_size,
            "[green]Downloaded[/green]"
            if is_downloaded
            else "[red]Not Downloaded[/red]",
        )

    console.print(table)


def download_models_command(
    names: Annotated[
        list[str] | None,
        typer.Argument(help="Model names to download."),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option(
            "--all", help="Download all available models (may take some time)"
        ),
    ] = False,
) -> None:
    """
    Download and cache the specified models for offline use.

    :param names: Model names to download
    :param all_models: If True, download all available models
    """
    if not all_models and (names is None or not names):
        console.print("[red]Error:[/red] No models specified for download.")
        console.print("Please either:")
        console.print("  1. Specify one or more model names to download")
        console.print("  2. Use [bold]--all[/bold] to download all available models")
        console.print("\nTo see available models, run: [bold]demucs models list[/bold]")
        raise typer.Exit(1)

    if all_models:
        models = get_models()
        model_names = list(models.keys())
    else:
        model_names = names

    _download_models_batch(model_names)


def remove_models_command(
    names: Annotated[
        list[str] | None,
        typer.Argument(help="Model names to remove."),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Remove all downloaded models"),
    ] = False,
) -> None:
    """
    Remove models from the cache to free up space.

    :param names: Model names to remove
    :param all_models: If True, remove all downloaded models
    """
    model_repo = ModelRepository()

    if all_models:
        cache_info = model_repo.get_cache_info()
        model_names = list(cache_info.keys())
    else:
        if names is None or not names:
            console.print(
                "[yellow]No models specified for removal. Please specify at least one model name.[/yellow]"
            )
            raise typer.Exit(1)
        else:
            model_names = names

    if not model_names:
        # Reached via --all with an empty cache: nothing to do, not an error.
        console.print("[yellow]No models found to remove.[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress_bar:
        task = progress_bar.add_task(
            "[yellow]Removing models...", total=len(model_names)
        )

        for name in model_names:
            progress_bar.update(task, description=f"[cyan]Removing {name}...[/cyan]")

            success = model_repo.remove_model(name)
            if success:
                console.print(f"[green]✓[/green] Removed model [bold]{name}[/bold]")
            else:
                console.print(
                    f"[yellow]![/yellow] Model [bold]{name}[/bold] not found in cache"
                )

            progress_bar.update(task, advance=1)

    console.print("[bold green]Model removal complete![/bold green]")


def _format_download_summary(
    name: str,
    model: Model | ModelEnsemble,
    models: dict,
    cache_info: dict,
    download_time: float,
) -> str:
    """
    Build the success summary line shown after a model finishes downloading.

    :param name: Model name
    :param model: Loaded model instance
    :param models: Dictionary of available model metadata
    :param cache_info: Cache info mapping from ``ModelRepository.get_cache_info``
    :param download_time: Elapsed download time in seconds
    :return: Rich-markup summary string
    """
    num_sources = len(model.sources)
    if name in models:
        layer_count = len(models[name]["models"])
        layer_word = "layer" if layer_count == 1 else "layers"
        model_type = f"{layer_count} {layer_word}"
    else:
        model_type = "Model"

    size_str = ""
    speed_str = ""
    if name in cache_info:
        size_bytes = cache_info[name]["size_bytes"]
        size_str = f" ({format_file_size(size_bytes)})"

        if download_time > 0.1:
            speed = size_bytes / download_time
            speed_str = f" at {format_file_size(speed)}/s"

    return (
        f"[green]✓[/green] [bold]{name}[/bold]: {model_type} with {num_sources} sources{size_str}{speed_str}"
    )


def _download_model_with_progress(name: str) -> bool:
    """
    Download a single model with progress display.

    :param name: Model name to download
    :return: True if successful, False otherwise
    """

    models = get_models()

    if name in models:
        layer_count = len(models[name]["models"])
        layer_word = "layer" if layer_count == 1 else "layers"
        console.print(
            f"[bold]Downloading {name} ({layer_count} {layer_word})...[/bold]"
        )

    with create_model_progress_bar() as progress_bar:
        try:
            start_time = time.time()

            if name in models:
                layer_count = len(models[name]["models"])
                layer_word = "layer" if layer_count == 1 else "layers"
                task = progress_bar.add_task(
                    f"[cyan]Downloading {name} ({layer_count} {layer_word})[/cyan]",
                    total=100,
                    completed=0,
                )
            else:
                task = progress_bar.add_task(
                    f"[cyan]Downloading {name}[/cyan]",
                    total=100,
                    completed=0,
                )

            callback = create_progress_callback(progress_bar, task)
            model_repo = ModelRepository()
            model = model_repo.get_model(name=name, progress_callback=callback)
            model.eval()

            progress_bar.remove_task(task)

            download_time = time.time() - start_time
            cache_info = model_repo.get_cache_info()

            console.print(
                _format_download_summary(name, model, models, cache_info, download_time)
            )
            return True

        except Exception as error:
            console.print(f"[red]✗[/red] [bold]{name}[/bold]: {error}")
            return False


def ensure_model_available(name: str) -> bool:
    """
    Ensure a model is available, downloading if necessary.

    :param name: Model name to check/download
    :return: True if model is available, False otherwise
    """
    model_repo = ModelRepository()
    cache_info = model_repo.get_cache_info()

    if name in cache_info:
        return True

    return _download_model_with_progress(name)


def _download_models_batch(model_names: list[str]) -> None:
    """
    Download multiple models, showing progress for each.

    :param model_names: List of model names to download
    """
    model_repo = ModelRepository()
    cache_info = model_repo.get_cache_info()

    models = get_models()

    to_download = []
    for name in model_names:
        if name in cache_info:
            if name in models:
                layer_count = len(models[name]["models"])
                layer_word = "layer" if layer_count == 1 else "layers"
                size_str = ""
                if name in cache_info:
                    size_bytes = cache_info[name]["size_bytes"]
                    size_str = f" ({format_file_size(size_bytes)})"
                console.print(
                    f"[green]✓[/green] [bold]{name}[/bold]: Already downloaded ({layer_count} {layer_word}{size_str})"
                )
        else:
            to_download.append(name)

    if not to_download:
        console.print("[green]All specified models are already downloaded.[/green]")
        return

    if len(to_download) > 1:
        total_layers = sum(
            len(models[name]["models"]) for name in to_download if name in models
        )
        console.print(
            f"[bold]Downloading {len(to_download)} models ({total_layers} total layers)...[/bold]"
        )

    with create_model_progress_bar() as progress_bar:
        for name in to_download:
            _download_single_model_in_batch(name, models, progress_bar)

    console.print("[bold green]Download complete![/bold green]")


def _download_single_model_in_batch(name: str, models: dict, progress_bar: Progress) -> None:
    """
    Download a single model within an existing progress bar context.

    :param name: Model name to download
    :param models: Dictionary of available model metadata
    :param progress_bar: Rich progress bar to update
    """

    try:
        start_time = time.time()

        if name in models:
            layer_count = len(models[name]["models"])
            layer_word = "layer" if layer_count == 1 else "layers"
            task = progress_bar.add_task(
                f"[cyan]Downloading {name} ({layer_count} {layer_word})[/cyan]",
                total=100,
                completed=0,
            )
        else:
            task = progress_bar.add_task(
                f"[cyan]Downloading {name}[/cyan]",
                total=100,
                completed=0,
            )

        callback = create_progress_callback(progress_bar, task)
        model_repo = ModelRepository()
        model = model_repo.get_model(name=name, progress_callback=callback)
        model.eval()

        progress_bar.remove_task(task)

        download_time = time.time() - start_time
        cache_info = model_repo.get_cache_info()

        console.print(
            _format_download_summary(name, model, models, cache_info, download_time)
        )

    except ModelLoadingError as error:
        console.print(f"[red]✗[/red] [bold]{name}[/bold]: {error}")
    except Exception as e:
        console.print(f"[red]✗[/red] [bold]{name}[/bold]: Unexpected error: {str(e)}")
