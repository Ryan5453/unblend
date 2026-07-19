# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Annotated

import typer
from rich.markup import escape
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
from ..repo import ModelRepository, get_cache_dir
from .progress import create_model_progress_bar, create_progress_callback
from .utils import console, format_file_size, get_models


def _model_layer_count(info: dict) -> int:
    """
    Return the number of independently downloaded files for a model.

    Demucs ensembles describe their files in ``models``; a RoFormer has one
    checkpoint instead. Keeping that registry difference here prevents CLI
    progress and summary code from assuming a Demucs-only metadata shape.

    :param info: One model's registry metadata.
    :return: Number of checkpoint files used by the model.
    """
    if info.get("backend") == "roformer":
        return 1
    return len(info["models"])


def list_models_command() -> None:
    """
    List all available models and show which ones are downloaded.
    """
    model_repo = ModelRepository()
    models = get_models()

    cache_info = model_repo.get_cache_info()

    table = Table(title="Available Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Layers", style="blue")
    table.add_column("Stems", style="yellow")
    table.add_column("License", style="dim")
    table.add_column("Size", style="magenta")
    table.add_column("Status", style="bright_green")

    for name in models.keys():
        info = models[name]

        layer_count = _model_layer_count(info)
        stems = ", ".join(info.get("sources", [])) or "N/A"
        license_label = info.get("license", "unknown")

        entry = cache_info.get(name)
        if entry is None:
            model_size = "N/A"
            status = "[red]Not Downloaded[/red]"
        elif entry["complete"]:
            model_size = format_file_size(entry["size_bytes"])
            status = "[green]Downloaded[/green]"
        else:
            # Partially cached (interrupted download or an --isolate-stem
            # specialist) — surface it instead of "Not Downloaded" with
            # invisible disk usage.
            model_size = format_file_size(entry["size_bytes"])
            status = (
                f"[yellow]Partial ({len(entry['layers'])}/"
                f"{entry['total_layers']} layers)[/yellow]"
            )

        table.add_row(
            name,
            str(layer_count) + (" layer" if layer_count == 1 else " layers"),
            stems,
            license_label,
            model_size,
            status,
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
    if all_models and names:
        console.print(
            "[red]Error:[/red] [bold]--all[/bold] and explicit model names are mutually exclusive."
        )
        raise typer.Exit(1)

    if not all_models and (names is None or not names):
        console.print("[red]Error:[/red] No models specified for download.")
        console.print("Please either:")
        console.print("  1. Specify one or more model names to download")
        console.print("  2. Use [bold]--all[/bold] to download all available models")
        console.print("\nTo see available models, run: [bold]unblend models list[/bold]")
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
    if all_models and names:
        console.print(
            "[red]Error:[/red] [bold]--all[/bold] and explicit model names are mutually exclusive."
        )
        raise typer.Exit(1)

    model_repo = ModelRepository()

    swept = 0
    if all_models:
        # get_cache_info includes partially-cached models, so interrupted
        # downloads are removed too; the sweep clears staging files left by
        # hard-killed downloads.
        model_names = list(model_repo.get_cache_info().keys())
        swept = model_repo.sweep_stale_downloads()
        if swept:
            console.print(
                f"[green]✓[/green] Removed {swept} leftover download temp "
                f"file{'s' if swept != 1 else ''}"
            )
    else:
        if names is None or not names:
            console.print(
                "[yellow]No models specified for removal. Please specify at least one model name.[/yellow]"
            )
            raise typer.Exit(1)
        else:
            model_names = names

    # Unknown model names are caller mistakes (typos), distinct from a known
    # model that just isn't cached — report them and exit nonzero.
    known_models = model_repo.list_models()
    unknown = [name for name in model_names if name not in known_models]
    for name in unknown:
        console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: Unknown model. "
            f"Available models: {', '.join(known_models)}"
        )
    model_names = [name for name in model_names if name in known_models]

    if not model_names:
        if unknown:
            raise typer.Exit(1)
        if swept:
            # Temp files were the only thing to clean up; already reported.
            return
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

        failed_removals = []
        for name in model_names:
            progress_bar.update(
                task, description=f"[cyan]Removing {escape(name)}...[/cyan]"
            )

            try:
                success = model_repo.remove_model(name)
            except ModelLoadingError as error:
                console.print(
                    f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}"
                )
                failed_removals.append(name)
                progress_bar.update(task, advance=1)
                continue
            if success:
                console.print(
                    f"[green]✓[/green] Removed model [bold]{escape(name)}[/bold]"
                )
            else:
                console.print(
                    f"[yellow]![/yellow] Model [bold]{escape(name)}[/bold] not found in cache"
                )

            progress_bar.update(task, advance=1)

    if unknown or failed_removals:
        raise typer.Exit(1)
    console.print("[bold green]Model removal complete![/bold green]")


def _format_download_summary(
    name: str,
    model: Model | ModelEnsemble,
    models: dict,
    cache_info: dict,
    download_time: float,
    layer_count: int | None = None,
) -> str:
    """
    Build the success summary line shown after a model finishes downloading.

    :param name: Model name
    :param model: Loaded model instance
    :param models: Dictionary of available model metadata
    :param cache_info: Cache info mapping from ``ModelRepository.get_cache_info``
    :param download_time: Elapsed download time in seconds
    :param layer_count: Layers actually downloaded; defaults to the model's
        full layer count from metadata (differs under ``only_load``)
    :return: Rich-markup summary string
    """
    num_sources = len(model.sources)
    if layer_count is None and name in models:
        layer_count = _model_layer_count(models[name])
    if layer_count is not None:
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

    return f"[green]✓[/green] [bold]{escape(name)}[/bold]: {model_type} with {num_sources} sources{size_str}{speed_str}"


def _download_model_with_progress(name: str, only_load: str | None = None) -> bool:
    """
    Download a single model with progress display.

    :param name: Model name to download
    :param only_load: Optional stem — restricts an ensemble download to the
        single specialist layer
    :return: True if successful, False otherwise
    """

    models = get_models()
    model_repo = ModelRepository()

    try:
        info = models.get(name)
        if info is not None and info.get("backend") == "roformer":
            layer_count = _model_layer_count(info)
        else:
            layer_count = len(model_repo.required_layers(name, only_load=only_load))
    except ModelLoadingError as error:
        console.print(f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}")
        return False

    layer_word = "layer" if layer_count == 1 else "layers"
    console.print(
        f"[bold]Downloading {escape(name)} ({layer_count} {layer_word})...[/bold]"
    )

    with create_model_progress_bar() as progress_bar:
        task = progress_bar.add_task(
            f"[cyan]Downloading {escape(name)} ({layer_count} {layer_word})[/cyan]",
            total=100,
            completed=0,
        )
        try:
            start_time = time.time()

            callback = create_progress_callback(progress_bar, task)
            model = model_repo.get_model(
                name=name, only_load=only_load, progress_callback=callback
            )
            model.eval()

            progress_bar.remove_task(task)

            download_time = time.time() - start_time
            cache_info = model_repo.get_cache_info()

            console.print(
                _format_download_summary(
                    name, model, models, cache_info, download_time, layer_count
                )
            )
            return True

        except Exception as error:
            progress_bar.remove_task(task)
            console.print(
                f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}"
            )
            return False


def ensure_model_available(name: str, only_load: str | None = None) -> bool:
    """
    Ensure a model is available, downloading if necessary.

    :param name: Model name to check/download
    :param only_load: Optional stem — when set, only the specialist layer for
        this stem needs to be (and will be) downloaded
    :return: True if model is available, False otherwise
    """
    model_repo = ModelRepository()
    models = model_repo.list_models()
    info = models.get(name)
    if info is None:
        console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: Unknown model. "
            f"Available models: {', '.join(models)}"
        )
        return False

    if info.get("backend") == "roformer":
        cached = model_repo.get_cache_info().get(name, {}).get("complete", False)
        if cached:
            return True
        return _download_model_with_progress(name, only_load=only_load)

    try:
        required = model_repo.required_layers(name, only_load=only_load)
    except ModelLoadingError as error:
        console.print(f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}")
        return False

    cache_dir = get_cache_dir()
    if all((cache_dir / f"{checksum}.th").exists() for checksum in required):
        # Existence-only — the download path sha256-verifies before moving into
        # cache, and ``get_model`` re-verifies on load. Re-hashing the whole
        # cache here would add ~1–3 s to every ``unblend separate`` invocation
        # on htdemucs_ft for the rare case of a user-corrupted cache file,
        # which the load-path catch already recovers from.
        return True

    return _download_model_with_progress(name, only_load=only_load)


def _download_models_batch(model_names: list[str]) -> None:
    """
    Download multiple models, showing progress for each. Exits nonzero if any
    name is unknown or any download fails, so scripts can detect it.

    :param model_names: List of model names to download
    :raises typer.Exit: If any model is unknown or fails to download
    """
    model_repo = ModelRepository()
    cache_info = model_repo.get_cache_info()

    models = get_models()

    # Unknown names are reported (and fail the command) without spinning up a
    # progress bar for them.
    unknown = [name for name in model_names if name not in models]
    for name in unknown:
        console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: Unknown model. "
            f"Available models: {', '.join(models)}"
        )

    to_download = []
    for name in model_names:
        if name in unknown:
            continue
        # Partially-cached models (interrupted downloads) still need the
        # download pass; get_model fetches only the missing layers.
        if cache_info.get(name, {}).get("complete"):
            layer_count = _model_layer_count(models[name])
            layer_word = "layer" if layer_count == 1 else "layers"
            size_bytes = cache_info[name]["size_bytes"]
            size_str = f" ({format_file_size(size_bytes)})"
            console.print(
                f"[green]✓[/green] [bold]{escape(name)}[/bold]: Already downloaded ({layer_count} {layer_word}{size_str})"
            )
        else:
            to_download.append(name)

    if not to_download:
        if unknown:
            raise typer.Exit(1)
        console.print("[green]All specified models are already downloaded.[/green]")
        return

    if len(to_download) > 1:
        total_layers = sum(_model_layer_count(models[name]) for name in to_download)
        console.print(
            f"[bold]Downloading {len(to_download)} models ({total_layers} total layers)...[/bold]"
        )

    failed = []
    with create_model_progress_bar() as progress_bar:
        for name in to_download:
            if not _download_single_model_in_batch(name, models, progress_bar):
                failed.append(name)

    if failed:
        console.print(
            f"[red]✗[/red] Failed to download: [bold]{', '.join(failed)}[/bold]"
        )
    if failed or unknown:
        raise typer.Exit(1)
    console.print("[bold green]Download complete![/bold green]")


def _download_single_model_in_batch(
    name: str, models: dict, progress_bar: Progress
) -> bool:
    """
    Download a single model within an existing progress bar context.

    :param name: Model name to download
    :param models: Dictionary of available model metadata
    :param progress_bar: Rich progress bar to update
    :return: True if successful, False otherwise
    """

    layer_count = _model_layer_count(models[name])
    layer_word = "layer" if layer_count == 1 else "layers"
    task = progress_bar.add_task(
        f"[cyan]Downloading {escape(name)} ({layer_count} {layer_word})[/cyan]",
        total=100,
        completed=0,
    )

    try:
        start_time = time.time()

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

    except ModelLoadingError as error:
        progress_bar.remove_task(task)
        console.print(f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}")
        return False
    except Exception as e:
        progress_bar.remove_task(task)
        console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: Unexpected error: "
            f"{escape(str(e))}"
        )
        return False
