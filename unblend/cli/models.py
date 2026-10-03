# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import errno
import os
import time
from pathlib import Path
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

from .._paths import name_encodable, name_fits
from ..apply import Model, ModelEnsemble
from ..exceptions import ModelLoadingError
from ..repo import ModelRepository, entry_weight_paths, get_cache_dir, stem_key
from .progress import create_model_progress_bar, create_progress_callback
from .utils import (
    complete_model_name,
    console,
    err_console,
    format_file_size,
    get_models,
)


def _model_file_count(name: str, repo: ModelRepository | None = None) -> int:
    """
    Return the number of distinct weight files a model is built from.

    :param name: Model name.
    :param repo: Registry to consult; a fresh one when omitted.
    :return: Number of checkpoint files used by the model, a shared one
        counted once.
    """
    return (repo or ModelRepository()).loaded_file_count(name)


def _local_bytes(name: str, repo: ModelRepository | None = None) -> int:
    """
    Return the size of a model's local (user-owned) weight files.

    :param name: Model name.
    :param repo: Registry to consult; a fresh one when omitted.
    :return: Bytes on disk, missing files counting as zero.
    """
    paths = (repo or ModelRepository()).local_artifacts(name)
    return sum(path.stat().st_size for path in paths if os.path.isfile(path))


def list_models_command() -> None:
    """
    List all available models and show which ones are downloaded.
    """
    model_repo = ModelRepository()
    models = get_models()

    cache_info = model_repo.get_cache_info()

    table = Table(title="Available Models", caption="Details: unblend models info NAME")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Stems", style="yellow")
    table.add_column("License", style="dim")
    table.add_column("Size", style="magenta", no_wrap=True)
    table.add_column("Status", style="bright_green", no_wrap=True)

    for name in models.keys():
        info = models[name]

        stems = ", ".join(info.get("sources", [])) or "N/A"
        license_label = info.get("license") or "unknown"

        # Locally-supplied weights are never "downloaded" — they are simply
        # present (or missing), so report them from disk rather than the cache.
        if model_repo.is_fully_local(name):
            paths = model_repo.local_artifacts(name)
            if all(os.path.isfile(path) for path in paths):
                model_size = format_file_size(
                    sum(path.stat().st_size for path in paths)
                )
                status = "[green]Local[/green]"
            else:
                model_size = "N/A"
                status = "[red]Missing[/red]"
            table.add_row(
                escape(name),
                escape(stems),
                escape(license_label),
                model_size,
                status,
            )
            continue

        # A mixed ensemble's local members count too.
        local_paths = model_repo.local_artifacts(name)
        missing_local = [path for path in local_paths if not os.path.isfile(path)]
        local_bytes = sum(
            path.stat().st_size for path in local_paths if os.path.isfile(path)
        )
        entry = cache_info.get(name)
        if missing_local:
            model_size = "N/A"
            status = "[red]Local file missing[/red]"
        elif entry is None:
            declared = sum(
                int(spec.get("size_bytes") or 0)
                for spec in model_repo.weight_files(name)
            )
            model_size = (
                f"[dim]{format_file_size(declared)}[/dim]" if declared else "N/A"
            )
            status = "[red]Not Downloaded[/red]"
        elif entry["complete"]:
            model_size = format_file_size(entry["size_bytes"] + local_bytes)
            status = "[green]Downloaded[/green]"
        else:
            # Partially cached (interrupted download or an --isolate-stem
            # specialist) — surface it instead of "Not Downloaded" with
            # invisible disk usage.
            model_size = format_file_size(entry["size_bytes"] + local_bytes)
            status = (
                f"[yellow]Partial ({len(entry['files'])}/"
                f"{entry['total_files']} files)[/yellow]"
            )

        table.add_row(
            escape(name),
            escape(stems),
            escape(license_label),
            model_size,
            status,
        )

    console.print(table)


def model_info_command(
    name: Annotated[
        str,
        typer.Argument(help="Model to describe", autocompletion=complete_model_name),
    ],
) -> None:
    """
    Show a model's stems, license terms and provenance.

    :param name: Model name
    :raises typer.Exit: If the model is unknown
    """
    repo = ModelRepository()
    info = repo.list_models().get(name)
    if info is None:
        err_console.print(f"[red]✗[/red] Unknown model: {escape(name)}")
        raise typer.Exit(1)

    # Resolved members know their architecture even when only the file header
    # states it.
    architectures = sorted(
        {
            m["architecture"]
            for m in repo._members.get(name, ())
            if m.get("architecture")
        }
    )
    referenced = [
        member["model"]
        for member in info.get("members") or []
        if isinstance(member, dict) and "model" in member
    ]
    console.print(f"[bold cyan]{escape(name)}[/bold cyan]")
    rows = [
        ("Architecture", ", ".join(architectures) or info.get("backend")),
        ("Members", ", ".join(referenced)),
        ("Stems", ", ".join(info.get("sources", []))),
        ("Files", str(_model_file_count(name, repo))),
        ("License", info.get("license") or "unknown"),
        ("License notes", info.get("license_note")),
        ("Provenance", info.get("provenance")),
    ]
    for label, value in rows:
        if value:
            console.print(f"[bold]{label}:[/bold] {escape(str(value).strip())}")


def unregister_model_command(
    name: Annotated[
        str,
        typer.Argument(help="Imported model to unregister", show_default=False),
    ],
    delete_weights: Annotated[
        bool,
        typer.Option(
            "--delete-weights",
            help="Also delete the model's local weights file(s); files another "
            "model uses, or in the download cache, are kept",
        ),
    ] = False,
) -> None:
    """
    Remove a model you added from the models file that defines it.

    :param name: Model name
    :param delete_weights: Also delete its local weights
    :raises typer.Exit: If no loaded models file defines the model
    """
    from ..exceptions import UnblendError
    from ..importer import _load_mapping, unregister_entry
    from ..repo import default_extra_models_files

    owner = None
    unreadable: list[tuple[Path, Exception]] = []
    for path in default_extra_models_files():
        if _certainly_missing(path):
            # A listed file not created yet, which the registry skips too.
            continue
        try:
            payload = _load_mapping(path)
        except Exception as error:
            unreadable.append((path, error))
            continue
        if isinstance(payload, dict) and name in (payload.get("models") or {}):
            owner = path
            break
    if owner is None and unreadable:
        path, error = unreadable[0]
        err_console.print(
            f"[red]✗[/red] Could not read {escape(str(path))} ({escape(str(error))}); "
            f"fix it before unregistering, as it may define {escape(name)}."
        )
        raise typer.Exit(1)
    if owner is None:
        where = (
            "built in"
            if name in ModelRepository(extra_models=[]).list_models()
            else "not defined in any loaded models file"
        )
        err_console.print(f"[red]✗[/red] {escape(name)} is {where}.")
        raise typer.Exit(1)
    try:
        removed = unregister_entry(owner, name)
    except UnblendError as error:
        err_console.print(f"[red]✗[/red] {escape(str(error))}")
        raise typer.Exit(1) from error
    console.print(
        f"[green]✓[/green] Unregistered [bold]{escape(name)}[/bold] from "
        f"{escape(str(owner))}"
    )
    # The files the removed entry itself names (read from the entry, so this
    # works for a models file the registry is skipping). Paths are compared
    # resolved but deleted as named: a symlink is removed, not its target.
    named = sorted(
        set(entry_weight_paths(removed, Path(os.path.realpath(owner)).parent))
    )
    # Only paths known absent are dropped: a file in a folder you can't enter
    # stays, so its deletion fails with a message instead of going unmentioned.
    named = [
        path for path in named if os.path.lexists(path) or not _certainly_missing(path)
    ]
    if not named:
        return
    still_used: set[Path] = set()
    unreadable = []
    for models_file in default_extra_models_files():
        paths = _paths_named_in(models_file)
        if paths is None:
            unreadable.append(models_file)
        else:
            still_used |= paths
    cache = Path(os.path.realpath(get_cache_dir()))
    kept: list[tuple[Path, str]] = []
    deletable: list[Path] = []
    for path in named:
        target = Path(os.path.realpath(path))
        if target.is_relative_to(cache):
            kept.append((path, "in the download cache; use 'models remove'"))
        elif target in still_used:
            kept.append((path, "another model still uses it"))
        elif unreadable:
            kept.append(
                (
                    path,
                    f"{', '.join(map(str, unreadable))} can't be read and may use it",
                )
            )
        else:
            deletable.append(path)
    for path, why in kept:
        console.print(f"[dim]Kept {escape(str(path))} ({escape(why)})[/dim]")
    if not delete_weights:
        if deletable:
            console.print(
                f"[dim]Kept {escape(', '.join(map(str, deletable)))} "
                "(pass --delete-weights to remove)[/dim]"
            )
        return
    for path in deletable:
        try:
            path.unlink()
        except OSError as error:
            err_console.print(
                f"[yellow]![/yellow] Couldn't delete {escape(str(path))}: "
                f"{escape(str(error))}"
            )
        else:
            console.print(f"[green]✓[/green] Deleted {escape(str(path))}")


def _certainly_missing(path: Path) -> bool:
    """
    Whether a file is known not to exist, as opposed to unreachable.

    ``os.path.exists`` is False for a file in a folder you can't enter too;
    callers that delete weights must treat that as "can't tell", not "absent".

    :param path: A file path.
    :return: True only if the path can't name an existing file.
    """
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except ValueError:
        return True  # an embedded NUL: no file can have that name
    except OSError as error:
        return error.errno == errno.ENAMETOOLONG
    return False


def _paths_named_in(models_file: Path) -> set[Path] | None:
    """
    Every local ``path`` a models file names, resolved as the registry would.

    :param models_file: A models file.
    :return: Resolved paths; empty for a missing file, None for one that
        can't be read or parsed (it may name anything).
    """
    from ..importer import _load_mapping

    if _certainly_missing(models_file):
        return set()
    try:
        payload = _load_mapping(models_file)
    except Exception:
        return None
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        return set()
    base = Path(os.path.realpath(models_file)).parent
    try:
        return {
            Path(os.path.realpath(path))
            for entry in models.values()
            for path in entry_weight_paths(entry, base)
        }
    except ValueError:
        # An embedded NUL: the registry refuses such a file, so treat it as
        # unreadable (it may name anything) rather than crash mid-unregister.
        return None


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
        err_console.print(
            "[red]Error:[/red] [bold]--all[/bold] and explicit model names are mutually exclusive."
        )
        raise typer.Exit(1)

    if not all_models and (names is None or not names):
        err_console.print("[red]Error:[/red] No models specified for download.")
        console.print("Please either:")
        console.print("  1. Specify one or more model names to download")
        console.print("  2. Use [bold]--all[/bold] to download all available models")
        console.print(
            "\nTo see available models, run: [bold]unblend models list[/bold]"
        )
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
    include_shared: Annotated[
        bool,
        typer.Option(
            "--include-shared",
            help="Also delete files that other models (e.g. an ensemble's members) use",
        ),
    ] = False,
) -> None:
    """
    Remove models from the cache to free up space.

    :param names: Model names to remove
    :param all_models: If True, remove all downloaded models
    :param include_shared: Also delete files other models share
    """
    if all_models and names:
        err_console.print(
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
            err_console.print(
                "[yellow]No models specified for removal. Name at least one, or "
                "pass --all.[/yellow]"
            )
            raise typer.Exit(1)
        else:
            model_names = names

    # Unknown model names are caller mistakes (typos), distinct from a known
    # model that just isn't cached — report them and exit nonzero.
    known_models = model_repo.list_models()
    unknown = [name for name in model_names if name not in known_models]
    for name in unknown:
        err_console.print(
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
        err_console.print("[yellow]No models found to remove.[/yellow]")
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

            kept = {
                path: users
                for path, users in model_repo.shared_artifacts(name).items()
                if os.path.isfile(path) and not set(users) <= set(model_names)
            }
            try:
                success = model_repo.remove_model(
                    name,
                    include_shared=include_shared or all_models,
                    also_removing=model_names,
                )
            except ModelLoadingError as error:
                err_console.print(
                    f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}"
                )
                failed_removals.append(name)
                progress_bar.update(task, advance=1)
                continue
            if kept and not (include_shared or all_models):
                users = sorted({user for group in kept.values() for user in group})
                err_console.print(
                    f"[yellow]![/yellow] Kept {len(kept)} file(s) of "
                    f"[bold]{escape(name)}[/bold] that "
                    f"{escape(', '.join(users))} also "
                    f"{'uses' if len(users) == 1 else 'use'}; pass --include-shared "
                    "to delete them too"
                )
            if success:
                console.print(
                    f"[green]✓[/green] Removed model [bold]{escape(name)}[/bold]"
                )
            elif model_repo.is_fully_local(name):
                err_console.print(
                    f"[yellow]![/yellow] [bold]{escape(name)}[/bold] uses local "
                    "files, not the cache; to drop it, run "
                    f"'unblend models unregister {escape(name)}'"
                )
            elif not kept:
                err_console.print(
                    f"[yellow]![/yellow] Model [bold]{escape(name)}[/bold] not found in cache"
                )

            progress_bar.update(task, advance=1)

    if unknown or failed_removals:
        raise typer.Exit(1)


def _format_download_summary(
    name: str,
    model: Model | ModelEnsemble,
    models: dict,
    cache_info: dict,
    download_time: float,
    file_count: int | None = None,
    only_load: str | None = None,
) -> str:
    """
    Build the success summary line shown after a model finishes downloading.

    :param name: Model name
    :param model: Loaded model instance
    :param models: Dictionary of available model metadata
    :param cache_info: Cache info mapping from ``ModelRepository.get_cache_info``
    :param download_time: Elapsed download time in seconds
    :param file_count: Files actually downloaded; defaults to the model's
        full file count from metadata (differs under ``only_load``)
    :param only_load: Stem isolated, if any; the size covers only the files
        loaded for it
    :return: Rich-markup summary string
    """
    num_sources = len(model.sources)
    if file_count is None and name in models:
        file_count = _model_file_count(name)
    if file_count is not None:
        file_word = "file" if file_count == 1 else "files"
        model_type = f"{file_count} {file_word}"
    else:
        model_type = "Model"

    size_str = ""
    speed_str = ""
    if name in cache_info:
        size_bytes = cache_info[name]["size_bytes"]
        # The files loaded (local ones too) give the size; the cache alone
        # gives the transfer speed.
        loaded = ModelRepository().loaded_bytes(name, only_load=only_load)
        size_str = f" ({format_file_size(loaded)})"

        if download_time > 0.1:
            speed = size_bytes / download_time
            speed_str = f" at {format_file_size(speed)}/s"

    return f"[green]✓[/green] [bold]{escape(name)}[/bold]: {model_type} with {num_sources} sources{size_str}{speed_str}"


def _download_model_with_progress(name: str, only_load: str | None = None) -> bool:
    """
    Download a single model with progress display.

    :param name: Model name to download
    :param only_load: Optional stem — restricts an ensemble download to the
        single specialist checkpoint
    :return: True if successful, False otherwise
    """

    models = get_models()
    model_repo = ModelRepository()

    try:
        file_count = model_repo.loaded_file_count(name, only_load=only_load)
    except ModelLoadingError as error:
        err_console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}"
        )
        return False

    file_word = "file" if file_count == 1 else "files"
    console.print(
        f"[bold]Downloading {escape(name)} ({file_count} {file_word})...[/bold]"
    )

    with create_model_progress_bar() as progress_bar:
        task = progress_bar.add_task(
            f"[cyan]Downloading {escape(name)} ({file_count} {file_word})[/cyan]",
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
                    name,
                    model,
                    models,
                    cache_info,
                    download_time,
                    file_count,
                    only_load=only_load,
                )
            )
            return True

        except Exception as error:
            progress_bar.remove_task(task)
            err_console.print(
                f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}"
            )
            return False


def ensure_model_available(name: str, only_load: str | None = None) -> bool:
    """
    Ensure a model is available, downloading if necessary.

    :param name: Model name to check/download
    :param only_load: Optional stem — when set, only the specialist checkpoint for
        this stem needs to be (and will be) downloaded
    :return: True if model is available, False otherwise
    """
    model_repo = ModelRepository()
    models = model_repo.list_models()
    info = models.get(name)
    if info is None:
        err_console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: Unknown model. "
            f"Available models: {', '.join(models)}"
        )
        return False

    if model_repo.is_fully_local(name):
        missing = [
            path
            for path in model_repo.local_artifacts(name)
            if not os.path.isfile(path)
        ]
        if not missing:
            return True
        for path in missing:
            err_console.print(
                f"[red]✗[/red] [bold]{escape(name)}[/bold]: Local checkpoint "
                f"does not exist: {escape(str(path))}"
            )
        return False

    try:
        required = model_repo.required_files(name, only_load=only_load)
    except ModelLoadingError as error:
        err_console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}"
        )
        return False

    cached_files = model_repo.get_cache_info().get(name, {}).get("files", {})
    if all(cached_files.get(key, {}).get("complete") for key in required):
        # Existence and size only: downloads are sha256-verified before
        # entering the cache and ``get_model`` re-verifies on load, so
        # re-hashing here would only add seconds to every run.
        return True

    return _download_model_with_progress(name, only_load=only_load)


def _download_models_batch(model_names: list[str]) -> None:
    """
    Download multiple models, showing progress for each. Exits nonzero if any
    name is unknown or any download fails, so scripts can detect it.

    :param model_names: List of model names to download
    :raises typer.Exit: If any model is unknown or fails to download
    """
    # A name given twice is downloaded once.
    model_names = list(dict.fromkeys(model_names))
    model_repo = ModelRepository()
    cache_info = model_repo.get_cache_info()

    models = get_models()

    # Unknown names are reported (and fail the command) without spinning up a
    # progress bar for them.
    problems = [name for name in model_names if name not in models]
    for name in problems:
        err_console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: Unknown model. "
            f"Available models: {', '.join(models)}"
        )

    to_download = []
    for name in model_names:
        if name in problems:
            continue
        missing = [
            path
            for path in model_repo.local_artifacts(name)
            if not os.path.isfile(path)
        ]
        if missing:
            err_console.print(
                f"[red]✗[/red] [bold]{escape(name)}[/bold]: Local weights "
                f"missing: {escape(', '.join(map(str, missing)))}"
            )
            problems.append(name)
            continue
        if model_repo.is_fully_local(name):
            console.print(
                f"[green]✓[/green] [bold]{escape(name)}[/bold]: Local "
                "weights, nothing to download"
            )
            continue
        # Partially-cached models (interrupted downloads) still need the
        # download pass; get_model fetches only the missing files.
        if cache_info.get(name, {}).get("complete"):
            file_count = _model_file_count(name, model_repo)
            file_word = "file" if file_count == 1 else "files"
            size_bytes = cache_info[name]["size_bytes"] + _local_bytes(name, model_repo)
            size_str = f", {format_file_size(size_bytes)}"
            console.print(
                f"[green]✓[/green] [bold]{escape(name)}[/bold]: Already downloaded ({file_count} {file_word}{size_str})"
            )
        else:
            to_download.append(name)

    if not to_download:
        if problems:
            raise typer.Exit(1)
        console.print("[green]All specified models are already downloaded.[/green]")
        return

    if len(to_download) > 1:
        # Models that share a checkpoint download it once.
        total_files = len(
            set().union(*(model_repo.loaded_files(name) for name in to_download))
        )
        console.print(
            f"[bold]Downloading {len(to_download)} models ({total_files} files)...[/bold]"
        )

    failed = []
    with create_model_progress_bar() as progress_bar:
        for name in to_download:
            if not _download_single_model_in_batch(name, models, progress_bar):
                failed.append(name)

    if failed:
        err_console.print(
            f"[red]✗[/red] Failed to download: [bold]{', '.join(failed)}[/bold]"
        )
    if failed or problems:
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

    file_count = _model_file_count(name)
    file_word = "file" if file_count == 1 else "files"
    task = progress_bar.add_task(
        f"[cyan]Downloading {escape(name)} ({file_count} {file_word})[/cyan]",
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
        err_console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: {escape(str(error))}"
        )
        return False
    except Exception as e:
        progress_bar.remove_task(task)
        err_console.print(
            f"[red]✗[/red] [bold]{escape(name)}[/bold]: Unexpected error: "
            f"{escape(str(e))}"
        )
        return False


def _loaded_models_files() -> list[Path]:
    """
    The user models files ``ModelRepository`` reads by default.

    :return: Resolved paths from ``UNBLEND_EXTRA_MODELS`` plus the default file.
    """
    from ..repo import default_models_file, listed_extra_models_files

    paths = [Path(os.path.realpath(path)) for path in listed_extra_models_files()]
    return [*paths, Path(os.path.realpath(default_models_file()))]


def import_model_command(
    checkpoint: Annotated[
        Path,
        typer.Argument(
            help="Checkpoint to import: .safetensors, or a .ckpt/.pt/.pth/.th holding only tensors",
            show_default=False,
        ),
    ],
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Name to register the model under"),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Training config to translate (Music-Source-Separation-Training YAML, or JSON)",
            show_default=False,
        ),
    ] = None,
    architecture: Annotated[
        str | None,
        typer.Option(
            "--architecture",
            "-a",
            help="Architecture; inferred from the weights when omitted",
            show_default=False,
        ),
    ] = None,
    stems: Annotated[
        list[str] | None,
        typer.Option(
            "--stem",
            help="Output stem name, in order (repeat). Overrides the config.",
            show_default=False,
        ),
    ] = None,
    samplerate: Annotated[
        int | None,
        typer.Option("--samplerate", help="Sample rate the weights operate at"),
    ] = None,
    segment_samples: Annotated[
        int | None,
        typer.Option("--segment-samples", help="Training chunk length in samples"),
    ] = None,
    license_label: Annotated[
        str,
        typer.Option("--license", help="License label recorded in the entry"),
    ] = "unknown",
    note: Annotated[
        str | None,
        typer.Option("--note", help="Provenance note recorded in the entry"),
    ] = None,
    # str, not Path: Path would drop the trailing slash that marks a folder.
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the converted weights "
            "(default: ~/.unblend/imported/NAME.safetensors)",
            show_default=False,
        ),
    ] = None,
    models_file: Annotated[
        Path | None,
        typer.Option(
            "--models-file",
            help="Models file to register in (default: ~/.unblend/models.yaml)",
            show_default=False,
        ),
    ] = None,
    register: Annotated[
        bool,
        typer.Option(
            "--register/--print",
            help="Register the entry, or print it (the .safetensors file is written either way)",
        ),
    ] = True,
) -> None:
    """
    Import a checkpoint from elsewhere: repackage it as Safetensors, prove it
    loads, and register it.

    :param checkpoint: Checkpoint to import
    :param name: Name to register the model under
    :param config: Training config to translate
    :param architecture: Architecture, inferred when omitted
    :param stems: Output stem names, overriding the config
    :param samplerate: Sample rate the weights operate at
    :param segment_samples: Training chunk length in samples
    :param license_label: License label recorded in the entry
    :param note: Provenance note recorded in the entry
    :param output: Where to write the converted weights
    :param models_file: Models file to register in
    :param register: Register the entry, or print it instead
    :raises typer.Exit: If the checkpoint cannot be imported
    """
    from ..exceptions import UnblendError
    from ..importer import (
        check_registrable,
        import_checkpoint,
        printable,
        register_entry,
        validate_model_name,
    )
    from ..repo import default_models_file

    try:
        validate_model_name(name)
    except UnblendError as error:
        err_console.print(f"[red]✗[/red] {escape(str(error))}")
        raise typer.Exit(1) from error

    if not os.path.isfile(checkpoint):
        if os.path.isdir(checkpoint):
            problem = "is a folder"
        elif os.path.exists(checkpoint):
            problem = "isn't a regular file"
        else:
            problem = "doesn't exist"
        err_console.print(
            f"[red]✗[/red] Checkpoint {escape(str(checkpoint))} {problem}."
        )
        raise typer.Exit(1)

    target = Path(
        os.path.realpath(os.path.expanduser(models_file or default_models_file()))
    )
    if register:
        try:
            check_registrable(target, name)
        except UnblendError as error:
            err_console.print(f"[red]✗[/red] {escape(str(error))}")
            raise typer.Exit(1) from error

    if output is not None and not output.strip():
        err_console.print("[red]✗[/red] --output is empty.")
        raise typer.Exit(1)
    # os.path, not Path: Path.expanduser() raises for an unknown ~user.
    artifact = (
        Path(os.path.expanduser(output))
        if output is not None
        else Path.home() / ".unblend" / "imported" / f"{name}.safetensors"
    )
    # A trailing slash (or a final "." or "..") names a folder, even one
    # that doesn't exist yet.
    if os.path.isdir(artifact) or (
        output is not None
        and (output.endswith(("/", os.sep)) or os.path.basename(output) in {".", ".."})
    ):
        artifact = artifact / f"{name}.safetensors"
    # The importer stages to ".NAME.<32 hex>.partial" beside it (42 more).
    too_long = not name_fits(artifact.name, reserve=42) or not all(
        name_fits(part) for part in artifact.parts
    )
    if too_long or not name_encodable(str(artifact)):
        err_console.print(
            f"[red]✗[/red] {escape(str(artifact))} has a file or folder name "
            "too long to write, or one this filesystem can't encode."
        )
        raise typer.Exit(1)
    try:
        repo = ModelRepository()
        in_use = [
            other
            for other in repo.list_models()
            if any(
                os.path.realpath(path) == os.path.realpath(artifact)
                or (
                    os.path.exists(path)
                    and os.path.exists(artifact)
                    and path.samefile(artifact)
                )
                for path in repo.local_artifacts(other)
            )
        ]
    except ModelLoadingError:
        in_use = []
    if in_use:
        err_console.print(
            f"[red]✗[/red] {escape(str(artifact))} holds the weights of "
            f"{escape(', '.join(in_use))}; choose another --output."
        )
        raise typer.Exit(1)
    if os.path.exists(artifact):
        err_console.print(
            f"[red]✗[/red] {escape(str(artifact))} already exists; choose another "
            "--output or delete it first."
        )
        raise typer.Exit(1)
    # Checked before the conversion, which can take a while.
    if stems and len({stem_key(stem) for stem in stems}) != len(stems):
        err_console.print(
            "[red]✗[/red] --stem names must be unique (ignoring case and Unicode form): "
            f"{escape(', '.join(stems))}"
        )
        raise typer.Exit(1)
    unsafe = [
        stem
        for stem in stems or []
        if not stem or stem in (".", "..") or any(c in stem for c in "/\\:\0")
    ]
    if unsafe:
        # The registry refuses these (stems become output file names).
        err_console.print(
            "[red]✗[/red] --stem names must be usable as file names: "
            f"{escape(', '.join(map(repr, unsafe)))}"
        )
        raise typer.Exit(1)
    folder = Path(os.path.abspath(artifact)).parent
    while not os.path.lexists(folder) and folder != folder.parent:
        folder = folder.parent
    if not os.path.isdir(folder) or not os.access(folder, os.W_OK):
        err_console.print(
            f"[red]✗[/red] Can't write {escape(str(artifact))} "
            f"({escape(str(folder))} isn't a writable folder)."
        )
        raise typer.Exit(1)

    try:
        entry, summary = import_checkpoint(
            checkpoint,
            artifact,
            config_path=config,
            architecture=architecture,
            sources=stems or None,
            samplerate=samplerate,
            segment_samples=segment_samples,
            license_label=license_label,
            note=note,
        )
    except (UnblendError, OSError) as error:
        err_console.print(f"[red]✗[/red] {escape(str(error))}")
        raise typer.Exit(1) from error

    # With --print, stdout carries only the YAML so it can be redirected.
    status = console if register else err_console
    status.print(
        f"[green]✓[/green] Loaded as [bold]{summary['architecture']}[/bold] and "
        f"strict-loaded {summary['tensors']} tensors"
    )
    if summary.get("dropped_config_keys"):
        status.print(
            "[dim]Ignored config keys the model doesn't take: "
            f"{escape(printable(', '.join(summary['dropped_config_keys'])))}[/dim]"
        )
    status.print(
        f"[green]✓[/green] Wrote [bold]{escape(str(artifact))}[/bold] "
        f"({format_file_size(summary['size_bytes'])})"
    )

    if not register:
        from ..importer import _dump_mapping

        typer.echo(_dump_mapping({"models": {name: entry}}, Path("entry.yaml")))
        return

    try:
        register_entry(target, name, entry)
    except UnblendError as error:
        # Don't leave converted weights behind that nothing refers to.
        artifact.unlink(missing_ok=True)
        err_console.print(f"[red]✗[/red] {escape(str(error))}")
        raise typer.Exit(1) from error
    console.print(
        f"[green]✓[/green] Registered [bold]{escape(name)}[/bold] in "
        f"{escape(str(target))}"
    )

    if target not in _loaded_models_files():
        err_console.print(
            f"\n[yellow]![/yellow] Add {escape(str(target))} to "
            "[bold]UNBLEND_EXTRA_MODELS[/bold] for Unblend to see it."
        )
    console.print(
        f"\nTry it: [bold]unblend separate --model {escape(name)} track.wav[/bold]"
    )
