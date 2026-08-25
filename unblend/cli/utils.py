# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import re
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

console = Console()

METADATA_PATH = Path(__file__).parent.parent / "metadata.yaml"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in a human-readable way.

    :param size_bytes: Size in bytes
    :return: Human-readable size string (e.g. "1.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_output_path(
    template: str,
    model: str,
    track: Path,
    stem: str,
    ext: str = "wav",
    now: datetime | None = None,
) -> Path:
    """
    Format output path template with variables.

    :param template: Path template with {variable} placeholders
    :param model: Model name
    :param track: Path to the source track
    :param stem: Stem name
    :param ext: Output file extension
    :param now: Timestamp used for {date}/{time}/{timestamp} substitutions. Pass
        a single value shared across an entire run so the collision pre-check and
        the actual writes resolve to identical paths; defaults to ``datetime.now()``.
    :return: Resolved output path
    """
    if now is None:
        now = datetime.now()
    stripped_track = track.name.rsplit(".", 1)[0]
    # Empty/dot components can collapse or escape a template directory after
    # path normalization. Preserve the full legal filename in those cases.
    safe_track = track.name if stripped_track in {"", ".", ".."} else stripped_track
    variables = {
        "model": model,
        "track": safe_track,
        "stem": stem,
        "ext": ext,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H-%M-%S"),
        "timestamp": str(int(now.timestamp())),
    }

    # Single pass so substituted values are never re-scanned (a track
    # literally named "my{stem}" must not have {stem} expanded inside it).
    pattern = re.compile("|".join(re.escape(f"{{{var}}}") for var in variables))
    formatted_path = pattern.sub(lambda m: variables[m.group(0)[1:-1]], template)

    return Path(formatted_path)


#: ``--model auto`` is not a registered model; it asks the CLI to pick one.
AUTO_MODEL = "auto"


def validate_model_name(value: str) -> str:
    """
    Accept any model the repository knows about, plus ``auto``.

    Typer callback. The choices used to be a hand-written enum, which could
    never name a model added through ``UNBLEND_EXTRA_MODELS`` — and had drifted
    from the shipped registry besides.

    :param value: The name the user passed.
    :return: The same name, once known.
    :raises typer.BadParameter: If no such model is registered.
    """
    if value == AUTO_MODEL:
        return value
    known = get_models()
    if value not in known:
        raise typer.BadParameter(
            f"{value!r} is not a known model. Choose one of: "
            f"{AUTO_MODEL}, {', '.join(known)}."
        )
    return value


def validate_model_names(values: list[str] | None) -> list[str] | None:
    """
    Validate a repeatable ``--model`` option against the registry.

    :param values: The names the user passed, if any.
    :return: The same names, once known.
    :raises typer.BadParameter: If any name is not registered.
    """
    if not values:
        return values
    return [validate_model_name(value) for value in values]


def complete_model_name(incomplete: str) -> list[str]:
    """
    Shell completion for model names, including locally-added models.

    :param incomplete: The partial name typed so far.
    :return: Matching model names.
    """
    candidates = [AUTO_MODEL, *get_models()]
    return [name for name in candidates if name.startswith(incomplete)]


def _combine_modes() -> list[str]:
    """
    Every ensemble combine mode, sorted.

    Imported lazily: ``unblend.apply`` pulls in torch, and the CLI's small
    helpers must stay importable without it.

    :return: The accepted mode names.
    """
    from ..apply import COMBINE_MODES

    return sorted(COMBINE_MODES)


def validate_combine_mode(value: str | None) -> str | None:
    """
    Accept any implemented ensemble combine mode.

    :param value: The mode the user passed, or ``None`` to use the model's own.
    :return: The same value, once known.
    :raises typer.BadParameter: If no such mode exists.
    """
    if value is None:
        return None
    modes = _combine_modes()
    if value not in modes:
        raise typer.BadParameter(
            f"{value!r} is not a known combine mode. Choose one of: {', '.join(modes)}."
        )
    return value


def complete_combine_mode(incomplete: str) -> list[str]:
    """
    Shell completion for combine modes.

    :param incomplete: The partial name typed so far.
    :return: Matching mode names.
    """
    return [mode for mode in _combine_modes() if mode.startswith(incomplete)]


def complete_stem_name(incomplete: str) -> list[str]:
    """
    Shell completion for stem names across every registered model.

    Only completion: which stems are actually valid depends on the model that
    ends up selected, so the check lives there.

    :param incomplete: The partial name typed so far.
    :return: Matching stem names.
    """
    stems = {stem for info in get_models().values() for stem in info.get("sources", [])}
    return sorted(stem for stem in stems if stem.startswith(incomplete))


def get_models() -> dict[str, dict]:
    """
    Get every model the repository knows about.

    Goes through ``ModelRepository`` rather than reading ``metadata.json``
    directly so models added via ``UNBLEND_EXTRA_MODELS`` appear in the CLI
    too — otherwise ``unblend models list`` would omit models that
    ``Separator`` can happily load.

    :return: Dictionary mapping model names to their metadata
    """
    # Imported here: unblend.repo pulls in the model modules, and importing it
    # at module scope would make the CLI's small helpers depend on torch.
    from ..repo import ModelRepository

    return ModelRepository().list_models()


def _looks_like_audio_file(path: Path) -> bool:
    """
    Heuristic check if a file might be audio based on extension.
    This is purely for performance in big folders, torchcodec will determine actual support.

    :param path: Path to check
    :return: True if the file extension matches a known audio format
    """
    return path.suffix.lower() in {
        ".mp3",
        ".wav",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".mp4",
        ".webm",
        ".mkv",
        ".avi",
        ".mov",
        ".wma",
        ".alac",
        ".aiff",
        ".aif",
        ".aifc",
        ".m4b",
        ".m4p",
        ".m4r",
        ".m4v",
    }


def expand_paths_to_audio_files(paths: list[Path]) -> tuple[list[Path], bool]:
    """
    Expand directory paths to include all audio files (recursively), keep
    regular files as-is.

    :param paths: List of file or directory paths
    :return: ``(audio_files, had_errors)`` — ``had_errors`` is True when any
        input path didn't resolve to audio (nonexistent path, or a directory
        with no audio files), so callers can exit nonzero
    """
    audio_files = []
    had_errors = False

    for path in paths:
        if path.is_file():
            # For individual files, just add them and let torchcodec handle validation
            # This allows users to try any file they want (including obscure formats FFmpeg can handle)
            audio_files.append(path)
        elif path.is_dir():
            # Recurse into the directory. Extension heuristic is the cheap
            # filter; probing every file with torchcodec would be slow on
            # large libraries. Dotfiles and dot-directories are skipped.
            found_files = [
                f
                for f in path.rglob("*")
                if f.is_file()
                and not any(part.startswith(".") for part in f.relative_to(path).parts)
                and _looks_like_audio_file(f)
            ]

            if found_files:
                found_files.sort()
                audio_files.extend(found_files)
            else:
                had_errors = True
                console.print(
                    f"[yellow]Warning:[/yellow] No audio files found in "
                    f"'{escape(str(path))}'"
                )
        else:
            had_errors = True
            console.print(
                f"[red]Error:[/red] Path '{escape(str(path))}' does not exist"
            )

    return audio_files, had_errors
