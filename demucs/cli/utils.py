# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()

METADATA_PATH = Path(__file__).parent.parent / "metadata.json"


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
    variables = {
        "model": model,
        "track": track.name.rsplit(".", 1)[0],
        "stem": stem,
        "ext": ext,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H-%M-%S"),
        "timestamp": str(int(now.timestamp())),
    }

    formatted_path = template
    for var, value in variables.items():
        formatted_path = formatted_path.replace(f"{{{var}}}", value)

    return Path(formatted_path)


def get_models() -> dict[str, dict]:
    """
    Get models from metadata.json.

    :return: Dictionary mapping model names to their metadata
    """
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return metadata["models"]


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
                    f"[yellow]Warning:[/yellow] No audio files found in '{path}'"
                )
        else:
            had_errors = True
            console.print(f"[red]Error:[/red] Path '{path}' does not exist")

    return audio_files, had_errors
