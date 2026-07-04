"""Unit tests for the path/formatting helpers in ``demucs.cli.utils``."""

from pathlib import Path

from demucs.cli.utils import (
    _looks_like_audio_file,
    expand_paths_to_audio_files,
    format_file_size,
    format_output_path,
)


def test_format_file_size_units() -> None:
    """
    Sizes are rendered with the largest fitting binary unit.
    """
    assert format_file_size(512) == "512 B"
    assert format_file_size(1536) == "1.5 KB"
    assert format_file_size(5 * 1024 * 1024) == "5.0 MB"
    assert format_file_size(3 * 1024**3) == "3.0 GB"


def test_format_output_path_substitutes_variables() -> None:
    """
    Template placeholders are replaced; ``{track}`` drops the extension.
    """
    out = format_output_path(
        "{model}/{track}/{stem}.{ext}",
        model="htdemucs",
        track=Path("/music/My Song.flac"),
        stem="vocals",
        ext="wav",
    )
    assert out == Path("htdemucs/My Song/vocals.wav")


def test_looks_like_audio_file_extension_check() -> None:
    """
    The heuristic matches known audio extensions case-insensitively.
    """
    assert _looks_like_audio_file(Path("track.MP3"))
    assert _looks_like_audio_file(Path("track.flac"))
    assert not _looks_like_audio_file(Path("notes.txt"))


def test_expand_paths_directory_and_file(tmp_path: Path) -> None:
    """
    Directories expand to their audio files (sorted, dotfiles skipped);
    explicit file paths pass through untouched.

    :param tmp_path: pytest temporary directory fixture
    """
    (tmp_path / "b.wav").write_bytes(b"")
    (tmp_path / "a.mp3").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / ".hidden.wav").write_bytes(b"")

    expanded, had_errors = expand_paths_to_audio_files([tmp_path])
    assert expanded == [tmp_path / "a.mp3", tmp_path / "b.wav"]
    assert not had_errors

    explicit = tmp_path / "notes.txt"
    assert expand_paths_to_audio_files([explicit]) == ([explicit], False)


def test_expand_paths_recurses_subdirectories(tmp_path: Path) -> None:
    """
    Audio files in nested subdirectories are picked up; dot-files and dot-
    directories are skipped at any depth.

    :param tmp_path: pytest temporary directory fixture
    """
    (tmp_path / "top.wav").write_bytes(b"")
    (tmp_path / "album").mkdir()
    (tmp_path / "album" / "track.flac").write_bytes(b"")
    (tmp_path / "album" / "disc 2").mkdir()
    (tmp_path / "album" / "disc 2" / "deeper.mp3").write_bytes(b"")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "skip-me.wav").write_bytes(b"")
    (tmp_path / "album" / ".hidden.wav").write_bytes(b"")

    expanded, had_errors = expand_paths_to_audio_files([tmp_path])
    assert expanded == [
        tmp_path / "album" / "disc 2" / "deeper.mp3",
        tmp_path / "album" / "track.flac",
        tmp_path / "top.wav",
    ]
    assert not had_errors


def test_expand_paths_flags_unresolvable_inputs(tmp_path: Path) -> None:
    """
    Nonexistent paths and audio-free directories set the error flag while
    still returning whatever did resolve.

    :param tmp_path: pytest temporary directory fixture
    """
    (tmp_path / "a.mp3").write_bytes(b"")

    expanded, had_errors = expand_paths_to_audio_files(
        [tmp_path, tmp_path / "missing.wav"]
    )
    assert expanded == [tmp_path / "a.mp3"]
    assert had_errors

    empty = tmp_path / "empty"
    empty.mkdir()
    assert expand_paths_to_audio_files([empty]) == ([], True)
