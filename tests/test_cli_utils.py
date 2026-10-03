"""
Unit tests for the path/formatting helpers in ``unblend.cli.utils``.
"""

from pathlib import Path

from unblend.cli.utils import (
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


def test_format_output_path_dot_components_keep_full_filename() -> None:
    """
    Legal names whose stripped stem is '.'/'..' cannot escape a template.
    """
    assert format_output_path(
        "out/{track}/{stem}.{ext}", "m", Path("...wav"), "vocals", "wav"
    ) == Path("out/...wav/vocals.wav")
    assert format_output_path(
        "out/{track}/{stem}.{ext}", "m", Path("..wav"), "vocals", "wav"
    ) == Path("out/..wav/vocals.wav")


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


def test_format_output_path_does_not_rescan_substituted_values() -> None:
    """
    Substitution is single-pass: a placeholder appearing literally in a
    track's filename must not be expanded inside the substituted value.
    """
    out = format_output_path(
        "out/{track}.wav", "m", Path("my{stem}.flac"), "vocals", "wav"
    )
    assert out == Path("out/my{stem}.wav")


def test_format_output_path_dotfile_track_keeps_name() -> None:
    """
    An extensionless dotfile track must not resolve {track} to "" (which
    would collapse a leading "{track}/" into an absolute path).
    """
    out = format_output_path(
        "{track}/{stem}.{ext}", "m", Path(".hidden"), "vocals", "wav"
    )
    assert out == Path(".hidden/vocals.wav")


def test_expand_paths_skips_the_output_root_inside_an_input_dir(tmp_path) -> None:
    """
    Re-running on a directory doesn't pick up the previous run's stems, but
    pointing at the output directory itself still works.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.cli.utils import expand_paths_to_audio_files

    (tmp_path / "song.wav").write_bytes(b"")
    out = tmp_path / "separated" / "htdemucs" / "song"
    out.mkdir(parents=True)
    (out / "vocals.wav").write_bytes(b"")
    root = (tmp_path / "separated").resolve()

    files, _ = expand_paths_to_audio_files([tmp_path], exclude=root)
    assert [f.name for f in files] == ["song.wav"]

    files, _ = expand_paths_to_audio_files([tmp_path / "separated"], exclude=root)
    assert [f.name for f in files] == ["vocals.wav"]


def test_parent_variable_distinguishes_same_named_tracks(tmp_path) -> None:
    """
    ``{parent}`` is the track's folder name, for layouts like MUSDB's
    ``*/mixture.wav``.

    :param tmp_path: pytest temporary directory fixture
    """
    from pathlib import Path

    from unblend.cli.utils import format_output_path

    a = format_output_path(
        "{parent}/{stem}.wav", "m", tmp_path / "song a" / "mixture.wav", "v"
    )
    b = format_output_path(
        "{parent}/{stem}.wav", "m", tmp_path / "song b" / "mixture.wav", "v"
    )
    assert (a, b) == (Path("song a/v.wav"), Path("song b/v.wav"))


def test_tilde_in_a_track_name_is_not_expanded() -> None:
    """
    Only the template's own ``~`` means home; a track called ``~.wav`` stays a
    literal folder name.
    """
    from pathlib import Path

    from unblend.cli.utils import format_output_path

    assert format_output_path("{track}/{stem}.wav", "m", Path("~.wav"), "v") == Path(
        "~/v.wav"
    )
    assert (
        format_output_path("~/o/{stem}.wav", "m", Path("t.wav"), "v")
        == Path.home() / "o" / "v.wav"
    )


def test_unknown_template_placeholders_are_reported() -> None:
    """
    A misspelled placeholder is reported instead of becoming a literal
    folder name.
    """
    from unblend.cli.utils import unknown_placeholders

    assert unknown_placeholders("out/{model}/{trak}/{stem}.{ext}") == ["{trak}"]
    assert unknown_placeholders("separated/{model}/{track}/{stem}.{ext}") == []
