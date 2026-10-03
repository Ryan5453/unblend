"""
Tests for the platform's file-name limits in ``unblend._paths``.
"""

import sys

import pytest

from unblend._paths import NAME_MAX, name_encodable, name_fits, name_length


@pytest.mark.parametrize(
    "platform, name, expected",
    [
        ("linux", "é", 2),  # UTF-8 bytes
        ("linux", "日", 3),
        ("linux", "a\ud800b", 5),  # a lone surrogate still has a length
        ("darwin", "é", 1),  # as given, as APFS counts it
        ("darwin", "e\u0301", 2),  # already decomposed
        ("darwin", "日", 1),
        ("darwin", "😀", 2),  # UTF-16 surrogate pair
        ("win32", "é", 1),
    ],
)
def test_name_length_uses_the_platform_unit(
    monkeypatch: pytest.MonkeyPatch, platform: str, name: str, expected: int
) -> None:
    """
    Names are measured as each platform's filesystems limit them.
    """
    monkeypatch.setattr(sys, "platform", platform)
    assert name_length(name) == expected


def test_name_fits_leaves_room_for_a_staging_suffix() -> None:
    """
    ``reserve`` counts against the limit, so the staged copy fits too.
    """
    assert name_fits("a" * NAME_MAX)
    assert not name_fits("a" * NAME_MAX, reserve=1)
    assert name_fits("a" * (NAME_MAX - 42), reserve=42)


def test_invalid_utf8_is_unencodable_only_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An argv name with invalid UTF-8 (a surrogate escape) is legal on Linux
    but refused by APFS.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    assert name_encodable("x\udcff.onnx")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert not name_encodable("x\udcff.onnx")
    assert name_encodable("é.onnx")
