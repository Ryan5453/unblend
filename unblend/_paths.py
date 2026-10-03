# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
File-name length limits, as the platform's filesystems count them.
"""

from __future__ import annotations

import os
import sys

#: The longest file or folder name: 255 bytes of UTF-8 on Linux filesystems,
#: 255 UTF-16 code units of the name as given on macOS (APFS) and Windows.
#: HFS+ counts the decomposed (NFD) form, so there a long accented name that
#: passes can still fail when it is written.
NAME_MAX = 255


def name_length(name: str) -> int:
    """
    Measure a single path component the way this platform limits it.

    :param name: A file or folder name (no separators).
    :return: Its length in the platform's unit.
    """
    if sys.platform in ("darwin", "win32"):
        return len(name.encode("utf-16-le", "surrogatepass")) // 2
    # surrogatepass: a lone surrogate (from a JSON file) still has a length;
    # name_encodable is what refuses it.
    return len(name.encode("utf-8", "surrogatepass"))


def name_fits(name: str, reserve: int = 0) -> bool:
    """
    Whether a name, plus ``reserve`` units added by a staging copy, fits.

    :param name: A file or folder name.
    :param reserve: Extra length a temporary sibling adds to the name.
    :return: True if the name (and its staging copy) can be created.
    """
    return name_length(name) + reserve <= NAME_MAX


def name_encodable(text: str) -> bool:
    """
    Whether a path can be created on this platform's filesystems.

    Arguments with invalid UTF-8 reach Python as surrogate escapes, which
    ``os.fsencode`` turns back into bytes: fine on Linux, but APFS (macOS)
    refuses names that aren't valid UTF-8.

    :param text: A path.
    :return: False for a path no file could be created at.
    """
    try:
        if sys.platform == "darwin":
            text.encode("utf-8")
        else:
            os.fsencode(text)
    except UnicodeEncodeError:
        return False
    return True
