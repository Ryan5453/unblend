"""
Packaging consistency checks.

``requirements.txt`` is a hand-maintained mirror of ``[project].dependencies``
in ``pyproject.toml`` (see the header in ``requirements.txt``): the former feeds
the Cog image, the latter the PyPI package. These run fully offline.
"""

import json
import re
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):
    pytest.skip("tomllib requires Python 3.11+", allow_module_level=True)

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def _normalize(requirement: str) -> str:
    """
    Strip all whitespace from a requirement string.

    This keeps ordering and minor formatting differences between the two files
    from registering as drift.

    :param requirement: a single dependency specifier
    :return: the specifier with all whitespace removed
    """
    return "".join(requirement.split())


def _read_requirements_txt() -> set[str]:
    """
    Read and normalize the dependency specifiers from ``requirements.txt``.

    :return: the set of normalized specifiers, with comments and blanks dropped
    """
    reqs: set[str] = set()
    for raw in (ROOT / "requirements.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            reqs.add(_normalize(line))
    return reqs


def _read_pyproject_dependencies() -> set[str]:
    """
    Read and normalize ``[project].dependencies`` from ``pyproject.toml``.

    :return: the set of normalized dependency specifiers
    """
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return {_normalize(dep) for dep in data["project"]["dependencies"]}


def test_requirements_txt_matches_pyproject_dependencies() -> None:
    """
    requirements.txt must mirror pyproject's [project].dependencies exactly.

    Guards against the two drifting so the Cog image and the PyPI package always
    install the same dependency set.
    """
    txt = _read_requirements_txt()
    pyproject = _read_pyproject_dependencies()
    assert txt == pyproject, (
        "requirements.txt and pyproject [project].dependencies have drifted.\n"
        f"Only in requirements.txt: {sorted(txt - pyproject)}\n"
        f"Only in pyproject.toml:   {sorted(pyproject - txt)}"
    )


def test_cog_model_url_matches_metadata() -> None:
    """
    The htdemucs layer URL baked into the Cog image must match metadata.json.

    cog.yaml's build.run commands execute before the repo is mounted, so the
    URL is necessarily hardcoded there; this guards it against drifting from
    the canonical entry that demucs.repo downloads from.
    """
    cog = (ROOT / "cog.yaml").read_text()
    match = re.search(r"curl -L -o /root/\.demucs/models/(\S+)\.th (\S+)", cog)
    assert match, "cog.yaml no longer bakes the htdemucs layer via curl"
    baked_checksum, baked_url = match.groups()

    base = re.search(
        r'^BASE_CDN_URL = "([^"]+)"',
        (ROOT / "demucs" / "repo.py").read_text(),
        re.MULTILINE,
    )
    assert base, "BASE_CDN_URL not found in demucs/repo.py"

    with open(ROOT / "demucs" / "metadata.json") as f:
        layers = json.load(f)["models"]["htdemucs"]["models"]
    assert len(layers) == 1, "cog.yaml bakes exactly one layer but htdemucs has more"

    assert baked_url == f"{base.group(1)}/{layers[0]['remote']}"
    assert baked_checksum == layers[0]["checksum"]
