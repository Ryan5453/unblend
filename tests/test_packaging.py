"""Offline packaging and Cog configuration consistency checks."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


_COG_EXPORT_COMMAND = (
    "uv export --locked --no-dev --no-hashes --no-emit-project "
    "--format requirements-txt"
)


def test_cog_uses_fully_locked_uv_export() -> None:
    """Cog installs the checked-in, fully pinned export of ``uv.lock``."""
    cog = (ROOT / "cog.yaml").read_text()
    assert 'python_requirements: "requirements-cog.txt"' in cog
    assert not (ROOT / "requirements.txt").exists()

    exported = (ROOT / "requirements-cog.txt").read_text()
    assert _COG_EXPORT_COMMAND in exported
    requirement_lines = [
        line.strip()
        for line in exported.splitlines()
        if line and not line[0].isspace() and not line.startswith("#")
    ]
    assert requirement_lines
    assert all("==" in line for line in requirement_lines)
    assert not any(line.startswith(("-e ", ".", "/")) for line in requirement_lines)


def test_cog_model_url_matches_metadata() -> None:
    """
    The htdemucs layer URL baked into the Cog image must match metadata.json.

    cog.yaml's build.run commands execute before the repo is mounted, so the
    URL is necessarily hardcoded there; this guards it against drifting from
    the canonical entry that unblend.repo downloads from.
    """
    cog = (ROOT / "cog.yaml").read_text()
    match = re.search(
        r"curl .*--output /root/\.unblend/models/(\S+\.safetensors) (https://\S+)",
        cog,
    )
    assert match, "cog.yaml no longer bakes the htdemucs Safetensors layer"
    baked_filename, baked_url = match.groups()

    with open(ROOT / "unblend" / "metadata.json") as f:
        layers = json.load(f)["models"]["htdemucs"]["models"]
    assert len(layers) == 1, "cog.yaml bakes exactly one layer but htdemucs has more"

    # Absolute remotes are used verbatim; relative ones resolve against the
    # Meta CDN — mirror unblend.repo's URL construction.
    remote = layers[0]["remote"]
    if "://" in remote:
        expected_url = remote
    else:
        base = re.search(
            r'^BASE_CDN_URL = "([^"]+)"',
            (ROOT / "unblend" / "repo.py").read_text(),
            re.MULTILINE,
        )
        assert base, "BASE_CDN_URL not found in unblend/repo.py"
        expected_url = f"{base.group(1)}/{remote}"

    assert baked_url == expected_url
    assert baked_filename == f"{layers[0]['checksum']}.safetensors"
    assert f"--max-filesize {layers[0]['size_bytes']}" in cog
    assert layers[0]["sha256"] in cog
