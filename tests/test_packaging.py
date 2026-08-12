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


def test_cog_requirements_have_no_environment_markers() -> None:
    """
    The Cog export must carry no environment markers.

    Cog rewrites each requirement line by matching ``([a-zA-Z0-9\\-_]+)==([^ ]+)``
    and re-emitting ``name==version``; ``[^ ]+`` stops at the space before the
    ``;``, so any marker is dropped silently. A marker-split pin like
    ``numpy==2.2.6 ; python_full_version < '3.11'`` plus its ``>= '3.11'``
    counterpart therefore reaches pip as two conflicting pins and the build
    fails with ResolutionImpossible. export_cog_requirements.py
    pre-evaluates markers for the image's environment to prevent exactly this.
    """
    exported = (ROOT / "requirements-cog.txt").read_text()
    requirement_lines = [
        line.strip()
        for line in exported.splitlines()
        if line and not line[0].isspace() and not line.startswith("#")
    ]

    assert not [line for line in requirement_lines if ";" in line]

    names = [line.split("==")[0].lower() for line in requirement_lines]
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, f"Cog would install conflicting pins for: {duplicates}"


def test_cog_cuda_not_newer_than_locked_wheels() -> None:
    """
    cog.yaml's ``cuda:`` must never be NEWER than the CUDA the wheels expect.

    The two deliberately disagree -- see the comment block in cog.yaml. Cog's
    torch matrix stops at 2.11.0, so our pin never matches and Cog emits no
    ``--extra-index-url``; pip takes PyPI's default and the ``nvidia-*-cuNN``
    pins are what the image really gets. ``cuda:`` only picks the base image,
    and it has to name a version with a published r8.im/cog-base, which caps it
    below the wheels. That understatement is safe because torch ships its own
    CUDA runtime in those wheels.

    The reverse would not be safe: a base image newer than the wheels means the
    system CUDA libs outrank what torch was built against, which surfaces as a
    loader error on GPU boot rather than anything visible at build time. So the
    invariant is an inequality, not equality.
    """
    cog = (ROOT / "cog.yaml").read_text()
    declared = re.search(r'^\s*cuda: "(\d+)\.', cog, re.MULTILINE)
    assert declared, "cog.yaml no longer declares a cuda version"
    declared_major = int(declared.group(1))

    exported = (ROOT / "requirements-cog.txt").read_text()
    # Two spellings carry the CUDA major: the `-cuNN` package-name suffix
    # (nvidia-cudnn-cu13) and cuda-toolkit's own version (13.0.3.0).
    majors = {
        int(m) for m in re.findall(r"^nvidia-\S*-cu(\d+)==", exported, re.MULTILINE)
    }
    majors |= {
        int(m) for m in re.findall(r"^cuda-toolkit==(\d+)\.", exported, re.MULTILINE)
    }
    assert majors, "no CUDA-versioned wheels found in the Cog export"
    assert len(majors) == 1, f"export mixes CUDA majors: {sorted(majors)}"
    wheel_major = majors.pop()

    assert declared_major <= wheel_major, (
        f"cog.yaml declares CUDA {declared_major}.x but the wheels are built "
        f"for CUDA {wheel_major}.x; the base image must not be newer than the "
        f"runtime torch expects"
    )

    # The understatement is only safe because torch carries its own runtime.
    # If these wheels ever stop being pinned, the base image becomes load
    # bearing and the version gap turns into a real mismatch.
    assert re.search(r"^nvidia-cudnn-cu\d+==", exported, re.MULTILINE), (
        "torch no longer pins a bundled CUDA runtime, so cog.yaml's cuda: "
        "field must match the wheels exactly"
    )


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
