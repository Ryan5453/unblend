"""
Shared fixtures.
"""

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """
    Turn off forced colour and styling before any test module imports the CLI.

    CLI assertions compare plain text, and Rich reads these variables when
    its consoles are created at import time, so a fixture would be too late.

    :param config: pytest configuration (unused).
    """
    del config
    # FORCE_COLOR and Rich 14's TTY_COMPATIBLE/TTY_INTERACTIVE force styled
    # output; NO_COLOR alone still lets bold through.
    for name in ("FORCE_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE"):
        os.environ.pop(name, None)
    os.environ["NO_COLOR"] = "1"
    # A narrow terminal's COLUMNS would rewrap messages the CLI tests match;
    # 80 is what CI renders at.
    os.environ["COLUMNS"] = "80"
    # Test modules import torchcodec directly; its macOS Python 3.10 wheel
    # writes default.profraw unless this is set before it loads.
    os.environ.setdefault("LLVM_PROFILE_FILE", os.devnull)


@pytest.fixture(autouse=True)
def _isolate_default_models_file(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """
    Point the always-loaded user models file at an empty temp location and
    clear ``UNBLEND_EXTRA_MODELS``, so a developer's own models can't leak into
    tests. The model cache is shared on purpose (slow tests reuse downloads).

    :param tmp_path_factory: pytest temp directory factory
    :param monkeypatch: pytest monkeypatch fixture
    :return: The substituted path (not created).
    """
    import unblend.repo

    path = tmp_path_factory.mktemp("home") / "models.yaml"
    monkeypatch.setattr(unblend.repo, "default_models_file", lambda: path)
    monkeypatch.delenv("UNBLEND_EXTRA_MODELS", raising=False)
    return path
