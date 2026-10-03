"""
Regression checks for the isolated upstream benchmark worker.
"""

import ast
import importlib.util
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# benchmark.py sits at the repo root, outside the package: load it by path so
# the suite also runs against an installed (non-editable) wheel.
_SPEC = importlib.util.spec_from_file_location("benchmark", ROOT / "benchmark.py")
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


def _worker_template() -> str:
    """
    Parse the upstream worker template out of benchmark.py's source.

    :return: The template string.
    """
    tree = ast.parse((ROOT / "benchmark.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_UPSTREAM_WORKER_TEMPLATE"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError("_UPSTREAM_WORKER_TEMPLATE not found")


def test_upstream_worker_imports_upstream_demucs() -> None:
    """
    The isolated worker must import the package installed in its venv.
    """
    template = _worker_template()
    assert "from demucs.api import Separator" in template
    assert "from unblend.api import Separator" not in template
    compile(template.replace("# __SHARED_SDR__", ""), "<upstream-worker>", "exec")


def test_upstream_venv_path_is_digest_contained(tmp_path, monkeypatch) -> None:
    """
    A hostile Git ref never becomes a filesystem path component.
    """
    root = tmp_path / "upstream-envs"
    monkeypatch.setattr(benchmark, "UPSTREAM_VENV_ROOT", root)

    path, marker = benchmark._upstream_venv_spec("../victim/../../outside", "3.11")

    assert path.parent == root.resolve()
    assert path.name.startswith("upstream-")
    assert "victim" not in path.name
    assert '"version": "../victim/../../outside"' in marker
    assert benchmark._upstream_venv_spec("../victim/../../outside", "3.11") == (
        path,
        marker,
    )
    assert benchmark._upstream_venv_spec("../victim/../../outside", "3.12")[0] != path


def test_upstream_venv_provisioning_is_serialized(tmp_path, monkeypatch) -> None:
    """
    Concurrent callers build one environment under the sibling file lock.
    """
    root = tmp_path / "upstream-envs"
    monkeypatch.setattr(benchmark, "UPSTREAM_VENV_ROOT", root)
    calls = 0
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def fake_provision(venv_dir, _version, _python_version, marker_text) -> None:
        nonlocal active, calls, max_active
        with state_lock:
            calls += 1
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").touch()
        (venv_dir / ".demucs-installed").write_text(marker_text)
        with state_lock:
            active -= 1

    monkeypatch.setattr(benchmark, "_provision_upstream_venv", fake_provision)
    barrier = threading.Barrier(3)
    results = []

    def ensure() -> None:
        barrier.wait()
        results.append(benchmark._ensure_upstream_venv("main", "3.11"))

    threads = [threading.Thread(target=ensure) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert results[0] == results[1]
    assert calls == 1
    assert max_active == 1
    assert Path(f"{results[0]}.lock").parent == root.resolve()


def test_upstream_payload_and_worker_propagate_track_seed(tmp_path) -> None:
    """
    Every upstream track gets the same stable seed the worker reports.
    """
    tracks = [
        benchmark.BenchmarkTrack(
            name="Track A",
            directory=tmp_path / "Track A",
            mixture_path=tmp_path / "Track A" / "mixture.wav",
            reference_stems=("vocals",),
        ),
        benchmark.BenchmarkTrack(
            name="Track B",
            directory=tmp_path / "Track B",
            mixture_path=tmp_path / "Track B" / "mixture.wav",
            reference_stems=("vocals",),
        ),
    ]

    payload = benchmark._build_upstream_tracks_payload(tracks, 1234)

    assert [item["track_seed"] for item in payload] == [
        benchmark._build_track_seed(1234, "Track A"),
        benchmark._build_track_seed(1234, "Track B"),
    ]
    template = _worker_template()
    assert "random.seed(track_seed)" in template
    assert "torch.manual_seed(track_seed)" in template
    assert '"track_seed": track_seed' in template


def test_six_stem_models_are_not_scored_on_other(tmp_path: Path) -> None:
    """
    A model with its own guitar/piano stems isn't scored on ``other``: its
    ``other`` leaves those out while MUSDB's includes them.

    :param tmp_path: pytest temporary directory fixture
    """
    from types import SimpleNamespace

    import torch

    stems = ("drums", "bass", "other", "vocals")
    track = benchmark.BenchmarkTrack(
        name="t",
        directory=tmp_path,
        mixture_path=tmp_path / "mixture.wav",
        reference_stems=stems,
    )
    audio = torch.ones(2, 100)
    separator = SimpleNamespace(
        model=SimpleNamespace(sources=[*stems, "guitar", "piano"]),
        _to_tensor=lambda _path: audio,
    )
    separated = SimpleNamespace(
        sources={name: audio for name in (*stems, "guitar", "piano")}
    )
    scores = benchmark._score_stems(separator, track, separated)
    assert set(scores) == {"drums", "bass", "vocals"}

    separator.model.sources = list(stems)
    assert set(benchmark._score_stems(separator, track, separated)) == set(stems)

    # A 6-stem reference set makes a 6-stem model's "other" comparable, and
    # a 4-stem model's not.
    six = (*stems, "guitar", "piano")
    track6 = benchmark.BenchmarkTrack(
        name="t",
        directory=tmp_path,
        mixture_path=tmp_path / "mixture.wav",
        reference_stems=six,
    )
    separator.model.sources = list(six)
    assert set(benchmark._score_stems(separator, track6, separated)) == set(six)
    separator.model.sources = list(stems)
    assert "other" not in benchmark._score_stems(separator, track6, separated)

    # Upstream runs score the same stems.
    payload = benchmark._build_upstream_tracks_payload([track], None, list(six))
    assert set(payload[0]["stem_paths"]) == {"drums", "bass", "vocals"}


def test_upstream_venv_without_uv_uses_the_requested_python(
    tmp_path, monkeypatch
) -> None:
    """
    Without uv, the venv is built from ``python<version>`` on PATH rather than
    the running interpreter, which may be too new for upstream's torch; if
    that isn't on PATH either, it says so.
    """
    import pytest

    calls = []
    found = {"python3.11": "/fake/bin/python3.11"}
    monkeypatch.setattr(benchmark.shutil, "which", found.get)
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda args, **kwargs: calls.append(args)
    )
    venv = tmp_path / "venv"
    venv.mkdir()
    benchmark._provision_upstream_venv(venv, "v4", "3.11", "marker")
    assert calls[0][:3] == ["/fake/bin/python3.11", "-m", "venv"]

    with pytest.raises(RuntimeError, match="python3.12"):
        benchmark._provision_upstream_venv(venv, "v4", "3.12", "marker")


def test_upstream_venv_setup_errors_are_reported_cleanly(tmp_path, monkeypatch) -> None:
    """
    A provisioning RuntimeError (e.g. no uv and no matching Python) becomes
    the CLI's "Failed to set up upstream demucs venv" error, not a traceback.
    """
    from typer.testing import CliRunner

    track = tmp_path / "musdb" / "t"
    track.mkdir(parents=True)
    import wave

    for name in ("mixture", "vocals"):
        with wave.open(str(track / f"{name}.wav"), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(44100)
            out.writeframes(b"\0" * 4 * 441)

    def fail(*args, **kwargs):
        raise RuntimeError("Neither uv nor python3.99 is on PATH")

    monkeypatch.setattr(benchmark, "_ensure_upstream_venv", fail)
    result = CliRunner().invoke(
        benchmark.app,
        ["--musdb-root", str(tmp_path / "musdb"), "--include-upstream"],
    )
    assert result.exit_code == 2, repr(result.exception)
    assert "Failed to set up upstream" in " ".join(result.output.split())
