"""Regression checks for the isolated upstream benchmark worker."""

import ast
import threading
import time
from pathlib import Path

import benchmark

ROOT = Path(__file__).resolve().parent.parent


def _worker_template() -> str:
    """Read the upstream worker template without importing benchmark.py."""
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
    """The isolated worker must import the package installed in its venv."""
    template = _worker_template()
    assert "from demucs.api import Separator" in template
    assert "from unblend.api import Separator" not in template
    compile(template.replace("# __SHARED_SDR__", ""), "<upstream-worker>", "exec")


def test_upstream_venv_path_is_digest_contained(tmp_path, monkeypatch) -> None:
    """A hostile Git ref never becomes a filesystem path component."""
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
    """Concurrent callers build one environment under the sibling file lock."""
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
    """Every upstream track gets the same stable seed the worker reports."""
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
