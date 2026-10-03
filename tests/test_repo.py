"""
Offline checks for ``unblend.repo`` integrity and safe-loading gates.

Registered models use explicit architectures plus tensor-only Safetensors
weights, verified by size and SHA-256 before anything reads them.
"""

import json
import os
import subprocess
import sys
import threading
import time
import warnings
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from unblend.exceptions import ModelLoadingError
from unblend.repo import (
    STAGING_PREFIX,
    STAGING_STALE_SECONDS,
    ModelRepository,
    check_checksum,
    check_size,
    get_cache_dir,
)

#: Fake layer digests for the metadata fixtures below. A cache file is named
#: after the first 16 characters of its artifact's sha256, so the tests derive
#: their expected filenames the same way the repository does.
FIRST_SHA = "abcd1234" + "a" * 56
FIRST_KEY = FIRST_SHA[:16]
SECOND_SHA = "ef012345" + "b" * 56
SECOND_KEY = SECOND_SHA[:16]


def _good_metadata() -> dict:
    """
    Minimal valid metadata blob accepted by ``ModelRepository.__init__``.

    :return: A metadata dict shaped like ``unblend/metadata.yaml``.
    """
    sources = ["drums", "bass", "other", "vocals"]
    return {
        "models": {
            "fakemodel": {
                "architecture": "htdemucs",
                "sources": sources,
                "config": {
                    "sources": sources,
                    "samplerate": 8000,
                    "segment": 1.0,
                    "nfft": 512,
                    "depth": 2,
                    "channels": 16,
                    "t_layers": 1,
                },
                "checkpoint": {
                    "format": "safetensors",
                    "url": "https://example.invalid/abcd.safetensors",
                    "sha256": FIRST_SHA,
                    "size_bytes": 1024,
                },
            }
        }
    }


def _write_metadata(tmp_path: Path, metadata: dict) -> Path:
    """
    Serialize a metadata dict to a temp file and return its path.

    :param tmp_path: pytest temporary directory fixture
    :param metadata: Metadata payload to write as JSON
    :return: Path to the written metadata file
    """
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    return path


def test_check_checksum_detects_corruption(tmp_path: Path) -> None:
    """
    A bit-flip in the file body trips the full-digest comparison.

    :param tmp_path: pytest temporary directory fixture
    """
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hello world")
    wrong = "0" * 64
    with pytest.raises(ModelLoadingError):
        check_checksum(path, wrong)


def test_check_checksum_passes_clean_file(tmp_path: Path) -> None:
    """
    A correctly-hashed file passes through silently.

    :param tmp_path: pytest temporary directory fixture
    """
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hello world")
    digest = sha256(b"hello world").hexdigest()
    # No exception → pass.
    check_checksum(path, digest)


def test_check_size_rejects_wrong_length(tmp_path: Path) -> None:
    """
    Trusted artifact sizes are enforced independently of checksums.
    """
    path = tmp_path / "blob.bin"
    path.write_bytes(b"1234")
    check_size(path, 4)
    with pytest.raises(ModelLoadingError, match="expected 5 bytes"):
        check_size(path, 5)


def test_demucs_download_rejects_wrong_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A declared size mismatch fails before model bytes are streamed.
    """

    class Response:
        """
        Minimal streaming response with a bad declared length.
        """

        headers = {"content-length": "5"}

        def __enter__(self):
            """
            Enter the fake response context.
            """
            return self

        def __exit__(self, *_args: object) -> None:
            """
            Leave the fake response context.
            """

        def raise_for_status(self) -> None:
            """
            Represent a successful HTTP status.
            """

        def iter_bytes(self, chunk_size: int):
            """
            Yield no bytes because the header should reject first.
            """
            del chunk_size
            return iter(())

    monkeypatch.setattr("unblend.repo.httpx.stream", lambda *_a, **_k: Response())
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    with pytest.raises(ModelLoadingError, match="expected 4"):
        repo._download_verified_file(
            "https://example.invalid/model",
            tmp_path / "cache" / "model.safetensors",
            "0" * 64,
            4,
        )


def test_roformer_download_rejects_chunked_size_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A chunked response cannot exceed metadata's expected artifact size.
    """

    class Response:
        """
        Minimal chunked response that lies by omitting Content-Length.
        """

        headers: dict[str, str] = {}

        def __enter__(self):
            """
            Enter the fake response context.
            """
            return self

        def __exit__(self, *_args: object) -> None:
            """
            Leave the fake response context.
            """

        def raise_for_status(self) -> None:
            """
            Represent a successful HTTP status.
            """

        def iter_bytes(self, chunk_size: int):
            """
            Yield five bytes against a four-byte limit.
            """
            del chunk_size
            return iter((b"123", b"45"))

    monkeypatch.setattr("unblend.repo.httpx.stream", lambda *_a, **_k: Response())
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    cache_path = tmp_path / "cache" / "model.safetensors"
    with pytest.raises(ModelLoadingError, match="exceeded"):
        repo._download_verified_file(
            "https://example.invalid/model", cache_path, "0" * 64, 4
        )
    assert not cache_path.exists()
    assert not list(cache_path.parent.glob("tmp*"))


def test_repository_rejects_short_sha256(tmp_path: Path) -> None:
    """
    A metadata entry with anything other than a full hexadecimal ``sha256``
    is rejected before any artifact can be loaded.

    :param tmp_path: pytest temporary directory fixture
    """
    bad = _good_metadata()
    bad["models"]["fakemodel"]["checkpoint"]["sha256"] = "a" * 8
    with pytest.raises(ModelLoadingError, match="sha256"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_repository_rejects_missing_sources(tmp_path: Path) -> None:
    """
    Metadata without a ``sources`` list is rejected — only_load resolution
    depends on it being available without downloading a layer first.

    :param tmp_path: pytest temporary directory fixture
    """
    bad = _good_metadata()
    del bad["models"]["fakemodel"]["sources"]
    with pytest.raises(ModelLoadingError, match="sources"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_repository_rejects_missing_models_top_key(tmp_path: Path) -> None:
    """
    Metadata without the top-level ``models`` key is rejected.

    :param tmp_path: pytest temporary directory fixture
    """
    with pytest.raises(ModelLoadingError, match="models"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, {"other": {}}))


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {"members": []},
        {"models": {"bad": []}},
        {"models": {"bad": {"backend": "unknown", "sources": ["x"]}}},
    ],
)
def test_repository_rejects_malformed_containers(
    tmp_path: Path, metadata: object
) -> None:
    """
    Malformed custom metadata always raises the package error type.
    """
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ModelLoadingError):
        ModelRepository(metadata_path=path)


def test_repository_rejects_empty_demucs_layers(tmp_path: Path) -> None:
    """
    A Demucs entry must contain at least one Safetensors artifact.
    """
    bad = _good_metadata()
    entry = bad["models"]["fakemodel"]
    del entry["checkpoint"]
    entry["members"] = []
    with pytest.raises(ModelLoadingError, match="at least two members"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_repository_rejects_malformed_roformer_fields(tmp_path: Path) -> None:
    """
    RoFormer architecture/config fields are validated before download.
    """
    bad = {
        "models": {
            "bad": {
                "architecture": ["bs_roformer"],
                "config": [],
                "sources": ["vocals"],
                "samplerate": 44100,
                "segment_samples": 44100,
                "checkpoint": {},
            }
        }
    }
    with pytest.raises(ModelLoadingError, match="architecture"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


@pytest.mark.parametrize("missing", ["samplerate", "segment_samples"])
def test_repository_wraps_missing_roformer_geometry(
    tmp_path: Path, missing: str
) -> None:
    """
    Missing required geometry raises ModelLoadingError, never raw KeyError.
    """
    entry = {
        "architecture": "bs_roformer",
        "config": {"dim": 16},
        "sources": ["vocals"],
        "samplerate": 44100,
        "segment_samples": 44100,
        "checkpoint": {
            "format": "safetensors",
            "url": "https://example.invalid/model.safetensors",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
    }
    entry.pop(missing)
    with pytest.raises(ModelLoadingError, match=missing):
        ModelRepository(
            metadata_path=_write_metadata(tmp_path, {"models": {"bad": entry}})
        )


def test_repository_accepts_well_formed_metadata(tmp_path: Path) -> None:
    """
    A correctly-shaped metadata file constructs cleanly.

    :param tmp_path: pytest temporary directory fixture
    """
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    expected = _good_metadata()["models"]
    listed = repo.list_models()
    assert set(listed) == set(expected)
    # ``backend`` is derived from ``architecture`` and added to the entry;
    # nothing else about a well-formed entry is rewritten.
    assert listed["fakemodel"] == {**expected["fakemodel"], "backend": "demucs"}


def test_get_cache_info_empty_cache(tmp_path: Path, monkeypatch: object) -> None:
    """
    ``get_cache_info`` returns an empty mapping when no layer files are on
    disk — no spurious zero-byte entries.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: cache_dir)
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    assert repo.get_cache_info() == {}


def test_get_cache_info_lists_present_layers(
    tmp_path: Path, monkeypatch: object
) -> None:
    """
    When a layer's cache file exists, ``get_cache_info`` reports its path and
    size. The summary aggregates only the *present* layers.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: cache_dir)
    (cache_dir / f"{FIRST_KEY}.safetensors").write_bytes(b"x" * 1024)

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    info = repo.get_cache_info()
    assert "fakemodel" in info
    assert info["fakemodel"]["size_bytes"] == 1024


def test_remove_model_returns_false_for_unknown(tmp_path: Path) -> None:
    """
    Removing a model not registered in metadata is a no-op returning False.

    :param tmp_path: pytest temporary directory fixture
    """
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    assert repo.remove_model("doesnotexist") is False


def test_remove_model_unlinks_cached_layers(
    tmp_path: Path, monkeypatch: object
) -> None:
    """
    ``remove_model`` deletes every cached layer file for the model and
    returns True; absent files are tolerated (only-load partial caches stay
    consistent).

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: cache_dir)
    layer = cache_dir / f"{FIRST_KEY}.safetensors"
    layer.write_bytes(b"x")

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    assert repo.remove_model("fakemodel") is True
    assert not layer.exists()


def test_get_model_redownloads_corrupt_cached_layer(
    tmp_path: Path, monkeypatch: object
) -> None:
    """
    A corrupt file already in the cache makes ``get_model`` discard it and
    re-invoke ``_download_and_load_layer``. The cache file is removed before
    the download runs so the next read isn't a stale half-correct blob.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    # Build a known-good repo against fake metadata. The first layer's
    # sha256 expects the registered digest, so any other content trips
    # check_checksum and exercises the redownload branch.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: cache_dir)

    corrupt_path = cache_dir / f"{FIRST_KEY}.safetensors"
    corrupt_path.write_bytes(b"this is not a real model checkpoint")

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))

    download_calls: list[dict] = []

    def fake_download_verified_file(self, **kwargs):
        """
        Record the call rather than hit the network.

        :param self: bound ``ModelRepository`` instance
        :param kwargs: forwarded download kwargs
        """
        download_calls.append(kwargs)

    monkeypatch.setattr(
        ModelRepository, "_download_verified_file", fake_download_verified_file
    )

    # get_model swallows the bad cache hit, removes the file, then hits the
    # mocked-out download path. After the recovery, the bag-of-models
    # assembly tries to introspect the placeholder and fails — that's fine;
    # we only need to verify the corrupt cache file is gone and the download
    # was attempted.
    try:
        repo.get_model("fakemodel")
    except Exception:
        pass

    assert not corrupt_path.exists(), (
        "Corrupt cache file should be unlinked before redownload"
    )
    assert len(download_calls) == 1
    assert download_calls[0]["cache_path"] == corrupt_path
    assert download_calls[0]["expected_sha256"] == FIRST_SHA


def test_repository_instances_coordinate_one_artifact_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Two repository instances recheck under one cross-process file lock.
    """
    cache = tmp_path / "cache"
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: cache)
    monkeypatch.setattr("unblend.repo.check_checksum", lambda *_args: None)
    model = SimpleNamespace(
        sources=["drums", "bass", "other", "vocals"], max_allowed_segment=1.0
    )
    # The fake download writes placeholder bytes, so stub reading and building.
    monkeypatch.setattr("unblend.repo._read_state", lambda _path: {})
    monkeypatch.setattr("unblend.repo._build_demucs_layer", lambda *_args: model)
    calls = 0
    calls_lock = threading.Lock()

    def fake_download(self: ModelRepository, **kwargs: object) -> None:
        """
        Populate the cache slowly enough for the other thread to wait.
        """
        nonlocal calls
        del self
        with calls_lock:
            calls += 1
        time.sleep(0.1)
        path = kwargs["cache_path"]
        assert isinstance(path, Path)
        path.write_bytes(b"x" * 1024)

    monkeypatch.setattr(ModelRepository, "_download_verified_file", fake_download)
    metadata_path = _write_metadata(tmp_path, _good_metadata())
    repos = [ModelRepository(metadata_path=metadata_path) for _ in range(2)]
    barrier = threading.Barrier(3)
    results: list[object] = []
    errors: list[BaseException] = []

    def load(repo: ModelRepository) -> None:
        barrier.wait()
        try:
            results.append(repo.get_model("fakemodel"))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=load, args=(repo,)) for repo in repos]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert len(results) == 2
    assert calls == 1


def test_only_load_requires_exclusive_specialist_weight(tmp_path: Path) -> None:
    """
    Repository cannot skip another layer that contributes to the stem.
    """
    metadata = _good_metadata()
    entry = metadata["models"]["fakemodel"]
    entry["members"] = [
        entry.pop("checkpoint"),
        {
            "format": "safetensors",
            "url": "https://example.invalid/ef.safetensors",
            "sha256": SECOND_SHA,
            "size_bytes": 2,
        },
    ]
    metadata["models"]["fakemodel"]["weights"] = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))
    assert repo.required_files("fakemodel", only_load="drums") == [
        FIRST_KEY,
        SECOND_KEY,
    ]


@pytest.mark.parametrize("stem", ["", "not-a-stem"])
def test_get_model_rejects_only_load_before_cache_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stem: str
) -> None:
    """
    Direct repository callers fail fast for invalid or empty stems.
    """
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    monkeypatch.setattr(
        repo,
        "_download_verified_file",
        lambda **_kwargs: pytest.fail("invalid only_load touched the downloader"),
    )
    with pytest.raises(ModelLoadingError, match="not found"):
        repo.get_model("fakemodel", only_load=stem)
    with pytest.raises(ModelLoadingError, match="not found"):
        repo.required_files("fakemodel", only_load=stem)


def test_artifact_lock_wraps_acquisition_filesystem_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Lock-path failures become stable repository-domain errors.
    """
    from unblend import repo as repo_module

    class BrokenLock:
        def __init__(self, _path: Path) -> None:
            pass

        def acquire(self, *, timeout: int) -> None:
            raise PermissionError("read-only cache")

    monkeypatch.setattr(repo_module, "FileLock", BrokenLock)
    with pytest.raises(ModelLoadingError, match="Could not create/acquire") as caught:
        with repo_module._artifact_lock(tmp_path / "cache" / "model.safetensors"):
            pytest.fail("lock acquisition unexpectedly succeeded")
    assert isinstance(caught.value.__cause__, PermissionError)


def test_roformer_materializes_state_while_cache_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A concurrent remove cannot unlink a checkpoint during load_file.
    """
    from contextlib import contextmanager

    from unblend import repo as repo_module

    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"registered")
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    metadata = {
        "models": {
            "tiny": {
                "architecture": "bs_roformer",
                "sources": ["vocals", "other"],
                "samplerate": 8000,
                "segment_samples": 8000,
                "config": {"dim": 1},
                "checkpoint": {
                    "format": "safetensors",
                    "url": "https://example.invalid/tiny.safetensors",
                    "sha256": digest,
                    "size_bytes": checkpoint.stat().st_size,
                },
            }
        }
    }
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))
    cache_path = repo_module._artifact_cache_path(
        repo.list_models()["tiny"]["checkpoint"]
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(checkpoint.read_bytes())

    locked = False

    @contextmanager
    def tracked_lock(_path: Path):
        nonlocal locked
        locked = True
        try:
            yield
        finally:
            locked = False

    def fake_load_file(path: Path, *, device: str) -> dict:
        assert path == cache_path
        assert device == "cpu"
        assert locked
        return {"weight": object()}

    monkeypatch.setattr(repo_module, "_artifact_lock", tracked_lock)
    monkeypatch.setattr(repo_module, "load_file", fake_load_file)
    monkeypatch.setattr(
        repo_module.backends,
        "build",
        lambda *_args, state, **_kwargs: (assert_state_materialized(state, locked)),
    )

    def assert_state_materialized(state: dict, still_locked: bool) -> SimpleNamespace:
        assert state == {"weight": state["weight"]}
        assert not still_locked
        return SimpleNamespace(sources=["vocals", "other"])

    loaded = repo.get_model("tiny")
    assert loaded.sources == ["vocals", "other"]


def test_get_cache_dir_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``UNBLEND_CACHE_DIR`` relocates the model cache away from ``~/.unblend``,
    with tilde expansion (Docker ENV / systemd values are not shell-expanded),
    and without creating the directory (that happens on first download).
    """
    # Resolved on both sides: the override is resolved, and tmp or HOME may
    # sit behind a symlink (/tmp on macOS, /home on some clusters).
    target = tmp_path / "custom-cache"
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(target))
    assert get_cache_dir() == target.resolve()
    assert not target.exists()

    monkeypatch.setenv("UNBLEND_CACHE_DIR", "~/some-demucs-cache")
    assert get_cache_dir() == (Path.home() / "some-demucs-cache").resolve()


def test_get_cache_info_reports_partial_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A model with some but not all layers cached is reported with
    ``complete: False`` and the cached subset's size, so ``models list`` and
    ``models remove --all`` account for its disk usage.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(cache))

    metadata = _good_metadata()
    entry = metadata["models"]["fakemodel"]
    entry["members"] = [
        entry.pop("checkpoint"),
        {
            "format": "safetensors",
            "url": "https://example.invalid/ef.safetensors",
            "sha256": SECOND_SHA,
            "size_bytes": 2,
        },
    ]
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))

    assert repo.get_cache_info() == {}

    (cache / f"{FIRST_KEY}.safetensors").write_bytes(b"x" * 1024)
    info = repo.get_cache_info()
    assert info["fakemodel"]["complete"] is False
    assert info["fakemodel"]["total_files"] == 2
    assert info["fakemodel"]["size_bytes"] == 1024
    assert list(info["fakemodel"]["files"]) == [FIRST_KEY]

    # A truncated file is listed (removal must find it) but not complete.
    (cache / f"{SECOND_KEY}.safetensors").write_bytes(b"y")
    info = repo.get_cache_info()
    assert info["fakemodel"]["files"][SECOND_KEY]["complete"] is False
    assert info["fakemodel"]["complete"] is False

    (cache / f"{SECOND_KEY}.safetensors").write_bytes(b"yy")
    info = repo.get_cache_info()
    assert info["fakemodel"]["complete"] is True
    assert info["fakemodel"]["size_bytes"] == 1026


def test_sweep_stale_downloads_removes_staging_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Only expired staging files are swept; active files are preserved.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(cache))

    stale = cache / f"{STAGING_PREFIX}stale.tmp"
    active = cache / f"{STAGING_PREFIX}active.tmp"
    stale.write_bytes(b"abandoned")
    active.write_bytes(b"active")
    old = time.time() - STAGING_STALE_SECONDS - 1
    os.utime(stale, (old, old))
    cached = cache / f"{FIRST_KEY}.safetensors"
    cached.write_bytes(b"cached layer")

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    assert repo.sweep_stale_downloads() == 1
    assert not stale.exists()
    assert active.exists()
    assert cached.exists()


def test_list_models_returns_copies() -> None:
    """
    Mutating a ``list_models`` result must not corrupt repository state.
    """
    repo = ModelRepository()
    listed = repo.list_models()
    name = next(iter(listed))
    listed[name]["checkpoint"] = {}
    assert repo.list_models()[name]["checkpoint"], "internal metadata was mutated"


@pytest.mark.parametrize(
    "make_exc, expect_wrapped",
    [
        (
            lambda cause: ModelLoadingError("could not read for verification"),
            False,
        ),
        (lambda cause: OSError(5, "I/O error"), True),
    ],
    ids=["MLE-with-cause", "raw-OSError"],
)
def test_get_model_preserves_cache_on_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_exc: object,
    expect_wrapped: bool,
) -> None:
    """
    Read failures (OSError-caused or raw OSError) are not corruption: the
    cached file must be KEPT, no redownload attempted, and the error must
    leave ``get_model`` as ``ModelLoadingError`` (wrapped exactly once).

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param make_exc: Factory building the exception the cache load raises
    :param expect_wrapped: Whether get_model wraps it (vs re-raising as-is)
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: cache_dir)

    cached = cache_dir / f"{FIRST_KEY}.safetensors"
    cached.write_bytes(b"x" * 1024)

    cause = OSError(13, "Permission denied")
    exc = make_exc(cause)  # type: ignore[operator]

    def raise_exc(*_args: object, **_kwargs: object) -> None:
        """
        Patched ``check_checksum`` raising the parametrized read failure.

        :param _args: ignored positional arguments
        :param _kwargs: ignored keyword arguments
        :raises Exception: the parametrized exception (OSError-caused)
        """
        if isinstance(exc, OSError):
            raise exc
        raise exc from cause

    monkeypatch.setattr("unblend.repo.check_checksum", raise_exc)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """
        Downloader stub that fails the test if recovery wrongly triggers.

        :param _args: ignored positional arguments
        :param _kwargs: ignored keyword arguments
        """
        pytest.fail("read failure must not trigger a redownload")

    monkeypatch.setattr(ModelRepository, "_download_verified_file", fail_download)

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    with pytest.raises(ModelLoadingError) as excinfo:
        repo.get_model("fakemodel")

    assert cached.exists(), "read failure must not unlink the cached file"
    if expect_wrapped:
        assert excinfo.value is not exc
        assert excinfo.value.__cause__ is exc
    else:
        assert excinfo.value is exc


def test_normal_import_does_not_install_demucs_aliases() -> None:
    """
    Ordinary package import coexists with a separately installed Demucs.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import unblend; "
            "assert 'demucs' not in sys.modules; "
            "assert 'demucs.htdemucs' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registered_layer_loads_safetensors_without_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Registered weights build strictly without calling ``torch.load``.
    """
    import torch
    from safetensors.torch import save_file

    from unblend import repo as repo_module
    from unblend.htdemucs import HTDemucs

    config = dict(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )
    model = HTDemucs(**config)
    packed = tmp_path / "layer.safetensors"
    save_file(dict(model.state_dict()), packed)
    digest = sha256(packed.read_bytes()).hexdigest()

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / f"{digest[:16]}.safetensors"
    cached.write_bytes(packed.read_bytes())
    monkeypatch.setattr(repo_module, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("registered loading used pickle"),
    )

    metadata = {
        "models": {
            "tiny": {
                "architecture": "htdemucs",
                "sources": ["a", "b"],
                "config": config,
                "checkpoint": {
                    "format": "safetensors",
                    "url": "https://example.invalid/layer.safetensors",
                    "sha256": digest,
                    "size_bytes": packed.stat().st_size,
                },
            }
        }
    }
    metadata_path = _write_metadata(tmp_path, metadata)

    repo = repo_module.ModelRepository(metadata_path)
    loaded = repo.get_model("tiny")
    assert isinstance(loaded, HTDemucs)
    assert loaded.sources == ["a", "b"]


def _tiny_demucs_layer(
    tmp_path: Path, sources: list[str] | None = None
) -> tuple[Path, dict]:
    """
    Save a small real HTDemucs checkpoint for the custom-model tests.

    :param tmp_path: pytest temporary directory fixture
    :param sources: Stems it should emit; two by default
    :return: ``(checkpoint path, constructor config)``
    """
    from safetensors.torch import save_file

    from unblend.htdemucs import HTDemucs

    config = dict(
        sources=list(sources) if sources else ["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )
    path = tmp_path / "layer.safetensors"
    save_file(dict(HTDemucs(**config).state_dict()), path)
    return path, config


def _tiny_scnet_checkpoint(tmp_path: Path, sources: list[str]) -> tuple[Path, dict]:
    """
    Save a small real SCNet, for testing ensembles that mix architectures.

    :param tmp_path: pytest temporary directory fixture
    :param sources: Stems the model should emit
    :return: ``(checkpoint path, constructor config)``
    """
    from safetensors.torch import save_file

    from unblend.scnet import SCNet

    config = dict(
        audio_channels=2,
        dims=[4, 8, 16, 32],
        nfft=512,
        hop_size=128,
        win_size=512,
        band_stride=[1, 2, 4],
        band_kernel=[3, 4, 4],
        conv_depths=[1, 1, 1],
        num_dplayer=2,
    )
    model = SCNet(sources=list(sources), **config)
    path = tmp_path / "scnet.safetensors"
    save_file(
        {key: value.contiguous() for key, value in model.state_dict().items()}, path
    )
    return path, config


def _extra_models_file(tmp_path: Path, entry: dict) -> Path:
    """
    Write a one-model ``UNBLEND_EXTRA_MODELS`` file.

    :param tmp_path: pytest temporary directory fixture
    :param entry: The model entry to register as ``custom``
    :return: Path to the written file
    """
    path = tmp_path / "extra-models.json"
    path.write_text(json.dumps({"models": {"custom": entry}}))
    return path


def test_demucs_model_loads_from_a_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A user-supplied Demucs checkpoint loads from disk, with the backend
    derived from its architecture and its license label passed through
    untouched.
    """
    from unblend.htdemucs import HTDemucs

    weights, config = _tiny_demucs_layer(tmp_path)
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    extra = _extra_models_file(
        tmp_path,
        {
            "architecture": "htdemucs",
            "license": "my own terms",
            "sources": config["sources"],
            "config": config,
            "checkpoint": {"format": "safetensors", "path": str(weights)},
        },
    )

    repo = ModelRepository(extra_models=extra)
    listed = repo.list_models()["custom"]
    assert listed["backend"] == "demucs", "backend should follow from architecture"
    assert listed["license"] == "my own terms", "license is a pass-through label"

    model = repo.get_model("custom")
    assert isinstance(model, HTDemucs)
    assert model.sources == ["a", "b"]

    # A file the user owns is not cache: nothing to fetch, nothing to account
    # for, and ``models remove`` must never unlink it.
    assert repo.is_fully_local("custom")
    assert repo.local_artifacts("custom") == [weights]
    assert repo.required_files("custom") == []
    assert repo.get_cache_info() == {}
    assert repo.remove_model("custom") is False
    assert weights.is_file()


def test_demucs_layer_from_a_url_is_fetched_once_then_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An https layer is downloaded once into the content-addressed cache and
    served from there afterwards, so repeated runs don't refetch it.
    """
    weights, config = _tiny_demucs_layer(tmp_path)
    payload = weights.read_bytes()
    digest = sha256(payload).hexdigest()
    cache = tmp_path / "cache"
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(cache))
    url = "https://example.invalid/layer.safetensors"
    extra = _extra_models_file(
        tmp_path,
        {
            "architecture": "htdemucs",
            "sources": config["sources"],
            "config": config,
            "checkpoint": {
                "format": "safetensors",
                "url": url,
                "sha256": digest,
                "size_bytes": len(payload),
            },
        },
    )

    downloads: list[str] = []

    def fake_download(
        self: ModelRepository, *, url: str, cache_path: Path, **_: object
    ) -> None:
        """
        Stand in for the network by promoting known-good bytes.
        """
        del self
        downloads.append(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)

    monkeypatch.setattr(ModelRepository, "_download_verified_file", fake_download)

    repo = ModelRepository(extra_models=extra)
    # No explicit ``checksum``: the cache filename comes from the digest.
    cached = cache / f"{digest[:16]}.safetensors"
    assert repo.required_files("custom") == [digest[:16]]

    repo.get_model("custom")
    assert downloads == [url]
    assert cached.is_file()

    # A second repository re-verifies the cached bytes instead of refetching.
    ModelRepository(extra_models=extra).get_model("custom")
    assert downloads == [url]

    entry = repo.get_cache_info()["custom"]
    assert entry["complete"] is True
    assert entry["size_bytes"] == len(payload)
    assert repo.remove_model("custom") is True
    assert not cached.exists()


def test_mixed_local_and_remote_layers_only_account_for_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An ensemble may mix a local layer with a downloaded one; only the remote
    layer is something the cache can report on.
    """
    weights, config = _tiny_demucs_layer(tmp_path)
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    extra = _extra_models_file(
        tmp_path,
        {
            "architecture": "htdemucs",
            "sources": config["sources"],
            "config": config,
            "members": [
                {"format": "safetensors", "path": str(weights)},
                {
                    "format": "safetensors",
                    "url": "https://example.invalid/second.safetensors",
                    "sha256": "b" * 64,
                    "size_bytes": 8,
                },
            ],
        },
    )

    repo = ModelRepository(extra_models=extra)
    assert repo.is_fully_local("custom") is False
    assert repo.local_artifacts("custom") == [weights]
    assert repo.required_files("custom") == ["b" * 16]


def test_entry_without_a_known_architecture_is_rejected(tmp_path: Path) -> None:
    """
    An entry naming neither a backend nor a known architecture fails.
    """
    bad = {
        "models": {
            "mystery": {
                "architecture": "wavenet",
                "sources": ["a"],
                "config": {"a": 1},
                "checkpoint": {
                    "format": "safetensors",
                    "path": "/models/mystery.safetensors",
                },
            }
        }
    }
    with pytest.raises(ModelLoadingError, match="known architecture"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_declared_backend_is_rejected(tmp_path: Path) -> None:
    """
    ``backend`` is derived from ``architecture``, never declared.

    Accepting it too would mean two sources of truth that can disagree.
    """
    bad = _good_metadata()
    bad["models"]["fakemodel"]["backend"] = "demucs"
    with pytest.raises(ModelLoadingError, match="declares a .backend."):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_declared_backend_is_rejected_on_a_member(tmp_path: Path) -> None:
    """
    The same rule applies inside a member spec.

    :param tmp_path: pytest temporary directory fixture
    """
    bad = _good_metadata()
    entry = bad["models"]["fakemodel"]
    artifact = entry.pop("checkpoint")
    entry["members"] = [{"backend": "demucs", **artifact}, dict(artifact)]
    with pytest.raises(ModelLoadingError, match="declares a .backend."):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_remote_artifact_spelling_is_rejected(tmp_path: Path) -> None:
    """
    ``url`` is the only spelling; the old ``remote`` key names nothing.

    :param tmp_path: pytest temporary directory fixture
    """
    bad = _good_metadata()
    layer = bad["models"]["fakemodel"]["checkpoint"]
    layer["remote"] = layer.pop("url")
    with pytest.raises(ModelLoadingError, match="unknown field.*remote"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_single_checkpoint_entries_are_validated_at_construction(
    tmp_path: Path,
) -> None:
    """
    Every backend's artifacts are checked up front, so an SCNet entry with an
    unverifiable download fails at construction rather than mid-``get_model``.
    """
    bad = {
        "models": {
            "custom_scnet": {
                "architecture": "scnet_masked",
                "sources": ["vocals", "other"],
                "samplerate": 44100,
                "segment_samples": 44100,
                "config": {"dims": [4, 8]},
                "checkpoint": {
                    "format": "safetensors",
                    "url": "https://example.invalid/x.safetensors",
                },
            }
        }
    }
    with pytest.raises(ModelLoadingError, match="sha256"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


@pytest.mark.parametrize(
    "checkpoint, expected",
    [
        (
            {
                "format": "safetensors",
                "url": "http://example.invalid/x.safetensors",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "https",
        ),
        (
            {
                "format": "safetensors",
                "path": "/models/x.safetensors",
                "url": "https://example.invalid/x.safetensors",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "pick one",
        ),
        ({"format": "safetensors"}, "local path or an https url"),
        ({"format": "pickle", "path": "/models/x.pt"}, "Safetensors"),
    ],
    ids=["plain-http", "both-sources", "no-source", "not-safetensors"],
)
def test_artifact_source_rules_are_enforced(
    tmp_path: Path, checkpoint: dict, expected: str
) -> None:
    """
    Every artifact is one Safetensors file, named locally or over https.
    """
    bad = {
        "models": {
            "custom_scnet": {
                "architecture": "scnet",
                "sources": ["vocals", "other"],
                "samplerate": 44100,
                "segment_samples": 44100,
                "config": {"dims": [4, 8]},
                "checkpoint": checkpoint,
            }
        }
    }
    with pytest.raises(ModelLoadingError, match=expected):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def _local_htdemucs_entry(weights: Path, config: dict) -> dict:
    """
    A one-checkpoint entry for the tiny local HTDemucs.

    :param weights: Path to the saved checkpoint.
    :param config: Its constructor config.
    :return: A registry entry.
    """
    return {
        "architecture": "htdemucs",
        "sources": config["sources"],
        "config": config,
        "checkpoint": {"format": "safetensors", "path": str(weights)},
    }


def test_ensemble_members_build_and_honour_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A ``members`` list builds an ensemble; a one-hot column still collapses to
    the single contributing member when a stem is isolated.
    """
    from unblend.apply import ModelEnsemble
    from unblend.htdemucs import HTDemucs

    weights, config = _tiny_demucs_layer(tmp_path)
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    artifact = {"format": "safetensors", "path": str(weights)}
    extra = _extra_models_file(
        tmp_path,
        {
            "architecture": "htdemucs",
            "sources": config["sources"],
            "config": config,
            "members": [{"checkpoint": artifact}, {"checkpoint": artifact}],
            "weights": [[1.0, 0.0], [0.0, 1.0]],
        },
    )

    repo = ModelRepository(extra_models=extra)
    ensemble = repo.get_model("custom")
    assert isinstance(ensemble, ModelEnsemble)
    assert len(ensemble.models) == 2

    # One contributor for stem "b", so isolating it builds that member alone.
    isolated = repo.get_model("custom", only_load="b")
    assert isinstance(isolated, HTDemucs)
    assert repo.required_files("custom", only_load="b") == []


def test_member_can_reference_another_registered_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An ensemble can be built out of models that are already registered, which
    is how a user combines shipped models without restating their config.
    """
    from unblend.apply import ModelEnsemble

    weights, config = _tiny_demucs_layer(tmp_path)
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    extra = tmp_path / "extra-models.json"
    extra.write_text(
        json.dumps(
            {
                "models": {
                    "base": _local_htdemucs_entry(weights, config),
                    "pair": {
                        "sources": config["sources"],
                        "combine": "max_wave",
                        "members": [{"model": "base"}, {"model": "base"}],
                    },
                }
            }
        )
    )

    repo = ModelRepository(extra_models=extra)
    ensemble = repo.get_model("pair")
    assert isinstance(ensemble, ModelEnsemble)
    assert len(ensemble.models) == 2
    assert ensemble.combine_mode == "max_wave"
    # The referenced model's own fields carried over, unstated by the ensemble.
    assert repo.list_models()["pair"]["backend"] == "demucs"
    assert repo.is_fully_local("pair")


def test_heterogeneous_members_report_the_ensemble_backend(tmp_path: Path) -> None:
    """
    An entry whose members are built by different families has no single
    backend, so it reports ``ensemble``.
    """
    metadata = {
        "models": {
            "mixed": {
                "sources": ["vocals", "other"],
                "members": [
                    {
                        "architecture": "htdemucs",
                        "config": {"sources": ["vocals", "other"]},
                        "checkpoint": {
                            "format": "safetensors",
                            "path": "/models/htdemucs.safetensors",
                        },
                    },
                    {
                        "architecture": "mel_band_roformer",
                        "config": {"dim": 16, "stereo": True},
                        "samplerate": 44100,
                        "segment_samples": 44100,
                        "checkpoint": {
                            "format": "safetensors",
                            "path": "/models/melband.safetensors",
                        },
                    },
                ],
            }
        }
    }
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))
    assert repo.list_models()["mixed"]["backend"] == "ensemble"


def test_ensemble_can_mix_architectures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An HTDemucs and an SCNet combine in one ensemble even though they disagree
    about normalisation: the ensemble takes raw audio and normalises around the
    member that wants it.
    """
    from unblend.apply import ModelEnsemble

    stems = ["drums", "bass", "other", "vocals"]
    demucs_weights, demucs_config = _tiny_demucs_layer(tmp_path, stems)
    scnet_weights, scnet_config = _tiny_scnet_checkpoint(tmp_path, stems)
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "cache"))
    extra = _extra_models_file(
        tmp_path,
        {
            "sources": stems,
            "combine": "max_wave",
            "weights": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            "members": [
                {
                    "architecture": "htdemucs",
                    "config": demucs_config,
                    "checkpoint": {
                        "format": "safetensors",
                        "path": str(demucs_weights),
                    },
                },
                {
                    "architecture": "scnet",
                    "config": scnet_config,
                    "samplerate": demucs_config["samplerate"],
                    "segment_samples": 4096,
                    "checkpoint": {
                        "format": "safetensors",
                        "path": str(scnet_weights),
                    },
                },
            ],
        },
    )

    repo = ModelRepository(extra_models=extra)
    assert repo.list_models()["custom"]["backend"] == "ensemble"

    ensemble = repo.get_model("custom")
    assert isinstance(ensemble, ModelEnsemble)
    # HTDemucs wants normalised audio, SCNet raw, so the caller supplies raw
    # and the HTDemucs member is normalised around its own pass.
    assert ensemble.member_normalization == [True, False]
    assert ensemble.external_normalization is False

    from unblend.htdemucs import HTDemucs
    from unblend.scnet import SCNet

    assert isinstance(ensemble.models[0], HTDemucs)
    assert isinstance(ensemble.models[1], SCNet)
    assert ensemble.sources == stems
    # What the members then receive is covered exactly, on affine stand-ins,
    # by test_apply's normalisation tests.


@pytest.mark.parametrize(
    "entries, expected",
    [
        (
            {
                "a": {
                    "sources": ["x"],
                    "members": [{"model": "nope"}, {"model": "nope"}],
                }
            },
            "references unknown model",
        ),
        (
            {
                "a": {"sources": ["x"], "members": [{"model": "b"}, {"model": "b"}]},
                "b": {"sources": ["x"], "members": [{"model": "a"}, {"model": "a"}]},
            },
            "reference cycle",
        ),
    ],
    ids=["unknown-reference", "cycle"],
)
def test_bad_member_references_are_rejected(
    tmp_path: Path, entries: dict, expected: str
) -> None:
    """
    A member reference must name a real, single, non-recursive model.
    """
    with pytest.raises(ModelLoadingError, match=expected):
        ModelRepository(metadata_path=_write_metadata(tmp_path, {"models": entries}))


def test_reference_to_an_ensemble_is_rejected(tmp_path: Path) -> None:
    """
    Members must be single models: nesting ensembles is not supported.
    """
    weights, config = _tiny_demucs_layer(tmp_path)
    metadata = {
        "models": {
            "base": _local_htdemucs_entry(weights, config),
            "pair": {
                "sources": config["sources"],
                "members": [{"model": "base"}, {"model": "base"}],
            },
            "nested": {
                "sources": config["sources"],
                "members": [{"model": "pair"}, {"model": "base"}],
            },
        }
    }
    with pytest.raises(ModelLoadingError, match="itself an ensemble"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))


def test_referenced_member_must_emit_the_same_stems(tmp_path: Path) -> None:
    """
    Members have to agree on stem names and order, so mismatches fail early.
    """
    weights, config = _tiny_demucs_layer(tmp_path)
    metadata = {
        "models": {
            "base": _local_htdemucs_entry(weights, config),
            "pair": {
                "sources": list(reversed(config["sources"])),
                "members": [{"model": "base"}, {"model": "base"}],
            },
        }
    }
    with pytest.raises(ModelLoadingError, match="same stems in the same order"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))


def test_entry_must_name_its_weights_exactly_once(tmp_path: Path) -> None:
    """
    ``checkpoint`` and ``members`` are mutually exclusive.
    """
    weights, config = _tiny_demucs_layer(tmp_path)
    entry = _local_htdemucs_entry(weights, config)
    entry["members"] = [{"checkpoint": entry["checkpoint"]}]
    with pytest.raises(ModelLoadingError, match="exactly one of"):
        ModelRepository(
            metadata_path=_write_metadata(tmp_path, {"models": {"custom": entry}})
        )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"weights": [[1.0, 1.0]]}, "one weight row per member"),
        ({"weights": [[1.0], [1.0]]}, "must contain 2 source weights"),
        ({"weights": [[1.0, 0.0], [1.0, 0.0]]}, "no member contributing"),
        (
            {"weights": [[0.5, 1.0], [1.0, 1.0]], "combine": "min_fft"},
            "participation mask",
        ),
        ({"combine": "telepathy"}, "Unknown ensemble combine mode"),
        (
            {"combine": "min_fft", "combine_params": {"n_fft": 1000}},
            "whole multiple",
        ),
    ],
    ids=[
        "row-count",
        "row-width",
        "orphan-stem",
        "non-binary-mask",
        "unknown-mode",
        "bad-geometry",
    ],
)
def test_ensemble_combination_is_validated_offline(
    tmp_path: Path, overrides: dict, expected: str
) -> None:
    """
    How members combine is checked when the repository is built, so a bad
    recipe never survives to a download.
    """
    weights, config = _tiny_demucs_layer(tmp_path)
    artifact = {"format": "safetensors", "path": str(weights)}
    entry = {
        "architecture": "htdemucs",
        "sources": config["sources"],
        "config": config,
        "members": [{"checkpoint": artifact}, {"checkpoint": artifact}],
        **overrides,
    }
    with pytest.raises(ModelLoadingError, match=expected):
        ModelRepository(
            metadata_path=_write_metadata(tmp_path, {"models": {"custom": entry}})
        )


def test_default_models_file_is_loaded_without_env(
    _isolate_default_models_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``~/.unblend/models.yaml`` (where ``models import`` writes) is read even
    when ``UNBLEND_EXTRA_MODELS`` is unset, and listing it there too doesn't
    load it twice.

    :param _isolate_default_models_file: the substituted default file path
    :param monkeypatch: pytest monkeypatch fixture
    """
    path = _isolate_default_models_file
    entry = {
        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
        "sources": ["drums", "bass", "other", "vocals"],
    }
    path.write_text(json.dumps({"models": {"custom": entry}}))
    monkeypatch.delenv("UNBLEND_EXTRA_MODELS", raising=False)
    assert "custom" in ModelRepository().list_models()

    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(path))
    assert "custom" in ModelRepository().list_models()


@pytest.mark.parametrize(
    "payload",
    [{"models": []}, {"models": {"broken": {"architecture": "htdemucs"}}}],
    ids=["not-a-mapping", "invalid-entry"],
)
def test_a_broken_default_models_file_is_skipped_with_a_warning(
    _isolate_default_models_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
) -> None:
    """
    The implicitly loaded default file can't take the built-in models down.

    :param _isolate_default_models_file: the substituted default file path
    :param monkeypatch: pytest monkeypatch fixture
    :param payload: A models file that fails validation
    """
    monkeypatch.delenv("UNBLEND_EXTRA_MODELS", raising=False)
    _isolate_default_models_file.write_text(json.dumps(payload))
    with pytest.warns(UserWarning, match="Ignoring"):
        models = ModelRepository().list_models()
    assert "htdemucs" in models


def test_default_file_clashing_with_an_env_file_is_skipped(
    _isolate_default_models_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A name defined both in an env-listed file and the default file doesn't
    break the registry; the explicit file wins.

    :param _isolate_default_models_file: the substituted default file path
    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    entry = {
        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
        "sources": ["drums", "bass", "other", "vocals"],
    }
    env_file = tmp_path / "env.json"
    env_file.write_text(json.dumps({"models": {"dup": entry}}))
    _isolate_default_models_file.write_text(json.dumps({"models": {"dup": entry}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(env_file))
    with pytest.warns(UserWarning, match="already registered"):
        assert "dup" in ModelRepository().list_models()


def test_remove_model_keeps_files_other_models_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Removing an ensemble doesn't delete its members' weights unless asked to,
    or unless every model using them is being removed too.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    from unblend.repo import _artifact_cache_path

    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: tmp_path)
    repo = ModelRepository(extra_models=[])
    paths = [
        _artifact_cache_path(s) for s in repo._artifacts("roformer_vocals_ensemble")
    ]

    def populate() -> None:
        """
        Put a placeholder file at every member's cache path.
        """
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    populate()
    assert set(repo.shared_artifacts("roformer_vocals_ensemble")) == set(paths)
    assert repo.remove_model("roformer_vocals_ensemble") is False
    assert all(path.exists() for path in paths)

    members = ["melband_roformer_kim", "bs_roformer_anvuew"]
    assert repo.remove_model("roformer_vocals_ensemble", also_removing=members)
    assert not any(path.exists() for path in paths)

    populate()
    assert repo.remove_model("roformer_vocals_ensemble", include_shared=True)
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize("segment", ["abc", -1, float("nan"), True, 1e-5])
def test_invalid_segment_is_rejected_at_construction(
    tmp_path: Path, segment: object
) -> None:
    """
    ``segment`` is validated with everything else, not at load or apply time.

    :param tmp_path: pytest temporary directory fixture
    :param segment: An invalid segment value
    """
    entry = {
        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
        "sources": ["drums", "bass", "other", "vocals"],
        "segment": segment,
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"models": {"bad": entry}}))
    with pytest.raises(
        ModelLoadingError, match="invalid segment|shorter than one sample"
    ):
        ModelRepository(extra_models=path)


def test_a_huge_ensemble_segment_loads_as_no_cap(tmp_path: Path) -> None:
    """
    ``segment: 1e307`` overflows when multiplied by a sample rate; it must
    load (a cap above every member's own), not crash every command.

    :param tmp_path: pytest temporary directory fixture
    """
    entry = {
        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
        "sources": ["drums", "bass", "other", "vocals"],
        "segment": 1e307,
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"models": {"huge": entry}}))
    assert "huge" in ModelRepository(extra_models=path).list_models()


def test_ensemble_members_with_different_sample_rates_fail_before_download(
    tmp_path: Path,
) -> None:
    """
    Declared geometry mismatches are caught at construction, not after every
    member has downloaded.

    :param tmp_path: pytest temporary directory fixture
    """
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    member_keys = {"architecture", "checkpoint", "config", "segment_samples"}
    other = {k: v for k, v in base.items() if k in member_keys}
    other["samplerate"] = 48000
    entry = {
        "members": [{"model": "scnet_small"}, other],
        "sources": base["sources"],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"models": {"mixed": entry}}))
    with pytest.raises(ModelLoadingError, match="disagree on sample rate"):
        ModelRepository(extra_models=path)


def test_relative_artifact_paths_resolve_against_the_models_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``path: weights.safetensors`` means the file next to the models file, not
    one in whatever directory unblend happens to run from.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    path = models_dir / "m.json"
    entry = {
        "architecture": "scnet",
        "sources": ["a"],
        "samplerate": 44100,
        "segment_samples": 44100,
        "config": {"dims": [4]},
        "checkpoint": {"format": "safetensors", "path": "w.safetensors"},
    }
    path.write_text(json.dumps({"models": {"rel": entry}}))
    monkeypatch.chdir(tmp_path)
    repo = ModelRepository(extra_models=path)
    assert repo.local_artifacts("rel") == [(models_dir / "w.safetensors").resolve()]


def test_missing_local_checkpoint_says_so(tmp_path: Path) -> None:
    """
    An entry that relies on its checkpoint's header, whose file is missing,
    reports the missing file rather than an unknown architecture.

    :param tmp_path: pytest temporary directory fixture
    """
    path = tmp_path / "m.json"
    entry = {
        "sources": ["a"],
        "checkpoint": {
            "format": "safetensors",
            "path": str(tmp_path / "gone.safetensors"),
        },
    }
    path.write_text(json.dumps({"models": {"gone": entry}}))
    with pytest.raises(ModelLoadingError, match="checkpoint file not found"):
        ModelRepository(extra_models=path)


def test_remove_model_ignores_ensembles_that_were_never_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``htdemucs``'s file is also an ensemble member, but if that ensemble's other
    member isn't cached nobody else is using it, so removal deletes it.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    from unblend.repo import _artifact_cache_path

    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: tmp_path)
    repo = ModelRepository(extra_models=[])
    (path,) = [_artifact_cache_path(s) for s in repo._artifacts("htdemucs")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    assert repo.shared_artifacts("htdemucs") == {}
    assert repo.remove_model("htdemucs") is True
    assert not path.exists()


@pytest.mark.parametrize("names", [["foo", "FOO"], ["auto"]])
def test_models_file_rejects_case_duplicates_and_auto(
    tmp_path: Path, names: list[str]
) -> None:
    """
    Names that collide case-insensitively, or the reserved ``auto``, fail.

    :param tmp_path: pytest temporary directory fixture
    :param names: Entry names to write
    """
    entry = {
        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
        "sources": ["drums", "bass", "other", "vocals"],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"models": {n: entry for n in names}}))
    with pytest.raises(ModelLoadingError):
        ModelRepository(extra_models=path)


def test_models_file_rejects_unknown_fields_and_versions(tmp_path: Path) -> None:
    """
    A misspelled field (``wieghts``) or an unknown ``version`` fails instead
    of silently falling back to defaults.

    :param tmp_path: pytest temporary directory fixture
    """
    entry = {
        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
        "sources": ["drums", "bass", "other", "vocals"],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"models": {"x": {**entry, "wieghts": [[1], [1]]}}}))
    with pytest.raises(ModelLoadingError, match="unknown field"):
        ModelRepository(extra_models=path)
    path.write_text(json.dumps({"version": 2, "models": {"x": entry}}))
    with pytest.raises(ModelLoadingError, match="version"):
        ModelRepository(extra_models=path)


def test_models_file_accepts_msst_style_configs(tmp_path: Path) -> None:
    """
    An MSST ``model:`` section pastes verbatim: ``!!python/tuple`` parses, and
    training-only keys are dropped with a warning.

    :param tmp_path: pytest temporary directory fixture
    """
    weights, config = _tiny_scnet_checkpoint(tmp_path, ["a", "b", "c", "d"])
    lines = "".join(
        f"      {k}: !!python/tuple {list(v)}\n"
        if isinstance(v, list)
        else f"      {k}: {v}\n"
        for k, v in config.items()
    )
    path = tmp_path / "m.yaml"
    path.write_text(
        "models:\n  mine:\n    architecture: scnet\n    sources: [a, b, c, d]\n"
        "    samplerate: 44100\n    segment_samples: 44100\n    config:\n"
        f"{lines}      flash_attn: true\n"
        f"    checkpoint: {{format: safetensors, path: {weights}}}\n"
    )
    with pytest.warns(UserWarning, match="flash_attn"):
        repo = ModelRepository(extra_models=path)
    assert repo.get_model("mine").sources == ["a", "b", "c", "d"]


def test_verified_cache_files_load_from_a_read_only_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An already-downloaded artifact loads without creating a lock file, so a
    read-only or shared cache directory works.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores permission bits")
    import hashlib

    from unblend.repo import _artifact_cache_path

    weights, config = _tiny_demucs_layer(tmp_path)
    data = weights.read_bytes()
    spec = {
        "format": "safetensors",
        "url": "https://example.invalid/w.safetensors",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: cache)
    _artifact_cache_path(spec).write_bytes(data)
    extra = _extra_models_file(
        tmp_path,
        {
            "architecture": "htdemucs",
            "sources": config["sources"],
            "config": config,
            "checkpoint": spec,
        },
    )
    os.chmod(cache, 0o555)
    try:
        model = ModelRepository(extra_models=extra).get_model("custom")
    finally:
        os.chmod(cache, 0o755)
    assert model.sources == config["sources"]
    assert not list(cache.glob(".*.lock"))


def test_download_resumes_after_a_dropped_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A connection that drops mid-download resumes with an HTTP Range request
    instead of starting over.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    import httpx

    payload = b"abcdefgh"
    calls: list[dict] = []

    class Response:
        """
        First attempt drops after four bytes; the resumed one sends the rest.
        """

        def __init__(self, headers: dict) -> None:
            self.range = headers.get("Range")
            self.status_code = 206 if self.range else 200
            body = payload[4:] if self.range else payload
            self.headers = {"content-length": str(len(body))}
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self, chunk_size: int):
            del chunk_size
            if self.range:
                yield self.body
            else:
                yield self.body[:4]
                raise httpx.ReadError("connection reset")

    def fake_stream(_method: str, _url: str, headers: dict, **_kwargs: object):
        calls.append(dict(headers))
        return Response(headers)

    monkeypatch.setattr("unblend.repo.httpx.stream", fake_stream)
    monkeypatch.setattr("unblend.repo.time.sleep", lambda _s: None)
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    target = tmp_path / "cache" / "model.safetensors"
    repo._download_verified_file(
        "https://example.invalid/model",
        target,
        sha256(payload).hexdigest(),
        len(payload),
    )
    assert target.read_bytes() == payload
    assert calls == [{}, {"Range": "bytes=4-"}]


def test_models_file_structure_typos_are_refused(tmp_path: Path) -> None:
    """
    A misspelled top-level key, a non-integer version, and unknown or
    ignored member fields are errors rather than silently dropped.

    :param tmp_path: pytest temporary directory fixture
    """
    path = tmp_path / "m.yaml"
    for text, match in (
        ("version: 1\nmodels: {}\nmodles: {}\n", "unknown top-level"),
        ("version: true\nmodels: {}\n", "version"),
        (
            "models:\n  ens:\n    sources: [drums, bass, other, vocals]\n"
            "    members: [{model: htdemucs, weight: 2}, {model: scnet_small}]\n",
            "unknown field",
        ),
    ):
        path.write_text(text)
        with pytest.raises(ModelLoadingError, match=match):
            ModelRepository(extra_models=path)


def test_uppercase_sha256_and_duplicate_listing_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An upper-case digest is normalized, and a file listed twice in
    ``UNBLEND_EXTRA_MODELS`` loads once instead of clashing with itself.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    entry = {k: v for k, v in base.items() if k != "backend"}
    entry["checkpoint"] = {
        **entry["checkpoint"],
        "sha256": entry["checkpoint"]["sha256"].upper(),
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"models": {"mine": entry}}))
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", os.pathsep.join([str(path)] * 2))
    repo = ModelRepository()
    assert repo.list_models()["mine"]["checkpoint"]["sha256"].islower()


def test_a_missing_listed_models_file_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A file ``UNBLEND_EXTRA_MODELS`` lists before it exists (the usual order
    before a first import) is skipped with a warning, not fatal.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(tmp_path / "later.yaml"))
    with pytest.warns(UserWarning, match="doesn't exist"):
        assert "htdemucs" in ModelRepository().list_models()


def test_a_broken_listed_file_is_blamed_not_the_default_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    With a valid default file present, an error in a listed file names the
    listed file instead of warning that the default one is being ignored.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param _isolate_default_models_file: the substituted default file path
    """
    _isolate_default_models_file.write_text("version: 1\nmodels: {}\n")
    listed = tmp_path / "listed.yaml"
    listed.write_text(
        "models:\n  ens:\n    sources: [drums, bass, other, vocals]\n"
        "    members: [{model: htdemucs}, {model: nope}]\n"
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ModelLoadingError, match="listed.yaml"):
            ModelRepository()


def test_entry_errors_name_their_models_file(tmp_path: Path) -> None:
    """
    An entry-level error (here an unknown field) names the models file the
    entry came from.

    :param tmp_path: pytest temporary directory fixture
    """
    path = tmp_path / "m.yaml"
    path.write_text("models:\n  mine:\n    sources: [a]\n    wieghts: []\n")
    with pytest.raises(ModelLoadingError, match=r"m\.yaml"):
        ModelRepository(extra_models=path)


@pytest.mark.parametrize("failure", ["after_last_byte", "server_error"])
def test_download_retries_transient_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """
    A connection dropped after the last byte completes without a Range
    request (which would get a 416), and a 5xx response is retried.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param failure: Which transient failure the first attempt hits.
    """
    import httpx

    payload = b"abcdefgh"
    calls: list[dict] = []

    class Response:
        """
        First attempt fails as ``failure`` says; later ones succeed.
        """

        def __init__(self, first: bool) -> None:
            self.first = first
            self.status_code = 503 if first and failure == "server_error" else 200
            self.headers = {"content-length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                request = httpx.Request("GET", "https://example.invalid/model")
                raise httpx.HTTPStatusError(
                    "unavailable",
                    request=request,
                    response=httpx.Response(self.status_code, request=request),
                )

        def iter_bytes(self, chunk_size: int):
            del chunk_size
            yield payload
            if self.first and failure == "after_last_byte":
                raise httpx.ReadError("connection reset")

    def fake_stream(_method: str, _url: str, headers: dict, **_kwargs: object):
        calls.append(dict(headers))
        return Response(first=len(calls) == 1)

    monkeypatch.setattr("unblend.repo.httpx.stream", fake_stream)
    monkeypatch.setattr("unblend.repo.time.sleep", lambda _s: None)
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    target = tmp_path / "cache" / "model.safetensors"
    repo._download_verified_file(
        "https://example.invalid/model",
        target,
        sha256(payload).hexdigest(),
        len(payload),
    )
    assert target.read_bytes() == payload
    assert calls == ([{}] if failure == "after_last_byte" else [{}, {}])


def test_checkpoint_field_typos_are_refused(tmp_path: Path) -> None:
    """
    A misspelled ``checkpoint`` field (``sha265``) is an error rather than
    silently skipping verification of a local file.

    :param tmp_path: pytest temporary directory fixture
    """
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "mine": {
                        "architecture": "htdemucs",
                        "sources": ["a", "b"],
                        "config": {"sources": ["a", "b"]},
                        "checkpoint": {
                            "format": "safetensors",
                            "path": str(weights),
                            "sha265": "0" * 64,
                        },
                    }
                }
            }
        )
    )
    with pytest.raises(ModelLoadingError, match="sha265"):
        ModelRepository(extra_models=path)


def test_member_artifact_fields_beside_checkpoint_are_refused(tmp_path: Path) -> None:
    """
    A member with a ``checkpoint:`` mapping and a sibling ``sha256`` is an
    error; the sibling would otherwise be dropped and never verified.

    :param tmp_path: pytest temporary directory fixture
    """
    base = ModelRepository(extra_models=[]).list_models()["htdemucs"]
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "bag": {
                        "sources": base["sources"],
                        "architecture": "htdemucs",
                        "config": base["config"],
                        "members": [
                            {
                                "checkpoint": {
                                    "format": "safetensors",
                                    "path": str(tmp_path / "a.safetensors"),
                                },
                                "sha256": "0" * 64,
                            },
                            {"model": "htdemucs"},
                        ],
                    }
                }
            }
        )
    )
    with pytest.raises(ModelLoadingError, match="unknown field.*sha256"):
        ModelRepository(extra_models=path)


def test_every_family_is_built_in_eval_mode(tmp_path: Path) -> None:
    """
    ``get_model`` returns HTDemucs, and an ensemble with every submodule, in
    eval mode like the other families, so a direct forward has dropout off
    (and doesn't fail on MPS attention).

    :param tmp_path: pytest temporary directory fixture
    """
    from safetensors.torch import save_file

    from unblend.htdemucs import HTDemucs

    config = dict(
        sources=["a", "b"],
        samplerate=8000,
        segment=1,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )
    weights = tmp_path / "w.safetensors"
    save_file(HTDemucs(**config).state_dict(), str(weights))
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "tiny": {
                        "architecture": "htdemucs",
                        "sources": ["a", "b"],
                        "config": config,
                        "checkpoint": {"format": "safetensors", "path": str(weights)},
                    },
                    "pair": {
                        "sources": ["a", "b"],
                        "members": [{"model": "tiny"}, {"model": "tiny"}],
                    },
                }
            }
        )
    )
    repo = ModelRepository(extra_models=path)
    for name in ("tiny", "pair"):
        model = repo.get_model(name)
        assert not any(module.training for module in model.modules()), name


@pytest.mark.parametrize(
    "architecture, config_patch, match",
    [
        ("htdemucs", {"cac": False}, "cac"),
        ("scnet", {"num_dplayer": 3}, "num_dplayer"),
    ],
)
def test_unbuildable_configs_fail_before_download(
    tmp_path: Path, architecture: str, config_patch: dict, match: str
) -> None:
    """
    Configs the constructors refuse are refused when the registry loads, so a
    remote entry never downloads weights it can't build.

    :param tmp_path: pytest temporary directory fixture
    :param architecture: Architecture of the entry.
    :param config_patch: Keys making the config unbuildable.
    :param match: Expected error fragment.
    """
    base = ModelRepository(extra_models=[]).list_models()[
        "htdemucs" if architecture == "htdemucs" else "scnet_small"
    ]
    entry = {k: v for k, v in base.items() if k != "backend"}
    entry["architecture"] = architecture
    entry["config"] = {**entry["config"], **config_patch}
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"models": {"bad": entry}}))
    with pytest.raises(ModelLoadingError, match=match):
        ModelRepository(extra_models=path)


def test_a_broken_default_file_is_named_when_a_listed_file_needs_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    When a listed ensemble uses a model from a default file that is broken by
    an unrelated entry, the error names the real fault in the default file
    rather than blaming the listed file for an "unknown" model.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param _isolate_default_models_file: the substituted default file path
    """
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    good = {k: v for k, v in base.items() if k != "backend"}
    _isolate_default_models_file.write_text(
        json.dumps({"models": {"d1": good, "d2": {**good, "architecture": "nonsense"}}})
    )
    listed = tmp_path / "b.json"
    listed.write_text(
        json.dumps(
            {
                "models": {
                    "eb": {
                        "sources": base["sources"],
                        "members": [{"model": "d1"}, {"model": "scnet_small"}],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    with pytest.raises(ModelLoadingError, match=r"d2(?s:.*)models\.yaml"):
        ModelRepository()


def test_a_listed_files_real_error_is_reported_not_a_missing_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    A listed ensemble that uses a default-file model and also has a real
    mistake (a weights row missing) reports that mistake, not a false
    "unknown model" from the default file being dropped.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param _isolate_default_models_file: the substituted default file path
    """
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    good = {k: v for k, v in base.items() if k != "backend"}
    _isolate_default_models_file.write_text(json.dumps({"models": {"mine": good}}))
    listed = tmp_path / "l.json"
    listed.write_text(
        json.dumps(
            {
                "models": {
                    "ens": {
                        "sources": base["sources"],
                        "members": [{"model": "mine"}, {"model": "scnet_small"}],
                        "weights": [[1, 1, 1, 1]],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ModelLoadingError, match="weight row"):
            ModelRepository()


def test_a_sound_default_file_is_not_blamed_for_a_listed_files_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    A default file whose ensemble uses a listed model is not warned about
    when the listed file is broken by an unrelated entry; the error names
    the listed file's fault.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param _isolate_default_models_file: the substituted default file path
    """
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    good = {k: v for k, v in base.items() if k != "backend"}
    listed = tmp_path / "l.json"
    listed.write_text(
        json.dumps(
            {"models": {"l_good": good, "l_bad": {**good, "architecture": "nope"}}}
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    _isolate_default_models_file.write_text(
        json.dumps(
            {
                "models": {
                    "dens": {
                        "sources": base["sources"],
                        "members": [{"model": "l_good"}, {"model": "scnet_small"}],
                    }
                }
            }
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ModelLoadingError, match=r"l_bad(?s:.*)l\.json"):
            ModelRepository()


def test_cancelling_registry_weights_fail_before_download(tmp_path: Path) -> None:
    """
    An extra-models ensemble whose weights cancel is refused when the
    registry loads, not after its members are downloaded.

    :param tmp_path: pytest temporary directory fixture
    """
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "cancel": {
                        "sources": base["sources"],
                        "members": [{"model": "scnet_small"}, {"model": "scnet_small"}],
                        "weights": [[1, 1, 1, 1], [-1, 1, 1, 1]],
                    }
                }
            }
        )
    )
    with pytest.raises(ModelLoadingError, match="non-zero total"):
        ModelRepository(extra_models=path)


def test_a_header_only_entry_with_an_unreadable_file_says_so(tmp_path: Path) -> None:
    """
    An entry that relies on its file's header, when the file is truncated,
    gets an error saying the header is unreadable rather than "unknown
    architecture None".

    :param tmp_path: pytest temporary directory fixture
    """
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"\x00" * 1000)
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "hdr": {
                        "sources": ["a", "b"],
                        "checkpoint": {"format": "safetensors", "path": str(weights)},
                    }
                }
            }
        )
    )
    with pytest.raises(
        ModelLoadingError, match=r"Model hdr doesn't state an architecture"
    ):
        ModelRepository(extra_models=path)


def test_a_number_too_large_for_a_float_is_a_clean_error(tmp_path: Path) -> None:
    """
    A 400-digit integer in a models file is reported as invalid, not raised
    as a raw OverflowError.

    :param tmp_path: pytest temporary directory fixture
    """
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    entry = {k: v for k, v in base.items() if k != "backend"}
    path = tmp_path / "m.yaml"
    for field in ("segment", "weights"):
        bad = dict(entry)
        if field == "segment":
            bad["segment"] = 10**400
        else:
            bad = {
                "sources": base["sources"],
                "members": [{"model": "scnet_small"}, {"model": "scnet_small"}],
                "weights": [[10**400, 1, 1, 1], [1, 1, 1, 1]],
            }
        path.write_text(yaml.safe_dump({"models": {"big": bad}}))
        with pytest.raises(ModelLoadingError):
            ModelRepository(extra_models=path)
    demucs = ModelRepository(extra_models=[]).list_models()["htdemucs"]
    demucs = {k: v for k, v in demucs.items() if k != "backend"}
    for key, value in (("samplerate", 10**400), ("segment", 1e305), ("segment", 1e-9)):
        bad = {**demucs, "config": {**demucs["config"], key: value}}
        bad.pop("samplerate", None)
        bad.pop("segment_samples", None)
        path.write_text(yaml.safe_dump({"models": {"big": bad}}))
        with pytest.raises(ModelLoadingError):
            ModelRepository(extra_models=path)


@pytest.mark.parametrize("case", ["members-int", "channels-list", "huge-rate"])
def test_nonsense_field_types_are_clean_errors(tmp_path: Path, case: str) -> None:
    """
    Values of the wrong type or size give a ``ModelLoadingError`` (so a broken
    default models file is skipped) rather than a raw TypeError/OverflowError.

    :param tmp_path: pytest temporary directory fixture
    :param case: Which field to break.
    """
    models = ModelRepository(extra_models=[]).list_models()
    if case == "members-int":
        entry = {k: v for k, v in models["htdemucs_ft"].items() if k != "backend"}
        entry["members"] = 5
    elif case == "channels-list":
        entry = {
            "sources": models["htdemucs"]["sources"],
            "members": [
                {"model": "htdemucs"},
                {
                    **{
                        k: v
                        for k, v in models["htdemucs"].items()
                        if k in ("architecture", "checkpoint")
                    },
                    "config": {**models["htdemucs"]["config"], "audio_channels": [2]},
                },
            ],
        }
    else:
        entry = {k: v for k, v in models["scnet_small"].items() if k != "backend"}
        entry["samplerate"] = 10**401
        entry["segment"] = 1.0
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"models": {"odd": entry}}))
    expected = {
        "members-int": "members",
        "channels-list": "invalid channel count",
        "huge-rate": "invalid samplerate",
    }[case]
    with pytest.raises(ModelLoadingError, match=expected):
        ModelRepository(extra_models=path)


def test_an_implicitly_mono_roformer_member_is_caught_before_download(
    tmp_path: Path,
) -> None:
    """
    RoFormer is mono unless its config sets ``stereo``; an ensemble pairing
    that with a stereo member is refused at construction, not after both
    checkpoints download.

    :param tmp_path: pytest temporary directory fixture
    """
    base = ModelRepository(extra_models=[]).list_models()["bs_roformer_anvuew"]
    member = {
        k: v
        for k, v in base.items()
        if k in ("architecture", "checkpoint", "samplerate", "segment_samples")
    }
    mono = {
        **member,
        "config": {k: v for k, v in base["config"].items() if k != "stereo"},
    }
    entry = {
        "sources": base["sources"],
        "members": [{"model": "bs_roformer_anvuew"}, mono],
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"models": {"mixed": entry}}))
    with pytest.raises(ModelLoadingError, match="channel count"):
        ModelRepository(extra_models=path)


@pytest.mark.parametrize(
    "case", ["combine-params-int", "bool-key", "nul-path", "nul-absolute-path"]
)
def test_unanticipated_shapes_are_model_loading_errors(
    tmp_path: Path, case: str
) -> None:
    """
    A shape no specific check covers (a non-mapping ``combine_params``, a
    YAML ``on:`` key read as a bool, a NUL in a path) still fails as a
    ``ModelLoadingError`` naming the file, so a broken default models file
    is skipped rather than crashing every command.

    :param tmp_path: pytest temporary directory fixture
    :param case: Which shape to write.
    """
    models = ModelRepository(extra_models=[]).list_models()
    if case == "combine-params-int":
        entry = {k: v for k, v in models["htdemucs_ft"].items() if k != "backend"}
        entry["combine_params"] = 1024
    elif case == "bool-key":
        entry = {k: v for k, v in models["scnet_small"].items() if k != "backend"}
        entry[True] = 1
    else:
        weights = "w\0.safetensors" if case == "nul-path" else str(tmp_path / "w\0.st")
        entry = {
            "sources": ["a", "b"],
            "architecture": "scnet",
            "samplerate": 44100,
            "segment_samples": 44100,
            "config": {"dims": [4, 8]},
            "checkpoint": {"format": "safetensors", "path": weights},
        }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"models": {"odd": entry}}))
    with pytest.raises(ModelLoadingError, match=str(path).replace("\\", "\\\\")):
        ModelRepository(extra_models=path)


def test_an_unreadable_local_weights_folder_is_missing_not_a_crash(
    tmp_path: Path,
) -> None:
    """
    ``Path.is_file()`` raises on EACCES; a local path in a locked folder
    must load the registry and report the file as missing.

    :param tmp_path: pytest temporary directory fixture
    """

    if os.geteuid() == 0:
        pytest.skip("root reads any folder")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "w.safetensors").write_bytes(b"x")
    entry = {
        "sources": ["a", "b"],
        "architecture": "scnet",
        "samplerate": 44100,
        "segment_samples": 44100,
        "config": {"dims": [4, 8]},
        "checkpoint": {
            "format": "safetensors",
            "path": str(locked / "w.safetensors"),
        },
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"models": {"locked": entry}}))
    os.chmod(locked, 0)
    try:
        repo = ModelRepository(extra_models=path)
        assert repo.loaded_bytes("locked") == 0
    finally:
        os.chmod(locked, 0o755)


def test_a_path_under_an_unknown_tilde_user_is_taken_literally(
    tmp_path: Path,
) -> None:
    """
    ``Path.expanduser()`` raises ``RuntimeError`` for ``~nosuchuser``; the
    registry keeps such a path as written (the file is then just missing).

    :param tmp_path: pytest temporary directory fixture
    """
    entry = {
        "sources": ["a", "b"],
        "architecture": "scnet",
        "samplerate": 44100,
        "segment_samples": 44100,
        "config": {"dims": [4, 8]},
        "checkpoint": {"format": "safetensors", "path": "~nosuchuserzz/w.st"},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"models": {"tilde": entry}}))
    repo = ModelRepository(extra_models=path)
    assert "tilde" in repo.list_models()
    assert repo.loaded_bytes("tilde") == 0


def test_an_htdemucs_config_its_constructor_asserts_on_is_a_load_error(
    tmp_path: Path,
) -> None:
    """
    ``t_heads: 5`` fails an ``assert`` in the constructor; that surfaces as a
    ``ModelLoadingError``, as for the other backends, not an AssertionError.

    :param tmp_path: pytest temporary directory fixture
    """
    import torch
    from safetensors.torch import save_file

    weights = tmp_path / "w.safetensors"
    save_file({"x": torch.zeros(1)}, str(weights))
    entry = {
        "architecture": "htdemucs",
        "sources": ["a", "b"],
        "config": {"sources": ["a", "b"], "t_heads": 5},
        "checkpoint": {"format": "safetensors", "path": str(weights)},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"models": {"uu": entry}}))
    repo = ModelRepository(extra_models=path)
    with pytest.raises(ModelLoadingError, match="Failed to build"):
        repo.get_model("uu")


def test_an_unknown_tilde_user_path_means_one_file_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``~nosuchuser/w`` stays relative, so it is anchored at the models file's
    folder like any relative path — the file the registry loads is the file
    ``unregister --delete-weights`` would consider.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    from unblend.repo import entry_weight_paths

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    entry = {
        "sources": ["a", "b"],
        "architecture": "scnet",
        "samplerate": 44100,
        "segment_samples": 44100,
        "config": {"dims": [4, 8]},
        "checkpoint": {"format": "safetensors", "path": "~nosuchuserzz/w.st"},
    }
    path = models_dir / "m.yaml"
    path.write_text(yaml.safe_dump({"models": {"rel": entry}}))
    monkeypatch.chdir(run_dir)
    (loaded,) = ModelRepository(extra_models=path).local_artifacts("rel")
    (named,) = entry_weight_paths(entry, models_dir.resolve())
    assert os.path.realpath(loaded) == os.path.realpath(named)


def test_a_symlink_loop_as_the_cache_dir_is_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``Path.resolve()`` raises on a loop (Python 3.10–3.12); the cache
    directory is resolved without it, so listing the cache just finds nothing.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    from unblend.repo import get_cache_dir

    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(tmp_path / "a"))
    get_cache_dir()
    assert ModelRepository(extra_models=[]).get_cache_info() == {}
