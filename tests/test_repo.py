"""
Offline checks for ``unblend.repo`` integrity and safe-loading gates.

Registered models use explicit architectures plus tensor-only Safetensors;
legacy pickle compatibility is tested as a separate opt-in boundary.
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
                "backend": "demucs",
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
                "models": [
                    {
                        "format": "safetensors",
                        "remote": "https://example.invalid/abcd.safetensors",
                        "sha256": FIRST_SHA,
                        "size_bytes": 1024,
                    }
                ],
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
    bad["models"]["fakemodel"]["models"][0]["sha256"] = "a" * 8
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
        {"models": []},
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
    bad["models"]["fakemodel"]["models"] = []
    with pytest.raises(ModelLoadingError, match="non-empty layer"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_repository_rejects_malformed_roformer_fields(tmp_path: Path) -> None:
    """
    RoFormer architecture/config fields are validated before download.
    """
    bad = {
        "models": {
            "bad": {
                "backend": "roformer",
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
        "backend": "roformer",
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
    assert repo.list_models() == _good_metadata()["models"]


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


def test_layer_sha256_lookup(tmp_path: Path) -> None:
    """
    Public ``layer_sha256`` returns the full 64-character digest for a known
    layer and raises ``KeyError`` for an unknown one.

    :param tmp_path: pytest temporary directory fixture
    """
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    assert repo.layer_sha256(FIRST_KEY) == FIRST_SHA
    with pytest.raises(KeyError):
        repo.layer_sha256("nothere")


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
    # (now mocked-out) download path. After the recovery, the bag-of-models
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
    metadata["models"]["fakemodel"]["models"].append(
        {
            "format": "safetensors",
            "remote": "https://example.invalid/ef.safetensors",
            "sha256": SECOND_SHA,
            "size_bytes": 2,
        }
    )
    metadata["models"]["fakemodel"]["weights"] = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))
    assert repo.required_layers("fakemodel", only_load="drums") == [
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
        repo.required_layers("fakemodel", only_load=stem)


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
                "backend": "roformer",
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
    cache_path = repo._checkpoint_cache_path(repo.list_models()["tiny"])
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
    # Construction now dispatches through the backend registry rather than
    # calling build_roformer directly, so patch the registry entry point.
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
    target = tmp_path / "custom-cache"
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(target))
    assert get_cache_dir() == target
    assert not target.exists()

    monkeypatch.setenv("UNBLEND_CACHE_DIR", "~/some-demucs-cache")
    assert get_cache_dir() == Path.home() / "some-demucs-cache"


def test_get_cache_dir_legacy_fallback_and_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The old variable warns, while the new name silently wins.
    """
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    monkeypatch.delenv("UNBLEND_CACHE_DIR", raising=False)
    monkeypatch.setenv("DEMUCS_CACHE_DIR", str(legacy))
    with pytest.warns(DeprecationWarning, match="UNBLEND_CACHE_DIR"):
        assert get_cache_dir() == legacy

    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(current))
    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter("always")
        assert get_cache_dir() == current
    assert not warnings_seen


def test_get_cache_info_reports_partial_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A model with some but not all layers cached is reported with
    ``complete: False`` and the cached subset's size — previously it was
    omitted entirely, hiding its disk usage from ``models list`` and
    ``models remove --all``.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(cache))

    metadata = _good_metadata()
    metadata["models"]["fakemodel"]["models"].append(
        {
            "format": "safetensors",
            "remote": "https://example.invalid/ef.safetensors",
            "sha256": SECOND_SHA,
            "size_bytes": 2,
        }
    )
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))

    assert repo.get_cache_info() == {}

    (cache / f"{FIRST_KEY}.safetensors").write_bytes(b"xxxx")
    info = repo.get_cache_info()
    assert info["fakemodel"]["complete"] is False
    assert info["fakemodel"]["total_layers"] == 2
    assert info["fakemodel"]["size_bytes"] == 4
    assert list(info["fakemodel"]["layers"]) == [FIRST_KEY]

    (cache / f"{SECOND_KEY}.safetensors").write_bytes(b"yy")
    info = repo.get_cache_info()
    assert info["fakemodel"]["complete"] is True
    assert info["fakemodel"]["size_bytes"] == 6


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
    listed[name]["models"] = []
    assert repo.list_models()[name]["models"], "internal metadata was mutated"


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
                "backend": "demucs",
                "architecture": "htdemucs",
                "sources": ["a", "b"],
                "config": config,
                "models": [
                    {
                        "format": "safetensors",
                        "remote": "https://example.invalid/layer.safetensors",
                        "sha256": digest,
                        "size_bytes": packed.stat().st_size,
                    }
                ],
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
        num_dplayer=1,
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
    derived from its architecture and its licence label passed through
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
            "models": [{"format": "safetensors", "path": str(weights)}],
        },
    )

    repo = ModelRepository(extra_models=extra)
    listed = repo.list_models()["custom"]
    assert listed["backend"] == "demucs", "backend should follow from architecture"
    assert listed["license"] == "my own terms", "licence is a pass-through label"

    model = repo.get_model("custom")
    assert isinstance(model, HTDemucs)
    assert model.sources == ["a", "b"]

    # A file the user owns is not cache: nothing to fetch, nothing to account
    # for, and ``models remove`` must never unlink it.
    assert repo.is_fully_local("custom")
    assert repo.local_artifacts("custom") == [weights]
    assert repo.required_layers("custom") == []
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
            "models": [
                {
                    "format": "safetensors",
                    "url": url,
                    "sha256": digest,
                    "size_bytes": len(payload),
                }
            ],
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
    assert repo.required_layers("custom") == [digest[:16]]

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
            "models": [
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
    assert repo.required_layers("custom") == ["b" * 16]


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


def test_backend_cannot_build_a_foreign_architecture(tmp_path: Path) -> None:
    """
    A backend/architecture pair from different families is rejected.
    """
    bad = _good_metadata()
    bad["models"]["fakemodel"]["architecture"] = "scnet"
    with pytest.raises(ModelLoadingError, match="cannot build"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_single_checkpoint_entries_are_validated_at_construction(
    tmp_path: Path,
) -> None:
    """
    Every backend's artifacts are checked up front. An SCNet entry with an
    unverifiable download used to construct fine and fail mid-``get_model``.
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
    assert repo.required_layers("custom", only_load="b") == []


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
                        "config": {"dim": 16},
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
            {"a": {"sources": ["x"], "members": [{"model": "nope"}]}},
            "references unknown model",
        ),
        (
            {
                "a": {"sources": ["x"], "members": [{"model": "b"}]},
                "b": {"sources": ["x"], "members": [{"model": "a"}]},
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
                "members": [{"model": "base"}],
            },
        }
    }
    with pytest.raises(ModelLoadingError, match="same stems in the same order"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))


def test_entry_must_name_its_weights_exactly_once(tmp_path: Path) -> None:
    """
    ``checkpoint``, ``members`` and ``models`` are mutually exclusive.
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
