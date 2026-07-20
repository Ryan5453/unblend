"""
Offline checks for ``unblend.repo`` integrity and safe-loading gates.

Registered models use explicit architectures plus tensor-only Safetensors;
legacy pickle compatibility is tested as a separate opt-in boundary.
"""

import json
import subprocess
import sys
import threading
from hashlib import sha256
from pathlib import Path

import pytest

from unblend.exceptions import ModelLoadingError
from unblend.repo import ModelRepository, check_checksum, check_size, get_cache_dir


def _good_metadata() -> dict:
    """
    Minimal valid metadata blob accepted by ``ModelRepository.__init__``.

    :return: A metadata dict shaped like ``unblend/metadata.json``.
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
                        "checksum": "abcd1234",
                        "sha256": "abcd1234" + "a" * 56,
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
    """Trusted artifact sizes are enforced independently of checksums."""
    path = tmp_path / "blob.bin"
    path.write_bytes(b"1234")
    check_size(path, 4)
    with pytest.raises(ModelLoadingError, match="expected 5 bytes"):
        check_size(path, 5)


def test_demucs_download_rejects_wrong_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared size mismatch fails before model bytes are streamed."""

    class Response:
        """Minimal streaming response with a bad declared length."""

        headers = {"content-length": "5"}

        def __enter__(self):
            """Enter the fake response context."""
            return self

        def __exit__(self, *_args: object) -> None:
            """Leave the fake response context."""

        def raise_for_status(self) -> None:
            """Represent a successful HTTP status."""

        def iter_bytes(self, chunk_size: int):
            """Yield no bytes because the header should reject first."""
            del chunk_size
            return iter(())

    monkeypatch.setattr("unblend.repo.httpx.stream", lambda *_a, **_k: Response())
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    with pytest.raises(ModelLoadingError, match="expected 4"):
        repo._download_and_load_layer(
            "https://example.invalid/model",
            tmp_path / "cache" / "model.safetensors",
            "0" * 64,
            4,
            _good_metadata()["models"]["fakemodel"],
        )


def test_roformer_download_rejects_chunked_size_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunked response cannot exceed metadata's expected artifact size."""

    class Response:
        """Minimal chunked response that lies by omitting Content-Length."""

        headers: dict[str, str] = {}

        def __enter__(self):
            """Enter the fake response context."""
            return self

        def __exit__(self, *_args: object) -> None:
            """Leave the fake response context."""

        def raise_for_status(self) -> None:
            """Represent a successful HTTP status."""

        def iter_bytes(self, chunk_size: int):
            """Yield five bytes against a four-byte limit."""
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
    """Malformed custom metadata always raises the package error type."""
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ModelLoadingError):
        ModelRepository(metadata_path=path)


def test_repository_rejects_empty_demucs_layers(tmp_path: Path) -> None:
    """A Demucs entry must contain at least one Safetensors artifact."""
    bad = _good_metadata()
    bad["models"]["fakemodel"]["models"] = []
    with pytest.raises(ModelLoadingError, match="non-empty layer"):
        ModelRepository(metadata_path=_write_metadata(tmp_path, bad))


def test_repository_rejects_malformed_roformer_fields(tmp_path: Path) -> None:
    """RoFormer architecture/config fields are validated before download."""
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
    (cache_dir / "abcd1234.safetensors").write_bytes(b"x" * 1024)

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
    layer = cache_dir / "abcd1234.safetensors"
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
    assert repo.layer_sha256("abcd1234") == "abcd1234" + "a" * 56
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
    # Build a known-good repo against fake metadata. The "abcd1234" layer's
    # sha256 expects the registered digest, so any other content trips
    # check_checksum and exercises the redownload branch.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("unblend.repo.get_cache_dir", lambda: cache_dir)

    corrupt_path = cache_dir / "abcd1234.safetensors"
    corrupt_path.write_bytes(b"this is not a real model checkpoint")

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))

    download_calls: list[dict] = []

    def fake_download_and_load_layer(self, **kwargs):
        """
        Record the call and return a placeholder rather than hit the network.

        :param self: bound ``ModelRepository`` instance
        :param kwargs: forwarded download kwargs
        :return: a stand-in object representing a loaded layer
        """
        download_calls.append(kwargs)
        return object()  # placeholder layer; never actually used downstream

    monkeypatch.setattr(
        ModelRepository, "_download_and_load_layer", fake_download_and_load_layer
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
    assert download_calls[0]["expected_checksum"] == "abcd1234" + "a" * 56


def test_only_load_requires_exclusive_specialist_weight(tmp_path: Path) -> None:
    """Repository cannot skip another layer that contributes to the stem."""
    metadata = _good_metadata()
    metadata["models"]["fakemodel"]["models"].append(
        {
            "format": "safetensors",
            "remote": "https://example.invalid/ef.safetensors",
            "checksum": "ef012345",
            "sha256": "ef012345" + "b" * 56,
            "size_bytes": 2,
        }
    )
    metadata["models"]["fakemodel"]["weights"] = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))
    assert repo.required_layers("fakemodel", only_load="drums") == [
        "abcd1234",
        "ef012345",
    ]


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
            "checksum": "ef012345",
            "sha256": "ef012345" + "b" * 56,
            "size_bytes": 2,
        }
    )
    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, metadata))

    assert repo.get_cache_info() == {}

    (cache / "abcd1234.safetensors").write_bytes(b"xxxx")
    info = repo.get_cache_info()
    assert info["fakemodel"]["complete"] is False
    assert info["fakemodel"]["total_layers"] == 2
    assert info["fakemodel"]["size_bytes"] == 4
    assert list(info["fakemodel"]["layers"]) == ["abcd1234"]

    (cache / "ef012345.safetensors").write_bytes(b"yy")
    info = repo.get_cache_info()
    assert info["fakemodel"]["complete"] is True
    assert info["fakemodel"]["size_bytes"] == 6


def test_sweep_stale_downloads_removes_staging_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``sweep_stale_downloads`` removes ``tmp*`` staging leftovers but not
    cached layer files.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("UNBLEND_CACHE_DIR", str(cache))

    (cache / "tmpabc123.safetensors").write_bytes(b"partial download")
    (cache / "abcd1234.safetensors").write_bytes(b"cached layer")

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    assert repo.sweep_stale_downloads() == 1
    assert not (cache / "tmpabc123.safetensors").exists()
    assert (cache / "abcd1234.safetensors").exists()


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

    cached = cache_dir / "abcd1234.safetensors"
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

    monkeypatch.setattr(ModelRepository, "_download_and_load_layer", fail_download)

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    with pytest.raises(ModelLoadingError) as excinfo:
        repo.get_model("fakemodel")

    assert cached.exists(), "read failure must not unlink the cached file"
    if expect_wrapped:
        assert excinfo.value is not exc
        assert excinfo.value.__cause__ is exc
    else:
        assert excinfo.value is exc


def test_load_model_accepts_upstream_demucs_pickles(tmp_path) -> None:
    """
    Explicit legacy loading temporarily installs the historical upstream
    module alias and restores ``sys.modules`` afterward. Registered models do
    not use this compatibility path.
    """
    import torch

    from unblend.htdemucs import HTDemucs
    from unblend.states import _legacy_demucs_aliases, load_model

    kwargs = dict(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )
    model = HTDemucs(**kwargs)

    # Re-point the class's pickle identity at the upstream module path so the
    # saved bytes are byte-authentic to a real upstream checkpoint.
    original_module = HTDemucs.__module__
    HTDemucs.__module__ = "demucs.htdemucs"
    try:
        package = {
            "klass": HTDemucs,
            "args": (),
            "kwargs": kwargs,
            "state": model.state_dict(),
        }
        path = tmp_path / "upstream_style.th"
        with _legacy_demucs_aliases():
            torch.save(package, path)
    finally:
        HTDemucs.__module__ = original_module

    previous_demucs = sys.modules.get("demucs")
    loaded = load_model(path)
    assert isinstance(loaded, HTDemucs)
    assert loaded.sources == ["a", "b"]
    assert sys.modules.get("demucs") is previous_demucs


def test_legacy_alias_context_serializes_concurrent_loads() -> None:
    """Concurrent legacy loads cannot restore aliases out of order."""
    from unblend.states import _legacy_demucs_aliases

    previous_demucs = sys.modules.get("demucs")
    previous_htdemucs = sys.modules.get("demucs.htdemucs")
    entered = threading.Event()
    release = threading.Event()
    second_done = threading.Event()

    def first() -> None:
        """Hold the alias context while the second thread attempts entry."""
        with _legacy_demucs_aliases():
            entered.set()
            release.wait(timeout=5)

    def second() -> None:
        """Enter only after the first context has restored its aliases."""
        entered.wait(timeout=5)
        with _legacy_demucs_aliases():
            assert "demucs.htdemucs" in sys.modules
        second_done.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert entered.wait(timeout=5)
    second_thread.start()
    assert not second_done.wait(timeout=0.05)
    release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert second_done.is_set()
    assert sys.modules.get("demucs") is previous_demucs
    assert sys.modules.get("demucs.htdemucs") is previous_htdemucs


def test_normal_import_does_not_install_demucs_aliases() -> None:
    """Ordinary package import coexists with a separately installed Demucs."""
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
    """Registered weights build strictly without calling ``torch.load``."""
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
                        "checksum": digest[:16],
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
