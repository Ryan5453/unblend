"""
Offline checks for ``demucs.repo`` integrity gates.

These cover the SHA-256 verification branch and metadata-shape requirements
that guard ``torch.load(weights_only=False)`` from running on tampered data.
"""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from demucs.exceptions import ModelLoadingError
from demucs.repo import ModelRepository, check_checksum


def _good_metadata() -> dict:
    """
    Minimal valid metadata blob accepted by ``ModelRepository.__init__``.

    :return: A metadata dict shaped like ``demucs/metadata.json``.
    """
    return {
        "models": {
            "fakemodel": {
                "sources": ["drums", "bass", "other", "vocals"],
                "models": [
                    {
                        "remote": "fake/abcd.th",
                        "checksum": "abcd1234",
                        "sha256": "a" * 64,
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


def test_repository_rejects_short_sha256(tmp_path: Path) -> None:
    """
    A metadata entry with anything other than a full 64-character ``sha256``
    is rejected at load time, since loading runs ``weights_only=False``.

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
    monkeypatch.setattr("demucs.repo.get_cache_dir", lambda: cache_dir)
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
    monkeypatch.setattr("demucs.repo.get_cache_dir", lambda: cache_dir)
    (cache_dir / "abcd1234.th").write_bytes(b"x" * 1024)

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
    monkeypatch.setattr("demucs.repo.get_cache_dir", lambda: cache_dir)
    layer = cache_dir / "abcd1234.th"
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
    assert repo.layer_sha256("abcd1234") == "a" * 64
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
    # sha256 expects the file to hash to "a"*64, so any other content trips
    # check_checksum and exercises the redownload branch.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("demucs.repo.get_cache_dir", lambda: cache_dir)

    corrupt_path = cache_dir / "abcd1234.th"
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
    assert download_calls[0]["expected_checksum"] == "a" * 64


@pytest.mark.parametrize(
    "exc",
    [
        ModelLoadingError("bad checksum"),
        RuntimeError("torch.load blew up"),
        EOFError("truncated pickle"),
        __import__("pickle").UnpicklingError("not a pickle"),
    ],
    ids=["ModelLoadingError", "RuntimeError", "EOFError", "UnpicklingError"],
)
def test_get_model_recovers_from_each_cache_exception_type(
    tmp_path: Path, monkeypatch: object, exc: Exception
) -> None:
    """
    The ``except`` block in ``get_model`` covers four exception types; each
    one must trigger the same recovery path (unlink + redownload). This
    drives every branch so a future narrowing of the catch can't silently
    leak a corrupt cache file.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param exc: Exception instance the cache-load path will raise
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("demucs.repo.get_cache_dir", lambda: cache_dir)

    bad_path = cache_dir / "abcd1234.th"
    bad_path.write_bytes(b"placeholder; we'll bypass real load")

    def raise_exc(*_args: object, **_kwargs: object) -> None:
        """
        Patched ``check_checksum`` that raises the parametrized exception so
        the cache-recovery branch fires regardless of file content.

        :param _args: ignored positional arguments
        :param _kwargs: ignored keyword arguments
        :raises Exception: the parametrized exception
        """
        raise exc

    # Force the failure inside the cache-load ``try`` block by making
    # ``check_checksum`` raise. The remaining downloader is stubbed out.
    monkeypatch.setattr("demucs.repo.check_checksum", raise_exc)

    download_calls: list[dict] = []

    def fake_download_and_load_layer(self, **kwargs):
        """
        Record the redownload call and return a placeholder layer.

        :param self: bound ``ModelRepository`` instance
        :param kwargs: forwarded download kwargs
        :return: stand-in layer object
        """
        download_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        ModelRepository, "_download_and_load_layer", fake_download_and_load_layer
    )

    repo = ModelRepository(metadata_path=_write_metadata(tmp_path, _good_metadata()))
    try:
        repo.get_model("fakemodel")
    except Exception:
        pass

    assert not bad_path.exists(), (
        f"Cache file should be unlinked after {type(exc).__name__}"
    )
    assert len(download_calls) == 1
