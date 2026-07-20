# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import json
import os
import shutil
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import httpx
from safetensors import SafetensorError
from safetensors.torch import load_file

from .apply import Model, ModelEnsemble
from .exceptions import ModelLoadingError
from .htdemucs import HTDemucs
from .roformer import build_roformer

BASE_CDN_URL = "https://dl.fbaipublicfiles.com/demucs"

# Prefix for download staging files in the cache dir. Shared by the writer
# (``_download_and_load_layer``) and the sweeper (``sweep_stale_downloads``)
# so the two can't silently drift apart.
STAGING_PREFIX = "tmp"
DOWNLOAD_DEADLINE_SECONDS = 2 * 60 * 60


def check_checksum(path: Path, checksum: str) -> None:
    """
    Verify that a file matches an expected SHA-256 checksum.

    :param path: Path to the file to check
    :param checksum: Full 64-character SHA-256 hex digest from metadata's
        ``sha256`` field
    :raises ModelLoadingError: If the actual digest does not match
    """
    sha = sha256()
    try:
        with open(path, "rb") as file:
            while True:
                buf = file.read(2**20)
                if not buf:
                    break
                sha.update(buf)
    except OSError as e:
        raise ModelLoadingError(
            f"Could not read {path} for checksum verification: {e}"
        ) from e
    actual_checksum = sha.hexdigest()
    if actual_checksum != checksum:
        raise ModelLoadingError(
            f"Invalid checksum for file {path}, "
            f"expected {checksum} but got {actual_checksum}"
        )


def check_size(path: Path, expected_size: int) -> None:
    """
    Verify an artifact has the exact byte length declared in metadata.

    :param path: Artifact path.
    :param expected_size: Required byte count.
    :raises ModelLoadingError: If the file cannot be read or has the wrong size.
    """
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise ModelLoadingError(f"Could not stat {path}: {exc}") from exc
    if actual_size != expected_size:
        raise ModelLoadingError(
            f"Invalid size for {path}, expected {expected_size} bytes but got "
            f"{actual_size}."
        )


def _load_demucs_layer(path: Path, model_info: dict) -> HTDemucs:
    """
    Build one allowlisted HTDemucs architecture from pickle-free weights.

    Constructor configuration lives in trusted registry metadata; the artifact
    contains tensors only. A malformed Safetensors file fails closed and never
    falls back to a pickle loader.

    :param path: Path to the verified Safetensors artifact.
    :param model_info: Registry entry containing architecture and config.
    :return: Strictly weight-loaded HTDemucs model.
    :raises ModelLoadingError: If construction or strict loading fails.
    """
    if model_info.get("architecture") != "htdemucs":
        raise ModelLoadingError(
            f"Unsupported Demucs architecture {model_info.get('architecture')!r}."
        )
    try:
        model = HTDemucs(**dict(model_info["config"]))
        model.load_state_dict(load_file(path, device="cpu"), strict=True)
        return model
    except (KeyError, TypeError, RuntimeError, SafetensorError, ValueError) as exc:
        raise ModelLoadingError(f"Failed to build HTDemucs from {path}: {exc}") from exc


def get_cache_dir() -> Path:
    """
    Get the cache directory for downloaded models.

    Honours the ``UNBLEND_CACHE_DIR`` environment variable (tilde-expanded and
    resolved); defaults to ``~/.unblend/models``. The directory is not created
    here — read paths work against a missing dir, and the download path
    creates it (surfacing a clear error if it can't).

    :return: Path to the cache directory
    """
    override = os.environ.get("UNBLEND_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".unblend" / "models"


class ModelRepository:
    """Repository system for accessing models."""

    def __init__(self, metadata_path: Path | None = None) -> None:
        """
        Initialize the model repository from metadata.json.

        :param metadata_path: Path to a metadata file; defaults to the shipped
            ``unblend/metadata.json``. Mainly useful in tests.
        :raises ModelLoadingError: If the metadata structure is invalid
        """
        if metadata_path is None:
            metadata_path = Path(__file__).parent / "metadata.json"
        self.metadata_path = metadata_path

        try:
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadingError(
                f"Could not read model metadata {self.metadata_path}: {exc}"
            ) from exc

        if not isinstance(self.metadata, dict) or not isinstance(
            self.metadata.get("models"), dict
        ):
            raise ModelLoadingError(
                "Invalid metadata structure: expected a top-level 'models' dictionary."
            )
        self._models = self.metadata["models"]
        if not self._models:
            raise ModelLoadingError("Model metadata must contain at least one model.")
        for model_name, model_info in self._models.items():
            if not isinstance(model_name, str) or not model_name:
                raise ModelLoadingError("Every model name must be a non-empty string.")
            if not isinstance(model_info, dict):
                raise ModelLoadingError(
                    f"Model {model_name} metadata must be a dictionary."
                )
            if model_info.get("backend") not in {"demucs", "roformer"}:
                raise ModelLoadingError(
                    f"Model {model_name} has unknown backend "
                    f"{model_info.get('backend')!r}."
                )
            sources = model_info.get("sources")
            if not (
                isinstance(sources, list)
                and sources
                and all(isinstance(source, str) and source for source in sources)
                and len(set(sources)) == len(sources)
            ):
                raise ModelLoadingError(
                    f"Model {model_name} must declare unique, non-empty sources."
                )

        # Generate layer URLs from model remote paths. The digest prefix is
        # the cache filename / URL key; the full sha256 is what downloads and
        # cache hits are verified against.
        self._layer_urls: dict[str, str] = {}
        self._layer_sha256: dict[str, str] = {}
        self._layer_sizes: dict[str, int] = {}
        for model_name, model_info in self._models.items():
            if model_info["backend"] != "demucs":
                continue
            sources = model_info["sources"]
            layers = model_info.get("models")
            if not (
                isinstance(layers, list)
                and layers
                and all(isinstance(layer, dict) for layer in layers)
            ):
                raise ModelLoadingError(
                    f"Demucs model {model_name} must contain a non-empty layer list."
                )
            if model_info.get("architecture") != "htdemucs":
                raise ModelLoadingError(
                    f"Demucs model {model_name} must declare architecture 'htdemucs'."
                )
            config = model_info.get("config")
            if not isinstance(config, dict) or config.get("sources") != sources:
                raise ModelLoadingError(
                    f"Demucs model {model_name} must declare a config whose "
                    "sources match metadata."
                )
            for model_entry in layers:
                if model_entry.get("format") != "safetensors":
                    raise ModelLoadingError(
                        f"Layer of model {model_name} must use Safetensors."
                    )
                checksum = model_entry.get("checksum")
                sha = model_entry.get("sha256")
                if (
                    not isinstance(sha, str)
                    or len(sha) != 64
                    or any(char not in "0123456789abcdef" for char in sha)
                    or not isinstance(checksum, str)
                    or len(checksum) < 8
                    or not sha.startswith(checksum)
                ):
                    raise ModelLoadingError(
                        f"Layer of model {model_name} has an invalid checksum/sha256."
                    )
                size = model_entry.get("size_bytes")
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise ModelLoadingError(
                        f"Layer {checksum} of model {model_name} is missing a "
                        "positive size_bytes value."
                    )
                remote_path = model_entry.get("remote")
                if not isinstance(remote_path, str) or not remote_path:
                    raise ModelLoadingError(
                        f"Layer {checksum} of model {model_name} has no remote URL."
                    )
                url = (
                    remote_path
                    if "://" in remote_path
                    else f"{BASE_CDN_URL}/{remote_path}"
                )
                if checksum in self._layer_urls and (
                    self._layer_urls[checksum] != url
                    or self._layer_sha256[checksum] != sha
                    or self._layer_sizes[checksum] != size
                ):
                    raise ModelLoadingError(
                        f"Layer checksum {checksum} has conflicting metadata."
                    )
                self._layer_urls[checksum] = url
                self._layer_sha256[checksum] = sha
                self._layer_sizes[checksum] = size

        # RoFormer entries are a single checkpoint artifact + an inline config,
        # loaded via build_roformer rather than the Demucs layer/CDN path
        # above (which skipped them, having no ``models`` list). Validate their
        # required fields up front so a malformed registry fails at load, not
        # mid-download.
        for model_name, model_info in self._models.items():
            if model_info.get("backend") != "roformer":
                continue
            architecture = model_info.get("architecture")
            if not isinstance(architecture, str) or architecture not in {
                "bs_roformer",
                "mel_band_roformer",
            }:
                raise ModelLoadingError(
                    f"RoFormer model {model_name} has an unknown architecture."
                )
            if (
                not isinstance(model_info.get("config"), dict)
                or not model_info["config"]
            ):
                raise ModelLoadingError(
                    f"RoFormer model {model_name} must declare a non-empty config."
                )
            for field in ("samplerate", "segment_samples"):
                value = model_info[field]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ModelLoadingError(
                        f"RoFormer model {model_name} has invalid {field}: {value}."
                    )
            checkpoint = model_info.get("checkpoint")
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("format") != "safetensors"
                or not str(checkpoint.get("url", "")).startswith("https://")
            ):
                raise ModelLoadingError(
                    f"RoFormer model {model_name} must declare a valid https "
                    "Safetensors checkpoint."
                )
            digest = checkpoint.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ModelLoadingError(
                    f"RoFormer model {model_name} is missing a valid sha256."
                )
            size = checkpoint.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ModelLoadingError(
                    f"RoFormer model {model_name} is missing positive size_bytes."
                )

    def _roformer_cache_path(self, model_info: dict) -> Path:
        """
        Content-addressed cache path for a RoFormer checkpoint.

        :param model_info: The model's registry entry.
        :return: ``<cache dir>/<sha256[:16]>.safetensors``.
        """
        digest = model_info["checkpoint"]["sha256"]
        return get_cache_dir() / f"{digest[:16]}.safetensors"

    def _layer_cache_path(self, checksum: str) -> Path:
        """
        Return the content-addressed cache path for a Demucs layer.

        :param checksum: Registered digest prefix.
        :return: ``<cache dir>/<checksum>.safetensors``.
        """
        return get_cache_dir() / f"{checksum}.safetensors"

    def get_cache_info(self) -> dict[str, dict]:
        """
        Get information about cached models, including partially-cached ones
        (e.g. an interrupted multi-layer download).

        :return: Dictionary mapping each model name with at least one cached
            layer to ``{"layers", "size_bytes", "total_layers", "complete"}``
        """
        cached_models = {}

        # Check which layer files are downloaded. Single stat per file — an
        # exists()-then-stat() pair would race a concurrent removal.
        cached_layers = {}
        for checksum in self._layer_urls:
            layer_path = self._layer_cache_path(checksum)
            try:
                size_bytes = layer_path.stat().st_size
            except OSError:
                continue
            cached_layers[checksum] = {
                "path": str(layer_path),
                "size_bytes": size_bytes,
            }

        for name, info in self._models.items():
            if info.get("backend") == "roformer":
                path = self._roformer_cache_path(info)
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    continue
                digest = info["checkpoint"]["sha256"][:16]
                cached_models[name] = {
                    "layers": {digest: {"path": str(path), "size_bytes": size_bytes}},
                    "size_bytes": size_bytes,
                    "total_layers": 1,
                    "complete": True,
                }
                continue
            if "models" not in info:
                continue
            components = {
                entry["checksum"]: cached_layers[entry["checksum"]]
                for entry in info["models"]
                if entry["checksum"] in cached_layers
            }
            if not components:
                continue
            cached_models[name] = {
                "layers": components,
                "size_bytes": sum(c["size_bytes"] for c in components.values()),
                "total_layers": len(info["models"]),
                "complete": len(components) == len(info["models"]),
            }

        return cached_models

    def sweep_stale_downloads(self) -> int:
        """
        Remove leftover download staging files from interrupted downloads.
        Files that vanish concurrently or can't be removed (e.g. held open by
        an in-flight download on Windows) are skipped. On POSIX an in-flight
        download's staging file *is* removed — that download then fails
        cleanly at verification, consistent with the cache wipe requested.

        :return: Number of files removed
        """
        removed = 0
        for tmp_path in get_cache_dir().glob(f"{STAGING_PREFIX}*"):
            try:
                tmp_path.unlink()
            except OSError:
                continue
            removed += 1
        return removed

    def _download_and_load_layer(
        self,
        url: str,
        cache_path: Path,
        expected_checksum: str,
        expected_size: int,
        model_info: dict,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        model_name: str = "",
        layer_index: int = 1,
        total_layers: int = 1,
    ) -> Model | ModelEnsemble:
        """
        Download and load a model layer from a URL.

        :param url: URL to download the layer from
        :param cache_path: Local path to cache the downloaded layer
        :param expected_checksum: Expected full 64-character SHA-256 digest for verification
        :param expected_size: Exact artifact size from trusted metadata.
        :param model_info: Architecture/config registry entry for construction.
        :param progress_callback: Optional callback for download progress updates
        :param model_name: Name of the model being downloaded
        :param layer_index: Index of the current layer (1-based)
        :param total_layers: Total number of layers to download
        :return: The loaded model
        :raises ModelLoadingError: If download or loading fails
        """
        # Stream the download straight into a temp file (no in-memory copy),
        # verify it, then move it into the cache. A partial or corrupt
        # download can never land at ``cache_path``.
        tmp_path: Path | None = None
        started = time.monotonic()
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=30.0
            ) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                if total_size and total_size != expected_size:
                    raise ModelLoadingError(
                        f"Download size for {url} is {total_size} bytes; "
                        f"expected {expected_size}."
                    )
                downloaded_size = 0

                # Notify callback about layer start
                if progress_callback:
                    progress_callback(
                        "layer_start",
                        {
                            "model_name": model_name,
                            "layer_index": layer_index,
                            "total_layers": total_layers,
                            "layer_size_bytes": total_size,
                        },
                    )

                # Stage the temp file inside the cache dir so the final
                # ``shutil.move`` is a same-filesystem atomic rename rather than
                # a cross-device copy (which a concurrent reader could observe
                # half-written).
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix=STAGING_PREFIX,
                    suffix=cache_path.suffix,
                    dir=cache_path.parent,
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    chunk_counter = 0
                    for chunk in response.iter_bytes(chunk_size=8192):
                        downloaded_size += len(chunk)
                        if downloaded_size > expected_size:
                            raise ModelLoadingError(
                                f"Download from {url} exceeded the expected "
                                f"{expected_size} bytes."
                            )
                        if time.monotonic() - started > DOWNLOAD_DEADLINE_SECONDS:
                            raise ModelLoadingError(
                                f"Download from {url} exceeded the "
                                f"{DOWNLOAD_DEADLINE_SECONDS}-second deadline."
                            )
                        tmp_file.write(chunk)
                        chunk_counter += 1

                        # Update progress every few chunks to avoid too frequent updates
                        if progress_callback and (chunk_counter % 20 == 0):
                            if total_size and total_size > 0:
                                progress_percent = (downloaded_size / total_size) * 100
                            else:
                                # Estimate progress based on chunks downloaded
                                chunk_count = downloaded_size // 8192
                                progress_percent = min(
                                    chunk_count * 0.5, 95
                                )  # Cap at 95%

                            progress_callback(
                                "layer_progress",
                                {
                                    "model_name": model_name,
                                    "layer_index": layer_index,
                                    "total_layers": total_layers,
                                    "progress_percent": progress_percent,
                                    "downloaded_bytes": downloaded_size,
                                    "total_bytes": total_size,
                                },
                            )

            if downloaded_size != expected_size:
                raise ModelLoadingError(
                    f"Download from {url} ended at {downloaded_size} bytes; "
                    f"expected {expected_size}."
                )

            # Verify integrity before making the tensor-only artifact visible
            # in the cache or constructing the allowlisted architecture.
            if progress_callback:
                progress_callback(
                    "layer_progress",
                    {
                        "model_name": model_name,
                        "layer_index": layer_index,
                        "total_layers": total_layers,
                        "progress_percent": 95,
                        "downloaded_bytes": downloaded_size,
                        "total_bytes": total_size,
                        "phase": "verifying",
                    },
                )
            check_size(tmp_path, expected_size)
            check_checksum(tmp_path, expected_checksum)

            # Bytes are verified — now load and move into the cache.
            layer = _load_demucs_layer(tmp_path, model_info)
            shutil.move(str(tmp_path), str(cache_path))
            tmp_path = None

            # Notify callback about layer completion
            if progress_callback:
                progress_callback(
                    "layer_complete",
                    {
                        "model_name": model_name,
                        "layer_index": layer_index,
                        "total_layers": total_layers,
                    },
                )

            return layer

        except httpx.HTTPError as e:
            raise ModelLoadingError(f"Failed to download {url}: {str(e)}")
        except ModelLoadingError:
            raise
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model: {str(e)}")
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    def _download_verified_file(
        self,
        url: str,
        cache_path: Path,
        expected_sha256: str,
        expected_size: int,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        model_name: str = "",
    ) -> None:
        """
        Stream a single file to the cache, verifying its SHA-256 before it
        lands. Used for RoFormer Safetensors artifacts (one file per model).

        :param url: Source URL.
        :param cache_path: Destination path in the cache.
        :param expected_sha256: Full 64-character digest to verify against.
        :param expected_size: Exact artifact size from trusted metadata.
        :param progress_callback: Optional callback (``layer_start`` /
            ``layer_progress`` / ``layer_complete`` events, ``total_layers=1``).
        :param model_name: Model name for progress payloads.
        :raises ModelLoadingError: On download or verification failure.
        """
        tmp_path: Path | None = None
        started = time.monotonic()
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=30.0
            ) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                if total_size and total_size != expected_size:
                    raise ModelLoadingError(
                        f"Download size for {url} is {total_size} bytes; "
                        f"expected {expected_size}."
                    )
                downloaded = 0
                if progress_callback:
                    progress_callback(
                        "layer_start",
                        {
                            "model_name": model_name,
                            "layer_index": 1,
                            "total_layers": 1,
                            "layer_size_bytes": total_size,
                        },
                    )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix=STAGING_PREFIX,
                    suffix=cache_path.suffix,
                    dir=cache_path.parent,
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    counter = 0
                    for chunk in response.iter_bytes(chunk_size=8192):
                        downloaded += len(chunk)
                        if downloaded > expected_size:
                            raise ModelLoadingError(
                                f"Download from {url} exceeded the expected "
                                f"{expected_size} bytes."
                            )
                        if time.monotonic() - started > DOWNLOAD_DEADLINE_SECONDS:
                            raise ModelLoadingError(
                                f"Download from {url} exceeded the "
                                f"{DOWNLOAD_DEADLINE_SECONDS}-second deadline."
                            )
                        tmp_file.write(chunk)
                        counter += 1
                        if progress_callback and counter % 40 == 0 and total_size > 0:
                            progress_callback(
                                "layer_progress",
                                {
                                    "model_name": model_name,
                                    "layer_index": 1,
                                    "total_layers": 1,
                                    "progress_percent": downloaded / total_size * 100,
                                    "downloaded_bytes": downloaded,
                                    "total_bytes": total_size,
                                },
                            )
            if downloaded != expected_size:
                raise ModelLoadingError(
                    f"Download from {url} ended at {downloaded} bytes; "
                    f"expected {expected_size}."
                )
            # Integrity gate before the file is visible in the cache.
            check_size(tmp_path, expected_size)
            check_checksum(tmp_path, expected_sha256)
            shutil.move(str(tmp_path), str(cache_path))
            tmp_path = None
            if progress_callback:
                progress_callback(
                    "layer_complete",
                    {"model_name": model_name, "layer_index": 1, "total_layers": 1},
                )
        except httpx.HTTPError as e:
            raise ModelLoadingError(f"Failed to download {url}: {e}")
        except ModelLoadingError:
            raise
        except Exception as e:
            raise ModelLoadingError(f"Failed to download/verify {url}: {e}")
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    def _get_roformer_model(
        self,
        name: str,
        model_info: dict,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Model:
        """
        Build a RoFormer model, downloading and caching its checkpoint if a
        verified copy isn't already present.

        The checkpoint is a tensor-only Safetensors artifact. Its exact size
        and SHA-256 are verified before strict loading into the allowlisted
        architecture declared in metadata.

        :param name: Model name.
        :param model_info: The model's registry entry.
        :param progress_callback: Optional download-progress callback.
        :return: The constructed, weight-loaded model in eval mode.
        :raises ModelLoadingError: If download, verification, or loading fails.
        """
        checkpoint = model_info["checkpoint"]
        expected = checkpoint["sha256"]
        cache_path = self._roformer_cache_path(model_info)

        if progress_callback:
            progress_callback("download_start", {"model_name": name, "total_layers": 1})

        cached = False
        if cache_path.exists():
            try:
                check_size(cache_path, checkpoint["size_bytes"])
                check_checksum(cache_path, expected)
                cached = True
            except ModelLoadingError as exc:
                if isinstance(exc.__cause__, OSError):
                    raise
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    raise ModelLoadingError(
                        f"Cached checkpoint {cache_path} failed verification and "
                        f"could not be removed: {cleanup_error}"
                    ) from None

        if cached:
            if progress_callback:
                progress_callback(
                    "layer_complete",
                    {
                        "model_name": name,
                        "layer_index": 1,
                        "total_layers": 1,
                        "cached": True,
                    },
                )
        else:
            self._download_verified_file(
                checkpoint["url"],
                cache_path,
                expected,
                checkpoint["size_bytes"],
                progress_callback,
                name,
            )

        if progress_callback:
            progress_callback(
                "download_complete", {"model_name": name, "total_layers": 1}
            )

        try:
            state = load_file(cache_path, device="cpu")
            return build_roformer(
                model_info["architecture"],
                dict(model_info["config"]),
                sources=list(model_info["sources"]),
                samplerate=int(model_info["samplerate"]),
                segment_samples=int(model_info["segment_samples"]),
                state=state,
            )
        except ModelLoadingError:
            raise
        except Exception as e:
            raise ModelLoadingError(f"Failed to build {name} from checkpoint: {e}")

    def _select_layers(
        self, name: str, only_load: str | None = None
    ) -> tuple[list[str], dict]:
        """
        Resolve which layer checksums ``get_model`` needs, honouring the
        ``only_load`` single-specialist optimisation for ensembles.

        :param name: Model name
        :param only_load: If specified, select only the layer specialized for
            this stem (when the model is an ensemble with a one-hot weight row)
        :return: ``(layer_checksums, model_info)``; ``model_info`` has its
            weights stripped when only the specialist layer was selected
        :raises ModelLoadingError: If the model is not found
        """
        if name not in self._models:
            raise ModelLoadingError(
                f"Could not find a model with name {name}. "
                f"Available models: {', '.join(self._models.keys())}"
            )

        model_info = self._models[name]
        if "models" not in model_info:
            # __init__ skips (rather than rejects) metadata entries without a
            # layer list, so custom metadata can reach here.
            raise ModelLoadingError(f"Model {name} has no 'models' list in metadata.")
        weights = model_info.get("weights")
        layer_checksums = [entry["checksum"] for entry in model_info["models"]]

        if only_load and weights and len(weights) > 1:
            # Stem names come from metadata; the constructor enforces ``sources``
            # is present on every registered model.
            stem_names = model_info["sources"]

            # An unknown stem falls through to the full model; validation
            # happens in Separator.
            if only_load in stem_names:
                # Find which model specializes in this stem
                stem_index = stem_names.index(only_load)
                model_index = None

                for i, weight_row in enumerate(weights):
                    if (
                        len(weight_row) > stem_index
                        and abs(weight_row[stem_index] - 1.0) < 1e-6
                        and all(
                            abs(w) < 1e-6
                            for j, w in enumerate(weight_row)
                            if j != stem_index
                        )
                        and all(
                            other_index == i or abs(other_row[stem_index]) < 1e-6
                            for other_index, other_row in enumerate(weights)
                        )
                    ):
                        model_index = i
                        break

                if model_index is not None:
                    # Load only the specialized model; remove weights so the
                    # single layer is treated as identity.
                    layer_checksums = [layer_checksums[model_index]]
                    model_info = dict(model_info)
                    model_info.pop("weights", None)

        return layer_checksums, model_info

    def required_layers(self, name: str, only_load: str | None = None) -> list[str]:
        """
        Return the layer checksums ``get_model(name, only_load)`` would load.
        Useful for cache checks without touching the network.

        :param name: Model name
        :param only_load: Optional stem for the single-specialist optimisation
        :return: List of layer checksums (cache filenames use Safetensors)
        :raises ModelLoadingError: If the model is not found
        """
        layer_checksums, _ = self._select_layers(name, only_load)
        return layer_checksums

    def layer_sha256(self, checksum: str) -> str:
        """
        Return the full 64-character SHA-256 the layer with the given short
        checksum is expected to hash to.

        :param checksum: Digest prefix that names the cache file.
        :return: Full 64-character SHA-256 digest from metadata.
        :raises KeyError: If ``checksum`` is not a registered layer.
        """
        return self._layer_sha256[checksum]

    def get_model(
        self,
        name: str,
        only_load: str | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Model | ModelEnsemble:
        """
        Get a model by name, downloading if necessary.

        :param name: Model name
        :param only_load: If specified, load only the model specialized for this stem
        :param progress_callback: Optional callback for download progress updates
        :return: The requested model
        :raises ModelLoadingError: If the model is not found or fails to load
        """
        info = self._models.get(name)
        if info is not None and info.get("backend") == "roformer":
            return self._get_roformer_model(name, info, progress_callback)

        layer_checksums, model_info = self._select_layers(name, only_load)

        # Download each layer
        layers = []
        total_layers = len(layer_checksums)

        # Notify callback about download start
        if progress_callback:
            progress_callback(
                "download_start",
                {
                    "model_name": name,
                    "total_layers": total_layers,
                },
            )

        for i, layer_checksum in enumerate(layer_checksums):
            if layer_checksum not in self._layer_urls:
                raise ModelLoadingError(f"Layer {layer_checksum} not found in metadata")

            url = self._layer_urls[layer_checksum]

            cache_path = self._layer_cache_path(layer_checksum)
            expected = self._layer_sha256[layer_checksum]
            expected_size = self._layer_sizes[layer_checksum]

            # Check if file exists and validate its integrity
            if cache_path.exists():
                try:
                    check_size(cache_path, expected_size)
                    check_checksum(cache_path, expected)

                    # Safetensors parsing and strict architecture loading.
                    layer = _load_demucs_layer(cache_path, model_info)
                    layers.append(layer)

                    # Notify callback about cached layer
                    if progress_callback:
                        progress_callback(
                            "layer_complete",
                            {
                                "model_name": name,
                                "layer_index": i + 1,
                                "total_layers": total_layers,
                                "cached": True,
                            },
                        )
                    continue
                except OSError as exc:
                    # A direct read failure racing a concurrent removal is not
                    # corruption — keep the cache and surface the error.
                    raise ModelLoadingError(
                        f"Could not read cached layer {cache_path}: {exc}"
                    ) from exc
                except ModelLoadingError as exc:
                    if isinstance(exc.__cause__, OSError):
                        # A file lock, EIO, or permissions failure is not
                        # corruption; preserve the possibly-valid cache.
                        raise
                    # The cached file is corrupt: discard and redownload.
                    # ``missing_ok`` covers a concurrent process unlinking it
                    # first.
                    try:
                        cache_path.unlink(missing_ok=True)
                    except OSError as e:
                        # NOTE: deliberately not ``from e`` — an OSError cause
                        # means "transient read failure" to the guard above,
                        # and this is a failed *corruption* cleanup.
                        raise ModelLoadingError(
                            f"Cached layer {cache_path} failed verification and "
                            f"could not be removed: {e}"
                        ) from None

            # Download and load the layer
            layer = self._download_and_load_layer(
                url=url,
                cache_path=cache_path,
                expected_checksum=expected,
                expected_size=expected_size,
                model_info=model_info,
                progress_callback=progress_callback,
                model_name=name,
                layer_index=i + 1,
                total_layers=total_layers,
            )
            layers.append(layer)
        # Notify callback about download completion
        if progress_callback:
            progress_callback(
                "download_complete",
                {
                    "model_name": name,
                    "total_layers": total_layers,
                },
            )

        # Optimization: Return raw model for single models with default weights
        weights = model_info.get("weights")
        segment = model_info.get("segment")

        # Check if this is a single model with identity weights (or no weights specified)
        if len(layers) == 1:
            is_identity_weights = weights is None or (
                len(weights) == 1
                and len(weights[0]) == len(layers[0].sources)
                and all(abs(w - 1.0) < 1e-6 for w in weights[0])
            )

            if is_identity_weights:
                # Return the raw model directly for better performance
                model = layers[0]

                # A metadata override can shorten, never enlarge, the model's
                # configured training segment.
                if segment is not None:
                    model.max_allowed_segment = min(
                        float(segment), float(model.max_allowed_segment)
                    )

                return model

        # Use ModelEnsemble for true ensembles or models with custom weights
        return ModelEnsemble(layers, weights, segment)

    def list_models(self) -> dict[str, dict]:
        """
        List all available models.

        :return: Dictionary mapping model names to their metadata (deep
            copies — mutating them does not affect repository state)
        """
        return {name: copy.deepcopy(info) for name, info in self._models.items()}

    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the cache.

        :param name: Model name
        :return: True if the model was removed, False if not found
        :raises ModelLoadingError: If a cached layer exists but can't be
            removed (e.g. permissions)
        """
        if name not in self._models:
            return False

        info = self._models[name]
        removed_any = False

        if info.get("backend") == "roformer":
            path = self._roformer_cache_path(info)
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return False
            except OSError as e:
                raise ModelLoadingError(
                    f"Could not remove cached checkpoint {path}: {e}"
                ) from e

        # Remove all layer files for this model
        for layer_info in self._models[name].get("models", []):
            layer_checksum = layer_info["checksum"]
            layer_path = self._layer_cache_path(layer_checksum)
            try:
                layer_path.unlink()
            except FileNotFoundError:
                continue
            except OSError as e:
                raise ModelLoadingError(
                    f"Could not remove cached layer {layer_path}: {e}"
                ) from e
            removed_any = True

        return removed_any
