# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import pickle
import shutil
import tempfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import httpx
import torch

from .apply import Model, ModelEnsemble
from .exceptions import ModelLoadingError
from .states import load_model

BASE_CDN_URL = "https://dl.fbaipublicfiles.com/demucs"


def check_checksum(path: Path, checksum: str) -> None:
    """
    Verify that a file matches an expected checksum.

    :param path: Path to the file to check
    :param checksum: Expected SHA-256 checksum (first 8 characters)
    :raises ModelLoadingError: If the actual checksum does not match the expected one
    """
    sha = sha256()
    with open(path, "rb") as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[:8]
    if actual_checksum != checksum:
        raise ModelLoadingError(
            f"Invalid checksum for file {path}, "
            f"expected {checksum} but got {actual_checksum}"
        )


def get_cache_dir() -> Path:
    """
    Get the cache directory for Demucs models.

    :return: Path to the cache directory
    """
    home = Path.home()
    cache_dir = home / ".demucs" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class ModelRepository:
    """Repository system for accessing models."""

    def __init__(self) -> None:
        """
        Initialize the model repository from metadata.json.

        :raises ModelLoadingError: If the metadata structure is invalid
        """
        # Determine metadata_path relative to this file
        current_file_path = Path(__file__)
        self.metadata_path = current_file_path.parent / "metadata.json"

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        if "models" in self.metadata:
            self._models = self.metadata["models"]
        else:
            raise ModelLoadingError(
                "Invalid metadata structure: 'models' key not found in metadata.json. "
                "The expected format is a top-level 'models' dictionary."
            )

        # Generate layer URLs from model remote paths
        self._layer_urls = {}
        for model_name, model_info in self._models.items():
            if "models" in model_info:
                for model_entry in model_info["models"]:
                    checksum = model_entry["checksum"]
                    remote_path = model_entry["remote"]
                    self._layer_urls[checksum] = f"{BASE_CDN_URL}/{remote_path}"

    def get_cache_info(self) -> dict[str, dict]:
        """
        Get information about cached models.

        :return: Dictionary mapping model names to their cache info
        """
        cache_dir = get_cache_dir()
        cached_models = {}

        # Check which layer files are downloaded
        cached_layers = {}
        for checksum, url in self._layer_urls.items():
            filename = f"{checksum}.th"
            layer_path = cache_dir / filename

            if layer_path.exists():
                size_bytes = layer_path.stat().st_size
                cached_layers[checksum] = {
                    "path": str(layer_path),
                    "size_bytes": size_bytes,
                }

        # Handle models
        for name, info in self._models.items():
            if "models" in info:
                component_layers = info["models"]
                all_cached = True
                total_size = 0
                components = {}

                for component in component_layers:
                    checksum = component["checksum"]
                    if checksum in cached_layers:
                        components[checksum] = cached_layers[checksum]
                        total_size += cached_layers[checksum]["size_bytes"]
                    else:
                        all_cached = False

                if all_cached:
                    cached_models[name] = {
                        "layers": components,
                        "size_bytes": total_size,
                    }

        return cached_models

    def _download_and_load_layer(
        self,
        url: str,
        cache_path: Path,
        expected_checksum: str,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        model_name: str = "",
        layer_index: int = 1,
        total_layers: int = 1,
    ) -> Model | ModelEnsemble:
        """
        Download and load a model layer from a URL.

        :param url: URL to download the layer from
        :param cache_path: Local path to cache the downloaded layer
        :param expected_checksum: Expected SHA-256 checksum for verification
        :param progress_callback: Optional callback for download progress updates
        :param model_name: Name of the model being downloaded
        :param layer_index: Index of the current layer (1-based)
        :param total_layers: Total number of layers to download
        :return: The loaded model
        :raises ModelLoadingError: If download or loading fails
        """
        # Download the file to memory first
        try:
            with httpx.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                # Download to memory first
                buffer = BytesIO()
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

                chunk_counter = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    buffer.write(chunk)
                    downloaded_size += len(chunk)
                    chunk_counter += 1

                    # Update progress every few chunks to avoid too frequent updates
                    if progress_callback and (chunk_counter % 20 == 0):
                        if total_size and total_size > 0:
                            progress_percent = (downloaded_size / total_size) * 100
                        else:
                            # Estimate progress based on chunks downloaded
                            chunk_count = downloaded_size // 8192
                            progress_percent = min(chunk_count * 0.5, 95)  # Cap at 95%

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

                buffer.seek(0)

                # Try to load as a PyTorch model directly from memory
                tmp_path: Path | None = None
                try:
                    # Notify callback about loading phase
                    if progress_callback:
                        progress_callback(
                            "layer_progress",
                            {
                                "model_name": model_name,
                                "layer_index": layer_index,
                                "total_layers": total_layers,
                                "progress_percent": 90,
                                "downloaded_bytes": downloaded_size,
                                "total_bytes": total_size,
                                "phase": "loading",
                            },
                        )

                    # Save to a temporary file first
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".th"
                    ) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        tmp_file.write(buffer.getvalue())

                    # Verify integrity BEFORE unpickling. ``load_model`` /
                    # ``torch.load`` use ``weights_only=False`` because the
                    # checkpoint format pickles the model class and its init
                    # args (``weights_only=True`` refuses to reconstruct those),
                    # which means loading can execute arbitrary code. The
                    # SHA-256 from the shipped metadata is the only integrity
                    # gate we have, so it must pass before the bytes reach the
                    # unpickler — not after, as it used to. The cache-load paths
                    # in ``get_model`` already check_checksum before torch.load.
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
                    check_checksum(tmp_path, expected_checksum)

                    # Bytes are verified — now load and move into the cache.
                    model_data = torch.load(
                        tmp_path, map_location="cpu", weights_only=False
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(tmp_path), str(cache_path))

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

                    return load_model(model_data)

                except Exception as e:
                    # Clean up temp file. ``tmp_path`` may still be None if the
                    # failure happened in the loading-phase callback above,
                    # before the temp file was created.
                    if tmp_path is not None and tmp_path.exists():
                        tmp_path.unlink()
                    raise ModelLoadingError(f"Failed to load model: {str(e)}")

        except httpx.HTTPError as e:
            raise ModelLoadingError(f"Failed to download {url}: {str(e)}")

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
        # Only load models, not individual layers
        if name not in self._models:
            raise ModelLoadingError(
                f"Could not find a model with name {name}. "
                f"Available models: {', '.join(self._models.keys())}"
            )

        model_info = self._models[name]
        weights = model_info.get("weights")
        layer_checksums = [entry["checksum"] for entry in model_info["models"]]

        # Check if we should load only a specific stem model
        if only_load and weights and len(weights) > 1:
            # This is a model ensemble - try to find the specialized model for this stem
            cache_dir = get_cache_dir()

            # Load first model to get stem names
            first_checksum = layer_checksums[0]
            first_cache_path = cache_dir / f"{first_checksum}.th"

            # Download first model if needed to get stem names
            if not first_cache_path.exists():
                if first_checksum not in self._layer_urls:
                    raise ModelLoadingError(
                        f"Layer {first_checksum} not found in metadata"
                    )

                url = self._layer_urls[first_checksum]
                first_model = self._download_and_load_layer(
                    url=url,
                    cache_path=first_cache_path,
                    expected_checksum=first_checksum,
                    progress_callback=None,
                    model_name=name,
                    layer_index=1,
                    total_layers=1,
                )
            else:
                try:
                    check_checksum(first_cache_path, first_checksum)
                    first_model = load_model(
                        torch.load(
                            first_cache_path, map_location="cpu", weights_only=False
                        )
                    )
                except (
                    ModelLoadingError,
                    RuntimeError,
                    EOFError,
                    pickle.UnpicklingError,
                ):
                    # The cached file is corrupt/unreadable: discard and redownload.
                    if first_cache_path.exists():
                        first_cache_path.unlink()
                    url = self._layer_urls[first_checksum]
                    first_model = self._download_and_load_layer(
                        url=url,
                        cache_path=first_cache_path,
                        expected_checksum=first_checksum,
                        progress_callback=None,
                        model_name=name,
                        layer_index=1,
                        total_layers=1,
                    )

            stem_names = first_model.sources

            if only_load not in stem_names:
                # Stem doesn't exist - fall through to load full model
                # Validation will happen in Separator
                pass
            else:
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
                    ):
                        model_index = i
                        break

                if model_index is not None:
                    # Load only the specialized model
                    layer_checksums = [layer_checksums[model_index]]
                    # Update model_info to indicate this is now a single-model config
                    # Remove weights so it's treated as identity
                    model_info = dict(model_info)  # Make a copy
                    model_info.pop("weights", None)  # Remove weights

        # Download each layer
        layers = []
        cache_dir = get_cache_dir()
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

            # Determine cache path
            filename = f"{layer_checksum}.th"
            cache_path = cache_dir / filename

            # Check if file exists and validate its integrity
            if cache_path.exists():
                try:
                    # Validate checksum
                    check_checksum(cache_path, layer_checksum)

                    # Try to load the model
                    layer = load_model(
                        torch.load(cache_path, map_location="cpu", weights_only=False)
                    )
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
                except (
                    ModelLoadingError,
                    RuntimeError,
                    EOFError,
                    pickle.UnpicklingError,
                ):
                    # The cached file is corrupt/unreadable: discard and redownload.
                    if cache_path.exists():
                        cache_path.unlink()

            # Download and load the layer
            layer = self._download_and_load_layer(
                url=url,
                cache_path=cache_path,
                expected_checksum=layer_checksum,
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

                # Apply segment override if needed
                if segment is not None:
                    # Embedded import to avoid circular dependency: repo -> htdemucs -> repo
                    from .htdemucs import HTDemucs

                    if (
                        not isinstance(model, HTDemucs)
                        or segment <= model.max_allowed_segment
                    ):
                        model.max_allowed_segment = segment

                return model

        # Use ModelEnsemble for true ensembles or models with custom weights
        return ModelEnsemble(layers, weights, segment)

    def list_models(self) -> dict[str, dict]:
        """
        List all available models.

        :return: Dictionary mapping model names to their metadata
        """
        result = {}

        # Add models
        for name, info in self._models.items():
            result[name] = info

        return result

    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the cache.

        :param name: Model name
        :return: True if the model was removed, False if not found
        """
        if name not in self._models:
            return False

        cache_dir = get_cache_dir()
        removed_any = False

        # Remove all layer files for this model
        for layer_info in self._models[name].get("models", []):
            layer_checksum = layer_info["checksum"]
            filename = f"{layer_checksum}.th"
            layer_path = cache_dir / filename
            if layer_path.exists():
                layer_path.unlink()
                removed_any = True

        return removed_any
