# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import math
import os
import tempfile
import time
import warnings
from contextlib import contextmanager
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from safetensors import SafetensorError
from safetensors.torch import load_file

# Importing the architecture modules is what registers their builders;
# the repository dispatches through backends rather than naming them.
from . import backends, scnet  # noqa: F401
from .apply import (
    COMBINE_DEFAULT,
    Model,
    ModelEnsemble,
    canonical_combine,
    resolve_combine_params,
    sole_contributor,
    validate_combine_weights,
)
from .config_io import load_mapping
from .exceptions import ModelLoadingError, ValidationError
from .htdemucs import HTDemucs

BASE_CDN_URL = "https://dl.fbaipublicfiles.com/demucs"

# Prefix for download staging files in the cache dir. Shared by the writer
# (``_download_verified_file``) and the sweeper (``sweep_stale_downloads``)
# so the two can't silently drift apart.
STAGING_PREFIX = ".unblend-download-"
DOWNLOAD_DEADLINE_SECONDS = 2 * 60 * 60
STAGING_STALE_SECONDS = DOWNLOAD_DEADLINE_SECONDS + 5 * 60
LOCK_TIMEOUT_SECONDS = DOWNLOAD_DEADLINE_SECONDS + 10 * 60


@contextmanager
def _artifact_lock(cache_path: Path) -> Iterator[None]:
    """
    Serialize validation, download, and promotion for one cache artifact.

    :param cache_path: Final content-addressed cache path.
    :return: Context manager yielding while this artifact's lock is held.
    """
    lock_path = cache_path.with_name(f".{cache_path.name}.lock")
    lock = FileLock(lock_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock.acquire(timeout=LOCK_TIMEOUT_SECONDS)
    except FileLockTimeout as exc:
        raise ModelLoadingError(
            f"Timed out waiting for model cache lock {lock_path}."
        ) from exc
    except OSError as exc:
        raise ModelLoadingError(
            f"Could not create/acquire model cache lock {lock_path}: {exc}"
        ) from exc
    try:
        # Deliberately keep the caller's exception outside the acquisition
        # handlers above: domain errors raised in the critical section must
        # retain their original type and context.
        yield
    finally:
        lock.release()


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


def _artifact_path(spec: dict) -> Path | None:
    """
    The local, user-owned file an artifact names, if it names one.

    :param spec: An artifact entry (a Demucs layer or a ``checkpoint``).
    :return: The expanded path, or ``None`` for a remote artifact.
    """
    local = spec.get("path")
    if not isinstance(local, str) or not local:
        return None
    return Path(local).expanduser()


def _artifact_url(spec: dict) -> str | None:
    """
    The URL an artifact is downloaded from, if it is remote.

    ``url`` is the canonical spelling. ``remote`` is the shipped Demucs
    layers' historical spelling and may be a path relative to Meta's CDN,
    which is where those weights have always lived.

    :param spec: An artifact entry (a Demucs layer or a ``checkpoint``).
    :return: An absolute URL, or ``None`` for a local artifact.
    """
    url = spec.get("url")
    if isinstance(url, str) and url:
        return url
    remote = spec.get("remote")
    if not isinstance(remote, str) or not remote:
        return None
    return remote if "://" in remote else f"{BASE_CDN_URL}/{remote}"


def _artifact_cache_key(spec: dict) -> str:
    """
    Cache filename stem for a remote artifact.

    Content-addressed, so two models naming the same checkpoint share one
    cached file whatever else their entries say.

    :param spec: A remote artifact entry.
    :return: The digest prefix that names its cache file.
    """
    return spec["sha256"][:16]


def _artifact_cache_path(spec: dict) -> Path:
    """
    Where a remote artifact is cached once downloaded.

    :param spec: A remote artifact entry.
    :return: ``<cache dir>/<digest prefix>.safetensors``.
    """
    return get_cache_dir() / f"{_artifact_cache_key(spec)}.safetensors"


def _validate_artifact(spec: object, label: str) -> None:
    """
    Check one weight artifact's registry entry before anything reads it.

    Every backend describes its weights the same way: a Safetensors file named
    either by a local ``path`` — a file the user already has, used where it
    lies — or by an https ``url``, downloaded once into the content-addressed
    cache and reused from then on. A download has to be verifiable, so remote
    artifacts require an exact ``sha256`` and ``size_bytes``; for a local file
    both are optional and enforced only when supplied.

    :param spec: The candidate artifact entry.
    :param label: Human-readable prefix for error messages.
    :raises ModelLoadingError: If the entry is malformed or unverifiable.
    """
    if not isinstance(spec, dict):
        raise ModelLoadingError(f"{label} must be a dictionary.")
    if spec.get("format") != "safetensors":
        raise ModelLoadingError(f"{label} must use Safetensors.")

    path = _artifact_path(spec)
    url = _artifact_url(spec)
    if spec.get("path") is not None and path is None:
        raise ModelLoadingError(
            f"{label} declares an invalid local path {spec['path']!r}."
        )
    if path is not None and url is not None:
        raise ModelLoadingError(
            f"{label} declares both a local path and a remote url; pick one."
        )
    if path is None and url is None:
        raise ModelLoadingError(f"{label} must declare a local path or an https url.")
    if url is not None and not url.startswith("https://"):
        raise ModelLoadingError(f"{label} must be served over https, got {url!r}.")

    digest = spec.get("sha256")
    if url is not None or digest is not None:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ModelLoadingError(f"{label} is missing a valid sha256.")
    size = spec.get("size_bytes")
    if url is not None or size is not None:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ModelLoadingError(f"{label} is missing a positive size_bytes value.")


def _emit(
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
    event: str,
    **payload: Any,
) -> None:
    """
    Deliver one progress event, if the caller asked for them.

    :param progress_callback: The caller's callback, or ``None``.
    :param event: Event name.
    :param payload: Event fields.
    """
    if progress_callback is not None:
        progress_callback(event, payload)


def _read_state(path: Path) -> dict:
    """
    Read a verified Safetensors artifact into a state dict.

    :param path: Path to the artifact.
    :return: Its tensors.
    :raises ModelLoadingError: If the file cannot be read.
    """
    try:
        return load_file(path, device="cpu")
    except (OSError, SafetensorError) as exc:
        raise ModelLoadingError(
            f"Failed to read verified checkpoint {path}: {exc}"
        ) from exc


def _build_demucs_layer(state: dict, member: dict, label: str) -> HTDemucs:
    """
    Build one allowlisted HTDemucs from pickle-free weights.

    Constructor configuration lives in trusted registry metadata; the artifact
    contains tensors only. A malformed Safetensors file fails closed and never
    falls back to a pickle loader.

    :param state: Tensors read from the verified artifact.
    :param member: Resolved member carrying architecture and config.
    :param label: Human-readable prefix for error messages.
    :return: Strictly weight-loaded HTDemucs model.
    :raises ModelLoadingError: If construction or strict loading fails.
    """
    if member.get("architecture") not in DEMUCS_ARCHITECTURES:
        raise ModelLoadingError(
            f"Unsupported Demucs architecture {member.get('architecture')!r}."
        )
    try:
        model = HTDemucs(**dict(member["config"]))
        model.load_state_dict(state, strict=True)
        return model
    except (KeyError, TypeError, RuntimeError, SafetensorError, ValueError) as exc:
        raise ModelLoadingError(f"Failed to build {label}: {exc}") from exc


#: The backend whose weights ship as a bag of per-source layers rather than a
#: single checkpoint, and the architectures it builds. Special-cased because it
#: predates the builder registry; every other family registers itself.
DEMUCS_BACKEND = "demucs"
DEMUCS_ARCHITECTURES = frozenset({"htdemucs"})

#: Backend reported for an entry whose members are not all built by one family
#: — an ensemble mixing, say, an HTDemucs with an SCNet.
ENSEMBLE_BACKEND = "ensemble"


def _known_backends() -> frozenset[str]:
    """
    Backend names ``metadata.yaml`` may declare.

    ``demucs`` is special-cased because its weights ship as a bag of layers
    rather than one checkpoint; everything else registers a builder, so adding
    an architecture does not require editing this module.

    :return: The accepted backend names.
    """
    return (
        frozenset({DEMUCS_BACKEND, ENSEMBLE_BACKEND})
        | backends.single_checkpoint_backends()
    )


def _architectures_for(backend: str) -> frozenset[str]:
    """
    Architectures a backend can build.

    :param backend: Backend name.
    :return: Its architecture names.
    """
    if backend == DEMUCS_BACKEND:
        return DEMUCS_ARCHITECTURES
    return backends.architectures(backend)


def _known_architectures() -> frozenset[str]:
    """
    Every architecture any registered backend can build.

    :return: The accepted architecture names.
    """
    return frozenset().union(
        *(_architectures_for(backend) for backend in _known_backends())
    )


def _backend_for(label: str, info: dict) -> str:
    """
    Resolve which backend builds a model, or one ensemble member.

    ``backend`` names the loader family (how the weights are packaged) and
    ``architecture`` the class within it. Since every architecture belongs to
    exactly one family, naming the architecture is enough — entries and members
    alike can leave the backend out.

    :param label: Human-readable prefix for error messages.
    :param info: A registry entry or a resolved member.
    :return: The backend name.
    :raises ModelLoadingError: If neither a known backend nor a known
        architecture is named.
    """
    declared = info.get("backend")
    if declared is not None:
        if declared not in _known_backends():
            raise ModelLoadingError(f"{label} has unknown backend {declared!r}.")
        return declared

    architecture = info.get("architecture")
    derived = None
    if isinstance(architecture, str):
        derived = (
            DEMUCS_BACKEND
            if architecture in DEMUCS_ARCHITECTURES
            else backends.backend_for_architecture(architecture)
        )
    if derived is None:
        raise ModelLoadingError(
            f"{label} names neither a backend nor a known architecture (got "
            f"architecture {architecture!r}; known architectures: "
            f"{', '.join(sorted(_known_architectures()))})."
        )
    return derived


def _entry_member_specs(model_name: str, model_info: dict) -> list[dict]:
    """
    A model's member specs exactly as written, before inheritance.

    An entry names its weights one of three ways: ``checkpoint`` for a single
    model, ``members`` for an ensemble, or ``models`` — the Demucs bags'
    original spelling, where each item *is* an artifact rather than a member
    wrapping one.

    :param model_name: Model name, for error messages.
    :param model_info: The model's registry entry.
    :return: The raw member specs, one per file the model loads.
    :raises ModelLoadingError: If zero or several spellings are used, or the
        list is empty or malformed.
    """
    present = [
        key
        for key in ("checkpoint", "members", "models")
        if model_info.get(key) is not None
    ]
    if len(present) != 1:
        raise ModelLoadingError(
            f"Model {model_name} must declare exactly one of 'checkpoint', "
            f"'members' or 'models'; found {present or ['none']}."
        )
    if present[0] == "checkpoint":
        return [{"checkpoint": model_info["checkpoint"]}]
    raw = model_info[present[0]]
    noun = "layer" if present[0] == "models" else "member"
    if not (
        isinstance(raw, list) and raw and all(isinstance(item, dict) for item in raw)
    ):
        raise ModelLoadingError(
            f"Model {model_name} must declare a non-empty {noun} list "
            f"under {present[0]!r}."
        )
    return raw


def _member_artifact(spec: dict) -> dict | None:
    """
    The weight artifact one member spec points at.

    :param spec: A raw member spec.
    :return: Its artifact entry, or ``None`` for a spec that references another
        registered model rather than naming a file.
    """
    if "checkpoint" in spec:
        return spec["checkpoint"]
    if "model" in spec:
        return None
    # A Demucs layer is the artifact: ``{"format": ..., "remote": ...}``.
    return spec


def _embedded_member_fields(artifact: object) -> dict:
    """
    Registry fields a local Safetensors artifact records about itself.

    ``unblend models import`` writes architecture, stems, geometry and config
    into the file's header, which is covered by its hash — so a file can carry
    its own description and an entry need only say where it is. Remote
    artifacts are not consulted: there is nothing on disk to read yet.

    :param artifact: A member's artifact entry.
    :return: The embedded fields, empty if there are none to read.
    """
    if not isinstance(artifact, dict):
        return {}
    path = _artifact_path(artifact)
    if path is None or not path.is_file():
        return {}
    # Imported lazily: this is a cold path, and importer pulls in the
    # architectures purely to verify them.
    from .importer import read_embedded_fields

    return read_embedded_fields(path)


def _member_field(field: str, spec: dict, model_info: dict, embedded: dict) -> Any:
    """
    Resolve one member field: what the member says, else the entry, else what
    the artifact records about itself.

    :param field: Field name.
    :param spec: The raw member spec.
    :param model_info: The entry the member belongs to.
    :param embedded: Fields the artifact describes about itself.
    :return: The resolved value, or ``None`` if nothing states it.
    """
    if field in spec:
        return spec[field]
    if field in model_info:
        return model_info[field]
    return embedded.get(field)


def _validate_entry(model_name: str, model_info: object) -> dict:
    """
    Validate the parts of a registry entry that stand alone.

    Everything that depends on the entry's members — their architectures, the
    weight matrix's shape, whether a referenced model exists — is checked once
    members are resolved, which needs the rest of the registry.

    ``license`` is deliberately unchecked: it is a free-form label passed
    through to ``models list`` and ``list_models``, not something Unblend can
    meaningfully validate.

    :param model_name: Model name.
    :param model_info: The candidate entry.
    :return: The entry.
    :raises ModelLoadingError: If the entry is malformed.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ModelLoadingError("Every model name must be a non-empty string.")
    if not isinstance(model_info, dict):
        raise ModelLoadingError(f"Model {model_name} metadata must be a dictionary.")

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
    return model_info


def _validate_member(label: str, member: dict) -> dict:
    """
    Validate one resolved member and record the backend that builds it.

    :param label: Human-readable prefix for error messages.
    :param member: A resolved member, with entry fields already inherited.
    :return: The member, with ``backend`` filled in.
    :raises ModelLoadingError: If the member is malformed.
    """
    architecture = member.get("architecture")
    backend = _backend_for(label, member)
    buildable = _architectures_for(backend)
    if not isinstance(architecture, str) or architecture not in buildable:
        raise ModelLoadingError(
            f"{label} declares architecture {architecture!r}, which the "
            f"{backend!r} backend cannot build; expected one of "
            f"{', '.join(sorted(buildable))}."
        )

    config = member.get("config")
    if not isinstance(config, dict) or not config:
        raise ModelLoadingError(f"{label} must declare a non-empty config.")

    if backend == DEMUCS_BACKEND:
        if config.get("sources") != member["sources"]:
            raise ModelLoadingError(
                f"{label} must declare a config whose sources match metadata."
            )
    else:
        for field in ("samplerate", "segment_samples"):
            value = member.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelLoadingError(f"{label} has invalid {field}: {value}.")

    _validate_artifact(
        member.get("artifact"), f"Checkpoint of {label[0].lower() + label[1:]}"
    )
    member["backend"] = backend
    return member


def _validate_entry_combination(
    model_name: str, model_info: dict, member_count: int
) -> None:
    """
    Validate how an entry's members are combined, without touching the network.

    :param model_name: Model name.
    :param model_info: The model's registry entry.
    :param member_count: How many members the entry resolved to.
    :raises ModelLoadingError: If the mode, its parameters, or the weight
        matrix cannot be used.
    """
    weights = model_info.get("weights")
    sources = model_info["sources"]

    if weights is not None:
        if not isinstance(weights, list) or len(weights) != member_count:
            raise ModelLoadingError(
                f"Model {model_name} must declare one weight row per member "
                f"({member_count}), got "
                f"{len(weights) if isinstance(weights, list) else weights!r}."
            )
        for row_index, row in enumerate(weights):
            if not isinstance(row, list) or len(row) != len(sources):
                raise ModelLoadingError(
                    f"Model {model_name} weight row {row_index} must contain "
                    f"{len(sources)} source weights."
                )
            for column, value in enumerate(row):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                ):
                    raise ModelLoadingError(
                        f"Model {model_name} weight [{row_index}][{column}] "
                        "must be a finite number."
                    )
        for column, source in enumerate(sources):
            if all(abs(float(row[column])) <= 1e-9 for row in weights):
                raise ModelLoadingError(
                    f"Model {model_name} has no member contributing to source "
                    f"{source!r}."
                )

    # These speak ValidationError; from the registry's side a bad mode or bad
    # STFT geometry is just malformed metadata.
    try:
        canonical_combine(model_info.get("combine", COMBINE_DEFAULT))
        validate_combine_weights(model_info.get("combine", COMBINE_DEFAULT), weights)
        resolve_combine_params(model_info.get("combine_params"))
    except ValidationError as exc:
        raise ModelLoadingError(f"Model {model_name}: {exc}") from exc


def get_cache_dir() -> Path:
    """
    Get the cache directory for downloaded models.

    ``UNBLEND_CACHE_DIR`` takes precedence. The former
    ``DEMUCS_CACHE_DIR`` name remains a deprecated fallback so renamed CLI
    aliases do not silently download into a different filesystem. Without
    either variable, the cache defaults to ``~/.unblend/models``.

    :return: Path to the cache directory
    """
    override = os.environ.get("UNBLEND_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    legacy_override = os.environ.get("DEMUCS_CACHE_DIR")
    if legacy_override:
        warnings.warn(
            "DEMUCS_CACHE_DIR is deprecated; use UNBLEND_CACHE_DIR instead. "
            "Legacy .th cache artifacts are not reused by unblend.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Path(legacy_override).expanduser().resolve()
    return Path.home() / ".unblend" / "models"


class ModelRepository:
    """Repository system for accessing models."""

    def __init__(
        self,
        metadata_path: Path | None = None,
        extra_models: "Path | str | list[Path | str] | None" = None,
    ) -> None:
        """
        Initialize the model repository.

        :param metadata_path: Path to a metadata file; defaults to the shipped
            ``unblend/metadata.yaml``. Mainly useful in tests.
        :param extra_models: Additional models files to overlay on the shipped
            registry, as a path or list of paths. Defaults to the paths in
            ``UNBLEND_EXTRA_MODELS`` (os.pathsep-separated).
        :raises ModelLoadingError: If the metadata structure is invalid
        """
        if metadata_path is None:
            metadata_path = Path(__file__).parent / "metadata.yaml"
        self.metadata_path = metadata_path

        try:
            self.metadata = load_mapping(Path(self.metadata_path))
        except (OSError, ValueError) as exc:
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
        self._merge_extra_models(extra_models)
        if not self._models:
            raise ModelLoadingError("Model metadata must contain at least one model.")

        # Validate what stands alone, then resolve every entry into members --
        # which may reference other entries -- and validate those. Shipped and
        # user-supplied entries go through exactly the same checks, so a
        # malformed registry fails here rather than part-way through a download.
        self._models = {
            model_name: _validate_entry(model_name, model_info)
            for model_name, model_info in self._models.items()
        }
        self._members: dict[str, list[dict]] = {}
        for model_name in self._models:
            self._resolve_members(model_name)

        for model_name, members in self._members.items():
            model_info = self._models[model_name]
            _validate_entry_combination(model_name, model_info, len(members))
            # An entry's backend follows from its members: one family if they
            # agree, ``ensemble`` if they don't.
            used = {member["backend"] for member in members}
            derived = used.pop() if len(used) == 1 else ENSEMBLE_BACKEND
            declared = model_info.get("backend")
            if declared is None:
                self._models[model_name] = {**model_info, "backend": derived}
            elif declared != derived:
                raise ModelLoadingError(
                    f"Model {model_name} declares backend {declared!r} but its "
                    f"members are built by {derived!r}."
                )
        self.metadata["models"] = self._models

        # Index every remote artifact by the digest prefix that names its cache
        # file; the full sha256 is what downloads and cache hits are verified
        # against. Local artifacts are files the user already owns, so there is
        # nothing to index. Two models naming the same artifact share one cache
        # entry, which is what makes an ensemble of registered models free.
        self._layer_urls: dict[str, str] = {}
        self._layer_sha256: dict[str, str] = {}
        self._layer_sizes: dict[str, int] = {}
        for model_name, members in self._members.items():
            for member in members:
                artifact = member["artifact"]
                url = _artifact_url(artifact)
                if url is None:
                    continue
                key = _artifact_cache_key(artifact)
                sha = artifact["sha256"]
                size = artifact["size_bytes"]
                if key in self._layer_urls and (
                    self._layer_urls[key] != url
                    or self._layer_sha256[key] != sha
                    or self._layer_sizes[key] != size
                ):
                    raise ModelLoadingError(f"Artifact {key} has conflicting metadata.")
                self._layer_urls[key] = url
                self._layer_sha256[key] = sha
                self._layer_sizes[key] = size

    def _resolve_members(
        self, model_name: str, resolving: tuple[str, ...] = ()
    ) -> list[dict]:
        """
        Expand one entry into fully-specified members.

        A member inherits anything it does not state from its entry, so a bag
        of same-architecture layers stays terse while a heterogeneous ensemble
        can spell each member out. A member may also name another registered
        model, which is how an ensemble is built from models that already ship.

        :param model_name: Model name.
        :param resolving: Entries currently being resolved, for cycle detection.
        :return: The resolved members, in load order.
        :raises ModelLoadingError: If a member is malformed, references an
            unknown or non-single model, or forms a reference cycle.
        """
        if model_name in self._members:
            return self._members[model_name]
        if model_name in resolving:
            chain = " -> ".join(resolving + (model_name,))
            raise ModelLoadingError(
                f"Model {model_name} is part of a member reference cycle: {chain}."
            )

        model_info = self._models[model_name]
        sources = list(model_info["sources"])
        resolved: list[dict] = []
        for index, spec in enumerate(
            _entry_member_specs(model_name, model_info), start=1
        ):
            label = f"Member {index} of model {model_name}"
            reference = spec.get("model")
            if reference is not None:
                if not isinstance(reference, str) or reference not in self._models:
                    raise ModelLoadingError(
                        f"{label} references unknown model {reference!r}."
                    )
                referenced = self._resolve_members(reference, resolving + (model_name,))
                if len(referenced) != 1:
                    raise ModelLoadingError(
                        f"{label} references {reference!r}, which is itself an "
                        "ensemble; a member must be a single model."
                    )
                member = dict(referenced[0])
                if member["sources"] != sources:
                    raise ModelLoadingError(
                        f"{label} references {reference!r}, whose sources "
                        f"{member['sources']} differ from {sources}; members "
                        "must emit the same stems in the same order."
                    )
                resolved.append(member)
                continue
            # A member that inherits the entry's architecture inherits its
            # declared backend too, so "backend": "demucs" with a foreign
            # architecture is reported as the mismatch it is rather than as a
            # missing field. A member naming its own architecture derives its
            # own backend, and an entry already marked as an ensemble has no
            # single backend to pass down.
            declared_backend = model_info.get("backend")
            artifact = _member_artifact(spec)
            # Precedence: what the member says, then the entry, then whatever
            # the file itself records.
            embedded = _embedded_member_fields(artifact)
            resolved.append(
                _validate_member(
                    label,
                    {
                        "backend": (
                            declared_backend
                            if "architecture" not in spec
                            and declared_backend != ENSEMBLE_BACKEND
                            else None
                        ),
                        "architecture": _member_field(
                            "architecture", spec, model_info, embedded
                        ),
                        "config": _member_field("config", spec, model_info, embedded),
                        "sources": sources,
                        "samplerate": _member_field(
                            "samplerate", spec, model_info, embedded
                        ),
                        "segment_samples": _member_field(
                            "segment_samples", spec, model_info, embedded
                        ),
                        "artifact": artifact,
                    },
                )
            )

        self._members[model_name] = resolved
        return resolved

    def _merge_extra_models(
        self, extra_models: "Path | str | list[Path | str] | None"
    ) -> None:
        """
        Overlay user-supplied model entries onto the shipped registry.

        Entries are added, never substituted: a file that reuses a built-in
        name is rejected rather than shadowing it, so dropping one in cannot
        silently swap the weights out from under existing code.

        :param extra_models: Paths to overlay, or ``None`` to read
            ``UNBLEND_EXTRA_MODELS``.
        :raises ModelLoadingError: If a file is unreadable, malformed, or
            redefines a name that already exists.
        """
        if extra_models is None:
            raw = os.environ.get("UNBLEND_EXTRA_MODELS", "")
            paths = [p for p in raw.split(os.pathsep) if p]
        elif isinstance(extra_models, (str, Path)):
            paths = [extra_models]
        else:
            paths = list(extra_models)

        for entry in paths:
            path = Path(entry).expanduser()
            try:
                payload = load_mapping(path)
            except (OSError, ValueError) as exc:
                raise ModelLoadingError(
                    f"Could not read extra models file {path}: {exc}"
                ) from exc
            models = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(models, dict) or not models:
                raise ModelLoadingError(
                    f"Extra models file {path} must contain a non-empty "
                    "'models' object."
                )
            for model_name, model_info in models.items():
                if model_name in self._models:
                    raise ModelLoadingError(
                        f"Extra models file {path} redefines built-in model "
                        f"{model_name!r}; choose a different name."
                    )
                self._models[model_name] = model_info
        self.metadata["models"] = self._models

    def _artifacts(self, name: str) -> list[dict]:
        """
        Every weight artifact a model loads, in member order.

        :param name: Model name.
        :return: The artifact entries, empty for an unknown model.
        """
        return [member["artifact"] for member in self._members.get(name, ())]

    def _checkpoint_cache_path(self, model_info: dict) -> Path:
        """
        Content-addressed cache path for a single-checkpoint backend.

        Shared by every backend registered in
        :func:`backends.single_checkpoint_backends` (RoFormer, SCNet, …).

        :param model_info: The model's registry entry.
        :return: ``<cache dir>/<sha256[:16]>.safetensors``.
        """
        return _artifact_cache_path(model_info["checkpoint"])

    def local_artifacts(self, name: str) -> list[Path]:
        """
        Paths of the weight artifacts a model reads from disk rather than the
        cache, in load order.

        Empty for a model whose weights Unblend downloads. Callers use this to
        report a user-supplied model's availability, which the cache
        deliberately knows nothing about.

        :param name: Model name.
        :return: The declared local paths, which may or may not exist yet.
        """
        info = self._models.get(name)
        if info is None:
            return []
        return [
            path
            for spec in self._artifacts(name)
            if (path := _artifact_path(spec)) is not None
        ]

    def is_fully_local(self, name: str) -> bool:
        """
        Whether every artifact a model needs is a file the user supplied.

        Such a model never touches the cache or the network, so "downloaded"
        is the wrong question to ask about it — only whether the files exist.

        :param name: Model name.
        :return: ``True`` if the model has artifacts and none are remote.
        """
        info = self._models.get(name)
        if info is None:
            return False
        specs = self._artifacts(name)
        return bool(specs) and all(_artifact_url(spec) is None for spec in specs)

    def get_cache_info(self) -> dict[str, dict]:
        """
        Get information about cached models, including partially-cached ones
        (e.g. an interrupted multi-layer download).

        Only artifacts Unblend downloaded are reported. A model whose weights
        the user supplied as local files is absent: it occupies none of the
        cache, and listing it here would make cache removal look authorized to
        delete a file Unblend does not own. Callers that need such a model's
        availability inspect its ``path`` entries directly.

        :return: Dictionary mapping each model name with at least one cached
            artifact to ``{"layers", "size_bytes", "total_layers", "complete"}``
        """
        cached_models = {}

        for name, info in self._models.items():
            remote = [
                spec
                for spec in self._artifacts(name)
                if _artifact_url(spec) is not None
            ]
            if not remote:
                continue

            # Single stat per file — an exists()-then-stat() pair would race a
            # concurrent removal.
            components = {}
            for spec in remote:
                path = _artifact_cache_path(spec)
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    continue
                components[_artifact_cache_key(spec)] = {
                    "path": str(path),
                    "size_bytes": size_bytes,
                }
            if not components:
                continue

            cached_models[name] = {
                "layers": components,
                "size_bytes": sum(c["size_bytes"] for c in components.values()),
                "total_layers": len(remote),
                "complete": len(components) == len(remote),
            }

        return cached_models

    def sweep_stale_downloads(self) -> int:
        """
        Remove staging files older than the maximum download lifetime.

        Active downloads continuously update their staging-file mtime and are
        bounded by ``DOWNLOAD_DEADLINE_SECONDS``. The additional grace period
        ensures a concurrent sweeper never unlinks an in-flight POSIX file.

        :return: Number of stale files removed
        """
        removed = 0
        cutoff = time.time() - STAGING_STALE_SECONDS
        for tmp_path in get_cache_dir().glob(f"{STAGING_PREFIX}*"):
            try:
                if tmp_path.stat().st_mtime >= cutoff:
                    continue
                tmp_path.unlink()
            except OSError:
                continue
            removed += 1
        return removed

    @contextmanager
    def _resolved_artifact(
        self,
        spec: dict,
        *,
        label: str,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        model_name: str = "",
        layer_index: int = 1,
        total_layers: int = 1,
    ) -> Iterator[Path]:
        """
        Yield a verified Safetensors path for one artifact, from wherever it
        lives. Every backend loads its weights through here.

        A local ``path`` is read where it lies: there is nothing to download,
        and copying a file the user already has into the cache would only
        duplicate it. Its size and digest are still checked when metadata
        declares them.

        A remote ``url`` is downloaded once into the content-addressed cache
        and reused by every later load, so pointing a model at a Hugging Face
        URL costs one fetch rather than one per run. The per-artifact lock is
        held for the body of the ``with`` block, so a concurrent
        ``models remove`` cannot unlink the file while the caller is still
        reading tensors out of it.

        :param spec: The artifact entry (a Demucs layer or a ``checkpoint``).
        :param label: Human-readable prefix for error messages.
        :param progress_callback: Optional download-progress callback.
        :param model_name: Model name for progress payloads.
        :param layer_index: 1-based index of this artifact within the model.
        :param total_layers: How many artifacts the model needs in total.
        :return: Context manager yielding the verified artifact path.
        :raises ModelLoadingError: If the artifact is missing, unreadable, or
            fails verification.
        """
        local = _artifact_path(spec)
        if local is not None:
            if not local.is_file():
                raise ModelLoadingError(
                    f"{label} declares a local checkpoint that does not exist: {local}"
                )
            if spec.get("size_bytes") is not None:
                check_size(local, spec["size_bytes"])
            if spec.get("sha256") is not None:
                check_checksum(local, spec["sha256"])
            _emit(
                progress_callback,
                "layer_complete",
                model_name=model_name,
                layer_index=layer_index,
                total_layers=total_layers,
                cached=True,
            )
            yield local
            return

        url = _artifact_url(spec)
        expected = spec["sha256"]
        expected_size = spec["size_bytes"]
        cache_path = _artifact_cache_path(spec)

        with _artifact_lock(cache_path):
            # Recheck only after taking the per-artifact lock: another process
            # may have completed and promoted the artifact while we waited.
            cached = False
            if cache_path.exists():
                try:
                    check_size(cache_path, expected_size)
                    check_checksum(cache_path, expected)
                    cached = True
                except OSError as exc:
                    # A read failure is not corruption — keep the file.
                    raise ModelLoadingError(
                        f"Could not read cached artifact {cache_path}: {exc}"
                    ) from exc
                except ModelLoadingError as exc:
                    if isinstance(exc.__cause__, OSError):
                        raise
                    try:
                        cache_path.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        raise ModelLoadingError(
                            f"Cached artifact {cache_path} failed verification "
                            f"and could not be removed: {cleanup_error}"
                        ) from None

            if cached:
                _emit(
                    progress_callback,
                    "layer_complete",
                    model_name=model_name,
                    layer_index=layer_index,
                    total_layers=total_layers,
                    cached=True,
                )
            else:
                self._download_verified_file(
                    url=url,
                    cache_path=cache_path,
                    expected_sha256=expected,
                    expected_size=expected_size,
                    progress_callback=progress_callback,
                    model_name=model_name,
                    layer_index=layer_index,
                    total_layers=total_layers,
                )
            yield cache_path

    def _download_verified_file(
        self,
        url: str,
        cache_path: Path,
        expected_sha256: str,
        expected_size: int,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        model_name: str = "",
        layer_index: int = 1,
        total_layers: int = 1,
    ) -> None:
        """
        Stream one artifact to the cache, verifying its SHA-256 before it
        lands. Shared by every backend: a Demucs bag downloads one layer per
        call, a single-checkpoint model exactly one.

        The download is staged in a temp file inside the cache directory and
        promoted with a single ``os.replace``, so a partial or corrupt
        download can never appear at ``cache_path``.

        :param url: Source URL.
        :param cache_path: Destination path in the cache.
        :param expected_sha256: Full 64-character digest to verify against.
        :param expected_size: Exact artifact size from trusted metadata.
        :param progress_callback: Optional callback (``layer_start`` /
            ``layer_progress`` / ``layer_complete`` events).
        :param model_name: Model name for progress payloads.
        :param layer_index: 1-based index of this artifact within the model.
        :param total_layers: How many artifacts the model needs in total.
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
                _emit(
                    progress_callback,
                    "layer_start",
                    model_name=model_name,
                    layer_index=layer_index,
                    total_layers=total_layers,
                    layer_size_bytes=total_size,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix=f"{STAGING_PREFIX}{cache_path.name}.",
                    suffix=".tmp",
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
                        if progress_callback and counter % 20 == 0 and total_size > 0:
                            _emit(
                                progress_callback,
                                "layer_progress",
                                model_name=model_name,
                                layer_index=layer_index,
                                total_layers=total_layers,
                                progress_percent=downloaded / total_size * 100,
                                downloaded_bytes=downloaded,
                                total_bytes=total_size,
                            )
            if downloaded != expected_size:
                raise ModelLoadingError(
                    f"Download from {url} ended at {downloaded} bytes; "
                    f"expected {expected_size}."
                )
            # Integrity gate before the file is visible in the cache.
            check_size(tmp_path, expected_size)
            check_checksum(tmp_path, expected_sha256)
            os.replace(tmp_path, cache_path)
            tmp_path = None
            _emit(
                progress_callback,
                "layer_complete",
                model_name=model_name,
                layer_index=layer_index,
                total_layers=total_layers,
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

    def _select_members(
        self, name: str, only_load: str | None = None
    ) -> tuple[list[dict], list[list[float]] | None]:
        """
        Resolve which members ``get_model`` needs, honouring the single-stem
        shortcut.

        When exactly one member contributes to the isolated stem, every combine
        mode reduces to that member's output — a mean of one, a median of one —
        so only it is fetched and run, and the weights come back as ``None`` so
        the caller builds it directly instead of a one-member ensemble.

        :param name: Model name.
        :param only_load: Stem to isolate, if any.
        :return: ``(members, weights)``.
        :raises ModelLoadingError: If the model or stem is unknown.
        """
        if name not in self._models:
            raise ModelLoadingError(
                f"Could not find a model with name {name}. "
                f"Available models: {', '.join(self._models.keys())}"
            )

        model_info = self._models[name]
        if only_load is not None and only_load not in model_info["sources"]:
            raise ModelLoadingError(
                f"Stem {only_load!r} not found in model {name}. Available "
                f"stems: {', '.join(model_info['sources'])}"
            )

        members = self._members[name]
        weights = model_info.get("weights")
        if only_load is None or weights is None or len(members) == 1:
            return members, weights

        index = sole_contributor(weights, model_info["sources"].index(only_load))
        if index is None:
            return members, weights
        return [members[index]], None

    def required_layers(self, name: str, only_load: str | None = None) -> list[str]:
        """
        Return the cache keys of the artifacts ``get_model(name, only_load)``
        would fetch. Useful for cache checks without touching the network.

        Artifacts the entry points at a local file are omitted: they are read
        where they lie and never enter the cache, so there is nothing for a
        caller to check for.

        :param name: Model name
        :param only_load: Optional stem for the single-specialist optimisation
        :return: List of artifact cache keys (cache files use Safetensors)
        :raises ModelLoadingError: If the model is not found
        """
        members, _ = self._select_members(name, only_load)
        return [
            _artifact_cache_key(member["artifact"])
            for member in members
            if _artifact_url(member["artifact"]) is not None
        ]

    def layer_sha256(self, cache_key: str) -> str:
        """
        Return the full 64-character SHA-256 the cached artifact with the given
        filename stem is expected to hash to.

        :param cache_key: Digest prefix that names the cache file.
        :return: Full 64-character SHA-256 digest from metadata.
        :raises KeyError: If ``cache_key`` is not a registered artifact.
        """
        return self._layer_sha256[cache_key]

    def get_model(
        self,
        name: str,
        only_load: str | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Model | ModelEnsemble:
        """
        Get a model by name, downloading whatever is not cached.

        Every model is a list of members: one for a single checkpoint, several
        for an ensemble. Each member's artifact is resolved the same way — read
        in place if the entry names a local file, downloaded once into the cache
        otherwise — and built by whichever backend owns its architecture.

        :param name: Model name
        :param only_load: If specified, load only the model specialized for this stem
        :param progress_callback: Optional callback for download progress updates
        :return: The requested model, or an ensemble of its members
        :raises ModelLoadingError: If the model is not found or fails to load
        """
        members, weights = self._select_members(name, only_load)
        model_info = self._models[name]
        total_layers = len(members)

        _emit(
            progress_callback,
            "download_start",
            model_name=name,
            total_layers=total_layers,
        )

        built: list[Model] = []
        for index, member in enumerate(members, start=1):
            label = (
                f"Member {index} of model {name}"
                if total_layers > 1
                else f"Model {name}"
            )
            with self._resolved_artifact(
                member["artifact"],
                label=label,
                progress_callback=progress_callback,
                model_name=name,
                layer_index=index,
                total_layers=total_layers,
            ) as path:
                # Materialise the tensors while the artifact lock is held, so a
                # concurrent ``models remove`` cannot unlink the file mid-read;
                # construction happens after the lock is released.
                state = _read_state(path)
            built.append(self._build_member(label, member, state))
            del state

        _emit(
            progress_callback,
            "download_complete",
            model_name=name,
            total_layers=total_layers,
        )

        # A metadata override can shorten, never enlarge, the configured
        # training segment.
        segment = model_info.get("segment")
        if len(built) == 1:
            # One member is its own output under every combine mode, so skip
            # the ensemble wrapper entirely.
            model = built[0]
            if segment is not None:
                model.max_allowed_segment = min(
                    float(segment), float(model.max_allowed_segment)
                )
            return model

        return ModelEnsemble(
            built,
            weights,
            segment,
            model_info.get("combine", COMBINE_DEFAULT),
            model_info.get("combine_params"),
        )

    def _build_member(self, label: str, member: dict, state: dict) -> Model:
        """
        Construct one member from its verified weights.

        :param label: Human-readable prefix for error messages.
        :param member: The resolved member.
        :param state: Tensors read from its artifact.
        :return: The constructed model in eval mode.
        :raises ModelLoadingError: If construction or strict loading fails.
        """
        if member["backend"] == DEMUCS_BACKEND:
            return _build_demucs_layer(state, member, label)
        try:
            return backends.build(
                member["backend"],
                member["architecture"],
                dict(member["config"]),
                sources=list(member["sources"]),
                samplerate=int(member["samplerate"]),
                segment_samples=int(member["segment_samples"]),
                state=state,
            )
        except ModelLoadingError:
            raise
        except Exception as exc:
            raise ModelLoadingError(f"Failed to build {label} from checkpoint: {exc}")

    def list_models(self) -> dict[str, dict]:
        """
        List all available models.

        :return: Dictionary mapping model names to their metadata (deep
            copies — mutating them does not affect repository state)
        """
        return {name: copy.deepcopy(info) for name, info in self._models.items()}

    def remove_model(self, name: str) -> bool:
        """
        Remove a model's downloaded artifacts from the cache.

        Local artifacts are left alone: they belong to the user, not to
        Unblend's cache, so a model whose weights are all local reports
        ``False`` rather than deleting anything.

        :param name: Model name
        :return: True if at least one cached artifact was removed, False if the
            model is unknown or had nothing cached
        :raises ModelLoadingError: If a cached artifact exists but can't be
            removed (e.g. permissions)
        """
        if name not in self._models:
            return False

        removed_any = False
        for spec in self._artifacts(name):
            if _artifact_url(spec) is None:
                continue
            path = _artifact_cache_path(spec)
            with _artifact_lock(path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as e:
                    raise ModelLoadingError(
                        f"Could not remove cached artifact {path}: {e}"
                    ) from e
                removed_any = True

        return removed_any
