# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import json
import os
import re
import tempfile
import time
import unicodedata
import warnings
from contextlib import contextmanager
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import httpx
import torch
import yaml
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file

from . import backends, scnet  # noqa: F401
from .apply import (
    COMBINE_DEFAULT,
    Model,
    ModelEnsemble,
    _finite,
    canonical_combine,
    check_weight_totals,
    resolve_combine_params,
    sole_contributor,
    validate_combine_weights,
)
from .exceptions import ModelLoadingError, ValidationError
from .htdemucs import HTDemucs


class _ConfigLoader(yaml.SafeLoader):
    """
    Safe YAML loading that also accepts ``!!python/tuple``, which
    Music-Source-Separation-Training configs use for band layouts.
    """


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: list(loader.construct_sequence(node)),
)


def _load_mapping(path: Path) -> Any:
    """
    Parse a YAML or JSON file.

    :param path: File to read.
    :return: Parsed contents.
    """
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.load(text, Loader=_ConfigLoader)


STAGING_PREFIX = ".unblend-download-"
DOWNLOAD_DEADLINE_SECONDS = 2 * 60 * 60
# Tries per artifact; a dropped connection resumes with an HTTP Range request.
DOWNLOAD_ATTEMPTS = 4
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


def _artifact_specs(info: object) -> list[dict]:
    """
    The mappings in a models-file entry that can name a weight artifact: its
    ``checkpoint``, and each member and member ``checkpoint``.

    :param info: A models-file entry.
    :return: The candidate artifact mappings.
    """
    if not isinstance(info, dict):
        return []
    specs = [info.get("checkpoint")]
    members = info.get("members")
    # A non-list is refused later with a proper message; don't iterate it.
    for member in members if isinstance(members, list) else []:
        if isinstance(member, dict):
            specs += [member, member.get("checkpoint")]
    return [spec for spec in specs if isinstance(spec, dict)]


def entry_weight_paths(info: object, base: Path) -> list[Path]:
    """
    The local weight files a models-file entry names, joined to ``base`` as
    the registry reads them but not normalised, so deleting one removes what
    the entry names (a symlink, not its target; ``link/..`` through the link).

    :param info: A models-file entry, as written.
    :param base: Directory of the file the entry came from.
    :return: The paths; compare them by ``os.path.realpath`` (``resolve()`` raises on a
        symlink loop).
    """
    paths = []
    for spec in _artifact_specs(info):
        local = spec.get("path")
        if isinstance(local, str) and local:
            path = Path(os.path.expanduser(local))
            paths.append(path if path.is_absolute() else base / path)
    return paths


def _anchor_relative_paths(info: object, base: Path) -> None:
    """
    Resolve relative artifact ``path`` values against the models file's
    directory, so an entry means the same thing wherever unblend runs.

    :param info: A models-file entry, updated in place.
    :param base: Directory of the file the entry came from.
    """
    for spec in _artifact_specs(info):
        local = spec.get("path")
        if isinstance(local, str) and "\0" in local:
            # No file can have such a name. A ValueError, so the merge boundary
            # reports it naming the models file (which is then skipped or
            # refused) instead of a later lookup crashing.
            raise ValueError(f"weights path {local!r} contains a NUL character")
        if isinstance(local, str) and local:
            # Expanded first, as entry_weight_paths does: "~/x" is absolute,
            # while an unknown "~user/x" stays relative and is anchored too.
            expanded = os.path.expanduser(local)
            if not Path(expanded).is_absolute():
                # realpath, unlike resolve(), doesn't raise on a symlink loop; the
                # load then reports the file as unreadable.
                spec["path"] = os.path.realpath(base / expanded)


def _artifact_path(spec: dict) -> Path | None:
    """
    The local, user-owned file an artifact names, if it names one.

    :param spec: An artifact entry (a ``checkpoint``, or one of ``members``).
    :return: The expanded path, or ``None`` for a remote artifact.
    """
    local = spec.get("path")
    if not isinstance(local, str) or not local:
        return None
    # os.path, not Path: Path.expanduser() raises for an unknown ~user.
    return Path(os.path.expanduser(local))


# Safetensors dtype strings for the float widths a checkpoint may carry.
_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E4M3": torch.float8_e4m3fn,
}


def artifact_storage_dtype(spec: dict) -> torch.dtype | None:
    """
    The float dtype an artifact stores its weights at, read from its header.

    Safetensors declares a dtype per tensor, so this reports the widest float
    dtype present, which is what the file needs to round-trip without loss.

    :param spec: An artifact entry (a ``checkpoint``, or one of ``members``).
    :return: The widest float dtype declared, or ``None`` if the file is
        absent, unreadable, or holds no float tensors.
    """
    path = _artifact_path(spec)
    if path is None:
        # A remote artifact is cached under its digest; without one there is
        # no file to inspect.
        if "sha256" not in spec:
            return None
        path = _artifact_cache_path(spec)
    try:
        with safe_open(path, framework="pt") as handle:
            declared = {handle.get_slice(key).get_dtype() for key in handle.keys()}
    except (OSError, SafetensorError):
        return None
    found = [dtype for name, dtype in _SAFETENSORS_DTYPES.items() if name in declared]
    if not found:
        return None
    return max(found, key=lambda dtype: dtype.itemsize)


def _artifact_url(spec: dict) -> str | None:
    """
    The URL an artifact is downloaded from, if it is remote.

    :param spec: An artifact entry (a member's weights, or a ``checkpoint``).
    :return: An absolute URL, or ``None`` for a local artifact.
    """
    url = spec.get("url")
    if not isinstance(url, str) or not url:
        return None
    return url


def _artifact_cache_key(spec: dict) -> str:
    """
    Cache filename stem for a remote artifact.

    Content-addressed, so two models naming the same checkpoint share one
    cached file whatever else their entries say.

    :param spec: A remote artifact entry.
    :return: The digest prefix that names its cache file.
    """
    return spec["sha256"][:16]


def _artifact_identity(spec: dict) -> str:
    """
    What makes two artifact entries the same file.

    :param spec: An artifact entry.
    :return: The real path of a local file, else the remote cache key.
    """
    path = _artifact_path(spec)
    # realpath, unlike resolve(), doesn't raise on a symlink loop.
    return os.path.realpath(path) if path is not None else _artifact_cache_key(spec)


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

    :param spec: The candidate artifact entry.
    :param label: Human-readable prefix for error messages.
    :raises ModelLoadingError: If the entry is malformed.
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
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
        ):
            raise ModelLoadingError(f"{label} is missing a valid sha256.")
        # Some tools print digests in upper case; cache keys use lower.
        spec["sha256"] = digest.lower()
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

    :param state: Tensors from the verified artifact.
    :param member: Resolved member carrying architecture and config.
    :param label: Human-readable prefix for error messages.
    :return: Weight-loaded HTDemucs model, in eval mode.
    :raises ModelLoadingError: If the architecture is not allowlisted or the
        weights do not load.
    """
    if member.get("architecture") not in DEMUCS_ARCHITECTURES:
        raise ModelLoadingError(
            f"Unsupported Demucs architecture {member.get('architecture')!r}."
        )
    try:
        model = HTDemucs(**dict(member["config"]))
        model.load_state_dict(state, strict=True)
        return model.eval()
    except Exception as exc:
        # As for the other backends: a config the constructor rejects any way
        # (an assert on t_heads, a zero divisor) is a load error.
        detail = str(exc) or type(exc).__name__
        raise ModelLoadingError(f"Failed to build {label}: {detail}") from exc


#: What an unanticipated value in a user's models file can raise while it is
#: merged and validated; converted to ModelLoadingError at those boundaries.
_USER_VALUE_ERRORS = (
    TypeError,
    ValueError,
    OverflowError,
    OSError,
    AttributeError,
    RuntimeError,  # pathlib's own path errors, e.g. a symlink loop in resolve()
)

DEMUCS_BACKEND = "demucs"
DEMUCS_ARCHITECTURES = frozenset({"htdemucs"})

ENSEMBLE_BACKEND = "ensemble"


def _known_backends() -> frozenset[str]:
    """
    Backend names a registry entry can resolve to.

    :return: The accepted backend names.
    """
    return frozenset({DEMUCS_BACKEND, ENSEMBLE_BACKEND}) | frozenset(backends._BUILDERS)


def _architectures_for(backend: str) -> frozenset[str]:
    """
    Architectures a backend can build.

    :param backend: Backend name.
    :return: Its architecture names.
    """
    if backend == DEMUCS_BACKEND:
        return DEMUCS_ARCHITECTURES
    return backends._ARCHITECTURES.get(backend, frozenset())


def _known_architectures() -> frozenset[str]:
    """
    Every architecture any registered backend can build.

    :return: The accepted architecture names.
    """
    return frozenset().union(
        *(_architectures_for(backend) for backend in _known_backends())
    )


def _reject_declared_backend(label: str, info: dict) -> None:
    """
    Reject a ``backend`` field: the loader family is derived, never declared.

    :param label: Human-readable prefix for error messages.
    :param info: A registry entry or a raw member spec.
    :raises ModelLoadingError: If the mapping declares a backend.
    """
    if isinstance(info, dict) and "backend" in info:
        raise ModelLoadingError(
            f"{label} declares a 'backend', which models files don't take: the "
            "loader family is derived from 'architecture'. Remove it (list_models "
            "adds it to its output)."
        )


def _backend_for(label: str, info: dict) -> str:
    """
    Resolve which backend builds a model, or one ensemble member.

    :param label: Human-readable prefix for error messages.
    :param info: A registry entry or a resolved member.
    :return: The backend name.
    :raises ModelLoadingError: If the architecture is unknown.
    """
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
            f"{label} has an unknown architecture {architecture!r} (known: "
            f"{', '.join(sorted(_known_architectures()))})."
        )
    return derived


# Fields of a weight artifact (a ``checkpoint:`` mapping).
_ARTIFACT_KEYS = frozenset({"format", "path", "sha256", "size_bytes", "url"})

# Fields a member that names its own weights may set: the artifact, inline
# or under ``checkpoint``, and the fields it may override from the entry.
_MEMBER_KEYS = frozenset(
    {
        "architecture",
        "checkpoint",
        "config",
        "format",
        "path",
        "samplerate",
        "segment_samples",
        "sha256",
        "size_bytes",
        "url",
    }
)


def _entry_member_specs(model_name: str, model_info: dict) -> list[dict]:
    """
    A model's member specs exactly as written, before inheritance.

    :param model_name: Model name, for error messages.
    :param model_info: The model's registry entry.
    :return: The raw member specs.
    :raises ModelLoadingError: If the entry declares neither or both of
        ``checkpoint`` and ``members``, or fewer than two members.
    """
    present = [
        key for key in ("checkpoint", "members") if model_info.get(key) is not None
    ]
    if len(present) != 1:
        raise ModelLoadingError(
            f"Model {model_name} must declare exactly one of 'checkpoint' or "
            f"'members'; found {present or ['none']}."
        )
    if present[0] == "checkpoint":
        return [{"checkpoint": model_info["checkpoint"]}]
    raw = model_info["members"]
    if not (
        isinstance(raw, list)
        and len(raw) > 1
        and all(isinstance(item, dict) for item in raw)
    ):
        raise ModelLoadingError(
            f"Model {model_name} must declare at least two members under "
            "'members'; a single set of weights is a 'checkpoint'."
        )
    for index, item in enumerate(raw, start=1):
        if "model" in item:
            allowed = {"model"}
        elif "checkpoint" in item:
            # Artifact fields belong inside the checkpoint mapping; beside it
            # they would be ignored (a stray `sha256` would go unchecked).
            allowed = _MEMBER_KEYS - _ARTIFACT_KEYS
        else:
            allowed = _MEMBER_KEYS
        # "backend" has its own, clearer error.
        unknown = sorted(str(key) for key in set(item) - allowed - {"backend"})
        if unknown:
            raise ModelLoadingError(
                f"Member {index} of model {model_name} has unknown field(s) "
                f"{', '.join(unknown)}; expected "
                + (
                    "only 'model' alongside 'model'."
                    if "model" in item
                    else f"some of {', '.join(sorted(allowed))}."
                )
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

    return spec


def _embedded_member_fields(artifact: object) -> dict:
    """
    Registry fields a local Safetensors artifact records about itself.

    :param artifact: A member's artifact entry.
    :return: The embedded fields, empty if none.
    """
    if not isinstance(artifact, dict):
        return {}
    path = _artifact_path(artifact)
    if path is None or not os.path.isfile(path):
        return {}

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


_ENTRY_KEYS = frozenset(
    {
        "architecture",
        "checkpoint",
        "combine",
        "combine_params",
        "config",
        "license",
        "license_note",
        "members",
        "provenance",
        "samplerate",
        "segment",
        "segment_samples",
        "sources",
        "weights",
    }
)


def stem_key(name: str) -> str:
    """
    The form two names collide in as filenames: case and Unicode
    normalization ignored (Unicode's canonical caseless match, as
    ``separate`` also compares output paths).

    :param name: A stem name or path.
    :return: Its comparison key.
    """
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", name).casefold())


def stem_name_problem(sources: object) -> str | None:
    """
    What's wrong with an entry's stem names, if anything.

    Shared by the registry and ``models import`` (which checks before it
    converts anything).

    :param sources: The declared stems.
    :return: A phrase completing "Model X ...", or None if they're usable.
    """
    if not (
        isinstance(sources, list)
        and sources
        and all(isinstance(source, str) and source for source in sources)
        and len({stem_key(source) for source in sources}) == len(sources)
    ):
        # Unique ignoring case and Unicode normalization: stems become
        # filenames, and separate refuses names that alias on a case- or
        # normalization-insensitive filesystem.
        return "must declare unique (ignoring case and Unicode form), non-empty sources"
    unsafe = [
        source
        for source in sources
        if source in (".", "..") or any(c in source for c in "/\\:\0")
    ]
    if unsafe:
        # Stem names become output filenames ({stem}).
        return f"has stem names that aren't safe as filenames: {unsafe}"
    return None


def _validate_entry(model_name: str, model_info: object) -> dict:
    """
    Validate the parts of a registry entry that stand alone.

    :param model_name: Model name.
    :param model_info: The candidate entry.
    :return: The entry.
    :raises ModelLoadingError: If the name, mapping, or sources are invalid.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ModelLoadingError("Every model name must be a non-empty string.")
    if not isinstance(model_info, dict):
        raise ModelLoadingError(f"Model {model_name} metadata must be a dictionary.")
    _reject_declared_backend(f"Model {model_name}", model_info)
    unknown = sorted(str(key) for key in set(model_info) - _ENTRY_KEYS)
    if unknown:
        # A typo (``wieghts``, ``segmnet``) would otherwise silently fall back
        # to a default.
        raise ModelLoadingError(
            f"Model {model_name} has unknown field(s) {', '.join(unknown)}; "
            f"expected some of {', '.join(sorted(_ENTRY_KEYS))}."
        )

    problem = stem_name_problem(model_info.get("sources"))
    if problem is not None:
        raise ModelLoadingError(f"Model {model_name} {problem}.")
    for field in ("license", "license_note", "provenance"):
        value = model_info.get(field)
        if value is not None and not isinstance(value, str):
            # YAML reads `license: 2.0` or `license: no` as a float or bool.
            raise ModelLoadingError(
                f"Model {model_name} has a non-text {field} ({value!r}); quote it."
            )
    segment = model_info.get("segment")
    if segment is not None and (
        isinstance(segment, bool)
        or not isinstance(segment, Real)
        or not _finite(segment)
        or segment <= 0
    ):
        raise ModelLoadingError(
            f"Model {model_name} has invalid segment {segment!r}; expected a "
            "positive number of seconds."
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
    from .importer import constructor_config

    kept, dropped = constructor_config(architecture, config)
    if dropped:
        # Training configs carry options (``flash_attn``) inference doesn't take.
        warnings.warn(
            f"{label}: ignoring config keys the model doesn't take: "
            f"{', '.join(dropped)}.",
            stacklevel=2,
        )
        member["config"] = config = kept

    # Refused by the constructors too, but checked here so a malformed entry
    # fails before its weights are downloaded.
    if not config.get("cac", True) and architecture in DEMUCS_ARCHITECTURES:
        raise ModelLoadingError(f"{label}: HTDemucs only supports cac: true.")
    if architecture in ("scnet", "scnet_masked"):
        layers = config.get("num_dplayer", 6)
        if (
            isinstance(layers, bool)
            or not isinstance(layers, int)
            or layers < 1
            or layers % 2
        ):
            raise ModelLoadingError(
                f"{label}: SCNet num_dplayer must be a positive even number, got {layers!r}."
            )

    if backend == DEMUCS_BACKEND:
        if config.get("sources") != member["sources"]:
            raise ModelLoadingError(
                f"{label} must declare a config whose sources match metadata."
            )
        # HTDemucs takes its geometry from the config (defaults as in its
        # constructor); entry-level fields may only restate it.
        rate = config.get("samplerate", 44100)
        seconds = config.get("segment", 10)
        if (
            isinstance(rate, bool)
            or not isinstance(rate, int)
            or not _finite(rate)
            or rate <= 0
        ):
            raise ModelLoadingError(f"{label} has invalid config samplerate: {rate}.")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not _finite(seconds)
            or seconds <= 0
        ):
            raise ModelLoadingError(f"{label} has invalid config segment: {seconds}.")
        if not _finite(seconds * rate):
            raise ModelLoadingError(
                f"{label} has a config segment of {seconds} s at {rate} Hz, "
                "too long to represent."
            )
        if seconds * rate < 1:
            raise ModelLoadingError(
                f"{label} has a config segment of {seconds} s, shorter than "
                f"one sample at {rate} Hz."
            )
        stated = {
            "samplerate": rate,
            "segment_samples": int(round(seconds * rate)),
        }
        for field, expected in stated.items():
            value = member.get(field)
            if value is not None and value != expected:
                raise ModelLoadingError(
                    f"{label} has {field} {value}, but its config gives "
                    f"{expected}; HTDemucs uses the config's samplerate and "
                    "segment."
                )
    else:
        for field in ("samplerate", "segment_samples"):
            value = member.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not _finite(value)
                or value <= 0
            ):
                raise ModelLoadingError(f"{label} has invalid {field}: {value}.")

    _validate_artifact(
        member.get("artifact"), f"Checkpoint of {label[0].lower() + label[1:]}"
    )
    member["backend"] = backend
    return member


_DEFAULT_CHANNELS = {
    "bs_roformer": 1,
    "mel_band_roformer": 1,
    "htdemucs": 2,
    "scnet": 2,
    "scnet_masked": 2,
}


def _declared_geometry(member: dict) -> tuple[int | None, int | None]:
    """
    A member's sample rate and channel count, as far as its entry states them.

    :param member: A resolved member.
    :return: ``(samplerate, channels)``, either ``None`` when not declared.
    """
    config = member.get("config") or {}
    if member.get("backend") == DEMUCS_BACKEND:
        rate = config.get("samplerate", 44100)
    else:
        rate = member.get("samplerate") or config.get("samplerate")
    channels = config.get("audio_channels")
    if channels is None and "stereo" in config:
        channels = 2 if config["stereo"] else 1
    if channels is None:
        # The constructors' defaults: RoFormer is mono unless stereo is set;
        # HTDemucs and SCNet take two channels.
        channels = _DEFAULT_CHANNELS.get(member.get("architecture"))
    return rate, channels


def _validate_member_compatibility(model_name: str, members: list[dict]) -> None:
    """
    Reject an ensemble whose members declare different sample rates or channel
    counts, before anything is downloaded.

    :param model_name: Model name.
    :param members: The entry's resolved members.
    :raises ModelLoadingError: If two members disagree.
    """
    if len(members) < 2:
        return
    geometry = [_declared_geometry(member) for member in members]
    for rate, channels in geometry:
        for value, label in ((rate, "sample rate"), (channels, "channel count")):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ModelLoadingError(
                    f"Model {model_name} has a member with invalid {label} {value!r}."
                )
    for index, label in ((0, "sample rate"), (1, "channel count")):
        values = {g[index] for g in geometry if g[index] is not None}
        if len(values) > 1:
            raise ModelLoadingError(
                f"Model {model_name}'s members disagree on {label}: {sorted(values)}."
            )


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
                    or not _finite(value)
                ):
                    raise ModelLoadingError(
                        f"Model {model_name} weight [{row_index}][{column}] "
                        "must be a finite number."
                    )
        # The same rule ModelEnsemble applies, checked before any download.
        try:
            check_weight_totals(weights, sources)
        except ValidationError as exc:
            raise ModelLoadingError(f"Model {model_name}: {exc}") from exc

    try:
        canonical_combine(model_info.get("combine", COMBINE_DEFAULT))
        validate_combine_weights(model_info.get("combine", COMBINE_DEFAULT), weights)
        resolve_combine_params(model_info.get("combine_params"))
    except ValidationError as exc:
        raise ModelLoadingError(f"Model {model_name}: {exc}") from exc


def get_cache_dir() -> Path:
    """
    Get the cache directory for downloaded models.

    ``UNBLEND_CACHE_DIR`` overrides; defaults to ``~/.unblend/models``.

    :return: Path to the cache directory
    """
    override = os.environ.get("UNBLEND_CACHE_DIR")
    if override:
        return Path(os.path.realpath(os.path.expanduser(override)))
    return Path.home() / ".unblend" / "models"


# Model names: also used as folder and file names.
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def default_models_file() -> Path:
    """
    The user models file that is always loaded when it exists, alongside any
    listed in ``UNBLEND_EXTRA_MODELS``. ``unblend models import`` writes here
    by default.

    :return: ``~/.unblend/models.yaml``.
    """
    return Path.home() / ".unblend" / "models.yaml"


def listed_extra_models_files() -> list[Path]:
    """
    The models files listed in ``UNBLEND_EXTRA_MODELS``.

    :return: The listed paths, ``~`` expanded, in order.
    """
    raw = os.environ.get("UNBLEND_EXTRA_MODELS", "")
    return [Path(os.path.expanduser(p)) for p in raw.split(os.pathsep) if p]


def default_extra_models_files() -> list[Path]:
    """
    The user models files a default ``ModelRepository()`` overlays.

    :return: ``UNBLEND_EXTRA_MODELS`` entries, then the default file if it
        exists and isn't already listed.
    """
    paths = listed_extra_models_files()
    default = default_models_file()
    if os.path.isfile(default) and Path(os.path.realpath(default)) not in {
        Path(os.path.realpath(p)) for p in paths
    }:
        paths.append(default)
    return paths


class ModelRepository:
    """
    Registry of known models: validates metadata, caches verified weights,
    and builds models from them.
    """

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
            ``UNBLEND_EXTRA_MODELS`` (os.pathsep-separated) plus
            ``~/.unblend/models.yaml`` when it exists.
        :raises ModelLoadingError: If the metadata structure is invalid
        """
        if metadata_path is None:
            metadata_path = Path(__file__).parent / "metadata.yaml"
        self.metadata_path = metadata_path

        try:
            self.metadata = _load_mapping(Path(self.metadata_path))
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
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
        self._origins: dict[str, Path] = {}
        self._current: str | None = None
        self._merge_extra_models(extra_models)
        if not self._models:
            raise ModelLoadingError("Model metadata must contain at least one model.")

        with self._naming_source():
            validated = {}
            for model_name, model_info in self._models.items():
                self._current = model_name
                validated[model_name] = _validate_entry(model_name, model_info)
            self._models = validated
            self._members: dict[str, list[dict]] = {}
            for model_name in self._models:
                self._current = model_name
                self._resolve_members(model_name)

            for model_name, members in self._members.items():
                self._current = model_name
                model_info = self._models[model_name]
                _validate_entry_combination(model_name, model_info, len(members))
                _validate_member_compatibility(model_name, members)
                segment = model_info.get("segment")
                # Every resolved member states its rate (HTDemucs defaults
                # to 44100); ModelEnsemble repeats the check for Python use.
                rates = [_declared_geometry(member)[0] for member in members]
                if segment is not None and any(
                    rate is not None and segment * rate < 1 for rate in rates
                ):
                    raise ModelLoadingError(
                        f"Model {model_name} has segment {segment!r} s, "
                        "shorter than one sample."
                    )

                used = {member["backend"] for member in members}
                derived = used.pop() if len(used) == 1 else ENSEMBLE_BACKEND
                self._models[model_name] = {**model_info, "backend": derived}
        self.metadata["models"] = self._models

        self._artifact_urls: dict[str, str] = {}
        self._artifact_sha256: dict[str, str] = {}
        self._artifact_sizes: dict[str, int] = {}
        for model_name, members in self._members.items():
            for member in members:
                artifact = member["artifact"]
                url = _artifact_url(artifact)
                if url is None:
                    continue
                key = _artifact_cache_key(artifact)
                sha = artifact["sha256"]
                size = artifact["size_bytes"]
                if key in self._artifact_urls:
                    # Cache files are keyed by content, so a second URL for the
                    # same bytes is fine; different bytes under one key aren't.
                    if (self._artifact_sha256[key], self._artifact_sizes[key]) != (
                        sha,
                        size,
                    ):
                        raise ModelLoadingError(
                            f"Model {model_name}: artifact {key} has a different "
                            "sha256/size than another model's."
                        )
                    continue
                self._artifact_urls[key] = url
                self._artifact_sha256[key] = sha
                self._artifact_sizes[key] = size

    def _resolve_members(
        self, model_name: str, resolving: tuple[str, ...] = ()
    ) -> list[dict]:
        """
        Expand one entry into fully-specified members.

        :param model_name: Model name.
        :param resolving: Entries being resolved, for cycle detection.
        :return: The resolved members, in load order.
        :raises ModelLoadingError: If a member reference is unknown, cyclic,
            an ensemble, or has mismatched sources.
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
            label = (
                f"Member {index} of model {model_name}"
                if "members" in model_info
                else f"Model {model_name}"
            )
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

            _reject_declared_backend(label, spec)
            artifact = _member_artifact(spec)
            checkpoint = spec.get("checkpoint")
            if isinstance(checkpoint, dict):
                unknown = sorted(str(key) for key in set(checkpoint) - _ARTIFACT_KEYS)
                if unknown:
                    # A typo such as `sha265:` would otherwise quietly turn
                    # off verification of a local file.
                    raise ModelLoadingError(
                        f"{label}'s checkpoint has unknown field(s) "
                        f"{', '.join(unknown)}; expected some of "
                        f"{', '.join(sorted(_ARTIFACT_KEYS))}."
                    )

            embedded = _embedded_member_fields(artifact)
            local = _artifact_path(artifact) if isinstance(artifact, dict) else None
            if (
                local is not None
                and not os.path.isfile(local)
                and _member_field("architecture", spec, model_info, embedded) is None
            ):
                raise ModelLoadingError(f"{label}: checkpoint file not found: {local}")
            if (
                local is not None
                and _member_field("architecture", spec, model_info, embedded) is None
            ):
                raise ModelLoadingError(
                    f"{label} doesn't state an architecture, and {local} doesn't "
                    "either (its header is unreadable, or it wasn't written by "
                    "'unblend models import'); add architecture and config to "
                    "the entry, or re-import the checkpoint."
                )
            resolved.append(
                _validate_member(
                    label,
                    {
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

    @contextmanager
    def _naming_source(self) -> Iterator[None]:
        """
        Add the models file to validation errors about an entry that came
        from one.

        :return: Context manager.
        :raises ModelLoadingError: Re-raised with the source file named.
        """
        try:
            yield
        except ModelLoadingError as exc:
            origin = self._origins.get(self._current or "")
            if origin is None:
                raise
            raise ModelLoadingError(f"{exc} (in {origin})") from exc
        except _USER_VALUE_ERRORS as exc:
            origin = self._origins.get(self._current or "")
            if origin is None:
                raise  # A built-in entry: a bug here, not bad input.
            # A shape no check anticipated, in a user's entry.
            raise ModelLoadingError(
                f"Model {self._current} holds a value unblend can't use "
                f"({type(exc).__name__}: {exc}) (in {origin})"
            ) from exc

    def _merge_extra_models(
        self, extra_models: "Path | str | list[Path | str] | None"
    ) -> None:
        """
        Overlay user-supplied model entries onto the shipped registry.

        Explicitly listed files must be valid, though a file
        ``UNBLEND_EXTRA_MODELS`` lists that doesn't exist yet is skipped with a
        warning. The implicit default file is skipped with a warning when it
        is broken or clashes and the listed files load without it, so a bad
        user file can't take the built-in models down with it.

        :param extra_models: Paths to overlay, or ``None`` to read
            ``UNBLEND_EXTRA_MODELS`` plus :func:`default_models_file`.
        :raises ModelLoadingError: If an explicitly listed file is invalid, or
            the default file is kept (the listed files need it) and is.
        """
        implicit: Path | None = None
        if extra_models is None:
            paths: list[Path | str] = []
            for path in listed_extra_models_files():
                if os.path.exists(path):
                    paths.append(path)
                else:
                    # Commonly set before the first `models import` creates
                    # the file; failing would take every command down.
                    warnings.warn(
                        f"UNBLEND_EXTRA_MODELS lists {path}, which doesn't "
                        "exist; skipping it.",
                        stacklevel=3,
                    )
            default = default_models_file()
            listed = {Path(os.path.realpath(Path(p))) for p in paths}
            if (
                os.path.isfile(default)
                and Path(os.path.realpath(default)) not in listed
            ):
                implicit = default
        elif isinstance(extra_models, (str, Path)):
            paths = [extra_models]
        else:
            paths = list(extra_models)

        builtin = set(self._models)
        merged: set[Path] = set()
        for entry in paths:
            path = Path(os.path.expanduser(entry))
            # Listing a file twice would otherwise clash with itself.
            if Path(os.path.realpath(path)) in merged:
                continue
            merged.add(Path(os.path.realpath(path)))
            self._merge_models_file(path, builtin)

        if implicit is not None:
            # The default file is skipped only when the listed files load
            # without it: then it is the one that can't join. When they don't,
            # it is kept, so this repository's own validation reports the real
            # fault wherever it is (a listed file may use the default's models,
            # and the default may use theirs).
            try:
                ModelRepository(
                    metadata_path=self.metadata_path,
                    extra_models=[*paths, implicit],
                )
            except ModelLoadingError as exc:
                try:
                    ModelRepository(
                        metadata_path=self.metadata_path, extra_models=paths
                    )
                except ModelLoadingError:
                    pass
                else:
                    warnings.warn(f"Ignoring {implicit}: {exc}", stacklevel=3)
                    implicit = None
            if implicit is not None:
                self._merge_models_file(implicit, builtin)
        self.metadata["models"] = self._models

    def _merge_models_file(self, path: Path, builtin: set[str]) -> None:
        """
        Add one models file's entries, all or nothing.

        :param path: The models file.
        :param builtin: Names shipped in ``metadata.yaml``.
        :raises ModelLoadingError: If the file is unreadable, empty, reuses a
            name that is already registered, or holds a value of a shape this
            code can't handle.
        """
        try:
            self._merge_models_file_unchecked(path, builtin)
        except _USER_VALUE_ERRORS as exc:
            # A hand-written file can hold shapes no check anticipated; they
            # must fail as a ModelLoadingError (so a broken default file is
            # skipped), not a traceback.
            raise ModelLoadingError(
                f"Extra models file {path} holds a value unblend can't use "
                f"({type(exc).__name__}: {exc})."
            ) from exc

    def _merge_models_file_unchecked(self, path: Path, builtin: set[str]) -> None:
        """
        :meth:`_merge_models_file` without the conversion of unexpected errors.

        :param path: The models file.
        :param builtin: Names shipped in ``metadata.yaml``.
        :raises ModelLoadingError: If the file is unreadable, empty, or reuses a
            name that is already registered.
        """
        try:
            payload = _load_mapping(path)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ModelLoadingError(
                f"Could not read extra models file {path}: {exc}"
            ) from exc
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, dict):
            raise ModelLoadingError(
                f"Extra models file {path} must contain a 'models' object."
            )
        unknown = sorted(str(key) for key in set(payload) - {"version", "models"})
        if unknown:
            raise ModelLoadingError(
                f"Extra models file {path} has unknown top-level field(s) "
                f"{', '.join(unknown)}; expected 'version' and 'models'."
            )
        version = payload.get("version", 1)
        if isinstance(version, bool) or version != 1:
            raise ModelLoadingError(
                f"Extra models file {path} has version {payload['version']!r}; "
                "this unblend reads version 1."
            )
        taken = {name.casefold(): name for name in self._models}
        here: set[str] = set()
        for model_name in models:
            if (
                not isinstance(model_name, str)
                or not _MODEL_NAME.fullmatch(model_name)
                or ".." in model_name
            ):
                # Names become output folders ({model}) and import filenames.
                raise ModelLoadingError(
                    f"Extra models file {path} has an invalid model name "
                    f"{model_name!r}: use letters, digits, '_', '-' and '.'."
                )
            if model_name.casefold() == "auto":
                raise ModelLoadingError(
                    f"Extra models file {path} uses the reserved name 'auto' "
                    "(it means auto-select on the command line)."
                )
            clash = taken.get(model_name.casefold())
            if clash is not None:
                kind = (
                    "built-in"
                    if clash in builtin
                    else "duplicate (names are case-insensitive)"
                    if clash in here
                    else "already registered"
                )
                raise ModelLoadingError(
                    f"Extra models file {path} redefines {kind} model "
                    f"{clash!r}; choose a different name."
                )
            taken[model_name.casefold()] = model_name
            here.add(model_name)
        for info in models.values():
            # The real file's folder, so a symlinked models file reads its
            # relative paths the same way register/unregister validate them.
            _anchor_relative_paths(info, Path(os.path.realpath(path)).parent)
        self._models.update(models)
        self._origins.update({model_name: path for model_name in models})

    def _artifacts(self, name: str) -> list[dict]:
        """
        Every weight artifact a model loads, in member order.

        :param name: Model name.
        :return: The artifact entries, empty for an unknown model.
        """
        return [member["artifact"] for member in self._members.get(name, ())]

    def weight_files(self, name: str) -> list[dict]:
        """
        A model's distinct weight artifacts, each listed once.

        An ensemble may use one checkpoint for two members; sizes and file
        counts should see it once.

        :param name: Model name.
        :return: The artifact entries in member order, without repeats.
        """
        unique: dict[str, dict] = {}
        for spec in self._artifacts(name):
            unique.setdefault(_artifact_identity(spec), spec)
        return list(unique.values())

    def local_artifacts(self, name: str) -> list[Path]:
        """
        Paths of weight artifacts a model reads from disk.

        :param name: Model name.
        :return: The declared local paths.
        """
        info = self._models.get(name)
        if info is None:
            return []
        return [
            path
            for spec in self.weight_files(name)
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
        Get information about cached models, including partially-cached ones.

        :return: Mapping of model names to ``{"files", "size_bytes",
            "total_files", "complete"}`` dicts.
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
                    # A truncated or corrupt file is listed (so removal still
                    # finds it) but isn't a usable copy.
                    "complete": size_bytes == spec.get("size_bytes"),
                }
            if not components:
                continue

            cached_models[name] = {
                "files": components,
                "size_bytes": sum(c["size_bytes"] for c in components.values()),
                # Unique files: an ensemble may use one checkpoint twice.
                "total_files": len({_artifact_cache_key(spec) for spec in remote}),
                "complete": len(components)
                == len({_artifact_cache_key(spec) for spec in remote})
                and all(c["complete"] for c in components.values()),
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
        file_index: int = 1,
        total_files: int = 1,
    ) -> Iterator[Path]:
        """
        Yield a verified Safetensors path for one artifact.

        :param spec: The artifact entry.
        :param label: Human-readable prefix for error messages.
        :param progress_callback: Optional progress callback.
        :param model_name: Model name for progress payloads.
        :param file_index: 1-based index within the model.
        :param total_files: How many artifacts the model needs.
        :return: Context manager yielding the verified path.
        """
        local = _artifact_path(spec)
        if local is not None:
            if not os.path.isfile(local):
                raise ModelLoadingError(
                    f"{label} declares a local checkpoint that does not exist: {local}"
                )
            if spec.get("size_bytes") is not None:
                check_size(local, spec["size_bytes"])
            if spec.get("sha256") is not None:
                check_checksum(local, spec["sha256"])
            _emit(
                progress_callback,
                "file_complete",
                model_name=model_name,
                file_index=file_index,
                total_files=total_files,
                cached=True,
            )
            yield local
            return

        url = _artifact_url(spec)
        expected = spec["sha256"]
        expected_size = spec["size_bytes"]
        cache_path = _artifact_cache_path(spec)

        # A read-only or shared cache can't hold our lock file. Downloads land
        # with an atomic rename, so a file already there is complete and can be
        # verified and read without the lock (the lock only guards against a
        # concurrent remove, which such a cache can't do either).
        if os.path.isfile(cache_path) and not os.access(cache_path.parent, os.W_OK):
            check_size(cache_path, expected_size)
            check_checksum(cache_path, expected)
            _emit(
                progress_callback,
                "file_complete",
                model_name=model_name,
                file_index=file_index,
                total_files=total_files,
                cached=True,
            )
            yield cache_path
            return

        with _artifact_lock(cache_path):
            cached = False
            if os.path.exists(cache_path):
                try:
                    check_size(cache_path, expected_size)
                    check_checksum(cache_path, expected)
                    cached = True
                except OSError as exc:
                    raise ModelLoadingError(
                        f"Could not read cached artifact {cache_path}: {exc}"
                    ) from exc
                except ModelLoadingError as exc:
                    # A read error keeps the file; only a mismatch is worth
                    # deleting it over.
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
                    "file_complete",
                    model_name=model_name,
                    file_index=file_index,
                    total_files=total_files,
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
                    file_index=file_index,
                    total_files=total_files,
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
        file_index: int = 1,
        total_files: int = 1,
    ) -> None:
        """
        Stream one artifact to the cache with SHA-256 verification.

        :param url: Source URL.
        :param cache_path: Destination path in the cache.
        :param expected_sha256: Digest to verify against.
        :param expected_size: Exact artifact size.
        :param progress_callback: Optional download-progress callback.
        :param model_name: Model name for progress payloads.
        :param file_index: 1-based index within the model.
        :param total_files: How many artifacts the model needs.
        """
        # Staging files from killed downloads are otherwise only swept by
        # ``models remove --all``.
        self.sweep_stale_downloads()
        tmp_path: Path | None = None
        started = time.monotonic()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                delete=False,
                prefix=f"{STAGING_PREFIX}{cache_path.name}.",
                suffix=".tmp",
                dir=cache_path.parent,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                downloaded = 0
                announced = False
                for attempt in range(DOWNLOAD_ATTEMPTS):
                    if downloaded == expected_size:
                        # Dropped after the last byte; a Range request would
                        # only get a 416.
                        break
                    # After a dropped connection, resume where it stopped.
                    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
                    try:
                        with httpx.stream(
                            "GET",
                            url,
                            headers=headers,
                            follow_redirects=True,
                            timeout=30.0,
                        ) as response:
                            response.raise_for_status()
                            if downloaded and response.status_code != 206:
                                # The server ignored Range: start over.
                                tmp_file.seek(0)
                                tmp_file.truncate()
                                downloaded = 0
                            remaining = int(response.headers.get("content-length", 0))
                            if remaining and downloaded + remaining != expected_size:
                                raise ModelLoadingError(
                                    f"Download size for {url} is "
                                    f"{downloaded + remaining} bytes; expected "
                                    f"{expected_size}."
                                )
                            total_size = expected_size
                            if not announced:
                                _emit(
                                    progress_callback,
                                    "file_start",
                                    model_name=model_name,
                                    file_index=file_index,
                                    total_files=total_files,
                                    file_size_bytes=total_size,
                                )
                                announced = True
                            counter = 0
                            for chunk in response.iter_bytes(chunk_size=8192):
                                downloaded += len(chunk)
                                if downloaded > expected_size:
                                    raise ModelLoadingError(
                                        f"Download from {url} exceeded the "
                                        f"expected {expected_size} bytes."
                                    )
                                if (
                                    time.monotonic() - started
                                    > DOWNLOAD_DEADLINE_SECONDS
                                ):
                                    raise ModelLoadingError(
                                        f"Download from {url} exceeded the "
                                        f"{DOWNLOAD_DEADLINE_SECONDS}-second "
                                        "deadline."
                                    )
                                tmp_file.write(chunk)
                                counter += 1
                                if progress_callback and counter % 20 == 0:
                                    _emit(
                                        progress_callback,
                                        "file_progress",
                                        model_name=model_name,
                                        file_index=file_index,
                                        total_files=total_files,
                                        progress_percent=downloaded / total_size * 100,
                                        downloaded_bytes=downloaded,
                                        total_bytes=total_size,
                                    )
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                        transient = isinstance(exc, httpx.TransportError) or (
                            exc.response.status_code >= 500
                            or exc.response.status_code == 429
                        )
                        if not transient or attempt == DOWNLOAD_ATTEMPTS - 1:
                            raise
                        tmp_file.flush()
                        time.sleep(min(2**attempt, 10))
            if downloaded != expected_size:
                raise ModelLoadingError(
                    f"Download from {url} ended at {downloaded} bytes; "
                    f"expected {expected_size}."
                )

            check_size(tmp_path, expected_size)
            check_checksum(tmp_path, expected_sha256)
            # NamedTemporaryFile creates 0600; a shared cache needs the umask's
            # normal permissions.
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp_path, 0o666 & ~umask)
            os.replace(tmp_path, cache_path)
            tmp_path = None
            _emit(
                progress_callback,
                "file_complete",
                model_name=model_name,
                file_index=file_index,
                total_files=total_files,
            )
        except httpx.HTTPError as e:
            raise ModelLoadingError(f"Failed to download {url}: {e}") from e
        except ModelLoadingError:
            raise
        except Exception as e:
            raise ModelLoadingError(f"Failed to download/verify {url}: {e}") from e
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                tmp_path.unlink()

    def _select_members(
        self, name: str, only_load: str | None = None
    ) -> tuple[list[dict], list[list[float]] | None]:
        """
        Resolve which members get_model needs.

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

    def loaded_files(self, name: str, only_load: str | None = None) -> set[str]:
        """
        The distinct weight files :meth:`get_model` would read.

        :param name: Model name.
        :param only_load: Optional stem to isolate.
        :return: One identity per file (real path if local, cache key if
            remote), so sets from several models can be merged.
        """
        members, _ = self._select_members(name, only_load)
        return {_artifact_identity(member["artifact"]) for member in members}

    def loaded_bytes(self, name: str, only_load: str | None = None) -> int:
        """
        Size on disk of the distinct files :meth:`get_model` would read.

        :param name: Model name.
        :param only_load: Optional stem to isolate.
        :return: Bytes of the local files and of the cached remote ones;
            missing files count as zero.
        """
        members, _ = self._select_members(name, only_load)
        files: dict[str, Path] = {}
        for member in members:
            spec = member["artifact"]
            local = _artifact_path(spec)
            files.setdefault(
                _artifact_identity(spec),
                local if local is not None else _artifact_cache_path(spec),
            )
        return sum(
            path.stat().st_size for path in files.values() if os.path.isfile(path)
        )

    def loaded_file_count(self, name: str, only_load: str | None = None) -> int:
        """
        How many distinct weight files :meth:`get_model` would read.

        :param name: Model name.
        :param only_load: Optional stem to isolate.
        :return: Local and remote files, a shared one counted once.
        """
        return len(self.loaded_files(name, only_load))

    def required_files(self, name: str, only_load: str | None = None) -> list[str]:
        """
        Cache keys of the remote artifacts :meth:`get_model` would load.

        :param name: Model name.
        :param only_load: Optional stem to isolate.
        :return: List of artifact cache keys.
        """
        members, _ = self._select_members(name, only_load)
        # dict.fromkeys: a checkpoint two members share is fetched once.
        return list(
            dict.fromkeys(
                _artifact_cache_key(member["artifact"])
                for member in members
                if _artifact_url(member["artifact"]) is not None
            )
        )

    def get_model(
        self,
        name: str,
        only_load: str | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Model | ModelEnsemble:
        """
        Get a model by name, downloading whatever is not cached.

        With ``only_load``, an ensemble whose weights give that stem to a single
        member loads only that member.

        :param name: Model name.
        :param only_load: Stem to isolate, if any.
        :param progress_callback: Optional download-progress callback.
        :return: The loaded model or ensemble.
        :raises ModelLoadingError: If the model is unknown, a download or
            verification fails, or the weights do not build.
        """
        members, weights = self._select_members(name, only_load)
        model_info = self._models[name]
        # Progress counts distinct files, each reported once: a member
        # reusing another's checkpoint emits nothing for it.
        file_indices: dict[str, int] = {}
        for member in members:
            key = _artifact_identity(member["artifact"])
            file_indices.setdefault(key, len(file_indices) + 1)
        total_files = len(file_indices)

        _emit(
            progress_callback,
            "download_start",
            model_name=name,
            total_files=total_files,
        )

        built: list[Model] = []
        reported: set[int] = set()
        for index, member in enumerate(members, start=1):
            label = (
                f"Member {index} of model {name}"
                if len(members) > 1
                else f"Model {name}"
            )
            file_index = file_indices[_artifact_identity(member["artifact"])]
            with self._resolved_artifact(
                member["artifact"],
                label=label,
                # A file already reported (just downloaded, perhaps) isn't
                # reported again as cached.
                progress_callback=progress_callback
                if file_index not in reported
                else None,
                model_name=name,
                file_index=file_index,
                total_files=total_files,
            ) as path:
                state = _read_state(path)
            reported.add(file_index)
            built.append(self._build_member(label, member, state))
            del state

        _emit(
            progress_callback,
            "download_complete",
            model_name=name,
            total_files=total_files,
        )

        segment = model_info.get("segment")
        if len(built) == 1:
            model = built[0]
            if segment is not None:
                model.max_allowed_segment = min(
                    float(segment), float(model.max_allowed_segment)
                )
            return model

        try:
            return ModelEnsemble(
                built,
                weights,
                segment,
                model_info.get("combine", COMBINE_DEFAULT),
                model_info.get("combine_params"),
            ).eval()
        except ValidationError as exc:
            raise ModelLoadingError(f"Model {name}: {exc}") from exc

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
            raise ModelLoadingError(
                f"Failed to build {label} from checkpoint: {exc}"
            ) from exc

    def list_models(self) -> dict[str, dict]:
        """
        List all available models.

        :return: Dictionary mapping model names to their metadata (deep
            copies — mutating them does not affect repository state)
        """
        return {name: copy.deepcopy(info) for name, info in self._models.items()}

    def shared_artifacts(self, name: str) -> dict[Path, list[str]]:
        """
        Cached artifacts of ``name`` that other fully downloaded models also use.

        :param name: Model name.
        :return: Cache path to the other models that share it.
        """
        mine = {
            _artifact_cache_path(spec)
            for spec in self._artifacts(name)
            if _artifact_url(spec) is not None
        }
        shared: dict[Path, list[str]] = {}
        for other in self._models:
            if other == name:
                continue
            remote = [
                _artifact_cache_path(spec)
                for spec in self._artifacts(other)
                if _artifact_url(spec) is not None
            ]
            # Only a model you actually have (every file cached) counts as a
            # user; a registered ensemble you never downloaded doesn't.
            if not remote or not all(os.path.isfile(path) for path in remote):
                continue
            for spec in self._artifacts(other):
                if _artifact_url(spec) is None:
                    continue
                path = _artifact_cache_path(spec)
                if path in mine and other not in shared.setdefault(path, []):
                    shared[path].append(other)
        return shared

    def remove_model(
        self,
        name: str,
        include_shared: bool = False,
        also_removing: Iterable[str] = (),
    ) -> bool:
        """
        Remove a model's downloaded artifacts from the cache.

        Files another fully downloaded model also uses are kept unless
        ``include_shared`` is set, or every model sharing them is in
        ``also_removing``; see :meth:`shared_artifacts`.

        :param name: Model name.
        :param include_shared: Also delete files other models share.
        :param also_removing: Models being removed in the same operation.
        :return: True if any cached artifact was removed.
        """
        if name not in self._models:
            return False

        removing = set(also_removing)
        keep = (
            set()
            if include_shared
            else {
                path
                for path, users in self.shared_artifacts(name).items()
                if not set(users) <= removing
            }
        )
        removed_any = False
        for spec in self._artifacts(name):
            if _artifact_url(spec) is None:
                continue
            path = _artifact_cache_path(spec)
            if path in keep:
                continue
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
