# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Bring a checkpoint from elsewhere into Unblend's registry.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from filelock import FileLock
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from . import backends, scnet  # noqa: F401  (registers the builders)
from ._paths import name_encodable, name_fits
from .exceptions import ModelLoadingError, ValidationError
from .htdemucs import HTDemucs
from .repo import _MODEL_NAME, _load_mapping, stem_name_problem


def _dump_mapping(payload: Any, path: Path) -> str:
    """
    Serialize a mapping to YAML or JSON based on path suffix.

    :param payload: Data to serialize.
    :param path: Destination path.
    :return: Serialized text.
    """
    if path.suffix == ".json":
        return json.dumps(payload, indent=2) + "\n"
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


EMBEDDED_FORMAT = "1"

EMBEDDED_FIELDS = ("architecture", "sources", "samplerate", "segment_samples", "config")

STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "state")

WRAPPER_PREFIXES = ("model.", "module.", "net.", "_orig_mod.")


def printable(text: str) -> str:
    """
    ``text`` without terminal control sequences, for messages that quote a
    file's own contents (key names, header fields, pickled globals): a
    crafted file could otherwise reset, retitle or clear the terminal.

    :param text: The text.
    :return: It with colour codes and every other control character except
        newline removed.
    """
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]", "", text)


def _load_failure_reason(exc: BaseException) -> str:
    """
    The informative line of a ``torch.load`` failure.

    torch rewraps a weights-only refusal in advice to retry with
    ``weights_only=False`` and to file an issue, which doesn't apply here;
    the unpickler's own error, which it chains as the context, names what
    was refused.

    :param exc: The exception ``torch.load`` raised.
    :return: One short line.
    """
    message = str(exc)
    if message.startswith("Weights only load failed") and exc.__context__ is not None:
        message = str(exc.__context__)
    message = printable(message)
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return "unreadable"
    line = lines[0].split(" was not an allowed global")[0]
    # "[enforce fail at inline_container.cc:180] . file in archive ..."
    line = re.sub(r"^\[enforce fail at [^\]]*\]\s*\.?\s*", "", line)
    # The first sentence; torch follows it with internal detail.
    return line.split(". ")[0].rstrip(".") or "unreadable"


def read_tensors(path: Path) -> dict[str, Tensor]:
    """
    Read a checkpoint's tensors, whatever container they arrived in.

    :param path: Path to a ``.safetensors``, ``.ckpt``, ``.pt``, ``.pth`` or ``.th`` file.
    :return: The state dict, unwrapped and un-prefixed.
    """
    if path.suffix.lower() == ".onnx":
        raise ValidationError(
            f"{path} is an ONNX graph, not a PyTorch checkpoint; it can't be "
            "imported. (UVR's .onnx models are MDX-Net or VR-arch, which Unblend "
            "doesn't implement.)"
        )
    if path.suffix.lower() == ".safetensors":
        try:
            return strip_wrapper_prefix(load_file(path, device="cpu"))
        except SafetensorError as exc:
            raise ValidationError(
                f"Could not read {path}: {printable(str(exc))}"
            ) from exc

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except EOFError as exc:
        raise ValidationError(
            f"Could not read {path}: the file is empty or truncated."
        ) from exc
    except Exception as exc:
        reason = _load_failure_reason(exc)
        # torch's message can't tell a refusal of pickled objects from a
        # truncated or corrupt file (a cut can land mid-pickle and read as a
        # disallowed global), so name both causes.
        hint = (
            " Official HTDemucs checkpoints don't need importing: Unblend ships "
            "their weights as built-in models."
            if "demucs.htdemucs.HTDemucs" in reason
            else ""
        )
        raise ValidationError(
            f"Could not read {path} as a tensor-only checkpoint ({reason}). "
            "The file is truncated or corrupt, or it pickles Python objects "
            "rather than just weights; loading those runs code from the file, "
            f"so inspect it before trusting it.{hint}"
        ) from exc

    state = loaded
    for key in STATE_DICT_KEYS:
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break

    if not isinstance(state, dict):
        raise ValidationError(f"{path} does not contain a state dict.")

    tensors = {
        key: value for key, value in state.items() if isinstance(value, torch.Tensor)
    }
    if not tensors:
        raise ValidationError(f"{path} contains no tensors.")

    return strip_wrapper_prefix(tensors)


def strip_wrapper_prefix(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    Remove a prefix a trainer added to every parameter name.

    Parameter names must match the architecture exactly, and a wrapper like
    Lightning's ``model.`` would break that on every key at once.

    :param state: The state dict as read.
    :return: The state dict with any uniform wrapper prefix removed.
    """
    for prefix in WRAPPER_PREFIXES:
        if state and all(key.startswith(prefix) for key in state):
            return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def read_config(path: Path) -> dict:
    """
    Read a model config file.

    :param path: Path to the config.
    :return: Its contents as a mapping.
    :raises ValidationError: If the file cannot be read or parsed.
    """
    try:
        loaded = _load_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"Could not read {path}: {printable(str(exc))}") from exc
    except RecursionError as exc:
        raise ValidationError(f"Could not read {path}: nested too deeply.") from exc

    if not isinstance(loaded, dict):
        raise ValidationError(f"{path} must contain a mapping.")
    return loaded


def constructor_config(architecture: str, config: dict) -> tuple[dict, list[str]]:
    """
    Keep only the config keys the architecture's constructor takes. Training
    configs carry options such as ``flash_attn`` that inference doesn't.

    :param architecture: Architecture to build.
    :param config: Candidate constructor kwargs.
    :return: ``(kept config, sorted dropped keys)``.
    """
    klass = _architecture_class(architecture)
    if klass is None:
        return dict(config), []
    # A subclass that forwards **kwargs (SCNetMasked) takes its parent's.
    params: dict[str, inspect.Parameter] = {}
    for cls in klass.__mro__:
        if "__init__" not in vars(cls) or cls is object:
            continue
        own = inspect.signature(cls.__init__).parameters
        params.update(own)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in own.values()):
            break
    else:
        return dict(config), []
    kept = {key: value for key, value in config.items() if key in params}
    return kept, sorted(str(key) for key in set(config) - set(kept))


def _architecture_class(architecture: str) -> type | None:
    """
    The class that builds an architecture, if it's one unblend implements.

    :param architecture: Architecture name.
    :return: The model class, or ``None``.
    """
    if architecture == "htdemucs":
        return HTDemucs
    from . import roformer

    return {**roformer._ARCHITECTURES, **scnet._ARCHITECTURES}.get(architecture)


def _seconds_to_samples(seconds: object, samplerate: object) -> object:
    """
    A config's ``training.segment`` (seconds) as a sample count.

    :param seconds: The segment length in seconds.
    :param samplerate: The final sample rate, if known.
    :return: The sample count, or ``seconds`` unchanged when it can't be
        converted, so validation reports it (or the missing rate) rather
        than storing seconds as samples. An integer too large to multiply
        as a float is passed through too; it is a valid-looking count, so
        it fails later, when the model is loaded or run with it.
    """
    if (
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and isinstance(samplerate, int)
        and not isinstance(samplerate, bool)
    ):
        try:
            samples = seconds * samplerate
            if math.isfinite(samples):
                return int(round(samples))
        except OverflowError:
            # Integers too large for a float.
            pass
    return seconds


def fields_from_config(raw: dict) -> dict[str, Any]:
    """
    Translate a training config into registry fields.

    :param raw: The parsed config file.
    :return: Registry fields such as sources, samplerate and segment length,
        plus ``segment_seconds`` when the length is only given in seconds
        (``import_checkpoint`` converts it once the sample rate is final).
    """
    fields: dict[str, Any] = {}

    model_section = raw.get("model")
    if isinstance(model_section, dict):
        fields["config"] = model_section
    elif isinstance(raw.get("htdemucs"), dict):
        # MSST's HTDemucs configs keep the constructor kwargs in a section
        # named after the architecture (``model: htdemucs`` points at it).
        fields["config"] = raw["htdemucs"]
        fields["architecture"] = "htdemucs"
    elif "config" in raw and isinstance(raw["config"], dict):
        fields["config"] = raw["config"]

    audio = raw.get("audio") if isinstance(raw.get("audio"), dict) else {}
    training = raw.get("training") if isinstance(raw.get("training"), dict) else {}

    # The first value given, so an explicit null doesn't hide a later one. A
    # value of the wrong type is kept, to be reported as invalid rather than
    # as missing.
    samplerate = next(
        (
            value
            for value in (
                audio.get("sample_rate"),
                training.get("samplerate"),
                raw.get("samplerate"),
            )
            if value is not None
        ),
        None,
    )
    if samplerate is not None:
        fields["samplerate"] = samplerate
    segment = next(
        (
            value
            for value in (audio.get("chunk_size"), raw.get("segment_samples"))
            if value is not None
        ),
        None,
    )
    if segment is not None:
        fields["segment_samples"] = segment
    elif training.get("segment") is not None:
        # In seconds: converted once the sample rate is final, which a
        # header or --samplerate may still supply.
        fields["segment_seconds"] = training["segment"]

    target = training.get("target_instrument")
    # Wrong-typed or non-string values are kept, for the stem-name check to
    # report; an empty list counts as not given.
    instruments = next(
        (
            value
            for value in (training.get("instruments"), raw.get("sources"))
            if value is not None and value != []
        ),
        None,
    )
    if isinstance(target, str) and target:
        complement = (
            "vocals" if target.casefold() in {"other", "instrumental"} else "other"
        )
        fields["sources"] = [target, complement]
    elif instruments is not None:
        fields["sources"] = instruments

    return fields


def read_embedded_fields(path: Path) -> dict[str, Any]:
    """
    Read the registry fields a Safetensors artifact records about itself.

    Only the header is read, so this costs a few kilobytes regardless of the
    file's size.

    :param path: Path to a Safetensors artifact.
    :return: The embedded fields, or an empty mapping if there are none.
    """
    try:
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return {}

    if metadata.get("unblend_format") != EMBEDDED_FORMAT:
        return {}

    fields: dict[str, Any] = {}
    for key in ("architecture",):
        if key in metadata:
            fields[key] = metadata[key]
    for key, kind in (("sources", list), ("config", dict)):
        if key in metadata:
            try:
                fields[key] = json.loads(metadata[key])
            except json.JSONDecodeError:
                return {}
            # A hand-edited header of the wrong shape isn't usable either.
            if not isinstance(fields[key], kind) or (
                key == "sources"
                and not all(isinstance(name, str) for name in fields[key])
            ):
                return {}
    for key in ("samplerate", "segment_samples"):
        if key in metadata:
            try:
                fields[key] = int(metadata[key])
            except ValueError:
                return {}
    return fields


def candidate_architectures(state: dict[str, Tensor], config: dict) -> list[str]:
    """
    Which architectures a checkpoint could be, narrowed by its parameter names.

    :param state: The checkpoint's tensors.
    :param config: The constructor config, if known.
    :return: Candidate architecture names, most likely first.
    """
    roots = {key.split(".")[0] for key in state}

    if {"crosstransformer", "tencoder", "tdecoder"} & roots:
        return ["htdemucs"]
    if "separation_net" in roots:
        return ["scnet_masked"] if {"mask_layer", "pos_embed_f"} & roots else ["scnet"]
    if "band_split" in roots:
        if "num_bands" in config:
            return ["mel_band_roformer"]
        if "freqs_per_bands" in config:
            return ["bs_roformer"]
        return ["bs_roformer", "mel_band_roformer"]
    return []


def _htdemucs_config(
    config: dict, sources: list[str], samplerate: int, segment_samples: int
) -> dict:
    """
    HTDemucs constructor kwargs with the entry's stems and geometry.

    HTDemucs takes its sample rate and segment (in seconds) as constructor
    kwargs, so the registry fields have to be written into the config.

    :param config: Constructor kwargs.
    :param sources: Output stem names.
    :param samplerate: Sample rate the weights operate at.
    :param segment_samples: Training chunk length in samples.
    :return: The completed kwargs.
    """
    segment = int(segment_samples) / int(samplerate)
    return {
        **dict(config),
        "sources": list(sources),
        "samplerate": int(samplerate),
        "segment": int(segment) if segment.is_integer() else segment,
    }


def build_and_verify(
    architecture: str,
    config: dict,
    *,
    sources: list[str],
    samplerate: int,
    segment_samples: int,
    state: dict[str, Tensor],
) -> torch.nn.Module:
    """
    Build the architecture and strict-load the checkpoint into it.

    :param architecture: Architecture to build.
    :param config: Constructor kwargs.
    :param sources: Output stem names.
    :param samplerate: Sample rate the weights operate at.
    :param segment_samples: Training chunk length in samples.
    :param state: The checkpoint's tensors.
    :return: The loaded, eval-mode model.
    """
    config, _dropped = constructor_config(architecture, config)
    try:
        if architecture == "htdemucs":
            # The entry's sources (config, --stem, or header) are the stems;
            # build_entry writes them into the config the same way.
            model = HTDemucs(
                **_htdemucs_config(config, sources, samplerate, segment_samples)
            )
            model.load_state_dict(state, strict=True)
            return model.eval()
        backend = backends.backend_for_architecture(architecture)
        if backend is None:
            raise ValidationError(f"Unknown architecture {architecture!r}.")
        return backends.build(
            backend,
            architecture,
            dict(config),
            sources=list(sources),
            samplerate=int(samplerate),
            segment_samples=int(segment_samples),
            state=state,
        )
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"Checkpoint does not load as {architecture}: {_brief(exc)}"
        ) from exc


def _brief(exc: Exception, lines: int = 12, line_chars: int = 600) -> str:
    """
    An exception's message, cut short. torch reports a strict load that
    doesn't fit as one very long line per category ("Missing key(s)",
    "Unexpected key(s)", size mismatches); each is cut to its start so every
    category still shows, and only the first ``lines`` lines are kept.

    :param exc: The exception.
    :param lines: Most lines kept.
    :param line_chars: Most characters kept per line.
    :return: The shortened message.
    """
    kept = []
    # Strict-load dumps quote the file's own key names.
    for line in printable(str(exc)).splitlines():
        if len(line) > line_chars:
            line = f"{line[:line_chars]} ... ({len(line) - line_chars} more characters)"
        kept.append(line)
    if len(kept) > lines:
        kept = [*kept[:lines], f"... ({len(kept) - lines} more lines)"]
    return "\n".join(kept)


def _check_runs(model: torch.nn.Module, segment_samples: int) -> None:
    """
    Run one silent segment through a strict-loaded model.

    Some config mistakes keep every weight shape, so the checkpoint loads and
    only the forward fails; this catches them before anything is registered.

    :param model: The loaded, eval-mode model.
    :param segment_samples: Training chunk length in samples.
    :raises ValidationError: If the forward fails or isn't finite.
    """
    from .apply import apply_model

    channels = int(getattr(model, "audio_channels", 2))
    try:
        with torch.inference_mode():
            # Through apply_model, which pads chunks as separation does.
            out = apply_model(model, torch.zeros(1, channels, segment_samples))
    except Exception as exc:
        raise ValidationError(
            f"The checkpoint loads but the model doesn't run with this config: {_brief(exc)}"
        ) from exc
    if not torch.isfinite(out).all():
        raise ValidationError(
            "The checkpoint loads but produces non-finite output on silence; "
            "check the config."
        )


def resolve_architecture(
    state: dict[str, Tensor],
    config: dict,
    *,
    sources: list[str],
    samplerate: int,
    segment_samples: int,
    architecture: str | None = None,
) -> tuple[str, torch.nn.Module]:
    """
    Determine which architecture a checkpoint is, by building it.

    :param state: The checkpoint's tensors.
    :param config: Constructor kwargs.
    :param sources: Output stem names.
    :param samplerate: Sample rate the weights operate at.
    :param segment_samples: Training chunk length in samples.
    :param architecture: Explicit architecture, or ``None`` to infer.
    :return: ``(architecture, loaded model)``.
    """
    if architecture is not None:
        return architecture, build_and_verify(
            architecture,
            config,
            sources=sources,
            samplerate=samplerate,
            segment_samples=segment_samples,
            state=state,
        )

    candidates = candidate_architectures(state, config)
    if not candidates:
        raise ValidationError(
            "These weights do not resemble any architecture Unblend "
            "implements (htdemucs, bs_roformer, mel_band_roformer, scnet, "
            "scnet_masked). MDX-Net and VR-arch models are different "
            "architectures, not different packaging."
        )

    failures = []
    for candidate in candidates:
        try:
            return candidate, build_and_verify(
                candidate,
                config,
                sources=sources,
                samplerate=samplerate,
                segment_samples=segment_samples,
                state=state,
            )
        except ValidationError as exc:
            failures.append(f"  {candidate}: {exc}")

    raise ValidationError(
        "Could not load the checkpoint as any matching architecture:\n"
        + "\n".join(failures)
    )


def write_artifact(
    state: dict[str, Tensor], path: Path, fields: dict[str, Any]
) -> None:
    """
    Write tensors as Safetensors, with the registry fields in the header.

    The header is covered by the file's own hash. A models-file entry can then
    omit these fields; any it does state take precedence over the header.

    The file is written beside ``path`` and then claimed with a hard link,
    so an interrupted write leaves no partial artifact and a concurrent
    import to the same path fails instead of overwriting.

    :param state: Tensors to write.
    :param path: Destination path.
    :param fields: Registry fields to embed.
    :raises ValidationError: If ``path`` already exists.
    """
    metadata = {"unblend_format": EMBEDDED_FORMAT}
    for key in EMBEDDED_FIELDS:
        if key not in fields:
            continue
        value = fields[key]
        metadata[key] = (
            json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    # safetensors refuses tensors that share storage (RoFormer checkpoints
    # alias their rotary ``freqs``), so give repeats their own copy.
    seen: set[int] = set()
    tensors = {}
    for key, value in state.items():
        value = value.contiguous()
        pointer = value.untyped_storage().data_ptr()
        tensors[key] = value.clone() if pointer in seen else value
        seen.add(pointer)
    staging = str(path.parent / f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        save_file(tensors, staging, metadata=metadata)
        # safetensors creates files 0600; give it the umask's normal mode.
        os.chmod(staging, _default_file_mode())
        try:
            os.link(staging, path)
        except FileExistsError:
            raise ValidationError(
                f"{path} already exists; choose another output path."
            ) from None
        except OSError:
            # No hard links on this filesystem: fall back to a checked rename.
            if os.path.exists(path):
                raise ValidationError(
                    f"{path} already exists; choose another output path."
                ) from None
            os.replace(staging, path)
    finally:
        Path(staging).unlink(missing_ok=True)


def build_entry(
    fields: dict[str, Any], artifact: Path, license_label: str, note: str | None
) -> dict:
    """
    Assemble the registry entry for an imported checkpoint.

    :param fields: Architecture, sources, geometry and config.
    :param artifact: Path to the written Safetensors file.
    :param license_label: Free-form license label for the entry.
    :param note: Optional provenance note.
    :return: A registry entry.
    """
    entry = {
        "architecture": fields["architecture"],
        "license": license_label,
        "sources": list(fields["sources"]),
        "samplerate": int(fields["samplerate"]),
        "segment_samples": int(fields["segment_samples"]),
        "config": dict(fields["config"]),
        "checkpoint": {
            "format": "safetensors",
            "path": str(Path(os.path.realpath(os.path.expanduser(artifact)))),
        },
    }
    if note:
        entry["provenance"] = note
    return entry


# ``--model auto`` asks the CLI to pick a model, so no model can be called that.
RESERVED_MODEL_NAMES = frozenset({"auto"})


def validate_model_name(name: str) -> None:
    """
    Reject names that can't be used safely as a model name and a filename.

    :param name: Proposed model name.
    :raises ValidationError: If the name has path characters or is reserved.
    """
    if not _MODEL_NAME.fullmatch(name) or ".." in name:
        raise ValidationError(
            f"Invalid model name {name!r}: use letters, digits, '_', '-' and "
            "'.', starting with a letter or digit."
        )
    if name.casefold() in RESERVED_MODEL_NAMES:
        raise ValidationError(f"{name!r} is reserved; choose another name.")


def check_registrable(models_file: Path, name: str) -> dict[str, Any]:
    """
    Confirm ``name`` can be added to ``models_file`` before any file is written.

    :param models_file: The user models file that will be written.
    :param name: Model name to register.
    :return: The file's current payload, or an empty one if it doesn't exist.
    :raises ValidationError: If the file exists but is not a models file,
        can't be written or doesn't load as it is, or the name is invalid,
        built in or already registered (here or in any models file loaded by
        default).
    """
    from .repo import ModelRepository, default_extra_models_files

    validate_model_name(name)
    # Case-insensitively: the name is also a filename, and macOS and Windows
    # filesystems would make ``mine`` and ``MINE`` the same weights file.
    key = name.casefold()
    if key in {n.casefold() for n in ModelRepository(extra_models=[]).list_models()}:
        raise ValidationError(f"{name!r} is a built-in model; choose another name.")
    try:
        registered = ModelRepository().list_models()
    except ModelLoadingError as exc:
        raise ValidationError(f"Could not read the current registry: {exc}") from exc
    existing = {n.casefold(): n for n in registered}.get(key)
    if existing is not None:
        # The registered spelling: unregister matches names exactly.
        raise ValidationError(
            f"{existing!r} is already registered; choose another name or run "
            f"'unblend models unregister {existing}'."
        )
    # Also names in a models file the registry is skipping right now: once
    # it's fixed, a clash would make the registry drop it whole.
    for other in default_extra_models_files():
        if Path(os.path.realpath(other)) == Path(os.path.realpath(models_file)):
            continue
        try:
            payload_other = _load_mapping(other)
        except Exception:
            continue
        other_models = (
            payload_other.get("models") if isinstance(payload_other, dict) else None
        )
        clash = (
            next((str(n) for n in other_models if str(n).casefold() == key), None)
            if isinstance(other_models, dict)
            else None
        )
        if clash is not None:
            raise ValidationError(
                f"{other} already defines a model named {clash!r}; choose another name."
            )

    payload: dict[str, Any] = {"version": 1, "models": {}}
    if os.path.isfile(models_file):
        try:
            existing = _load_mapping(models_file)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ValidationError(f"Could not read {models_file}: {exc}") from exc
        if not isinstance(existing, dict) or not isinstance(
            existing.get("models"), dict
        ):
            raise ValidationError(
                f"{models_file} is not a models file: it has no 'models' object."
            )
        payload = existing

    existing = {str(n).casefold(): str(n) for n in payload["models"]}.get(key)
    if existing is not None:
        raise ValidationError(
            f"{models_file} already defines {existing!r}; choose another name or "
            f"run 'unblend models unregister {existing}'."
        )
    # Also checked now, before an import spends time converting: the file
    # (and its sibling lock and staging files) must be writable.
    if os.path.exists(models_file) and not os.path.isfile(models_file):
        raise ValidationError(f"{models_file} is not a file.")
    if not name_encodable(str(models_file)):
        raise ValidationError(f"{models_file} can't be encoded as a file name here.")
    if os.path.islink(os.path.realpath(models_file)):
        # realpath stops at a symlink loop and returns the link itself;
        # register_entry refuses it too, but only after the conversion.
        raise ValidationError(f"{models_file} is a symlink loop.")
    # The staging copy is ".STEM.XXXXXXXX" + suffix (the name + 10) and the
    # lock is NAME + ".lock" (+ 5).
    if not name_fits(models_file.name, reserve=10) or not all(
        name_fits(part) for part in models_file.parts
    ):
        raise ValidationError(f"{models_file} has a name too long to write beside.")
    folder = models_file.parent
    while not os.path.lexists(folder) and folder != folder.parent:
        folder = folder.parent
    if not os.path.isdir(folder) or not os.access(folder, os.W_OK):
        raise ValidationError(f"Can't write {models_file}: {folder} isn't writable.")
    # The file itself needn't be writable: it's replaced by a rename.
    if os.path.isfile(models_file):
        # Checked now, before an import spends time converting weights, so a
        # file that is already broken isn't blamed on the new entry.
        try:
            ModelRepository(
                extra_models=[*_other_loaded_models_files(models_file), models_file]
            )
        except ModelLoadingError as exc:
            raise ValidationError(
                f"{models_file} doesn't load as it is; fix it first: {exc}"
            ) from exc
    return payload


def _register_entry(models_file: Path, name: str, entry: dict) -> None:
    """
    Add an entry to a user models file, creating it if needed.

    :param models_file: The user models file to write.
    :param name: Model name to register.
    :param entry: The registry entry.
    :raises ValidationError: See :func:`check_registrable`; also if the
        updated file fails the registry's own validation, in which case
        nothing is written.
    """
    from .repo import ModelRepository

    payload = check_registrable(models_file, name)
    payload["models"][name] = entry
    models_file.parent.mkdir(parents=True, exist_ok=True)
    staged = _staging_path(models_file)
    _write_staged(staged, _dump_mapping(payload, models_file), models_file)
    try:
        # Validate the new file in place of the old one, alongside the others
        # loaded by default, before it replaces anything.
        others = _other_loaded_models_files(models_file)
        try:
            ModelRepository(extra_models=[*others, staged]).list_models()[name]
        except ModelLoadingError as exc:
            message = str(exc).replace(str(staged), str(models_file))
            raise ValidationError(f"The new entry does not load: {message}") from exc
        if os.path.isfile(models_file):
            # The rewrite drops YAML comments; keep the previous file.
            backup = models_file.with_name(models_file.name + ".bak")
            backup.write_bytes(models_file.read_bytes())
        os.replace(staged, models_file)
    finally:
        staged.unlink(missing_ok=True)


def _other_loaded_models_files(models_file: Path) -> list[Path]:
    """
    The models files a default registry loads besides ``models_file``, to
    validate an updated ``models_file`` against.

    The implicit default file is left out when the registry skips it today
    (the current files, ``models_file`` included, load without it but not
    with it), so a broken one doesn't block edits elsewhere. A default file that loads
    today is kept, so an edit that would break it (unregistering a model one
    of its ensembles uses) is refused.

    :param models_file: The file being updated.
    :return: The other files, ``UNBLEND_EXTRA_MODELS`` entries first.
    """
    from .repo import ModelRepository, default_models_file, listed_extra_models_files

    target = Path(os.path.realpath(models_file))
    listed = listed_extra_models_files()
    # A listed file that doesn't exist yet (often this one) is skipped, as the
    # registry skips it.
    others = [
        path
        for path in listed
        if Path(os.path.realpath(path)) != target and os.path.exists(path)
    ]
    default = default_models_file()
    if (
        os.path.isfile(default)
        and Path(os.path.realpath(default)) != target
        and Path(os.path.realpath(default))
        not in {Path(os.path.realpath(path)) for path in listed}
    ):
        # Mirrors the registry: the default file is skipped only when the
        # rest load without it.
        current = [models_file] if os.path.isfile(models_file) else []
        try:
            ModelRepository(extra_models=[*others, *current, default])
        except ModelLoadingError:
            try:
                ModelRepository(extra_models=[*others, *current])
            except ModelLoadingError:
                pass
            else:
                return others
        others.append(default)
    return others


def _default_file_mode() -> int:
    """
    The mode a newly created file gets under the current umask.

    :return: ``0o666`` minus the umask.
    """
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _write_staged(staged: Path, text: str, models_file: Path) -> None:
    """
    Write an updated models file's staging copy with the original's mode.

    ``mkstemp`` creates 0600, which would otherwise replace a shared file's
    0644 and lock other users out of it.

    :param staged: The staging path.
    :param text: File contents.
    :param models_file: The file it will replace.
    """
    staged.write_text(text)
    mode = (
        models_file.stat().st_mode & 0o777
        if os.path.exists(models_file)
        else _default_file_mode()
    )
    os.chmod(staged, mode)


def _staging_path(models_file: Path) -> Path:
    """
    A fresh sibling path to write an updated models file to before it replaces
    the original (unique, so concurrent writers don't collide).

    :param models_file: The file that will be replaced.
    :return: An unused path in the same directory, with the same suffix.
    """
    models_file.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{models_file.stem}.",
        suffix=models_file.suffix,
        dir=models_file.parent,
    )
    os.close(fd)
    return Path(name)


def _unregister_entry(models_file: Path, name: str) -> dict:
    """
    Remove an entry from a user models file, keeping the old file as ``.bak``.

    :param models_file: The user models file holding the entry.
    :param name: Model name to remove.
    :return: The removed entry.
    :raises ValidationError: If the file can't be read or has no such entry.
    """
    try:
        payload = _load_mapping(models_file)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"Could not read {models_file}: {exc}") from exc
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict) or name not in models:
        raise ValidationError(f"{models_file} does not define {name!r}.")
    entry = models.pop(name)
    from .repo import ModelRepository

    others = _other_loaded_models_files(models_file)
    staged = _staging_path(models_file)
    try:
        _write_staged(staged, _dump_mapping(payload, models_file), models_file)
        # The registry without this entry must still load: another entry may
        # list it as a member.
        try:
            ModelRepository(extra_models=[*others, staged])
        except ModelLoadingError as exc:
            message = str(exc).replace(str(staged), str(models_file))
            try:
                ModelRepository(extra_models=[*others, models_file])
            except ModelLoadingError as before:
                # Not this removal's doing: the files were already broken, so
                # report what is wrong with them as they are.
                raise ValidationError(
                    f"Can't unregister {name!r}: the models files don't load "
                    f"even with it: {before}"
                ) from exc
            raise ValidationError(
                f"Can't unregister {name!r}; the remaining models don't load "
                f"without it: {message}"
            ) from exc
        backup = models_file.with_name(models_file.name + ".bak")
        backup.write_bytes(models_file.read_bytes())
        # An emptied file stays (with ``models: {}``): it may be listed in
        # UNBLEND_EXTRA_MODELS, which would warn about a missing file.
        os.replace(staged, models_file)
    finally:
        staged.unlink(missing_ok=True)
    return entry


def _models_file_lock(models_file: Path) -> FileLock:
    """
    Serialize read-modify-write updates of one models file across processes.

    :param models_file: The file being updated.
    :return: A lock on a sibling ``.lock`` file.
    """
    models_file.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(models_file.with_name(models_file.name + ".lock")))


def register_entry(models_file: Path, name: str, entry: dict) -> None:
    """
    Add an entry to a user models file, creating it if needed.

    :param models_file: The user models file to write.
    :param name: Model name to register.
    :param entry: The registry entry.
    :raises ValidationError: See :func:`check_registrable`; also if the
        updated file fails the registry's own validation, in which case
        nothing is written, or if the file or its lock can't be written.
    """
    # Through a symlink, update (and lock) the file it points at rather than
    # replacing the link with a plain file.
    models_file = Path(os.path.realpath(os.path.expanduser(models_file)))
    if os.path.islink(models_file):
        # realpath stops at a symlink loop and returns the link itself.
        raise ValidationError(f"{models_file} is a symlink loop.")
    try:
        with _models_file_lock(models_file):
            _register_entry(models_file, name, entry)
    except OSError as exc:
        raise ValidationError(f"Could not update {models_file}: {exc}") from exc


def unregister_entry(models_file: Path, name: str) -> dict:
    """
    Remove an entry from a user models file, keeping the old file as ``.bak``.

    :param models_file: The user models file holding the entry.
    :param name: Model name to remove.
    :return: The removed entry.
    :raises ValidationError: If the file can't be read or written, has no
        such entry, or the remaining models don't load without it.
    """
    models_file = Path(os.path.realpath(os.path.expanduser(models_file)))
    try:
        with _models_file_lock(models_file):
            return _unregister_entry(models_file, name)
    except OSError as exc:
        raise ValidationError(f"Could not update {models_file}: {exc}") from exc


def _missing_hint(missing: list[str], config_path: Path | None) -> str:
    """
    How to supply the fields an import is missing.

    :param missing: The missing field names.
    :param config_path: The training config given, if any.
    :return: The end of the error message.
    """
    others = [key for key in missing if key != "config"]
    if config_path is None:
        hint = "pass a training config with --config"
        if not others:
            return hint + "."
        return hint + (
            ", or give it with its own option."
            if len(others) == 1
            else ", or give them with their own options."
        )
    if not others:
        return "the training config has no model section."
    hint = "the training config doesn't give " + ("it" if len(others) == 1 else "them")
    return (
        hint
        + "; give "
        + ("it" if len(others) == 1 else "them")
        + (" with its own option." if len(others) == 1 else " with their own options.")
    )


def import_checkpoint(
    checkpoint: Path,
    artifact: Path,
    *,
    config_path: Path | None = None,
    architecture: str | None = None,
    sources: Iterable[str] | None = None,
    samplerate: int | None = None,
    segment_samples: int | None = None,
    license_label: str = "unknown",
    note: str | None = None,
) -> tuple[dict, dict[str, Any]]:
    """
    Repackage a checkpoint into a verified Safetensors artifact and an entry.

    :param checkpoint: The checkpoint to import.
    :param artifact: Where to write the Safetensors file.
    :param config_path: A training config to translate, if any.
    :param architecture: Explicit architecture, or ``None`` to infer.
    :param sources: Explicit stem names.
    :param samplerate: Explicit sample rate.
    :param segment_samples: Explicit training chunk length.
    :param license_label: License label for the entry.
    :param note: Provenance note for the entry.
    :return: ``(registry entry, summary)``.
    """
    state = read_tensors(checkpoint)

    fields: dict[str, Any] = {}
    if checkpoint.suffix.lower() == ".safetensors":
        fields.update(read_embedded_fields(checkpoint))
    if config_path is not None:
        fields.update(fields_from_config(read_config(config_path)))
    if architecture is not None:
        fields["architecture"] = architecture
    if sources is not None:
        fields["sources"] = [str(name) for name in sources]
    if samplerate is not None:
        fields["samplerate"] = samplerate
    seconds = fields.pop("segment_seconds", None)
    if segment_samples is not None:
        fields["segment_samples"] = segment_samples
    elif seconds is not None:
        fields["segment_samples"] = _seconds_to_samples(
            seconds, fields.get("samplerate")
        )

    # A given 0 or negative number is caught below as invalid, not "missing".
    missing = [
        key
        for key in ("sources", "samplerate", "segment_samples", "config")
        if key not in fields
        or fields[key] is None
        or (key in ("sources", "config") and not fields[key])
    ]
    if missing:
        raise ValidationError(
            f"Missing {', '.join(missing)}: " + _missing_hint(missing, config_path)
        )
    for key in ("samplerate", "segment_samples"):
        value = fields[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{key} must be a positive integer, got {value!r}.")
    # The registry's own stem-name rules, checked before converting: it would
    # refuse the entry afterwards, with the artifact already written.
    problem = stem_name_problem(fields["sources"])
    if problem is not None:
        raise ValidationError(f"The stem list {fields['sources']!r} {problem}.")

    resolved, model = resolve_architecture(
        state,
        fields["config"],
        sources=fields["sources"],
        samplerate=fields["samplerate"],
        segment_samples=fields["segment_samples"],
        architecture=fields.get("architecture"),
    )
    fields["architecture"] = resolved
    _check_runs(model, int(fields["segment_samples"]))
    fields["config"], dropped = constructor_config(resolved, fields["config"])
    if resolved == "htdemucs":
        # In the config too, so the artifact's embedded fields rebuild it.
        fields["config"] = _htdemucs_config(
            fields["config"],
            fields["sources"],
            fields["samplerate"],
            fields["segment_samples"],
        )
    del model

    write_artifact(state, artifact, fields)
    entry = build_entry(fields, artifact, license_label, note)
    digest = hashlib.sha256()
    with open(artifact, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    entry["checkpoint"]["sha256"] = digest.hexdigest()
    entry["checkpoint"]["size_bytes"] = artifact.stat().st_size
    summary = {
        "dropped_config_keys": dropped,
        "architecture": resolved,
        "tensors": len(state),
        "size_bytes": artifact.stat().st_size,
    }
    return entry, summary
