# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Bring a checkpoint from elsewhere into Unblend's registry.

Community weights ship as a pickled ``.ckpt`` plus a YAML config; Unblend loads
tensor-only Safetensors described by a registry entry. Nothing about the
*weights* differs — every architecture's parameter names are pinned to its
reference implementation — so importing is repackaging, not conversion:

1. read the checkpoint's tensors, whatever container they arrived in,
2. translate the config into registry fields,
3. **build the model and strict-load the tensors into it**, so an entry is
   never written unless it demonstrably works,
4. write Safetensors (with the registry fields embedded in its header, so the
   file describes itself) and a registry entry.

Step 3 is the point: a mistranslated config or a mislabelled architecture fails
here, naming what did not fit, rather than at separation time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from . import backends, scnet  # noqa: F401  (registers the builders)
from .config_io import dump_mapping, load_mapping
from .exceptions import ValidationError
from .htdemucs import HTDemucs

#: Marker written into a Safetensors header alongside the registry fields, so a
#: file Unblend produced can be told apart from any other Safetensors.
EMBEDDED_FORMAT = "1"

#: Registry fields carried in the artifact header. ``config`` and ``sources``
#: are JSON, the rest are plain strings, because Safetensors metadata is a flat
#: ``str -> str`` map.
EMBEDDED_FIELDS = ("architecture", "sources", "samplerate", "segment_samples", "config")

#: Wrapper keys a training framework may nest the real state dict under.
STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "state")

#: Prefixes a trainer may have added to every parameter name. Stripped only
#: when *every* key carries the same one, so a model whose own modules happen
#: to start with these is untouched.
WRAPPER_PREFIXES = ("model.", "module.", "net.", "_orig_mod.")


def read_tensors(path: Path) -> dict[str, Tensor]:
    """
    Read a checkpoint's tensors, whatever container they arrived in.

    Safetensors is read directly. Anything else goes through
    ``torch.load(weights_only=True)``, which refuses to unpickle arbitrary
    classes — so a checkpoint that needs code execution to load is rejected
    here rather than run.

    :param path: Path to a ``.safetensors``, ``.ckpt``, ``.pt`` or ``.th`` file.
    :return: The state dict, unwrapped and un-prefixed.
    :raises ValidationError: If the file cannot be read, or holds no tensors.
    """
    if path.suffix == ".safetensors":
        return load_file(path, device="cpu")

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValidationError(
            f"Could not read {path} as a tensor-only checkpoint: {exc}\n"
            "If it needs weights_only=False it contains pickled objects, not "
            "just weights; inspect it before trusting it."
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
        loaded = load_mapping(path)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Could not read {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValidationError(f"{path} must contain a mapping.")
    return loaded


def fields_from_config(raw: dict) -> dict[str, Any]:
    """
    Translate a training config into registry fields.

    Recognises the layout ZFTurbo's Music-Source-Separation-Training uses,
    which is what most community RoFormer and SCNet weights ship with: the
    ``model:`` section *is* the constructor config, and the geometry lives
    under ``audio:`` and ``training:``. A config already in Unblend's own shape
    passes through.

    ``training.target_instrument`` marks a single-head model, whose second
    source Unblend synthesises as ``mixture - prediction``; the complement is
    named ``other``, or ``vocals`` for a model that predicts everything else.

    :param raw: The parsed config file.
    :return: Any of ``sources``, ``samplerate``, ``segment_samples`` and
        ``config`` the file determines.
    """
    fields: dict[str, Any] = {}

    model_section = raw.get("model")
    if isinstance(model_section, dict):
        fields["config"] = model_section
    elif "config" in raw and isinstance(raw["config"], dict):
        fields["config"] = raw["config"]

    audio = raw.get("audio") if isinstance(raw.get("audio"), dict) else {}
    training = raw.get("training") if isinstance(raw.get("training"), dict) else {}

    samplerate = audio.get("sample_rate", raw.get("samplerate"))
    if isinstance(samplerate, int):
        fields["samplerate"] = samplerate
    segment = audio.get("chunk_size", raw.get("segment_samples"))
    if isinstance(segment, int):
        fields["segment_samples"] = segment

    target = training.get("target_instrument")
    instruments = training.get("instruments", raw.get("sources"))
    if isinstance(target, str) and target:
        complement = "vocals" if target in {"other", "instrumental"} else "other"
        fields["sources"] = [target, complement]
    elif isinstance(instruments, list) and instruments:
        fields["sources"] = [str(name) for name in instruments]

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
    for key in ("sources", "config"):
        if key in metadata:
            try:
                fields[key] = json.loads(metadata[key])
            except json.JSONDecodeError:
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

    Every family is unmistakable from its keys. Within the SCNet family the
    masked variant is identifiable too, since it carries weights plain SCNet
    has no slot for. BS- and Mel-Band RoFormer share their key *names*, so they
    are told apart by the config, or — failing that — by trying both.

    :param state: The checkpoint's tensors.
    :param config: The constructor config, if known.
    :return: Candidate architecture names, most likely first; empty if the
        checkpoint resembles nothing Unblend implements.
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

    This is the import's proof: a config that disagrees with the weights, or a
    mislabelled architecture, fails here — with the mismatch named — instead of
    producing a registry entry that only breaks later.

    :param architecture: Architecture to build.
    :param config: Constructor kwargs.
    :param sources: Output stem names.
    :param samplerate: Sample rate the weights operate at.
    :param segment_samples: Training chunk length in samples.
    :param state: The checkpoint's tensors.
    :return: The constructed, weight-loaded model.
    :raises ValidationError: If it cannot be built or does not strict-load.
    """
    try:
        if architecture == "htdemucs":
            model = HTDemucs(**dict(config))
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
            f"Checkpoint does not load as {architecture}: {exc}"
        ) from exc


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

    An explicit name is verified rather than trusted. Otherwise the candidates
    its parameter names allow are tried in turn and the one that strict-loads
    wins.

    :param state: The checkpoint's tensors.
    :param config: Constructor kwargs.
    :param sources: Output stem names.
    :param samplerate: Sample rate the weights operate at.
    :param segment_samples: Training chunk length in samples.
    :param architecture: Explicit architecture, or ``None`` to infer.
    :return: ``(architecture, loaded model)``.
    :raises ValidationError: If nothing builds, with each candidate's failure.
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

    The header is covered by the file's own hash, so a config embedded here is
    verified along with the weights — which a JSON entry beside it is not.

    :param state: Tensors to write.
    :param path: Destination path.
    :param fields: Registry fields to embed.
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
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        str(path),
        metadata=metadata,
    )


def build_entry(
    fields: dict[str, Any], artifact: Path, license_label: str, note: str | None
) -> dict:
    """
    Assemble the registry entry for an imported checkpoint.

    :param fields: Architecture, sources, geometry and config.
    :param artifact: Path to the written Safetensors file.
    :param license_label: Free-form licence label for the entry.
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
        "checkpoint": {"format": "safetensors", "path": str(artifact)},
    }
    if fields["architecture"] == "htdemucs":
        # A Demucs config carries its own geometry and must agree with the
        # entry's stems; the other backends take theirs as arguments.
        entry["config"] = {**entry["config"], "sources": list(fields["sources"])}
    if note:
        entry["provenance"] = note
    return entry


def register_entry(models_file: Path, name: str, entry: dict) -> None:
    """
    Add an entry to a user models file, creating it if needed.

    :param models_file: The ``UNBLEND_EXTRA_MODELS`` file to write.
    :param name: Model name to register.
    :param entry: The registry entry.
    :raises ValidationError: If the file exists but is not a models file, or
        already defines this name.
    """
    payload: dict[str, Any] = {"version": 1, "models": {}}
    if models_file.is_file():
        try:
            existing = load_mapping(models_file)
        except (OSError, ValueError) as exc:
            raise ValidationError(f"Could not read {models_file}: {exc}") from exc
        if not isinstance(existing, dict) or not isinstance(
            existing.get("models"), dict
        ):
            raise ValidationError(
                f"{models_file} is not a models file: it has no 'models' object."
            )
        payload = existing

    if name in payload["models"]:
        raise ValidationError(
            f"{models_file} already defines {name!r}; choose another name or "
            "remove the existing entry."
        )

    payload["models"][name] = entry
    models_file.parent.mkdir(parents=True, exist_ok=True)
    models_file.write_text(dump_mapping(payload, models_file))


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

    Fields are taken from, in order of precedence: explicit arguments, the
    config file, and whatever the checkpoint already records about itself.

    :param checkpoint: The checkpoint to import.
    :param artifact: Where to write the Safetensors file.
    :param config_path: A training config to translate, if any.
    :param architecture: Explicit architecture, or ``None`` to infer.
    :param sources: Explicit stem names.
    :param samplerate: Explicit sample rate.
    :param segment_samples: Explicit training chunk length.
    :param license_label: Licence label for the entry.
    :param note: Provenance note for the entry.
    :return: ``(registry entry, summary)`` where the summary reports the
        architecture, tensor count and artifact size.
    :raises ValidationError: If a required field is missing, or the weights do
        not load as any architecture.
    """
    state = read_tensors(checkpoint)

    fields: dict[str, Any] = {}
    if checkpoint.suffix == ".safetensors":
        fields.update(read_embedded_fields(checkpoint))
    if config_path is not None:
        fields.update(fields_from_config(read_config(config_path)))
    if architecture is not None:
        fields["architecture"] = architecture
    if sources is not None:
        fields["sources"] = [str(name) for name in sources]
    if samplerate is not None:
        fields["samplerate"] = samplerate
    if segment_samples is not None:
        fields["segment_samples"] = segment_samples

    missing = [
        key
        for key in ("sources", "samplerate", "segment_samples", "config")
        if key not in fields or not fields[key]
    ]
    if missing:
        raise ValidationError(
            f"Missing {', '.join(missing)}. Pass a training config with "
            "--config, or supply them explicitly."
        )

    resolved, model = resolve_architecture(
        state,
        fields["config"],
        sources=fields["sources"],
        samplerate=fields["samplerate"],
        segment_samples=fields["segment_samples"],
        architecture=fields.get("architecture"),
    )
    fields["architecture"] = resolved
    del model

    write_artifact(state, artifact, fields)
    entry = build_entry(fields, artifact, license_label, note)
    summary = {
        "architecture": resolved,
        "tensors": len(state),
        "size_bytes": artifact.stat().st_size,
    }
    return entry, summary
