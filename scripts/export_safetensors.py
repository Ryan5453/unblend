#!/usr/bin/env python3
"""Convert verified model artifacts into pickle-free Safetensors weights."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert constructor metadata to JSON-compatible primitives."""
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported config value {value!r} ({type(value).__name__})")


def _extract_checkpoint_state(raw: Any) -> dict[str, torch.Tensor]:
    """Extract a tensor state dict from common plain/Lightning checkpoints."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Checkpoint must be a non-empty mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        nested = raw.get(key)
        if isinstance(nested, dict) and nested:
            raw = nested
            break
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("Every state-dict key must be a string")
    prefixes = ("model.", "module.", "net.")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        state[key] = value
    if not state:
        raise ValueError("Checkpoint contains no tensor state")
    return state


def _prepare_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Move tensors to contiguous CPU storage and break shared-storage aliases."""
    prepared: dict[str, torch.Tensor] = {}
    seen_storage: set[tuple[int, int]] = set()
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Safetensors state must map string keys to tensors")
        tensor = value.detach().cpu().contiguous()
        storage = tensor.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes())
        if identity in seen_storage:
            tensor = tensor.clone()
        seen_storage.add(identity)
        prepared[key] = tensor
    return prepared


def _load_source(
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Load one verified source and return its state plus optional constructor config."""
    if args.tensor_package:
        source = Path(args.tensor_package)
        if _sha256(source) != args.expected_sha256:
            raise ValueError("Tensor-package SHA-256 does not match --expected-sha256")
        package = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        if (
            not isinstance(package, dict)
            or package.get("format") != "unblend-htdemucs-v1"
        ):
            raise ValueError("Source is not an unblend HTDemucs tensor package")
        return dict(package["state"]), dict(package["config"])

    source = Path(args.checkpoint)
    if _sha256(source) != args.expected_sha256:
        raise ValueError("Checkpoint SHA-256 does not match --expected-sha256")
    raw = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    return _extract_checkpoint_state(raw), None


def main() -> None:
    """Parse arguments, convert one artifact, and print manifest metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tensor-package")
    source.add_argument("--checkpoint")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    args = parser.parse_args()

    state, config = _load_source(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(_prepare_state(state), args.output)
    if config is not None and args.config_output is not None:
        args.config_output.write_text(json.dumps(_json_value(config), indent=2) + "\n")

    print(
        json.dumps(
            {
                "path": str(args.output),
                "size_bytes": args.output.stat().st_size,
                "sha256": _sha256(args.output),
                "tensor_count": len(state),
                "element_count": sum(tensor.numel() for tensor in state.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
