# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import inspect
import sys
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

from . import htdemucs as _htdemucs

_LEGACY_ALIAS_LOCK = threading.RLock()


@contextmanager
def _legacy_demucs_aliases() -> Iterator[None]:
    """
    Temporarily expose historical module names while loading a legacy pickle.

    Normal ``import unblend`` never modifies ``sys.modules['demucs']``. The
    aliases exist only inside this explicit compatibility boundary and are
    restored even when unpickling fails.

    :return: Context manager yielding while aliases are installed.
    """
    with _LEGACY_ALIAS_LOCK:
        missing = object()
        previous_demucs = sys.modules.get("demucs", missing)
        previous_htdemucs = sys.modules.get("demucs.htdemucs", missing)
        sys.modules["demucs"] = sys.modules[__package__]
        sys.modules["demucs.htdemucs"] = _htdemucs
        try:
            yield
        finally:
            if previous_demucs is missing:
                sys.modules.pop("demucs", None)
            else:
                sys.modules["demucs"] = previous_demucs
            if previous_htdemucs is missing:
                sys.modules.pop("demucs.htdemucs", None)
            else:
                sys.modules["demucs.htdemucs"] = previous_htdemucs


# Known deprecated parameters that are present in older model checkpoints
# but are no longer used in the current model classes. These are silently ignored.
_DEPRECATED_PARAMS = frozenset(
    {
        # Legacy Wiener filtering parameters
        "wiener_iters",
        "end_iters",
        "wiener_residual",
        # Removed sparse attention parameters (xformers APIs deprecated in 0.0.34)
        "t_sparse_self_attn",
        "t_sparse_cross_attn",
        "t_mask_type",
        "t_mask_random_seed",
        "t_sparse_attn_window",
        "t_global_window",
        "t_sparsity",
        "t_auto_sparsity",
    }
)


def load_tensor_package(package: dict) -> torch.nn.Module:
    """
    Build an HTDemucs from an unblend tensor package: a plain
    ``{"format": "unblend-htdemucs-v1", "config": {...}, "state": {...}}``
    dict holding only tensors and primitives, loadable with
    ``torch.load(weights_only=True)`` — no class pickling, no code execution
    on load, no dependence on historical module paths.

    :param package: The loaded tensor package.
    :return: The constructed model with weights loaded (strictly).
    """
    model = _htdemucs.HTDemucs(**package["config"])
    set_state(model, package["state"])
    return model


def load_model(
    path_or_package: dict | str | Path, strict: bool = False
) -> torch.nn.Module:
    """
    Load a model from a serialized dict or a file path.

    :param path_or_package: A dict (already loaded) or path to a serialized model file
    :param strict: If True, do not drop unknown constructor kwargs. Weights are
        always loaded strictly regardless of this flag.
    :return: The loaded model with state restored
    :raises ValueError: If path_or_package is not a dict, str, or Path
    """
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            # weights_only=False is required: the checkpoint pickles the model
            # class object plus its init args/kwargs (reconstructed below from
            # package["klass"]/["args"]/["kwargs"]), which weights_only=True
            # refuses to unpickle. This means loading can execute arbitrary
            # code, so callers must only pass files whose integrity has been
            # verified independently. Registered models never use this legacy
            # path; ModelRepository accepts Safetensors only.
            with _legacy_demucs_aliases():
                package = torch.load(path, "cpu", weights_only=False)
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    # Filtering deprecated keys must not mutate a caller-owned package.
    kwargs = dict(package["kwargs"])

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                if key not in _DEPRECATED_PARAMS:
                    warnings.warn("Dropping nonexistent parameter " + key, stacklevel=2)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    # Always load weights strictly: ``strict`` here only governs whether unknown
    # *constructor* kwargs are dropped (above). Once the module is built, its
    # parameter keys must match the checkpoint's ``state`` exactly — a missing or
    # renamed weight tensor is a genuine corruption/mismatch we want to surface,
    # not silently leave at random initialization.
    set_state(model, state)
    return model


def set_state(model: torch.nn.Module, state: dict, strict: bool = True) -> None:
    """
    Set the state dict on a model.

    :param model: The model to load state into
    :param state: The state dict to load
    :param strict: Forwarded to ``load_state_dict``; if False, missing/unexpected
        keys are tolerated rather than raising.
    """
    model.load_state_dict(state, strict=strict)
