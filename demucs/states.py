# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import inspect
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

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


def load_model(path_or_package: dict | str | Path, strict: bool = False) -> torch.nn.Module:
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
            # verified — ModelRepository checksums every layer (SHA-256 against
            # the shipped metadata) before handing the path/bytes here.
            package = torch.load(path, "cpu", weights_only=False)
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                if key not in _DEPRECATED_PARAMS:
                    warnings.warn(
                        "Dropping nonexistent parameter " + key, stacklevel=2
                    )
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


def capture_init(init: Callable) -> Callable:
    """
    Decorator that captures the args and kwargs passed to __init__.

    :param init: The __init__ method to wrap
    :return: Wrapped __init__ that stores args/kwargs on the instance
    """

    @functools.wraps(init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Store the init args/kwargs on the instance, then call the wrapped init.

        :param args: Positional arguments forwarded to the wrapped ``__init__``
        :param kwargs: Keyword arguments forwarded to the wrapped ``__init__``
        """
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
