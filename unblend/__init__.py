# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from importlib.metadata import PackageNotFoundError, version

# Suppress PyTorch internal resize warnings from STFT/iSTFT
# These are benign and originate from PyTorch issue #134323
warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized since it had shape",
    category=UserWarning,
)

try:
    __version__ = version("unblend")
except PackageNotFoundError:
    # Running from a source tree without an installed distribution.
    __version__ = "0.0.0+unknown"

from .api import (
    SeparatedSources,
    Separator,
    default_device,
    default_dtype,
    get_version,
    select_model,
)
from .apply import Model, ModelEnsemble, apply_model, apply_model_multi
from .exceptions import (
    DemucsError,
    LoadAudioError,
    ModelLoadingError,
    UnblendError,
    ValidationError,
)
from .repo import ModelRepository

__all__ = [
    "__version__",
    "Separator",
    "SeparatedSources",
    "ModelRepository",
    "Model",
    "ModelEnsemble",
    "apply_model",
    "apply_model_multi",
    "default_device",
    "default_dtype",
    "get_version",
    "select_model",
    "UnblendError",
    "DemucsError",
    "LoadAudioError",
    "ModelLoadingError",
    "ValidationError",
]
