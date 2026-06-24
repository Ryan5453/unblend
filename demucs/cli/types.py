# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum


class DeviceType(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class ModelName(str, Enum):
    auto = "auto"
    htdemucs = "htdemucs"
    htdemucs_ft = "htdemucs_ft"
    htdemucs_6s = "htdemucs_6s"


class StemName(str, Enum):
    drums = "drums"
    bass = "bass"
    other = "other"
    vocals = "vocals"
    guitar = "guitar"  # Only provided by htdemucs_6s
    piano = "piano"  # Only provided by htdemucs_6s


class ClipMode(str, Enum):
    rescale = "rescale"
    clamp = "clamp"
    tanh = "tanh"
    none = "none"


class Precision(str, Enum):
    auto = "auto"
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"
