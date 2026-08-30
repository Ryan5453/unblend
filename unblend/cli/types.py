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


class ExportPrecision(str, Enum):
    native = "native"
    fp32 = "fp32"
    bf16 = "bf16"
    fp16 = "fp16"
    fp8_e5m2 = "fp8_e5m2"
    fp8_e4m3 = "fp8_e4m3"
