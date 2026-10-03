# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import errno
import json
import logging
import math
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import onnx

from ._paths import name_encodable, name_fits
from .blocks import pad1d, spectro
from .exceptions import ModelLoadingError, ValidationError
from .htdemucs import HTDemucs
from .repo import ModelRepository, artifact_storage_dtype
from .roformer import (
    Attention,
    FeedForward,
    MaskEstimator,
    MelBandRoformer,
    RMSNorm,
    _RoformerBase,
)
from .scnet import FeatureConversion, SCNet
from .scnet import GroupNorm as SCNetGroupNorm


class HTDemucsONNXWrapper(nn.Module):
    """
    Wrapper that makes HTDemucs compatible with ONNX export.
    """

    def __init__(self, model: HTDemucs) -> None:
        """
        Initialize the ONNX wrapper.

        :param model: The HTDemucs model to wrap for ONNX export
        """
        super().__init__()
        self.model = model
        self.sources = model.sources
        self.samplerate = model.samplerate
        self.audio_channels = model.audio_channels
        self.nfft = model.nfft
        self.hop_length = model.hop_length

    def forward(
        self, spec_real: torch.Tensor, spec_imag: torch.Tensor, mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.

        :param spec_real: Real part of spectrogram [B, C, Fq, T]
        :param spec_imag: Imaginary part of spectrogram [B, C, Fq, T]
        :param mix: Raw audio waveform [B, C, samples]
        :return: Tuple of (out_spec_real, out_spec_imag, out_wave) separated spectrograms and waveforms
        """
        B, C, Fq, T = spec_real.shape
        samples = mix.shape[-1]

        x = torch.stack([spec_real, spec_imag], dim=2).reshape(B, C * 2, Fq, T)

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        meant = mix.mean(dim=(1, 2), keepdim=True)
        stdt = mix.std(dim=(1, 2), keepdim=True)
        xt = (mix - meant) / (1e-5 + stdt)

        x, xt = self.model.forward_core(x, xt)

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        out_spec_real = x[:, :, 0::2, :, :]
        out_spec_imag = x[:, :, 1::2, :, :]

        xt = xt.view(B, S, -1, samples)
        xt = xt * stdt[:, None] + meant[:, None]

        return out_spec_real, out_spec_imag, xt


class RoformerONNXWrapper(nn.Module):
    """
    Wrapper that makes RoFormer compatible with ONNX.
    """

    def __init__(
        self,
        model: _RoformerBase,
        *,
        attention_query_chunk_size: int = 64,
        attention_head_chunk_size: int = 4,
        feedforward_hidden_chunk_size: int = 384,
    ) -> None:
        """
        Wrap a RoFormer and set its ONNX chunk sizes.

        :param model: The RoFormer model to wrap.
        :param attention_query_chunk_size: Max query rows per attention chunk.
        :param attention_head_chunk_size: Max heads projected at once.
        :param feedforward_hidden_chunk_size: Max expanded MLP features at once.
        :raises ValueError: If a chunk size is not positive.
        """
        super().__init__()
        self.model = model
        self.sources = model.sources
        self.samplerate = model.samplerate
        self.audio_channels = model.audio_channels
        self.num_stems = model.num_stems

        if attention_query_chunk_size <= 0:
            raise ValueError(
                "attention_query_chunk_size must be positive, got "
                f"{attention_query_chunk_size}"
            )
        if attention_head_chunk_size <= 0:
            raise ValueError(
                "attention_head_chunk_size must be positive, got "
                f"{attention_head_chunk_size}"
            )
        if feedforward_hidden_chunk_size <= 0:
            raise ValueError(
                "feedforward_hidden_chunk_size must be positive, got "
                f"{feedforward_hidden_chunk_size}"
            )
        for module in model.modules():
            if isinstance(module, Attention):
                module.onnx_query_chunk_size = attention_query_chunk_size
                module.onnx_head_chunk_size = attention_head_chunk_size
            elif isinstance(module, FeedForward):
                module.onnx_hidden_chunk_size = feedforward_hidden_chunk_size
            elif isinstance(module, MaskEstimator):
                module.onnx_safe_glu = True
            elif isinstance(module, RMSNorm):
                module.onnx_safe = True

        if isinstance(model, MelBandRoformer):
            n_selected = int(model.freq_indices.numel())
            denom = model.num_bands_per_freq.repeat_interleave(
                model.audio_channels
            ).clamp(min=1e-8)
            averaging = torch.zeros(int(denom.numel()), n_selected)
            averaging[model.freq_indices, torch.arange(n_selected)] = 1.0
            averaging = averaging / denom[:, None]
            self.register_buffer("mel_averaging_matrix", averaging, persistent=False)
        else:
            self.mel_averaging_matrix = None

    def forward(
        self, spec_real: torch.Tensor, spec_imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.

        :param spec_real: Real part of the mixture STFT ``[B, C, F, T]``.
        :param spec_imag: Imaginary part of the mixture STFT ``[B, C, F, T]``.
        :return: ``(out_spec_real, out_spec_imag)`` masked per-stem
            spectrograms, each ``[B, num_stems, C, F, T]``.
        """
        m = self.model
        B, C, F, T = spec_real.shape

        st = torch.stack([spec_real, spec_imag], dim=-1)
        st = st.permute(0, 2, 1, 3, 4).reshape(B, F * C, T, 2)

        if self.mel_averaging_matrix is not None:
            x = st.index_select(1, m.freq_indices)
        else:
            x = st
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)

        x = m.band_split(x)
        x = m._run_transformers(x)
        if not isinstance(m, MelBandRoformer):
            x = m.final_norm(x)

        masks = torch.stack([head(x) for head in m.mask_estimators], dim=1)
        masks = masks.view(B, self.num_stems, T, -1, 2).permute(0, 1, 3, 2, 4)

        mask_real = masks[..., 0]
        mask_imag = masks[..., 1]
        if self.mel_averaging_matrix is not None:
            mask_real = torch.matmul(self.mel_averaging_matrix, mask_real)
            mask_imag = torch.matmul(self.mel_averaging_matrix, mask_imag)

        spec_r = st[..., 0].unsqueeze(1)
        spec_i = st[..., 1].unsqueeze(1)
        out_r = spec_r * mask_real - spec_i * mask_imag
        out_i = spec_r * mask_imag + spec_i * mask_real

        out_r = out_r.view(B, self.num_stems, F, C, T).permute(0, 1, 3, 2, 4)
        out_i = out_i.view(B, self.num_stems, F, C, T).permute(0, 1, 3, 2, 4)

        if m.zero_dc:
            zeros = torch.zeros_like(out_r[..., :1, :])
            out_r = torch.cat([zeros, out_r[..., 1:, :]], dim=-2)
            out_i = torch.cat([zeros, out_i[..., 1:, :]], dim=-2)
        return out_r, out_i


class SCNetONNXWrapper(nn.Module):
    """
    Wrapper that makes SCNet exportable.
    """

    def __init__(self, model: "SCNet") -> None:
        """
        Initialize the ONNX wrapper.

        :param model: The SCNet model to wrap for export.
        """
        super().__init__()
        self.model = model
        for module in model.modules():
            if isinstance(module, (FeatureConversion, SCNetGroupNorm)):
                module.onnx_safe = True

    def forward(
        self, spec_real: torch.Tensor, spec_imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run everything between the transforms.

        :param spec_real: Real part of the mixture STFT ``[B, C, F, T]``.
        :param spec_imag: Imaginary part of the mixture STFT ``[B, C, F, T]``.
        :return: ``(out_real, out_imag)`` per-stem spectrograms, each
            ``[B, num_stems, C, F, T]``.
        """
        model = self.model
        batch, channels, freq, frames = spec_real.shape

        packed = torch.stack([spec_real, spec_imag], dim=2).reshape(
            batch, channels * 2, freq, frames
        )

        stems = len(model.sources)
        n = model.dims[0]

        if hasattr(model, "mask_layer"):
            pos_f = model.pos_embed_f[:, :, :freq, :]

            mixture = packed.repeat(1, stems, 1, 1)
            mask = model.mask_layer(model.forward_core(packed + pos_f.float()))

            pairs = (batch * stems * channels, 2, freq, frames)
            mixture = mixture.view(batch, n, -1, freq, frames).reshape(*pairs)
            mask = mask.view(batch, n, -1, freq, frames).reshape(*pairs)

            real = mixture[:, 0] * mask[:, 0] - mixture[:, 1] * mask[:, 1]
            imag = mixture[:, 0] * mask[:, 1] + mixture[:, 1] * mask[:, 0]
        else:
            decoded = model.forward_core(packed)
            decoded = decoded.view(batch, n, -1, freq, frames)
            decoded = decoded.reshape(-1, 2, freq, frames)
            real, imag = decoded[:, 0], decoded[:, 1]

        out_real = real.reshape(batch, stems, channels, freq, frames)
        out_imag = imag.reshape(batch, stems, channels, freq, frames)
        return out_real.contiguous(), out_imag.contiguous()


def compute_scnet_stft_for_export(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    normalized: bool,
    window: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute STFT for SCNet export.

    :param audio: Input audio ``[B, C, samples]``.
    :param n_fft: FFT size.
    :param hop_length: Hop length.
    :param win_length: Window length.
    :param normalized: Whether the STFT is normalised.
    :param window: ``"none"`` or ``"hann"``, from export metadata.
    :return: ``(real, imag)`` spectrograms ``[B, C, F, T]``.
    :raises ValueError: If ``window`` is not recognised.
    """
    if window not in ("none", "hann"):
        raise ValueError(f"unknown STFT window {window!r}; expected none or hann")
    batch, channels, samples = audio.shape
    z = torch.stft(
        audio.reshape(batch * channels, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=(
            torch.hann_window(n_fft, periodic=True, device=audio.device)
            if window == "hann"
            # Explicit ones: what torch uses for None, without its warning.
            else torch.ones(win_length, device=audio.device)
        ),
        center=True,
        normalized=normalized,
        return_complex=True,
    )
    z = z.view(batch, channels, z.shape[-2], z.shape[-1])
    return z.real.contiguous(), z.imag.contiguous()


def compute_roformer_stft_for_export(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    normalized: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute STFT for RoFormer export.

    :param audio: Input audio ``[B, C, samples]``.
    :param n_fft: FFT size.
    :param hop_length: Hop length.
    :param win_length: Window length.
    :param normalized: Whether the STFT is normalised.
    :return: ``(real, imag)`` spectrograms ``[B, C, F, T]``.
    """
    B, C, samples = audio.shape
    window = torch.hann_window(win_length, device=audio.device)
    z = torch.stft(
        audio.reshape(B * C, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=normalized,
        return_complex=True,
    )
    z = z.view(B, C, z.shape[-2], z.shape[-1])
    return z.real.contiguous(), z.imag.contiguous()


def compute_stft_for_export(
    audio: torch.Tensor, nfft: int, hop_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute STFT for model input, matching HTDemucs preprocessing.

    :param audio: Input audio [B, C, samples]
    :param nfft: FFT size
    :param hop_length: Hop length
    :return: Tuple of (real, imag) spectrograms [B, C, Fq, T]
    """

    le = int(math.ceil(audio.shape[-1] / hop_length))
    pad = hop_length // 2 * 3

    padded = pad1d(
        audio, (pad, pad + le * hop_length - audio.shape[-1]), mode="reflect"
    )

    z = spectro(padded, nfft, hop_length)

    z = z[..., :-1, :]

    if z.shape[-1] != le + 4:
        raise RuntimeError(
            f"STFT frame count {z.shape[-1]} does not match expected {le + 4}"
        )
    z = z[..., 2 : 2 + le]

    real = z.real
    imag = z.imag

    return real, imag


# Weight storage precisions the exporter can write. These labels are the
# ``--precision`` values, the ONNX metadata spelling, and the export filename
# suffix. "native" resolves to whatever the checkpoint's header declares.
_STORAGE_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e4m3": torch.float8_e4m3fn,
}
_STORAGE_LABELS: dict[torch.dtype, str] = {
    dt: name for name, dt in _STORAGE_DTYPES.items()
}

_EXPORT_PRECISIONS = ("native", *_STORAGE_DTYPES)

# Families whose fp16 export converts compute as well as storage. Everything
# else gets the weight-only rewrite, keeping arithmetic in fp32. Only fp16 has
# a mixed-precision converter, so narrower storage falls back to weight-only.
_MIXED_PRECISION_FAMILIES = frozenset({"roformer"})


def _validate_export_precision(precision: str) -> None:
    """
    Reject an unknown ``precision`` before any download happens.

    :param precision: Caller's choice.
    :raises ValidationError: If it is not one of ``_EXPORT_PRECISIONS``.
    """
    if precision not in _EXPORT_PRECISIONS:
        raise ValidationError(
            f"Invalid precision {precision!r}. Choose one of "
            f"{', '.join(_EXPORT_PRECISIONS)}."
        )


#: What ``_atomic_onnx_path``'s staging name adds: "." + name + ".XXXXXXXX.tmp.onnx".
_STAGING_NAME_EXTRA = 19


def _check_output_path(output_path: str, what: str = "Output path") -> None:
    """
    Refuse an export path that can't name a file.

    :param output_path: The path to write, as given (``~`` is not expanded).
    :param what: How to name the path in the error.
    :raises ValidationError: If it is empty, can't be a file name (NUL,
        unencodable, too long), ends in a separator or in ``.`` or ``..``, is
        an existing folder, or lies under something that isn't a folder.
    """
    text = str(output_path)
    if "\0" in text:
        raise ValidationError(f"{what} {text!r} contains a NUL character.")
    if not name_encodable(text):
        raise ValidationError(f"{what} {text!r} can't be encoded as a file name.")
    if (
        not text.strip()
        or text.endswith(("/", os.sep))
        # "nope/.." isn't a folder while nope is missing, but can't be written.
        or os.path.basename(text) in {".", ".."}
        # os.path.isdir, not Path.is_dir(): the latter raises on a name the
        # OS refuses as too long.
        or os.path.isdir(text)
    ):
        raise ValidationError(f"{what} {text!r} doesn't name a file.")
    # The staging copy adds 19 to the file name (see _atomic_onnx_path).
    if not name_fits(os.path.basename(text), reserve=_STAGING_NAME_EXTRA):
        raise ValidationError(f"{what} {text!r} has a file name too long to write.")
    if not all(name_fits(part) for part in Path(text).parts):
        raise ValidationError(f"{what} {text!r} has a folder name too long to write.")
    # The nearest existing ancestor must be a folder ("afile/x.onnx" fails
    # otherwise, but only after the model has loaded).
    parent = Path(text).parent
    while not os.path.lexists(parent) and parent != parent.parent:
        parent = parent.parent
    if os.path.lexists(parent) and not os.path.isdir(parent):
        raise ValidationError(
            f"{what} {text!r} is inside {str(parent)!r}, not a folder."
        )


def default_output_path(
    model_name: str, storage: torch.dtype, static_batch: bool = False
) -> str:
    """
    The file name an export gets when no output path is given.

    :param model_name: Registry model name.
    :param storage: Resolved weight storage dtype.
    :param static_batch: Whether the batch axis is fixed.
    :return: ``{model}_{precision}[_static].onnx``.
    """
    static_suffix = "_static" if static_batch else ""
    return f"{model_name}_{_STORAGE_LABELS[storage]}{static_suffix}.onnx"


def _resolve_export_precision(precision: str, model_info: dict) -> torch.dtype:
    """
    Resolve a validated ``precision`` choice to a weight storage dtype.

    ``"native"`` is the dtype the checkpoint declares in its Safetensors
    header: fp16 for HTDemucs, because Demucs' ``serialize_model`` rounded
    those weights at release and loading merely widens them back, and fp32 for
    RoFormer and SCNet. A checkpoint saved at fp8 exports at fp8.

    :param precision: One of ``_EXPORT_PRECISIONS``.
    :param model_info: Registry entry, read only for ``"native"``.
    :return: Dtype to store weights at; fp32 means no conversion.
    """
    if precision != "native":
        return _STORAGE_DTYPES[precision]
    declared = artifact_storage_dtype(model_info["checkpoint"])
    if declared is None or declared not in _STORAGE_LABELS:
        # Unreadable header, or a width ONNX cannot express: fp32 is the only
        # choice that cannot silently lose anything.
        return torch.float32
    return declared


# Storage dtype -> (numpy dtype factory, ONNX elem type, minimum opset). The
# fp8 types only exist from opset 19, so those exports are traced at 19.
def _onnx_storage_spec(dtype: torch.dtype) -> tuple[object, int, int]:
    """
    Resolve a torch storage dtype to its ONNX representation.

    :param dtype: Reduced-precision storage dtype.
    :return: ``(numpy dtype, TensorProto elem type, minimum opset)``.
    :raises ValueError: If ONNX has no representation for the dtype.
    """
    import numpy as np
    from onnx import TensorProto

    if dtype is torch.float16:
        return np.float16, TensorProto.FLOAT16, 1
    try:
        import ml_dtypes
    except ImportError:
        raise ImportError(
            "ml_dtypes is required to export weights narrower than fp16. "
            "Install unblend with the 'onnx' extra."
        ) from None
    table: dict[torch.dtype, tuple[object, int, int]] = {
        torch.bfloat16: (ml_dtypes.bfloat16, TensorProto.BFLOAT16, 1),
        torch.float8_e4m3fn: (ml_dtypes.float8_e4m3fn, TensorProto.FLOAT8E4M3FN, 19),
        torch.float8_e5m2: (ml_dtypes.float8_e5m2, TensorProto.FLOAT8E5M2, 19),
    }
    if dtype not in table:
        raise ValueError(f"No ONNX storage representation for {dtype}.")
    return table[dtype]


def _trace_opset(requested: int, family_min: int, storage: torch.dtype) -> int:
    """
    The opset to trace at: the request, raised to what the family and the
    weight storage type need.

    :param requested: Caller's opset.
    :param family_min: Minimum opset the architecture's export needs.
    :param storage: Weight storage dtype.
    :return: Opset to pass to the tracer.
    """
    storage_min = 1 if storage is torch.float32 else _onnx_storage_spec(storage)[2]
    return max(requested, family_min, storage_min)


def _parameter_fingerprints(model: nn.Module) -> set[tuple]:
    """
    Fingerprints of a model's learned parameters (and their transposes, which
    the exporters store for MatMul), for telling weights apart from constants
    the exporter folded into initializers.

    :param model: The exported model.
    :return: ``(shape, digest)`` pairs.
    """
    import hashlib

    fingerprints = set()
    for param in model.parameters():
        array = param.detach().float().cpu().numpy()
        views = [array, array.T] if array.ndim == 2 else [array]
        for view in views:
            view = view.copy(order="C")
            digest = hashlib.sha1(view.tobytes()).hexdigest()
            fingerprints.add((view.shape, digest))
    return fingerprints


def _convert_weight_storage(
    onnx_model: "onnx.ModelProto", dtype: torch.dtype, model: nn.Module
) -> None:
    """
    Rewrite ONNX weight initializers at a narrower storage dtype.

    Arithmetic is untouched: each converted initializer is followed by a
    ``Cast`` back to fp32 (and, for fp8, a ``Mul`` by its per-tensor scale),
    so only the stored bytes shrink. Only the model's learned parameters are
    narrowed: constants the exporter folded into initializers (SCNet's DFT
    basis, Mel-RoFormer's band-averaging matrix) stay fp32. This is the
    weight-only path used by every family except RoFormer at fp16.

    :param onnx_model: Loaded ONNX model; modified in place.
    :param dtype: Storage dtype for the weights.
    :param model: The exported model, whose parameters identify the weights.
    """
    import hashlib

    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    np_dtype, elem_type, min_opset = _onnx_storage_spec(dtype)
    suffix = "_" + _STORAGE_LABELS[dtype]
    fp8_max = (
        float(torch.finfo(dtype).max)
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        else None
    )

    weight_op_inputs = {
        "Conv": (1, 2),
        "ConvTranspose": (1, 2),
        "MatMul": (0, 1),
        "Gemm": (0, 1, 2),
        "LSTM": (1, 2, 3),
        "GRU": (1, 2, 3),
        "RNN": (1, 2, 3),
    }

    rearranging_ops = {
        "Reshape",
        "Concat",
        "Transpose",
        "Squeeze",
        "Unsqueeze",
        "Identity",
        "Slice",
        "Split",
    }
    initializer_names = {init.name for init in onnx_model.graph.initializer}
    producer = {
        out: node for node in onnx_model.graph.node for out in node.output if out
    }

    def source_initializers(name: str, seen: set[str]) -> set[str]:
        """Initializers feeding ``name`` through rearranging ops only.

        :param name: Graph value to trace back from.
        :param seen: Names already visited, guarding against cycles.
        :return: Names of initializers that ultimately supply ``name``.
        """
        if name in initializer_names:
            return {name}
        node = producer.get(name)
        if node is None or node.op_type not in rearranging_ops or name in seen:
            return set()
        seen.add(name)
        found: set[str] = set()
        for value in node.input:
            if value:
                found |= source_initializers(value, seen)
        return found

    weight_init_names: set[str] = set()
    for node in onnx_model.graph.node:
        for idx in weight_op_inputs.get(node.op_type, ()):
            if idx < len(node.input) and node.input[idx]:
                weight_init_names |= source_initializers(node.input[idx], set())

    existing_outputs = {n.output[0] for n in onnx_model.graph.node if n.output}
    existing_inputs = {i.name for i in onnx_model.graph.input}

    learned = _parameter_fingerprints(model)

    def is_learned(array: "np.ndarray") -> bool:
        """
        Whether an initializer holds one of the model's parameters.

        :param array: The initializer's values.
        :return: True for a learned weight, False for a folded constant.
        """
        array = np.ascontiguousarray(array, dtype=np.float32)
        return (array.shape, hashlib.sha1(array.tobytes()).hexdigest()) in learned

    new_inits = []
    new_cast_nodes = []
    for init in onnx_model.graph.initializer:
        if (
            init.name in weight_init_names
            and init.data_type == TensorProto.FLOAT
            and init.name not in existing_outputs
            and init.name not in existing_inputs
            and is_learned(numpy_helper.to_array(init))
        ):
            weights = numpy_helper.to_array(init)
            scale = None
            if fp8_max is not None:
                # fp8 has so few exponent steps that unscaled small weights
                # lose most of their precision, so each tensor is scaled to
                # fill the format's range and multiplied back after the Cast
                # (about 4 dB closer to fp32 output for e4m3 on HTDemucs).
                peak = float(np.abs(weights).max()) if weights.size else 0.0
                if peak > 0:
                    scale = np.float32(peak / fp8_max)
                    weights = weights / scale
            arr = weights.astype(np_dtype)
            stored_name = init.name + suffix
            stored = numpy_helper.from_array(arr, name=stored_name)
            stored.data_type = elem_type
            new_inits.append(stored)
            cast_output = init.name if scale is None else init.name + "_unscaled"
            new_cast_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[stored_name],
                    outputs=[cast_output],
                    to=TensorProto.FLOAT,
                    name=init.name + "_cast_to_fp32",
                )
            )
            if scale is not None:
                new_inits.append(
                    numpy_helper.from_array(
                        np.array(scale, dtype=np.float32), name=init.name + "_scale"
                    )
                )
                new_cast_nodes.append(
                    helper.make_node(
                        "Mul",
                        inputs=[cast_output, init.name + "_scale"],
                        outputs=[init.name],
                        name=init.name + "_rescale",
                    )
                )
        else:
            new_inits.append(init)

    if not new_cast_nodes:
        raise RuntimeError(
            f"{_STORAGE_LABELS[dtype]} export requested but no fp32 weight "
            "initializers were converted — the exporter's op/initializer "
            "layout likely changed. Refusing to write a mislabeled model."
        )

    onnx_model.graph.ClearField("initializer")
    onnx_model.graph.initializer.extend(new_inits)

    original_nodes = list(onnx_model.graph.node)
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_cast_nodes + original_nodes)

    if min_opset > 1:
        for opset in onnx_model.opset_import:
            if opset.domain in ("", "ai.onnx") and opset.version < min_opset:
                # Raising the number after tracing would leave ops in their
                # older signatures; export_to_onnx traces at min_opset instead.
                raise RuntimeError(
                    f"{dtype} storage needs a graph traced at opset "
                    f">= {min_opset}, got {opset.version}."
                )
        # fp8 tensor types require IR version 9 or newer.
        onnx_model.ir_version = max(onnx_model.ir_version, 9)


def _convert_roformer_to_fp16(onnx_model: "onnx.ModelProto") -> None:
    """
    Convert RoFormer graph to mixed precision.

    :param onnx_model: Loaded ONNX model; modified in place.
    """
    from onnx import TensorProto

    try:
        from onnxconverter_common.float16 import convert_float_to_float16
    except ImportError:
        raise ImportError(
            "onnxconverter-common is required for RoFormer fp16 export. "
            "Install unblend with the 'onnx' extra."
        ) from None

    # Pow squares RMSNorm's input: in fp16 anything above 256 overflows to inf
    # and zeroes that token, and real activations reach ~340.
    blocked_ops = {
        "Clip",
        "Cos",
        "Pow",
        "Reciprocal",
        "ReduceMean",
        "Sin",
        "Softmax",
        "Sqrt",
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"onnxconverter_common\.float16",
        )
        converted = convert_float_to_float16(
            onnx_model,
            keep_io_types=True,
            op_block_list=sorted(blocked_ops),
        )
    onnx_model.CopyFrom(converted)

    if not any(
        init.data_type == TensorProto.FLOAT16 for init in onnx_model.graph.initializer
    ):
        raise RuntimeError(
            "RoFormer fp16 export produced no float16 initializers; refusing "
            "to write a model mislabeled as fp16."
        )


def _materialize_nonlast_broadcast_muls(onnx_model: "onnx.ModelProto") -> int:
    """
    Replicate size-1 Mul operands for WebGPU.

    :param onnx_model: Loaded ONNX model; modified in place.
    :return: Number of Mul nodes rewritten.
    """
    from onnx import helper, shape_inference

    inferred = shape_inference.infer_shapes(
        onnx_model, strict_mode=False, data_prop=True
    )
    dims: dict[str, list[int | None]] = {}
    for collection in (
        inferred.graph.input,
        inferred.graph.output,
        inferred.graph.value_info,
    ):
        for value in collection:
            shape = value.type.tensor_type.shape
            dims[value.name] = [
                d.dim_value if d.HasField("dim_value") else None for d in shape.dim
            ]

    def plan(a: str, b: str) -> tuple[str, list[tuple[int, int]]] | None:
        """
        Decide how to fix one ``Mul``, or return ``None`` to leave it alone.

        :param a: Name of the first ``Mul`` operand.
        :param b: Name of the second ``Mul`` operand.
        :return: ``(small_operand, [(axis, copies), ...])``, or ``None``.
        """
        da, db = dims.get(a), dims.get(b)
        if not da or not db or len(da) != len(db):
            return None
        rank = len(da)
        for k in range(rank):
            if da[k] is None or db[k] is None or da[k] == db[k]:
                continue
            if 1 not in (da[k], db[k]):
                return None
        a_small = all(
            da[k] is None or db[k] is None or da[k] <= db[k] for k in range(rank)
        )
        b_small = all(
            da[k] is None or db[k] is None or db[k] <= da[k] for k in range(rank)
        )
        if a_small and not b_small:
            small, sd, fd = a, da, db
        elif b_small and not a_small:
            small, sd, fd = b, db, da
        else:
            return None
        axes = [
            (k, fd[k])
            for k in range(rank - 1)
            if sd[k] == 1 and isinstance(fd[k], int) and fd[k] > 1
        ]
        return (small, axes) if axes else None

    new_nodes = []
    rewritten = 0
    for node in onnx_model.graph.node:
        if node.op_type == "Mul" and len(node.input) == 2:
            fix = plan(node.input[0], node.input[1])
            if fix is not None:
                small, axes = fix
                current = small
                for axis, copies in axes:
                    out = f"unblend_wgpu_bcast_{rewritten}_ax{axis}"
                    new_nodes.append(
                        helper.make_node(
                            "Concat",
                            [current] * copies,
                            [out],
                            axis=axis,
                            name=out,
                        )
                    )
                    current = out
                for i, name in enumerate(node.input):
                    if name == small:
                        node.input[i] = current
                        break
                rewritten += 1
        new_nodes.append(node)

    if rewritten:
        onnx_model.graph.ClearField("node")
        onnx_model.graph.node.extend(new_nodes)
    return rewritten


def _materialize_matmul_rank_mismatch(onnx_model: "onnx.ModelProto") -> int:
    """
    Pad 2-D MatMul operand for WebGPU.

    :param onnx_model: Loaded ONNX model; modified in place.
    :return: Number of MatMul nodes rewritten.
    """
    import numpy as np
    from onnx import helper, numpy_helper, shape_inference

    inferred = shape_inference.infer_shapes(
        onnx_model, strict_mode=False, data_prop=True
    )
    dims: dict[str, int] = {}
    for collection in (
        inferred.graph.input,
        inferred.graph.output,
        inferred.graph.value_info,
    ):
        for value in collection:
            dims[value.name] = len(value.type.tensor_type.shape.dim)
    for init in onnx_model.graph.initializer:
        dims[init.name] = len(init.dims)

    new_nodes = []
    new_inits = []
    rewritten = 0
    for node in onnx_model.graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            left, right = node.input
            left_rank, right_rank = dims.get(left), dims.get(right)
            if left_rank == 2 and right_rank is not None and right_rank > 2:
                extra = right_rank - 2
                axes_name = f"unblend_wgpu_matmul_rank_axes_{rewritten}"
                out_name = f"unblend_wgpu_matmul_rank_unsqueeze_{rewritten}"
                new_inits.append(
                    numpy_helper.from_array(
                        np.arange(extra, dtype=np.int64), name=axes_name
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "Unsqueeze", [left, axes_name], [out_name], name=out_name
                    )
                )
                node.input[0] = out_name
                rewritten += 1
        new_nodes.append(node)

    if rewritten:
        onnx_model.graph.ClearField("node")
        onnx_model.graph.node.extend(new_nodes)
        onnx_model.graph.initializer.extend(new_inits)
    return rewritten


@contextmanager
def _atomic_onnx_path(output_path: str) -> Iterator[str]:
    """
    Yield a sibling staging path and atomically publish it on success.

    :param output_path: Caller-requested final ONNX path.
    :return: Context manager yielding a temporary sibling filename.
    """
    destination = Path(output_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp collapses ".." as text; stage in the folder the kernel will
        # resolve the destination to (through any symlink), so the final
        # rename stays within one folder.
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp.onnx",
            dir=os.path.realpath(destination.parent),
        )
    except OSError as exc:
        if exc.errno != errno.EILSEQ:
            raise
        # APFS refuses some valid UTF-8 (unassigned code points); no static
        # check matches the OS's Unicode tables, so name the user's path here.
        raise ValidationError(
            f"Output path {output_path!r} has a character this filesystem "
            "can't store in a file name."
        ) from None
    os.close(fd)
    staging = Path(raw_path)

    def sidecars() -> list[Path]:
        """
        Find external-data paths private to this random staging prefix.

        :return: Sibling sidecar files/directories created by the exporter.
        """

        return [
            candidate
            for candidate in staging.parent.iterdir()
            if candidate != staging and candidate.name.startswith(staging.stem)
        ]

    try:
        yield str(staging)
        external_files = sidecars()
        if external_files:
            names = ", ".join(path.name for path in external_files)
            raise RuntimeError(
                "External-data ONNX exports are not supported by this "
                f"single-file publisher; exporter created: {names}"
            )

        with open(staging, "rb+") as file:
            file.flush()
            os.fsync(file.fileno())

        # mkstemp creates the file 0600; publish it with normal permissions.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(staging, 0o666 & ~umask)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
        for candidate in sidecars():
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                candidate.unlink(missing_ok=True)


def _strip_exporter_metadata(onnx_model: "onnx.ModelProto") -> None:
    """
    Remove the dynamo exporter's debug metadata: per-node stack traces, FX
    node names and class hierarchies, the per-tensor bookkeeping on inputs,
    outputs and value_info, and its ``pkg.torch.*`` model properties.

    They carry the exporting machine's absolute file paths, add megabytes to
    every browser download, and change the file whenever a source line moves.

    :param onnx_model: Loaded ONNX ``ModelProto``; modified in place.
    """

    def clean_graph(graph: "onnx.GraphProto") -> None:
        """
        Strip one graph and every subgraph its nodes hold.

        :param graph: Graph to clean in place.
        """
        del graph.metadata_props[:]
        for node in graph.node:
            del node.metadata_props[:]
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    clean_graph(attribute.g)
                for subgraph in attribute.graphs:
                    clean_graph(subgraph)

    clean_graph(onnx_model.graph)
    # Tensor-level entries carry the exporter's own bookkeeping too
    # (original node names, export signatures, optimizer provenance).
    graph = onnx_model.graph
    for value in (*graph.value_info, *graph.input, *graph.output):
        del value.metadata_props[:]
    for function in onnx_model.functions:
        del function.metadata_props[:]
        for node in function.node:
            del node.metadata_props[:]
    kept = [p for p in onnx_model.metadata_props if not p.key.startswith("pkg.torch")]
    del onnx_model.metadata_props[:]
    onnx_model.metadata_props.extend(kept)


def _add_metadata(onnx_model: "onnx.ModelProto", metadata: dict[str, str]) -> None:
    """
    Attach key/value pairs to an ONNX model's ``metadata_props``, after
    stripping the exporter's own debug metadata.

    :param onnx_model: Loaded ONNX ``ModelProto``; modified in place.
    :param metadata: String key/value pairs to embed.
    """
    _strip_exporter_metadata(onnx_model)
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value


def _export_metadata(
    model: nn.Module,
    *,
    family: str,
    architecture: str,
    segment_samples: int,
    stft: dict,
    stft_window: str,
    storage: torch.dtype,
    static_batch: bool,
    license_label: str | None,
) -> dict[str, str]:
    """
    Build the metadata every export embeds, in the one shared spelling.

    Keys are unprefixed, ``sources`` is JSON, and booleans are ``"true"``
    or ``"false"``, so a consumer reads any Unblend export the same way.

    :param model: Model being exported.
    :param family: Loader family (``demucs``, ``roformer``, ``scnet``).
    :param architecture: Registry architecture name.
    :param segment_samples: Samples the graph expects per call.
    :param stft: STFT geometry, with the ``torch.stft`` keyword names.
    :param stft_window: Analysis window, ``"hann"`` or ``"none"``.
    :param storage: Dtype the weights were stored at. Whether that also
        converts arithmetic follows from ``family``.
    :param static_batch: Whether the batch axis was traced fixed at 1.
    :param license_label: Registry license label, embedded when set.
    :return: String key/value pairs for ``_add_metadata``.
    """
    metadata = {
        "sources": json.dumps(list(model.sources)),
        "sample_rate": str(model.samplerate),
        "audio_channels": str(model.audio_channels),
        "weight_precision": _STORAGE_LABELS[storage],
        "compute_precision": "fp16"
        if storage is torch.float16 and family in _MIXED_PRECISION_FAMILIES
        else "fp32",
        "model_family": family,
        "architecture": architecture,
        "segment_samples": str(segment_samples),
        "stft_n_fft": str(int(stft["n_fft"])),
        "stft_hop_length": str(int(stft["hop_length"])),
        "stft_win_length": str(int(stft["win_length"])),
        "stft_normalized": "true" if stft["normalized"] else "false",
        "stft_window": stft_window,
        "batch_mode": "static" if static_batch else "dynamic",
        "external_normalization": (
            "true" if getattr(model, "external_normalization", True) else "false"
        ),
    }
    if license_label:
        metadata["license"] = license_label
    return metadata


@contextmanager
def _quiet_dynamo_export() -> "Iterator[None]":
    """
    Silence the dynamo exporter's warnings that users can't act on.

    nn.LSTM re-assigns its ``_flat_weights`` while being traced, so
    torch.export warns once per LSTM and lists every weight (pages of output
    for SCNet XL). Torch's own deprecation notices, the axis-rename note for
    a batch-1 dummy input, and the registry's "torchvision is not installed"
    log lines are noise here too.

    :return: Context manager.
    """
    registration = logging.getLogger("torch.onnx._internal.exporter._registration")
    level = registration.level
    registration.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            for message, category in (
                (r"(?s).*_flat_weights.*were assigned during export", UserWarning),
                (r"(?s).*_check_is_size", FutureWarning),
                (r"(?s).*isinstance\(treespec, LeafSpec\)", FutureWarning),
                (r"(?s).*The axis name: .* will not be used", UserWarning),
            ):
                warnings.filterwarnings("ignore", message=message, category=category)
            yield
    finally:
        registration.setLevel(level)


def _export_roformer_to_onnx(
    model: _RoformerBase,
    output_path: str,
    *,
    opset_version: int,
    storage: torch.dtype,
    license_label: str | None = None,
    static_batch: bool = False,
) -> str:
    """
    Export RoFormer to ONNX.

    :param model: The model to export.
    :param output_path: Path to save the ONNX model.
    :param opset_version: Requested opset; raised to at least 18.
    :param storage: Weight storage dtype; fp16 selects the browser-oriented
        mixed-precision path.
    :param license_label: License to embed in metadata.
    :param static_batch: Trace with fixed batch=1 instead of dynamic batch.
    :return: Path to the exported ONNX model.
    """
    try:
        import onnx
        import onnxscript  # noqa: F401  (required by the dynamo exporter)
    except ImportError:
        raise ImportError(
            "The 'onnx' and 'onnxscript' packages are required for RoFormer "
            "ONNX export. Install them with: uv pip install 'unblend[onnx]'"
        ) from None

    model.eval()
    wrapper = RoformerONNXWrapper(model).eval()

    segment_samples = int(round(model.max_allowed_segment * model.samplerate))
    stft = model.stft_kwargs

    trace_batch = 1 if static_batch else 2
    dummy_audio = torch.randn(trace_batch, model.audio_channels, segment_samples)
    dummy_real, dummy_imag = compute_roformer_stft_for_export(
        dummy_audio,
        n_fft=stft["n_fft"],
        hop_length=stft["hop_length"],
        win_length=stft["win_length"],
        normalized=stft["normalized"],
    )

    # One eager pass fills each rotary module's table cache. The export then
    # reads the cached pair as one shared constant; tables first built inside
    # the traced graph aren't cached, and would be rebuilt (and stored) once
    # per attention layer.
    with torch.no_grad():
        wrapper(dummy_real[:1], dummy_imag[:1])

    with _atomic_onnx_path(output_path) as staging_path:
        dynamic_shapes = None
        if not static_batch:
            batch = torch.export.Dim("batch")

            dynamic_shapes = {"spec_real": {0: batch}, "spec_imag": {0: batch}}
        with _quiet_dynamo_export():
            program = torch.onnx.export(
                wrapper,
                (dummy_real, dummy_imag),
                input_names=["spec_real", "spec_imag"],
                output_names=["out_spec_real", "out_spec_imag"],
                dynamic_shapes=dynamic_shapes,
                opset_version=_trace_opset(opset_version, 18, storage),
                dynamo=True,
                verbose=False,
            )
        program.save(staging_path)

        onnx_model = onnx.load(staging_path)

        _materialize_nonlast_broadcast_muls(onnx_model)
        _materialize_matmul_rank_mismatch(onnx_model)
        if storage is torch.float16:
            _convert_roformer_to_fp16(onnx_model)
        elif storage is not torch.float32:
            _convert_weight_storage(onnx_model, storage, model)

        architecture = (
            "mel_band_roformer" if isinstance(model, MelBandRoformer) else "bs_roformer"
        )
        metadata = _export_metadata(
            model,
            family="roformer",
            architecture=architecture,
            segment_samples=segment_samples,
            stft=stft,
            stft_window="hann",
            storage=storage,
            static_batch=static_batch,
            license_label=license_label,
        )
        metadata["num_stems"] = str(model.num_stems)
        metadata["output_complement"] = "true" if model.output_complement else "false"
        _add_metadata(onnx_model, metadata)

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, staging_path)

        onnx.checker.check_model(onnx.load(staging_path))
    return output_path


def _scnet_architecture(model: "SCNet") -> str:
    """
    Registry architecture name for an SCNet instance.

    :param model: Model being exported.
    :return: Registry architecture name.
    """
    from .scnet import SCNetMasked

    if isinstance(model, SCNetMasked):
        return "scnet_masked"
    return "scnet"


def _export_scnet_to_onnx(
    model: "SCNet",
    output_path: str,
    *,
    opset_version: int,
    storage: torch.dtype,
    license_label: str | None,
    static_batch: bool,
) -> str:
    """
    Export SCNet to ONNX.

    :param model: The SCNet to export.
    :param output_path: Path to save the ONNX model.
    :param opset_version: Requested opset; raised to 18.
    :param storage: Weight storage dtype.
    :param license_label: License recorded in metadata.
    :param static_batch: Trace with fixed batch of 1.
    :return: Path to the exported ONNX model.
    """
    try:
        import onnx
        import onnxscript  # noqa: F401  (required by the dynamo exporter)
    except ImportError:
        raise ImportError(
            "ONNX export of SCNet needs 'onnx' and 'onnxscript'. "
            "Install them with: uv pip install 'unblend[onnx]'"
        ) from None

    opset_version = _trace_opset(opset_version, 18, storage)
    model.eval()
    wrapper = SCNetONNXWrapper(model).eval()

    segment = int(round(model.max_allowed_segment * model.samplerate))
    from .scnet import stft_padding

    padding = stft_padding(segment, model.hop_length)
    device = next(model.parameters()).device
    audio = torch.zeros(1, model.audio_channels, segment + padding, device=device)
    spec_real, spec_imag = compute_scnet_stft_for_export(
        audio,
        int(model.stft_config["n_fft"]),
        int(model.stft_config["hop_length"]),
        int(model.stft_config["win_length"]),
        bool(model.stft_config["normalized"]),
    )

    dynamic_shapes = None
    if not static_batch:
        batch = torch.export.Dim("batch")
        dynamic_shapes = ({0: batch}, {0: batch})

    with _atomic_onnx_path(output_path) as staging, _quiet_dynamo_export():
        program = torch.onnx.export(
            wrapper,
            (spec_real, spec_imag),
            input_names=["spec_real", "spec_imag"],
            output_names=["out_spec_real", "out_spec_imag"],
            opset_version=opset_version,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            verbose=False,
        )
        program.save(staging)
        onnx_model = onnx.load(staging)
        if storage is not torch.float32:
            _convert_weight_storage(onnx_model, storage, model)
        window = "hann" if hasattr(model, "window") else "none"
        metadata = _export_metadata(
            model,
            family="scnet",
            architecture=_scnet_architecture(model),
            segment_samples=segment + padding,
            stft=model.stft_config,
            stft_window=window,
            storage=storage,
            static_batch=static_batch,
            license_label=license_label,
        )
        metadata["logical_segment_samples"] = str(segment)
        metadata["stft_pad_samples"] = str(padding)
        _add_metadata(onnx_model, metadata)
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, staging)
    return output_path


_EXPORTERS: dict[type, "Callable[..., str]"] = {}


def register_exporter(family: type, exporter: "Callable[..., str]") -> None:
    """
    Register an ONNX exporter for an architecture family.

    :param family: The model class the exporter handles.
    :param exporter: Callable with ``_export_*_to_onnx``'s signature.
    """
    _EXPORTERS[family] = exporter


register_exporter(_RoformerBase, _export_roformer_to_onnx)
register_exporter(SCNet, _export_scnet_to_onnx)


def _reject_multi_checkpoint(models: dict[str, dict], model_name: str) -> None:
    """
    Reject entries that hold more than one checkpoint, before any download.

    ONNX export traces one graph per checkpoint, so a Demucs bag or an
    ensemble has nothing single to trace. Checking the registry entry first
    means the failure costs a lookup instead of a full weight download.

    :param models: Registry entries, as ``ModelRepository.list_models``
        returns them.
    :param model_name: Name being exported.
    :raises ModelLoadingError: If the name is unknown.
    :raises ValidationError: If the entry holds several checkpoints.
    """
    info = models.get(model_name)
    if info is None:
        raise ModelLoadingError(
            f"Could not find a model with name {model_name}. "
            f"Available models: {', '.join(models)}"
        )

    specs = info.get("members") or []
    if len(specs) < 2:
        return

    referenced = [
        spec["model"]
        for spec in specs
        if isinstance(spec, dict) and isinstance(spec.get("model"), str)
    ]
    hint = (
        f" Export its members separately: {', '.join(referenced)}."
        if len(referenced) == len(specs)
        else ""
    )
    raise ValidationError(
        f"Model {model_name} holds {len(specs)} checkpoints; ONNX export traces "
        f"a single graph and cannot represent a bag or an ensemble.{hint}"
    )


def validate_export_request(
    model_name: str, opset_version: int, precision: str
) -> tuple[ModelRepository, dict]:
    """
    Check an export's arguments against the registry, before any download.

    :param model_name: Model to export.
    :param opset_version: Requested opset.
    :param precision: Requested weight precision.
    :return: ``(repository, registry entry)``.
    :raises ValidationError: If precision or opset is invalid, or the model
        holds several checkpoints.
    :raises ModelLoadingError: If the model is unknown.
    :raises ImportError: If the ``onnx`` extra isn't installed.
    """
    try:
        import onnx.defs
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with: uv pip install 'unblend[onnx]'"
        ) from None

    _validate_export_precision(precision)
    from torch.onnx import _constants as torch_onnx

    max_opset = min(onnx.defs.onnx_opset_version(), torch_onnx.ONNX_MAX_OPSET)
    if (
        isinstance(opset_version, bool)
        or not isinstance(opset_version, int)
        or not 17 <= opset_version <= max_opset
    ):
        raise ValidationError(
            f"opset_version must be an integer from 17 to {max_opset} (the "
            f"newest this torch and onnx support), got {opset_version!r}."
        )

    repo = ModelRepository()
    models = repo.list_models()
    _reject_multi_checkpoint(models, model_name)
    model_info = models.get(model_name, {})
    # Before any download: HTDemucs uses the TorchScript exporter.
    if (
        model_info.get("architecture") == "htdemucs"
        and opset_version > torch_onnx.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
    ):
        raise ValidationError(
            "HTDemucs exports with torch's TorchScript exporter, which supports "
            f"opset <= {torch_onnx.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET}."
        )
    return repo, model_info


def export_to_onnx(
    model_name: str = "htdemucs",
    output_path: str | None = None,
    opset_version: int = 17,
    precision: str = "native",
    static_batch: bool = False,
) -> str:
    """
    Export model to ONNX.

    ``precision="native"`` stores weights at whatever precision the
    checkpoint actually carries, which is fp16 for the HTDemucs family and
    fp32 for RoFormer and SCNet. ``"fp16"`` and ``"fp32"`` force the choice.
    Note that fp16 storage means weight-only fp16 for HTDemucs and SCNet but
    mixed precision for RoFormer; see onnx.md.

    :param model_name: Name of the model to export.
    :param output_path: Path to save the ONNX model (defaults to
        ``{model_name}_{precision}.onnx``, with ``_static`` appended under
        ``static_batch``).
    :param opset_version: ONNX opset version (raised to 18 for RoFormer and
        SCNet, and to 19 for fp8 storage).
    :param precision: ``"native"``, ``"fp32"``, ``"fp16"``, ``"bf16"``,
        ``"fp8_e4m3"`` or ``"fp8_e5m2"``.
    :param static_batch: Trace with fixed batch=1; browser deployment only.
    :return: Path to the exported ONNX model.
    :raises ValidationError: If precision or opset is invalid, the model
        holds several checkpoints or isn't an exportable architecture, or the
        output path can't name a file here (see ``_check_output_path``; a
        character the filesystem refuses surfaces when the file is created).
    :raises ModelLoadingError: If the model is unknown or can't be loaded.
    """
    try:
        import onnx
        import onnx.defs
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with: uv pip install 'unblend[onnx]'"
        ) from None

    repo, model_info = validate_export_request(model_name, opset_version, precision)
    if output_path is not None:
        # Before the model is loaded: a bad explicit path fails fast.
        _check_output_path(output_path)
    model = repo.get_model(model_name)

    storage = _resolve_export_precision(precision, model_info)
    if storage is torch.float8_e5m2:
        warnings.warn(
            "fp8_e5m2 keeps only two mantissa bits: on HTDemucs stems drift to "
            "about -7 to 25 dB SNR against fp32, which audibly breaks separation. "
            "Prefer fp16 or bf16.",
            UserWarning,
            stacklevel=2,
        )
    if storage is not torch.float32:
        # Trace at the storage type's minimum opset; bumping the number after
        # tracing would leave ops in their older signatures.
        opset_version = max(opset_version, _onnx_storage_spec(storage)[2])

    if output_path is None:
        # Named from the checkpoint's real precision, so checked only now
        # (still before the trace).
        output_path = default_output_path(model_name, storage, static_batch)
        try:
            _check_output_path(output_path, "Default output path")
        except ValidationError as exc:
            raise ValidationError(f"{exc} Pass output_path (-o) instead.") from None

    for family, exporter in _EXPORTERS.items():
        if isinstance(model, family):
            return exporter(
                model,
                output_path,
                opset_version=opset_version,
                storage=storage,
                license_label=model_info.get("license"),
                static_batch=static_batch,
            )

    if not isinstance(model, HTDemucs):
        raise ValidationError(
            f"Model {model_name} is not a supported model type. "
            f"Expected HTDemucs, a RoFormer, or SCNet, got {type(model).__name__}"
        )
    wrapper = HTDemucsONNXWrapper(model)

    model.eval()
    wrapper.eval()

    sample_rate = model.samplerate
    segment_samples = int(round(model.max_allowed_segment * sample_rate))
    nfft = model.nfft
    hop_length = model.hop_length

    batch_size = 1
    audio_channels = model.audio_channels

    dummy_audio = torch.randn(batch_size, audio_channels, segment_samples)
    dummy_spec_real, dummy_spec_imag = compute_stft_for_export(
        dummy_audio, nfft, hop_length
    )

    with _atomic_onnx_path(output_path) as staging_path:
        # The TorchScript tracer warns about every Python branch it bakes in.
        # Here they all read static geometry, never the dynamic batch axis, so
        # the warnings are noise; they are silenced for this call only.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
            # At opset >= 19 (fp8, or an explicit --opset) the exporter notes
            # it can't fold strided Slices; the graph is correct regardless.
            warnings.filterwarnings(
                "ignore", message=r"(?s).*Constant folding - Only steps=1"
            )
            torch.onnx.export(
                wrapper,
                (dummy_spec_real, dummy_spec_imag, dummy_audio),
                staging_path,
                input_names=["spec_real", "spec_imag", "audio"],
                output_names=["out_spec_real", "out_spec_imag", "out_wave"],
                dynamic_axes=None
                if static_batch
                else {
                    "spec_real": {0: "batch"},
                    "spec_imag": {0: "batch"},
                    "audio": {0: "batch"},
                    "out_spec_real": {0: "batch"},
                    "out_spec_imag": {0: "batch"},
                    "out_wave": {0: "batch"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False,
            )

        onnx_model = onnx.load(staging_path)

        if storage is not torch.float32:
            _convert_weight_storage(onnx_model, storage, model)

        metadata = _export_metadata(
            model,
            family="demucs",
            architecture="htdemucs",
            segment_samples=segment_samples,
            stft={
                "n_fft": nfft,
                "hop_length": hop_length,
                "win_length": nfft,
                "normalized": True,
            },
            stft_window="hann",
            storage=storage,
            static_batch=static_batch,
            license_label=model_info.get("license"),
        )
        # Demucs-only STFT trimming: pre-pad, then drop two frames per side
        # and the top frequency bin. See onnx.md.
        metadata["stft_pad_samples"] = str(hop_length // 2 * 3)
        metadata["stft_frame_trim"] = "2"
        _add_metadata(onnx_model, metadata)

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, staging_path)
        onnx.checker.check_model(onnx.load(staging_path))

    return output_path
