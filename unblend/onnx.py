# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import onnx

from .blocks import pad1d, spectro
from .htdemucs import HTDemucs
from .repo import ModelRepository
from .roformer import (
    Attention,
    FeedForward,
    MaskEstimator,
    MelBandRoformer,
    RMSNorm,
    _RoformerBase,
)
from .scnet import FeatureConversion, SCNet


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
            Initialize the ONNX wrapper.
            :param model: The RoFormer model to wrap.
        :param attention_query_chunk_size: Max query rows per attention chunk.
        :param attention_head_chunk_size: Max heads projected at once.
        :param feedforward_hidden_chunk_size: Max expanded MLP features at once.
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
            if isinstance(module, FeatureConversion):
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
            if freq > model.max_f:
                repeats = math.ceil(freq / model.max_f)
                pos_f = model.pos_embed_f.repeat(1, 1, repeats, 1)[:, :, :freq, :]
            else:
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

    :param audio: Input audio ``[B, C, samples]``. :param n_fft: FFT size.
        :param hop_length: Hop length. :param win_length: Window length.
        :param normalized: Whether the STFT is normalised. :param window:
        ``"none"`` or ``"hann"``; from export metadata.
    :return: ``(real, imag)`` spectrograms ``[B, C, F, T]``.
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
            else None
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

    :param audio: Input audio ``[B, C, samples]``. :param n_fft: FFT size.
        :param hop_length: Hop length. :param win_length: Window length.
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


def _convert_weights_to_fp16(onnx_model: "onnx.ModelProto") -> None:
    """
    Rewrite ONNX weights as float16.

    :param onnx_model: Loaded ONNX model; modified in place.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

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

    new_inits = []
    new_cast_nodes = []
    for init in onnx_model.graph.initializer:
        if (
            init.name in weight_init_names
            and init.data_type == TensorProto.FLOAT
            and init.name not in existing_outputs
            and init.name not in existing_inputs
        ):
            arr = numpy_helper.to_array(init).astype(np.float16)
            fp16_name = init.name + "_fp16"
            new_inits.append(numpy_helper.from_array(arr, name=fp16_name))
            new_cast_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[fp16_name],
                    outputs=[init.name],
                    to=TensorProto.FLOAT,
                    name=init.name + "_cast_to_fp32",
                )
            )
        else:
            new_inits.append(init)

    if not new_cast_nodes:
        raise RuntimeError(
            "fp16 export requested but no fp32 weight initializers were "
            "converted — the exporter's op/initializer layout likely "
            "changed. Refusing to write a model mislabeled as fp16."
        )

    onnx_model.graph.ClearField("initializer")
    onnx_model.graph.initializer.extend(new_inits)

    original_nodes = list(onnx_model.graph.node)
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_cast_nodes + original_nodes)


def _convert_roformer_to_fp16(onnx_model: "onnx.ModelProto") -> None:
    """
    Convert RoFormer graph to mixed precision.

    :param onnx_model: Loaded ONNX model; modified in place.
    """
    import warnings

    from onnx import TensorProto

    try:
        from onnxconverter_common.float16 import convert_float_to_float16
    except ImportError:
        raise ImportError(
            "onnxconverter-common is required for RoFormer fp16 export. "
            "Install unblend with the 'onnx' extra."
        )

    blocked_ops = {
        "Clip",
        "Cos",
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

        :param a: Name of the first ``Mul`` operand. :param b: Name of the
            second ``Mul`` operand.
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.onnx", dir=destination.parent
    )
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

        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
        for candidate in sidecars():
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                candidate.unlink(missing_ok=True)


def _add_metadata(onnx_model: "onnx.ModelProto", metadata: dict[str, str]) -> None:
    """
    Attach key/value pairs to an ONNX model's ``metadata_props``.

    :param onnx_model: Loaded ONNX ``ModelProto``; modified in place.
    :param metadata: String key/value pairs to embed.
    """
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
    fp16: bool,
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
    :param fp16: Whether weights were stored as fp16.
    :param static_batch: Whether the batch axis was traced fixed at 1.
    :param license_label: Registry license label, embedded when set.
    :return: String key/value pairs for ``_add_metadata``.
    """
    metadata = {
        "sources": json.dumps(list(model.sources)),
        "sample_rate": str(model.samplerate),
        "audio_channels": str(model.audio_channels),
        "precision": "fp16" if fp16 else "fp32",
        "model_family": family,
        "architecture": architecture,
        "segment_samples": str(segment_samples),
        "stft_n_fft": str(int(stft["n_fft"])),
        "stft_hop_length": str(int(stft["hop_length"])),
        "stft_win_length": str(int(stft["win_length"])),
        "stft_normalized": "true" if stft["normalized"] else "false",
        "stft_window": stft_window,
        "batch_mode": "static" if static_batch else "dynamic",
    }
    if license_label:
        metadata["license"] = license_label
    return metadata


def _export_roformer_to_onnx(
    model: _RoformerBase,
    output_path: str,
    *,
    opset_version: int,
    fp16: bool,
    license_label: str | None = None,
    static_batch: bool = False,
) -> str:
    """
    Export RoFormer to ONNX.

    :param model: The model to export. :param output_path: Path to save the
        ONNX model. :param opset_version: Requested opset; clamped up to 18.
        :param fp16: Use browser-oriented mixed precision. :param license_label:
        License to embed in metadata. :param static_batch: Trace with fixed
        batch=1 instead of dynamic batch.
    :return: Path to the exported ONNX model.
    """
    try:
        import onnx
        import onnxscript  # noqa: F401  (required by the dynamo exporter)
    except ImportError:
        raise ImportError(
            "The 'onnx' and 'onnxscript' packages are required for RoFormer "
            "ONNX export. Install them with: uv pip install unblend[onnx]"
        )

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

    with _atomic_onnx_path(output_path) as staging_path:
        dynamic_shapes = None
        if not static_batch:
            batch = torch.export.Dim("batch")

            dynamic_shapes = {"spec_real": {0: batch}, "spec_imag": {0: batch}}
        program = torch.onnx.export(
            wrapper,
            (dummy_real, dummy_imag),
            input_names=["spec_real", "spec_imag"],
            output_names=["out_spec_real", "out_spec_imag"],
            dynamic_shapes=dynamic_shapes,
            opset_version=max(opset_version, 18),
            dynamo=True,
        )
        program.save(staging_path)

        onnx_model = onnx.load(staging_path)

        _materialize_nonlast_broadcast_muls(onnx_model)
        _materialize_matmul_rank_mismatch(onnx_model)
        if fp16:
            _convert_roformer_to_fp16(onnx_model)

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
            fp16=fp16,
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
    fp16: bool,
    license_label: str | None,
    static_batch: bool,
) -> str:
    """
    Export SCNet to ONNX.

    :param model: The SCNet to export.
    :param output_path: Path to save the ONNX model.
    :param opset_version: Requested opset; raised to 18.
    :param fp16: Convert weights to fp16.
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
            "Install them with: uv pip install unblend[onnx]"
        )

    opset_version = max(opset_version, 18)
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

    with _atomic_onnx_path(output_path) as staging:
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
        if fp16:
            _convert_weights_to_fp16(onnx_model)
        window = "hann" if hasattr(model, "window") else "none"
        metadata = _export_metadata(
            model,
            family="scnet",
            architecture=_scnet_architecture(model),
            segment_samples=segment + padding,
            stft=model.stft_config,
            stft_window=window,
            fp16=fp16,
            static_batch=static_batch,
            license_label=license_label,
        )
        metadata["logical_segment_samples"] = str(segment)
        metadata["stft_pad_samples"] = str(padding)
        metadata["external_normalization"] = (
            "true" if model.external_normalization else "false"
        )
        _add_metadata(onnx_model, metadata)
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
    :raises ValueError: If the name is unknown or holds several checkpoints.
    """
    info = models.get(model_name)
    if info is None:
        raise ValueError(
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
    raise ValueError(
        f"Model {model_name} holds {len(specs)} checkpoints; ONNX export traces "
        f"a single graph and cannot represent a bag or an ensemble.{hint}"
    )


def export_to_onnx(
    model_name: str = "htdemucs",
    output_path: str | None = None,
    opset_version: int = 17,
    fp16: bool = False,
    static_batch: bool = False,
) -> str:
    """
    Export model to ONNX.

    :param model_name: Name of the model to export. :param output_path: Path to
        save the ONNX model (defaults to ``{model_name}.onnx``).
    :param opset_version: ONNX opset version (raised to 18 for RoFormer and
        SCNet). :param fp16: Use weight-only (HTDemucs) or mixed precision
        (RoFormer). :param static_batch: Trace with fixed batch=1; browser
        deployment only.
    :return: Path to the exported ONNX model.
    :raises ValueError: If the model is unknown, holds several checkpoints, or
        is not one of the exportable architectures.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with: uv pip install unblend[onnx]"
        )

    if output_path is None:
        output_path = f"{model_name}.onnx"

    repo = ModelRepository()
    models = repo.list_models()
    _reject_multi_checkpoint(models, model_name)
    model = repo.get_model(model_name)
    model_info = models.get(model_name, {})

    for family, exporter in _EXPORTERS.items():
        if isinstance(model, family):
            return exporter(
                model,
                output_path,
                opset_version=opset_version,
                fp16=fp16,
                license_label=model_info.get("license"),
                static_batch=static_batch,
            )

    if not isinstance(model, HTDemucs):
        raise ValueError(
            f"Model {model_name} is not a supported model type. "
            f"Expected HTDemucs, a RoFormer, or SCNet, got {type(model).__name__}"
        )
    if not model.cac:
        raise ValueError(
            f"Model {model_name} does not use complex-as-channels (cac=False); "
            "the ONNX wrapper hardcodes CaC spectrogram packing."
        )
    wrapper = HTDemucsONNXWrapper(model)

    model.eval()
    wrapper.eval()

    sample_rate = model.samplerate
    segment_samples = int(model.max_allowed_segment * sample_rate)
    nfft = model.nfft
    hop_length = model.hop_length

    batch_size = 1
    audio_channels = model.audio_channels

    dummy_audio = torch.randn(batch_size, audio_channels, segment_samples)
    dummy_spec_real, dummy_spec_imag = compute_stft_for_export(
        dummy_audio, nfft, hop_length
    )

    with _atomic_onnx_path(output_path) as staging_path:
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

        if fp16:
            _convert_weights_to_fp16(onnx_model)

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
            fp16=fp16,
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
