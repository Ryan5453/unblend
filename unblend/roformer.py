# Copyright (c) 2023 Phil Wang (lucidrains/BS-RoFormer)
# Copyright (c) 2024 Roman Solovyev (ZFTurbo/Music-Source-Separation-Training)
# Copyright (c) 2025-present Ryan Fahey

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
RoFormer architectures for source separation.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backends import ASSModel, CustomKernelModule
from .exceptions import ValidationError

DEFAULT_FREQS_PER_BANDS: tuple[int, ...] = (
    *(2,) * 24,
    *(4,) * 12,
    *(12,) * 8,
    *(24,) * 8,
    *(48,) * 8,
    128,
    129,
)


class RMSNorm(CustomKernelModule):
    def __init__(self, dim: int) -> None:
        """
        Root-mean-square LayerNorm with a learnable gain.

        :param dim: Feature dimension to normalise over (last axis).
        """
        super().__init__()
        self.dim = dim
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

        self.onnx_safe = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to unit RMS over the last axis and apply the gain.

        :param x: Input of shape ``[..., dim]``. :return: Normalised tensor.
        """
        if self.onnx_safe:
            working = x.float()
            mean_square = working.square().mean(dim=-1, keepdim=True)

            normalized = working * torch.rsqrt(mean_square.clamp_min(1e-12))
            return (normalized * self.gamma.float()).to(x.dtype)
        if (
            self.use_custom_kernels
            and x.device.type == "mps"
            and not torch.is_grad_enabled()
        ):
            from .metal import metal_rms_norm

            return metal_rms_norm(x, self.gamma, self.scale)
        return F.rms_norm(x, (self.dim,), self.gamma, eps=1e-12)


def _binary_concat(tensors: list[Tensor], *, dim: int) -> Tensor:
    """
    Concatenate through a tree whose nodes have at most two inputs.

    :param tensors: Tensors to concatenate; must be non-empty.
    :param dim: Dimension to concatenate along.
    :return: The concatenated tensor.
    :raises ValueError: If ``tensors`` is empty.
    """
    if not tensors:
        raise ValueError("cannot concatenate an empty tensor list")
    while len(tensors) > 1:
        tensors = [
            torch.cat(tensors[index : index + 2], dim=dim)
            for index in range(0, len(tensors), 2)
        ]
    return tensors[0]


def _binary_sum(tensors: list[Tensor]) -> Tensor:
    """
    Sum through a balanced tree so no long Add chain stays live.

    :param tensors: Tensors to sum; must be non-empty and broadcast-compatible.
    :return: The summed tensor.
    :raises ValueError: If ``tensors`` is empty.
    """
    if not tensors:
        raise ValueError("cannot sum an empty tensor list")
    while len(tensors) > 1:
        tensors = [
            (
                tensors[index] + tensors[index + 1]
                if index + 1 < len(tensors)
                else tensors[index]
            )
            for index in range(0, len(tensors), 2)
        ]
    return tensors[0]


class RotaryEmbedding(CustomKernelModule):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        Rotary position embedding (RoPE, Su et al.

        :param dim: Per-head rotation dimensionality. :param theta: Inverse-frequency base.
        """
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

        self._cos_sin_cache: dict[
            tuple[int, torch.device, torch.dtype], tuple[Tensor, Tensor]
        ] = {}

        self._compiled_cos: Tensor | None = None
        self._compiled_sin: Tensor | None = None

    def _cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """
        ``(cos, sin)`` rotation tables of shape ``[seq_len, dim // 2]`` in the requested working dtype.

        :param seq_len: Sequence length to build tables for. :param device: Device for the tables. :param dtype: Working dtype of queries/keys. :return: Cached ``(cos, sin)`` pair.
        """
        key = (seq_len, device, dtype)
        cached = self._cos_sin_cache.get(key)
        if cached is None:
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            angles = positions[:, None] * self.freqs.to(
                device=device, dtype=torch.float32
            )
            cached = (angles.cos().to(dtype), angles.sin().to(dtype))
            self._cos_sin_cache[key] = cached
        return cached

    def prime_compiled(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """
        Bind this axis's rotation table to frozen attributes ahead of ``torch.compile`` capture, so the compiled trunk reads a constant instead of doing a Python dict lookup per attention.

        :param seq_len: Sequence length to bind tables for. :param device: Capture device. :param dtype: Working dtype of queries/keys.
        """
        cos, sin = self._cos_sin(seq_len, device, dtype)
        self._compiled_cos = cos
        self._compiled_sin = sin

    def _apply(
        self, fn: Callable[[Tensor], Tensor], recurse: bool = True
    ) -> "RotaryEmbedding":
        """
        Apply a dtype/device transform, then invalidate derived caches.

        :param fn: Tensor transformation supplied by ``nn.Module.to``/``half``.
        :param recurse: Whether child modules should also be transformed.
        :return: This module after the successful transformation.
        """
        result = super()._apply(fn, recurse=recurse)
        self._cos_sin_cache.clear()
        self._compiled_cos = None
        self._compiled_sin = None
        return result

    def _load_from_state_dict(self, *args: object, **kwargs: object) -> None:
        """
        Drop cached rotation tables when weights (``freqs``) are replaced.

        :param args: Forwarded to ``nn.Module._load_from_state_dict``.
        :param kwargs: Forwarded to ``nn.Module._load_from_state_dict``.
        """
        self._cos_sin_cache.clear()
        self._compiled_cos = None
        self._compiled_sin = None
        super()._load_from_state_dict(*args, **kwargs)

    def rotate_queries_or_keys(self, t: Tensor) -> Tensor:
        """
        Apply rotary rotation over the sequence axis of ``t``.

        :param t: Queries or keys of shape ``[..., seq, dim]``. :return: Rotated tensor.
        """
        if torch.compiler.is_compiling() and self._compiled_cos is not None:
            cos, sin = self._compiled_cos, self._compiled_sin
        else:
            cos, sin = self._cos_sin(t.shape[-2], t.device, t.dtype)

        if (
            self.use_custom_kernels
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and t.dtype in (torch.float16, torch.bfloat16)
        ):
            if t.device.type == "cuda":
                from .cuda import fused_roformer_rotary

                return fused_roformer_rotary(t, cos, sin)
            if t.device.type == "mps":
                from .metal import metal_rotary

                return metal_rotary(t, cos, sin)

        x1, x2 = t.unflatten(-1, (-1, 2)).unbind(dim=-1)
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.flatten(-2)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        """
        Pre-norm MLP block (RMSNorm -> Linear -> GELU -> Linear).

        :param dim: Input/output feature dimension.
        :param mult: Hidden-layer expansion factor.
        :param dropout: Dropout probability (inactive in eval mode).
        """
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

        self.onnx_hidden_chunk_size: int | None = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the MLP block.

        :param x: Input of shape ``[..., dim]``.
        :return: Output of the same shape.
        """
        chunk_size = self.onnx_hidden_chunk_size
        if chunk_size is None:
            return self.net(x)
        if chunk_size <= 0:
            raise ValueError(
                f"onnx_hidden_chunk_size must be positive or None, got {chunk_size}"
            )

        first = self.net[1]
        second = self.net[4]
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise RuntimeError("RoFormer feed-forward layout changed")

        normalized = self.net[0](x)
        partials: list[Tensor] = []
        for start in range(0, first.out_features, chunk_size):
            end = min(start + chunk_size, first.out_features)
            hidden = F.linear(
                normalized,
                first.weight[start:end],
                None if first.bias is None else first.bias[start:end],
            )
            hidden = F.gelu(hidden)
            hidden = self.net[3](hidden)
            partials.append(F.linear(hidden, second.weight[:, start:end], bias=None))

        out = _binary_sum(partials)
        if second.bias is not None:
            out = out + second.bias
        return self.net[5](out)


def _scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
    dropout: float,
    training: bool,
) -> Tensor:
    """
    Run RoFormer self-attention with the fastest measured backend path.

    :param query: Queries ``[batch, heads, sequence, dim]``. :param key: Keys matching ``query`` shape. :param value: Values matching ``query`` shape. :param scale: Dot-product scale. :param dropout: Dropout probability. :param training: Whether the module is training. :return: Attention output.
    """
    if query.device.type == "mps" and not training:
        weights = (query * scale) @ key.transpose(-1, -2)
        return weights.softmax(dim=-1) @ value
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=dropout if training else 0.0,
    )


def _chunked_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
    dropout: float,
    training: bool,
    query_chunk_size: int | None,
) -> Tensor:
    """
    Run exact attention without materialising the full score matrix.

    :param query: Queries ``[batch, heads, sequence, dim]``. :param key: Keys matching ``query`` shape. :param value: Values matching ``query`` shape. :param scale: Dot-product scale.
    :param dropout: Attention dropout probability. :param training: Whether the module is training. :param query_chunk_size: Max query rows per score tensor. :return: Attention output.
    """
    if query_chunk_size is None or query.shape[-2] <= query_chunk_size:
        return _scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            dropout=dropout,
            training=training,
        )
    if query_chunk_size <= 0:
        raise ValueError(
            f"query_chunk_size must be positive or None, got {query_chunk_size}"
        )

    chunks = [
        _scaled_dot_product_attention(
            query[..., start : start + query_chunk_size, :],
            key,
            value,
            scale=scale,
            dropout=dropout,
            training=training,
        )
        for start in range(0, query.shape[-2], query_chunk_size)
    ]
    return _binary_concat(chunks, dim=-2)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_embed: RotaryEmbedding | None = None,
    ) -> None:
        """
        Pre-norm multi-head self-attention with per-head sigmoid gating, as used by both RoFormer variants.

        :param dim: Feature dimension. :param heads: Number of heads. :param dim_head: Dimension per head. :param dropout: Attention/projection dropout. :param rotary_embed: Shared rotary embedding, or ``None``.
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.dropout = dropout

        self.onnx_query_chunk_size: int | None = None
        self.onnx_head_chunk_size: int | None = None

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def _forward_chunk(self, x: Tensor) -> Tensor:
        """
        Run the ordinary all-head attention path.

        :param x: Input of shape ``[batch, sequence, dim]``.
        :return: Attention output of the same shape.
        """
        batch, seq, _ = x.shape
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(batch, seq, self.heads, -1).transpose(1, 2) for t in (q, k, v)
        )

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = _chunked_scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            dropout=self.dropout,
            training=self.training,
            query_chunk_size=self.onnx_query_chunk_size,
        )

        gates = self.to_gates(x)
        out = out * gates.transpose(1, 2).unsqueeze(-1).sigmoid()

        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.to_out(out)

    def _forward_head_chunks(self, x: Tensor, head_chunk_size: int) -> Tensor:
        """
        Run exact attention in independently projected head groups.

        :param x: Input of shape ``[batch, sequence, dim]``. :param head_chunk_size: Heads projected per group. :return: Attention output of the same shape.
        """
        batch, seq, _ = x.shape
        x = self.norm(x)
        dim_inner = self.to_qkv.out_features // 3
        dim_head = dim_inner // self.heads
        gates = self.to_gates(x).sigmoid().transpose(1, 2).unsqueeze(-1)
        out_weight = self.to_out[0].weight

        partials: list[Tensor] = []
        for head_start in range(0, self.heads, head_chunk_size):
            head_end = min(head_start + head_chunk_size, self.heads)
            feature_start = head_start * dim_head
            feature_end = head_end * dim_head
            group_heads = head_end - head_start

            q = F.linear(
                x,
                self.to_qkv.weight[feature_start:feature_end],
                bias=None,
            )
            k = F.linear(
                x,
                self.to_qkv.weight[dim_inner + feature_start : dim_inner + feature_end],
                bias=None,
            )
            v = F.linear(
                x,
                self.to_qkv.weight[
                    2 * dim_inner + feature_start : 2 * dim_inner + feature_end
                ],
                bias=None,
            )
            q, k, v = (
                tensor.view(batch, seq, group_heads, dim_head).transpose(1, 2)
                for tensor in (q, k, v)
            )

            if self.rotary_embed is not None:
                q = self.rotary_embed.rotate_queries_or_keys(q)
                k = self.rotary_embed.rotate_queries_or_keys(k)

            out = _chunked_scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scale,
                dropout=self.dropout,
                training=self.training,
                query_chunk_size=self.onnx_query_chunk_size,
            )
            out = out * gates[:, head_start:head_end]
            out = out.transpose(1, 2).reshape(batch, seq, -1)
            partials.append(
                F.linear(
                    out,
                    out_weight[:, feature_start:feature_end],
                    bias=None,
                )
            )

        return self.to_out[1](_binary_sum(partials))

    def forward(self, x: Tensor) -> Tensor:
        """
        Run gated multi-head attention over the sequence axis.

        :param x: Input of shape ``[batch, sequence, dim]``. :return: Attention output.
        """
        chunk_size = self.onnx_head_chunk_size
        if chunk_size is None or chunk_size >= self.heads:
            return self._forward_chunk(x)
        if chunk_size <= 0:
            raise ValueError(
                f"onnx_head_chunk_size must be positive or None, got {chunk_size}"
            )
        return self._forward_head_chunks(x, chunk_size)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        norm_output: bool = True,
        rotary_embed: RotaryEmbedding | None = None,
    ) -> None:
        """
        Stack of pre-norm attention + feed-forward blocks with residuals.

        :param dim: Feature dimension. :param depth: Number of attention/FF pairs. :param dim_head: Dimension per head. :param heads: Number of heads.
        :param attn_dropout: Attention dropout probability. :param ff_dropout: Feed-forward dropout probability. :param ff_mult: Feed-forward expansion factor. :param norm_output: Whether to RMS-normalise the output. :param rotary_embed: Shared rotary embedding for every block.
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the transformer stack.

        :param x: Input of shape ``[batch, seq, dim]``.
        :return: Output of the same shape.
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]) -> None:
        """
        Project each frequency band (real/imag interleaved bins) into the
        shared feature dimension.

        :param dim: Output feature dimension.
        :param dim_inputs: Input width of each band.
        """
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([])
        for dim_in in dim_inputs:
            self.to_features.append(
                nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Split ``x`` into bands and project each to the feature dimension.

        :param x: Input of shape ``[batch, time, sum(dim_inputs)]``.
        :return: Band features of shape ``[batch, time, bands, dim]``.
        """
        outs = []
        start = 0
        for width, to_feature in zip(self.dim_inputs, self.to_features):
            outs.append(to_feature(x[..., start : start + width]).unsqueeze(-2))
            start += width
        return _binary_concat(outs, dim=-2)


def MLP(
    dim_in: int,
    dim_out: int,
    dim_hidden: int | None = None,
    hidden_layers: int = 0,
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    """
    Build a Linear/activation MLP as a flat ``nn.Sequential``.

    :param dim_in: Input feature dimension. :param dim_out: Output feature dimension. :param dim_hidden: Hidden feature dimension (defaults to ``dim_in``). :param hidden_layers: Number of hidden layers. :param activation: Activation module class. :return: The assembled ``nn.Sequential``.
    """
    dim_hidden = dim_hidden or dim_in
    dims = (dim_in, *((dim_hidden,) * hidden_layers), dim_out)
    net: list[nn.Module] = []
    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        net.append(nn.Linear(layer_dim_in, layer_dim_out))
        if ind < len(dims) - 2:
            net.append(activation())
    return nn.Sequential(*net)


class MaskEstimator(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_inputs: tuple[int, ...],
        mlp_hidden_layers: int,
        mlp_expansion_factor: int = 4,
    ) -> None:
        """
        Per-band MLP heads producing complex masks (via GLU) for one stem.

        :param dim: Feature dimension. :param dim_inputs: Input width of each band. :param mlp_hidden_layers: Hidden layer count per band MLP. :param mlp_expansion_factor: Hidden width multiplier over ``dim``.
        """
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = nn.ModuleList([])

        self.onnx_safe_glu = False
        dim_hidden = dim * mlp_expansion_factor
        for dim_in in dim_inputs:
            self.to_freqs.append(
                nn.Sequential(
                    MLP(
                        dim,
                        dim_in * 2,
                        dim_hidden=dim_hidden,
                        hidden_layers=mlp_hidden_layers,
                    ),
                    nn.GLU(dim=-1),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Estimate per-band masks and concatenate along the frequency axis.

        :param x: Band features of shape ``[batch, time, bands, dim]``.
        :return: Masks of shape ``[batch, time, sum(dim_inputs)]``.
        """
        outs = []
        for index, mlp in enumerate(self.to_freqs):
            band = x.select(dim=-2, index=index)
            if self.onnx_safe_glu:
                projected = mlp[0](band)
                width = self.dim_inputs[index]
                outs.append(
                    projected[..., :width] * projected[..., width : 2 * width].sigmoid()
                )
            else:
                outs.append(mlp(band))
        return _binary_concat(outs, dim=-1)


def _slaney_mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int) -> Tensor:
    """
    Slaney-style mel filter bank, replicating ``librosa.filters.mel`` with default arguments (``htk=False``, ``norm="slaney"``, ``fmin=0``, ``fmax=sample_rate / 2``) in float64.

    :param sample_rate: Audio sample rate. :param n_fft: STFT size (bank spans ``n_fft // 2 + 1`` bins). :param n_mels: Number of mel bands. :return: Filter bank of shape ``[n_mels, n_fft // 2 + 1]``.
    """

    def hz_to_mel(freq: Tensor) -> Tensor:
        """
        Convert Hz to Slaney mels (linear below 1 kHz, log above).

        :param freq: Frequencies in Hz.
        :return: Frequencies in Slaney mels.
        """
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        log_region = freq >= min_log_hz
        mels = torch.where(
            log_region,
            min_log_mel + torch.log(freq.clamp(min=min_log_hz) / min_log_hz) / logstep,
            mels,
        )
        return mels

    def mel_to_hz(mels: Tensor) -> Tensor:
        """
        Convert Slaney mels back to Hz.

        :param mels: Frequencies in Slaney mels.
        :return: Frequencies in Hz.
        """
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        log_region = mels >= min_log_mel
        freqs = torch.where(
            log_region,
            min_log_hz * torch.exp(logstep * (mels - min_log_mel)),
            freqs,
        )
        return freqs

    fmax = sample_rate / 2
    n_freqs = 1 + n_fft // 2
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs, dtype=torch.float64)

    max_mel = hz_to_mel(torch.tensor([fmax], dtype=torch.float64))[0]
    mel_points = torch.linspace(0.0, float(max_mel), n_mels + 2, dtype=torch.float64)
    mel_f = mel_to_hz(mel_points)

    fdiff = mel_f[1:] - mel_f[:-1]
    ramps = mel_f[:, None] - fft_freqs[None, :]

    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights = weights * enorm[:, None]
    return weights


class _RoformerBase(ASSModel):
    """
    Shared base for RoFormer variants.
    """

    core_name = "_run_transformers"

    external_normalization = False

    sources: list[str]
    samplerate: int = 44100
    max_allowed_segment: float = 8.0

    def _init_common(
        self,
        *,
        dim: int,
        depth: int,
        stereo: bool,
        num_stems: int,
        time_transformer_depth: int,
        freq_transformer_depth: int,
        linear_transformer_depth: int,
        dim_head: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        norm_transformer_output: bool,
        skip_connection: bool,
        stft_n_fft: int,
        stft_hop_length: int,
        stft_win_length: int,
        stft_normalized: bool,
        zero_dc: bool,
    ) -> None:
        """
        Build the transformer trunk and record the STFT configuration.

        :param dim: Feature dimension. :param depth: Number of (time, frequency) transformer pairs. :param stereo: Whether audio is stereo. :param num_stems: Number of mask-estimator heads. :param time_transformer_depth: Blocks per time transformer. :param freq_transformer_depth: Blocks per frequency transformer. :param linear_transformer_depth: Unsupported; must be 0. :param dim_head: Dimension per head. :param heads: Number of heads.
        :param attn_dropout: Attention dropout probability. :param ff_dropout: Feed-forward dropout probability. :param norm_transformer_output: Whether to normalise transformer outputs. :param skip_connection: Sum earlier block outputs into each block. :param stft_n_fft: STFT size. :param stft_hop_length: STFT hop length. :param stft_win_length: STFT window length. :param stft_normalized: Whether ``torch.stft`` normalises. :param zero_dc: Zero the DC bin before the iSTFT.
        """
        if linear_transformer_depth != 0:
            raise ValidationError(
                "linear_transformer_depth != 0 is not supported (no shipped "
                "checkpoint uses linear attention)."
            )

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.skip_connection = skip_connection
        self.zero_dc = zero_dc

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            norm_output=norm_transformer_output,
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed,
                            **transformer_kwargs,
                        ),
                        Transformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed,
                            **transformer_kwargs,
                        ),
                    ]
                )
            )

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )
        self.stft_win_length = stft_win_length

        self.sources = [f"stem_{i}" for i in range(num_stems)]
        self.output_complement = False

    def configure_inference(
        self,
        *,
        sources: list[str],
        samplerate: int,
        segment_samples: int,
    ) -> None:
        """
        Attach the checkpoint-specific inference interface.

        :param sources: Output stem names, in order. :param samplerate: Sample rate the checkpoint was trained at. :param segment_samples: Training chunk length in samples.
        """
        if (
            isinstance(samplerate, bool)
            or not isinstance(samplerate, int)
            or samplerate <= 0
        ):
            raise ValidationError(
                f"samplerate must be a positive integer, got {samplerate}"
            )
        if (
            isinstance(segment_samples, bool)
            or not isinstance(segment_samples, int)
            or segment_samples <= 0
        ):
            raise ValidationError(
                f"segment_samples must be a positive integer, got {segment_samples}"
            )
        if len(sources) == self.num_stems:
            self.output_complement = False
        elif self.num_stems == 1 and len(sources) == 2:
            self.output_complement = True
        else:
            raise ValidationError(
                f"{len(sources)} source names for a model with "
                f"{self.num_stems} mask head(s); expected "
                f"{self.num_stems} or, for single-stem models, 2."
            )
        self.sources = list(sources)
        self.samplerate = samplerate
        self.max_allowed_segment = segment_samples / samplerate

        self.match_input_audio_length = True

    def _stft_window(self, device: torch.device) -> Tensor:
        """
        Hann window for the model's STFT, on the requested device.

        :param device: Device to allocate the window on.
        :return: Float32 Hann window of length ``stft_win_length``.
        """
        return torch.hann_window(self.stft_win_length, device=device)

    prefers_power_of_two_batch = True

    def prefill_inference_caches(self) -> None:
        """
        Materialize the rotary tables before CUDAGraph compilation.
        """
        segment_length = int(round(self.max_allowed_segment * self.samplerate))
        hop_length = int(self.stft_kwargs["hop_length"])
        sequence_lengths = (
            segment_length // hop_length + 1,
            len(self.band_split.dim_inputs),
        )
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        seen: set[int] = set()
        for transformer_pair in self.layers:
            for axis, transformer in enumerate(transformer_pair):
                for attention, _feed_forward in transformer.layers:
                    rotary = attention.rotary_embed
                    if rotary is None or id(rotary) in seen:
                        continue
                    rotary.prime_compiled(sequence_lengths[axis], device, dtype)
                    seen.add(id(rotary))

    def _run_transformers(self, x: Tensor) -> Tensor:
        """
        Axial attention over band features: each depth runs a transformer
        along time (per band) then along bands (per frame).

        :param x: Band features ``[batch, time, bands, dim]``.
        :return: Transformed features of the same shape.
        """
        store: list[Tensor] = []
        for i, (time_transformer, freq_transformer) in enumerate(self.layers):
            if self.skip_connection:
                for previous in store:
                    x = x + previous

            batch, frames, bands, dim = x.shape
            x = x.transpose(1, 2).reshape(batch * bands, frames, dim)
            x = time_transformer(x)
            x = x.view(batch, bands, frames, dim).transpose(1, 2)
            x = x.reshape(batch * frames, bands, dim)
            x = freq_transformer(x)
            x = x.view(batch, frames, bands, dim)

            if self.skip_connection:
                store.append(x)
        return x

    def _zero_dc_bin(self, stft: Tensor) -> Tensor:
        """
        Zero the complex STFT's DC bin through its real view (MPS lacks
        complex ``index_fill``).

        :param stft: Complex STFT ``[batch, frequencies, frames]``.
        :return: STFT with the DC bin zeroed.
        """
        dc_index = torch.zeros(1, dtype=torch.long, device=stft.device)
        real_stft = torch.view_as_real(stft).index_fill(1, dc_index, 0.0)
        return torch.view_as_complex(real_stft)

    def _finalize_output(self, recon: Tensor, mix: Tensor) -> Tensor:
        """
        Normalise the reconstruction to the ``apply_model`` output contract, adding the mixture-complement stem when configured.

        :param recon: Per-stem reconstruction ``[batch, stems, channels, T]``. :param mix: Input mixture ``[batch, channels, T_in]``. :return: Stems with the mixture-complement stem appended when configured.
        """
        if self.output_complement:
            complement = mix[..., : recon.shape[-1]].unsqueeze(1) - recon
            recon = torch.cat([recon, complement], dim=1)
        return recon

    def _check_channels(self, raw_audio: Tensor) -> None:
        """
        Validate the channel count against the model's stereo setting.

        :param raw_audio: Input mixture ``[batch, channels, samples]``.
        :raises ValidationError: On a channel/config mismatch.
        """
        channels = raw_audio.shape[1]
        if channels != self.audio_channels:
            raise ValidationError(
                f"Model expects {self.audio_channels} channel(s) "
                f"(stereo={self.stereo}), got {channels}."
            )


class BSRoformer(_RoformerBase):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        linear_transformer_depth: int = 0,
        freqs_per_bands: Iterable[int] = DEFAULT_FREQS_PER_BANDS,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        zero_dc: bool = True,
        mask_estimator_depth: int = 2,
        mlp_expansion_factor: int = 4,
        skip_connection: bool = False,
    ) -> None:
        """
        Band-Split RoFormer: fixed hand-designed frequency bands over the full-resolution spectrogram.

        :param dim: Feature dimension. :param depth: Number of (time, frequency) transformer pairs. :param stereo: Whether audio is stereo. :param num_stems: Number of mask-estimator heads. :param time_transformer_depth: Blocks per time transformer. :param freq_transformer_depth: Blocks per frequency transformer. :param linear_transformer_depth: Unsupported; must be 0. :param freqs_per_bands: STFT bins per band; must sum to all bins. :param dim_head: Dimension per head. :param heads: Number of heads.
        :param attn_dropout: Attention dropout probability. :param ff_dropout: Feed-forward dropout probability. :param stft_n_fft: STFT size. :param stft_hop_length: STFT hop length. :param stft_win_length: STFT window length. :param stft_normalized: Whether ``torch.stft`` normalises. :param zero_dc: Zero the DC bin before the iSTFT. :param mask_estimator_depth: Depth reference for the mask MLPs. :param mlp_expansion_factor: Mask-MLP hidden width multiplier. :param skip_connection: Sum earlier block outputs into each block.
        """
        super().__init__()
        self._init_common(
            dim=dim,
            depth=depth,
            stereo=stereo,
            num_stems=num_stems,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            linear_transformer_depth=linear_transformer_depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            norm_transformer_output=False,
            skip_connection=skip_connection,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
            stft_normalized=stft_normalized,
            zero_dc=zero_dc,
        )

        self.final_norm = RMSNorm(dim)

        freqs_per_bands = tuple(freqs_per_bands)
        n_freqs = stft_n_fft // 2 + 1
        if len(freqs_per_bands) < 2 or sum(freqs_per_bands) != n_freqs:
            raise ValidationError(
                f"freqs_per_bands must sum to {n_freqs} for n_fft={stft_n_fft}; "
                f"got sum {sum(freqs_per_bands)} over {len(freqs_per_bands)} bands."
            )

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in freqs_per_bands
        )
        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    mlp_hidden_layers=mask_estimator_depth - 1,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
                for _ in range(num_stems)
            ]
        )

    def forward(self, raw_audio: Tensor) -> Tensor:
        """
        Separate one mixture chunk.

        :param raw_audio: Mixture of shape ``[batch, channels, samples]``.
        :return: Stems of shape ``[batch, len(self.sources), channels,
            samples]``.
        """
        self._check_channels(raw_audio)
        batch, channels, _ = raw_audio.shape
        device = raw_audio.device

        audio = raw_audio.float().reshape(batch * channels, -1)
        window = self._stft_window(device)
        stft_repr = torch.stft(
            audio, **self.stft_kwargs, window=window, return_complex=True
        )
        stft_repr = torch.view_as_real(stft_repr)
        n_freqs, n_frames = stft_repr.shape[-3], stft_repr.shape[-2]

        stft_repr = (
            stft_repr.view(batch, channels, n_freqs, n_frames, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, n_freqs * channels, n_frames, 2)
        )

        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, n_frames, -1)

        x = x.type(self.band_split.to_features[0][1].weight.dtype)
        x = self.band_split(x)
        x = self._run_transformers(x)
        x = self.final_norm(x)

        masks = torch.stack([head(x) for head in self.mask_estimators], dim=1)

        masks = masks.view(batch, self.num_stems, n_frames, -1, 2).permute(
            0, 1, 3, 2, 4
        )

        stft_complex = torch.view_as_complex(stft_repr).unsqueeze(1)
        masks_complex = torch.view_as_complex(masks.float().contiguous())
        stft_out = stft_complex * masks_complex

        stft_out = (
            stft_out.view(batch, self.num_stems, n_freqs, channels, n_frames)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * self.num_stems * channels, n_freqs, n_frames)
        )
        if self.zero_dc:
            stft_out = self._zero_dc_bin(stft_out)

        recon = torch.istft(
            stft_out,
            **self.stft_kwargs,
            window=window,
            return_complex=False,
            length=audio.shape[-1],
        )
        recon = recon.view(batch, self.num_stems, channels, -1)

        return self._finalize_output(recon, raw_audio.float()).type(raw_audio.dtype)


class MelBandRoformer(_RoformerBase):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        linear_transformer_depth: int = 0,
        num_bands: int = 60,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        sample_rate: int = 44100,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        zero_dc: bool = True,
        mask_estimator_depth: int = 1,
        mlp_expansion_factor: int = 4,
        skip_connection: bool = False,
        match_input_audio_length: bool = False,
    ) -> None:
        """
        Mel-Band RoFormer: overlapping frequency bands derived from a Slaney mel filter bank instead of a hand-designed split.

        :param dim: Feature dimension. :param depth: Number of (time, frequency) transformer pairs. :param stereo: Whether audio is stereo. :param num_stems: Number of mask-estimator heads. :param time_transformer_depth: Blocks per time transformer. :param freq_transformer_depth: Blocks per frequency transformer. :param linear_transformer_depth: Unsupported; must be 0. :param num_bands: Number of mel bands. :param dim_head: Dimension per head. :param heads: Number of heads. :param attn_dropout: Attention dropout probability.
        :param ff_dropout: Feed-forward dropout probability. :param sample_rate: Sample rate used to place the mel bands. :param stft_n_fft: STFT size. :param stft_hop_length: STFT hop length. :param stft_win_length: STFT window length. :param stft_normalized: Whether ``torch.stft`` normalises. :param zero_dc: Zero the DC bin before the iSTFT. :param mask_estimator_depth: Depth reference for the mask MLPs. :param mlp_expansion_factor: Mask-MLP hidden width multiplier. :param skip_connection: Sum earlier block outputs into each block. :param match_input_audio_length: Pad the iSTFT output to the input audio length.
        """
        super().__init__()
        self._init_common(
            dim=dim,
            depth=depth,
            stereo=stereo,
            num_stems=num_stems,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            linear_transformer_depth=linear_transformer_depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            norm_transformer_output=True,
            skip_connection=skip_connection,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
            stft_normalized=stft_normalized,
            zero_dc=zero_dc,
        )
        self.match_input_audio_length = match_input_audio_length

        n_freqs = stft_n_fft // 2 + 1
        mel_filter_bank = _slaney_mel_filter_bank(sample_rate, stft_n_fft, num_bands)

        mel_filter_bank[0][0] = 1.0
        mel_filter_bank[-1, -1] = 1.0

        freqs_per_band = mel_filter_bank > 0
        if not bool(freqs_per_band.any(dim=0).all()):
            raise ValidationError(
                "Invalid mel banding: every frequency bin must be covered by "
                "at least one band."
            )

        repeated_freq_indices = torch.arange(n_freqs).repeat(num_bands, 1)
        freq_indices = repeated_freq_indices[freqs_per_band]
        if stereo:
            freq_indices = (freq_indices[:, None] * 2 + torch.arange(2)).flatten()

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = freqs_per_band.sum(dim=1)
        num_bands_per_freq = freqs_per_band.sum(dim=0)
        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        freqs_per_bands_with_complex = tuple(
            2 * int(f) * self.audio_channels for f in num_freqs_per_band.tolist()
        )
        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    mlp_hidden_layers=mask_estimator_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
                for _ in range(num_stems)
            ]
        )

    def forward(self, raw_audio: Tensor) -> Tensor:
        """
        Separate one mixture chunk.

        :param raw_audio: Mixture of shape ``[batch, channels, samples]``.
        :return: Stems of shape ``[batch, len(self.sources), channels,
            samples]``.
        """
        self._check_channels(raw_audio)
        batch, channels, raw_len = raw_audio.shape
        device = raw_audio.device
        istft_length = raw_len if self.match_input_audio_length else None

        audio = raw_audio.float().reshape(batch * channels, -1)
        window = self._stft_window(device)
        stft_repr = torch.stft(
            audio, **self.stft_kwargs, window=window, return_complex=True
        )
        stft_repr = torch.view_as_real(stft_repr)
        n_freqs, n_frames = stft_repr.shape[-3], stft_repr.shape[-2]

        stft_repr = (
            stft_repr.view(batch, channels, n_freqs, n_frames, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, n_freqs * channels, n_frames, 2)
        )

        x = stft_repr.index_select(1, self.freq_indices)

        x = x.permute(0, 2, 1, 3).reshape(batch, n_frames, -1)

        x = x.type(self.band_split.to_features[0][1].weight.dtype)
        x = self.band_split(x)
        x = self._run_transformers(x)

        masks = torch.stack([head(x) for head in self.mask_estimators], dim=1)

        masks = masks.view(batch, self.num_stems, n_frames, -1, 2).permute(
            0, 1, 3, 2, 4
        )
        masks = masks.float()

        scatter_index = self.freq_indices.view(1, 1, -1, 1, 1).expand(
            batch, self.num_stems, -1, n_frames, 2
        )
        masks_summed = torch.zeros(
            batch,
            self.num_stems,
            n_freqs * channels,
            n_frames,
            2,
            device=device,
            dtype=masks.dtype,
        ).scatter_add_(2, scatter_index, masks.contiguous())

        denom = self.num_bands_per_freq.repeat_interleave(channels).view(1, 1, -1, 1)
        masks_averaged = torch.view_as_complex(masks_summed) / denom.clamp(min=1e-8)

        stft_out = torch.view_as_complex(stft_repr).unsqueeze(1) * masks_averaged

        stft_out = (
            stft_out.view(batch, self.num_stems, n_freqs, channels, n_frames)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * self.num_stems * channels, n_freqs, n_frames)
        )
        if self.zero_dc:
            stft_out = self._zero_dc_bin(stft_out)

        recon = torch.istft(
            stft_out,
            **self.stft_kwargs,
            window=window,
            return_complex=False,
            length=istft_length,
        )
        recon = recon.view(batch, self.num_stems, channels, -1)

        return self._finalize_output(recon, raw_audio.float()).type(raw_audio.dtype)


_ARCHITECTURES: dict[str, type[_RoformerBase]] = {
    "bs_roformer": BSRoformer,
    "mel_band_roformer": MelBandRoformer,
}


def build_roformer(
    architecture: str,
    config: dict,
    *,
    sources: list[str],
    samplerate: int,
    segment_samples: int,
    state: dict | None = None,
) -> _RoformerBase:
    """
    Construct a RoFormer variant from registry metadata and (optionally) load a checkpoint into it.

    :param architecture: ``"bs_roformer"`` or ``"mel_band_roformer"``. :param config: Constructor kwargs from checkpoint metadata. :param sources: Output stem names, in order. :param samplerate: Sample rate the checkpoint operates at. :param segment_samples: Training chunk length in samples. :param state: Checkpoint state dict to load strictly, or ``None``. :return: The constructed model in eval mode.
    """
    klass = _ARCHITECTURES.get(architecture)
    if klass is None:
        raise ValidationError(
            f"Unknown roformer architecture {architecture!r}; expected one "
            f"of {sorted(_ARCHITECTURES)}."
        )
    model = klass(**config)
    model.configure_inference(
        sources=sources, samplerate=samplerate, segment_samples=segment_samples
    )
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model.eval()


from . import backends as _backends  # noqa: E402  (avoids an import cycle)

_backends.register_backend("roformer", build_roformer, _ARCHITECTURES)
