# Copyright (c) 2023 Phil Wang (lucidrains/BS-RoFormer)
# Copyright (c) 2024 Roman Solovyev (ZFTurbo/Music-Source-Separation-Training)
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# BS-RoFormer and Mel-Band RoFormer (Wang et al., ByteDance 2023,
# arXiv:2309.02612 / 2310.01809), reimplemented in plain PyTorch from the
# MIT-licensed reference lineage above. Community checkpoints are trained
# against that lineage, so every module attribute name, ``nn.Sequential``
# position, and parameter shape below is pinned to it — ``load_state_dict``
# with ``strict=True`` must keep accepting those checkpoints verbatim.
# Everything else (einops/beartype/rotary-embedding-torch/librosa
# dependencies, the training-loss branches) is deliberately not carried over.

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .exceptions import ValidationError

# The standard 62-band split over the 1025 STFT bins of n_fft=2048, from the
# BS-RoFormer reference implementation. Community checkpoints (including
# BS-RoFormer-SW) train against exactly this layout.
DEFAULT_FREQS_PER_BANDS: tuple[int, ...] = (
    *(2,) * 24,
    *(4,) * 12,
    *(12,) * 8,
    *(24,) * 8,
    *(48,) * 8,
    128,
    129,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Root-mean-square LayerNorm with a learnable gain.

        :param dim: Feature dimension to normalise over (last axis).
        """
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to unit RMS over the last axis and apply the gain.

        The norm is computed in float32 regardless of input dtype: under
        fp16 the squared-sum inside ``F.normalize`` can overflow to inf for
        large activations (this also matches autocast semantics, which run
        normalisation ops in fp32). MPS inference uses an equivalent fused
        Metal reduction; autograd and other devices retain the native path.

        :param x: Input of shape ``[..., dim]``.
        :return: Normalised tensor of the same shape and dtype.
        """
        if x.device.type == "mps" and not torch.is_grad_enabled():
            from .metal import metal_rms_norm

            return metal_rms_norm(x, self.gamma, self.scale)
        normed = F.normalize(x.float(), dim=-1) * self.scale * self.gamma.float()
        return normed.type(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        Rotary position embedding (RoPE, Su et al. 2021), matching the
        ``rotary-embedding-torch`` package the reference models were trained
        with: interleaved pair rotation with per-pair inverse frequencies.

        ``freqs`` is an ``nn.Parameter`` (frozen) because the reference
        package stores it that way — checkpoints therefore contain a
        ``rotary_embed.freqs`` entry under every attention path, and the
        parameter must exist here for ``strict=True`` loading.

        :param dim: Rotation dimensionality (the per-head dimension).
        :param theta: Base for the inverse-frequency spectrum.
        """
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        # Phase table cache, keyed by (seq_len, device). Chunked inference
        # calls every attention block with the same one or two sequence
        # lengths thousands of times; recomputing cos/sin per call measured
        # ~12% of the whole forward on CPU. Plain dict (not a buffer) so
        # ``state_dict`` stays checkpoint-identical.
        self._phase_cache: dict[tuple[int, torch.device], Tensor] = {}
        # Inductor does not generate native code for complex multiplication.
        # CUDA's compiled path uses this equivalent real-valued cos/sin table;
        # eager CPU/MPS/CUDA retain the faster complex representation above.
        self._rotation_cache: dict[tuple[int, torch.device], Tensor] = {}

    def _phases(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Complex rotation phases ``e^{i·pos·freq}`` for one sequence length.

        :param seq_len: Sequence length to build (or fetch) phases for.
        :param device: Device the phases must live on.
        :return: Complex64 tensor of shape ``[seq_len, dim // 2]``.
        """
        key = (seq_len, device)
        cached = self._phase_cache.get(key)
        if cached is None:
            # Always float32: rotation runs in fp32 regardless of model dtype
            # (and ``torch.polar`` has no half kernel — a model cast to fp16
            # casts ``freqs`` with it).
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            angles = positions[:, None] * self.freqs.to(
                device=device, dtype=torch.float32
            )
            cached = torch.polar(torch.ones_like(angles), angles)
            self._phase_cache[key] = cached
        return cached

    def _rotations(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Real-valued ``[cos, sin]`` rotation table for Inductor graphs.

        :param seq_len: Sequence length to build (or fetch) rotations for.
        :param device: Device the rotations must live on.
        :return: Float32 tensor of shape ``[seq_len, dim // 2, 2]``.
        """
        key = (seq_len, device)
        cached = self._rotation_cache.get(key)
        if cached is None:
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            angles = positions[:, None] * self.freqs.to(
                device=device, dtype=torch.float32
            )
            cached = torch.stack((angles.cos(), angles.sin()), dim=-1)
            self._rotation_cache[key] = cached
        return cached

    def _load_from_state_dict(self, *args: object, **kwargs: object) -> None:
        """
        Drop cached phases when weights (``freqs``) are replaced.

        :param args: Forwarded to ``nn.Module._load_from_state_dict``.
        :param kwargs: Forwarded to ``nn.Module._load_from_state_dict``.
        """
        self._phase_cache.clear()
        self._rotation_cache.clear()
        super()._load_from_state_dict(*args, **kwargs)

    def rotate_queries_or_keys(self, t: Tensor) -> Tensor:
        """
        Apply rotary rotation over the sequence axis of ``t``.

        Implemented as one complex multiply on the interleaved pairs: for a
        pair ``(x1, x2)`` and angle ``θ``, ``(x1 + i·x2) · e^{iθ}`` is
        exactly ``(x1·cosθ − x2·sinθ, x1·sinθ + x2·cosθ)`` — the reference
        package's interleaved rotate-half, in a third of the kernels.

        :param t: Queries or keys of shape ``[..., seq, dim]``.
        :return: Rotated tensor of the same shape and dtype.
        """
        seq_len = t.shape[-2]
        pairs = t.float().unflatten(-1, (-1, 2))
        if torch.compiler.is_compiling():
            rotations = self._rotations(seq_len, t.device)
            real, imaginary = pairs.unbind(dim=-1)
            cosine, sine = rotations.unbind(dim=-1)
            rotated = torch.stack(
                (
                    real * cosine - imaginary * sine,
                    real * sine + imaginary * cosine,
                ),
                dim=-1,
            )
            return rotated.flatten(-2).type(t.dtype)

        phases = self._phases(seq_len, t.device)
        complex_pairs = torch.view_as_complex(pairs.contiguous())
        return torch.view_as_real(complex_pairs * phases).flatten(-2).type(t.dtype)


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

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the MLP block.

        :param x: Input of shape ``[..., dim]``.
        :return: Output of the same shape.
        """
        return self.net(x)


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

    PyTorch 2.10's fused MPS SDPA is about 32% slower than explicit
    matmul/softmax/matmul at both RoFormer sequence shapes on an M2 Max.
    Evaluation therefore uses the explicit path on MPS; training and other
    devices retain native SDPA, including its dropout semantics.

    :param query: Queries of shape ``[batch, heads, sequence, dim]``.
    :param key: Keys with the same shape as ``query``.
    :param value: Values with the same shape as ``query``.
    :param scale: Query/key dot-product scale.
    :param dropout: Training dropout probability.
    :param training: Whether the containing attention module is training.
    :return: Attention output with the same shape as ``query``.
    """
    if query.device.type == "mps" and not training:
        weights = (query @ key.transpose(-1, -2)) * scale
        return weights.softmax(dim=-1) @ value
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=dropout if training else 0.0,
    )


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
        Pre-norm multi-head self-attention with per-head sigmoid gating,
        as used by both RoFormer variants.

        :param dim: Input/output feature dimension.
        :param heads: Number of attention heads.
        :param dim_head: Dimension per head.
        :param dropout: Attention/projection dropout (inactive in eval mode).
        :param rotary_embed: Shared rotary embedding module, or ``None`` to
            skip position rotation.
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.dropout = dropout

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Run gated multi-head attention over the sequence axis.

        :param x: Input of shape ``[batch, seq, dim]``.
        :return: Output of the same shape.
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

        out = _scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            dropout=self.dropout,
            training=self.training,
        )

        gates = self.to_gates(x)
        out = out * gates.transpose(1, 2).unsqueeze(-1).sigmoid()

        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.to_out(out)


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

        :param dim: Feature dimension.
        :param depth: Number of attention/FF pairs.
        :param dim_head: Dimension per attention head.
        :param heads: Number of attention heads.
        :param attn_dropout: Attention dropout probability.
        :param ff_dropout: Feed-forward dropout probability.
        :param ff_mult: Feed-forward hidden expansion factor.
        :param norm_output: Whether to RMS-normalise the stack output
            (``True`` in Mel-Band RoFormer; ``False`` in BS-RoFormer, which
            applies a single shared ``final_norm`` instead).
        :param rotary_embed: Shared rotary embedding for every block.
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
        for split_input, to_feature in zip(
            x.split(self.dim_inputs, dim=-1), self.to_features
        ):
            outs.append(to_feature(split_input))
        return torch.stack(outs, dim=-2)


def MLP(
    dim_in: int,
    dim_out: int,
    dim_hidden: int | None = None,
    hidden_layers: int = 0,
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    """
    Build a Linear/activation MLP as a flat ``nn.Sequential``.

    The two reference implementations interpret their ``depth`` argument
    differently (BS-RoFormer builds ``depth`` Linears, Mel-Band builds
    ``depth + 1``), so this helper takes the unambiguous hidden-layer count
    and each caller translates. The Sequential indices (Linear at even
    positions) are what checkpoint keys address — do not restructure.

    :param dim_in: Input feature dimension.
    :param dim_out: Output feature dimension.
    :param dim_hidden: Hidden feature dimension (defaults to ``dim_in``).
    :param hidden_layers: Number of hidden Linear layers between input and
        output.
    :param activation: Activation module class placed between Linears.
    :return: The assembled ``nn.Sequential``.
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

        :param dim: Input feature dimension.
        :param dim_inputs: Output width of each band (real/imag interleaved).
        :param mlp_hidden_layers: Hidden Linear count per band MLP (see
            ``MLP`` for the BS vs Mel-Band ``depth`` translation).
        :param mlp_expansion_factor: Hidden width multiplier over ``dim``.
        """
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = nn.ModuleList([])
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
        for band_features, mlp in zip(x.unbind(dim=-2), self.to_freqs):
            outs.append(mlp(band_features))
        return torch.cat(outs, dim=-1)


def _slaney_mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int) -> Tensor:
    """
    Slaney-style mel filter bank, replicating ``librosa.filters.mel`` with
    default arguments (``htk=False``, ``norm="slaney"``, ``fmin=0``,
    ``fmax=sample_rate / 2``) in float64.

    Mel-Band RoFormer derives its band layout from the *support pattern*
    (nonzero positions) of this matrix; the reference implementation computes
    it with librosa. Replicating the algorithm in float64 keeps the support
    pattern bit-identical without carrying the librosa dependency.

    :param sample_rate: Audio sample rate the model operates at.
    :param n_fft: STFT size (the bank spans ``n_fft // 2 + 1`` bins).
    :param n_mels: Number of mel bands.
    :return: Filter bank of shape ``[n_mels, n_fft // 2 + 1]`` (float64).
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

    # Slaney normalisation: scale each filter to constant energy per band.
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights = weights * enorm[:, None]
    return weights


class _RoformerBase(nn.Module):
    """
    Shared inference plumbing for the two RoFormer variants: STFT/iSTFT
    bookkeeping, the axial (time/frequency) transformer loop, and the
    ``apply_model`` interface contract (``sources`` / ``samplerate`` /
    ``max_allowed_segment`` attributes plus a ``[B, S, C, T]`` forward
    output, optionally adding a mixture-minus-prediction complement stem
    for single-stem checkpoints).
    """

    # Separator consults this: RoFormer checkpoints are trained on raw
    # (unnormalised) audio, so the Demucs-style outer mean/std normalisation
    # must be skipped for them.
    external_normalization = False

    # Inference-interface attributes (set by ``configure_inference`` or the
    # repository loader; class-level defaults keep bare constructions usable).
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

        :param dim: Feature dimension.
        :param depth: Number of (time, frequency) transformer pairs.
        :param stereo: Whether the model consumes stereo audio.
        :param num_stems: Number of mask-estimator heads.
        :param time_transformer_depth: Blocks per time transformer.
        :param freq_transformer_depth: Blocks per frequency transformer.
        :param linear_transformer_depth: Optional linear-attention blocks —
            unsupported (no shipped checkpoint uses them).
        :param dim_head: Attention head dimension.
        :param heads: Attention head count.
        :param attn_dropout: Attention dropout probability.
        :param ff_dropout: Feed-forward dropout probability.
        :param norm_transformer_output: Per-transformer output norm flag
            (the BS/Mel structural difference).
        :param skip_connection: Sum every earlier block's output into each
            block input (rarely used by community configs, but present).
        :param stft_n_fft: STFT size.
        :param stft_hop_length: STFT hop.
        :param stft_win_length: STFT window length.
        :param stft_normalized: Whether ``torch.stft`` normalises.
        :param zero_dc: Zero the DC bin before the iSTFT.
        :raises ValidationError: If ``linear_transformer_depth`` is nonzero.
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

        # One rotary embedding per axis, shared across every depth — the
        # reference models are built this way, and checkpoints repeat the
        # shared ``freqs`` under each attention path.
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

        # Interface defaults; overridden per checkpoint by the loader.
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

        For single-stem checkpoints, passing two source names (e.g.
        ``["vocals", "other"]``) enables the complement output: the second
        stem is computed as ``mixture - prediction`` per chunk, which under
        the linear overlap-add in ``apply_model`` equals the full-track
        mixture-minus — the standard way these checkpoints produce their
        second stem.

        :param sources: Output stem names, in order.
        :param samplerate: Sample rate the checkpoint was trained at.
        :param segment_samples: Training chunk length in samples; becomes
            ``max_allowed_segment`` (seconds) for the tiling in
            ``apply_model``.
        :raises ValidationError: If source count or numeric metadata is invalid.
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
        # Tiled inference in ``apply_model`` overlap-adds fixed-length chunks,
        # so every forward must return exactly the input chunk length. Mel-Band
        # reads this flag in its iSTFT; BS-RoFormer always matches input length.
        self.match_input_audio_length = True

    def _stft_window(self, device: torch.device) -> Tensor:
        """
        Hann window for the model's STFT, on the requested device.

        :param device: Device to allocate the window on.
        :return: Float32 Hann window of length ``stft_win_length``.
        """
        return torch.hann_window(self.stft_win_length, device=device)

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
        Zero the complex STFT's DC frequency bin through its real view.

        MPS does not implement ``index_fill`` for complex tensors. The
        equivalent operation on the trailing real/imaginary representation
        works on CPU, CUDA, and MPS while keeping the STFT itself complex.

        :param stft: Complex STFT ``[batch, frequencies, frames]``.
        :return: STFT with frequency bin zero set to zero.
        """
        dc_index = torch.zeros(1, dtype=torch.long, device=stft.device)
        real_stft = torch.view_as_real(stft).index_fill(1, dc_index, 0.0)
        return torch.view_as_complex(real_stft)

    def _finalize_output(self, recon: Tensor, mix: Tensor) -> Tensor:
        """
        Normalise the reconstruction to the ``apply_model`` output contract,
        adding the mixture-complement stem when configured.

        :param recon: Per-stem reconstruction ``[batch, stems, channels, T]``.
        :param mix: The input mixture ``[batch, channels, T_in]`` in the
            model's working dtype.
        :return: ``[batch, len(self.sources), channels, T]``.
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
        Band-Split RoFormer: fixed hand-designed frequency bands over the
        full-resolution spectrogram.

        Parameter names mirror the reference implementation so checkpoint
        config dicts construct this class directly.

        :param dim: Feature dimension.
        :param depth: Number of (time, frequency) transformer pairs.
        :param stereo: Whether the model consumes stereo audio.
        :param num_stems: Number of mask-estimator heads.
        :param time_transformer_depth: Blocks per time transformer.
        :param freq_transformer_depth: Blocks per frequency transformer.
        :param linear_transformer_depth: Unsupported; must be 0.
        :param freqs_per_bands: STFT bins per band; must sum to
            ``stft_n_fft // 2 + 1``.
        :param dim_head: Attention head dimension.
        :param heads: Attention head count.
        :param attn_dropout: Attention dropout probability.
        :param ff_dropout: Feed-forward dropout probability.
        :param stft_n_fft: STFT size.
        :param stft_hop_length: STFT hop.
        :param stft_win_length: STFT window length.
        :param stft_normalized: Whether ``torch.stft`` normalises.
        :param zero_dc: Zero the DC bin before the iSTFT.
        :param mask_estimator_depth: Reference ``depth`` for the mask MLPs
            (builds ``depth`` Linears, i.e. ``depth - 1`` hidden layers).
        :param mlp_expansion_factor: Mask-MLP hidden width multiplier.
        :param skip_connection: Sum earlier block outputs into each block.
        :raises ValidationError: If the band widths don't cover the STFT
            bins exactly.
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
                    # Reference BS ``MLP`` builds ``depth`` Linear layers.
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

        # STFT runs in float32 regardless of model dtype: complex-half
        # support is incomplete across backends, and the STFT is a
        # negligible fraction of the compute.
        audio = raw_audio.float().reshape(batch * channels, -1)
        window = self._stft_window(device)
        stft_repr = torch.stft(
            audio, **self.stft_kwargs, window=window, return_complex=True
        )
        stft_repr = torch.view_as_real(stft_repr)
        n_freqs, n_frames = stft_repr.shape[-3], stft_repr.shape[-2]

        # Interleave channels into frequency (f-major, channel-minor), the
        # reference layout for band splitting: 'b s f t c -> b (f s) t c'.
        stft_repr = (
            stft_repr.view(batch, channels, n_freqs, n_frames, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, n_freqs * channels, n_frames, 2)
        )

        # 'b f t c -> b t (f c)'
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, n_frames, -1)

        x = x.type(self.band_split.to_features[0][1].weight.dtype)
        x = self.band_split(x)
        x = self._run_transformers(x)
        x = self.final_norm(x)

        masks = torch.stack([head(x) for head in self.mask_estimators], dim=1)
        # 'b n t (f c) -> b n f t c'
        masks = masks.view(batch, self.num_stems, n_frames, -1, 2).permute(
            0, 1, 3, 2, 4
        )

        stft_complex = torch.view_as_complex(stft_repr).unsqueeze(1)
        masks_complex = torch.view_as_complex(masks.float().contiguous())
        stft_out = stft_complex * masks_complex

        # 'b n (f s) t -> (b n s) f t'
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
        Mel-Band RoFormer: overlapping frequency bands derived from a
        Slaney mel filter bank instead of a hand-designed split.

        Parameter names mirror the reference implementation so checkpoint
        config dicts construct this class directly.

        :param dim: Feature dimension.
        :param depth: Number of (time, frequency) transformer pairs.
        :param stereo: Whether the model consumes stereo audio.
        :param num_stems: Number of mask-estimator heads.
        :param time_transformer_depth: Blocks per time transformer.
        :param freq_transformer_depth: Blocks per frequency transformer.
        :param linear_transformer_depth: Unsupported; must be 0.
        :param num_bands: Number of mel bands.
        :param dim_head: Attention head dimension.
        :param heads: Attention head count.
        :param attn_dropout: Attention dropout probability.
        :param ff_dropout: Feed-forward dropout probability.
        :param sample_rate: Sample rate used to place the mel bands.
        :param stft_n_fft: STFT size.
        :param stft_hop_length: STFT hop.
        :param stft_win_length: STFT window length.
        :param stft_normalized: Whether ``torch.stft`` normalises.
        :param zero_dc: Zero the DC bin before the iSTFT.
        :param mask_estimator_depth: Reference ``depth`` for the mask MLPs
            (builds ``depth + 1`` Linears, i.e. ``depth`` hidden layers).
        :param mlp_expansion_factor: Mask-MLP hidden width multiplier.
        :param skip_connection: Sum earlier block outputs into each block.
        :param match_input_audio_length: Pad the iSTFT output to the exact
            input length (reference flag; shipped chunk sizes are
            hop-divisible so the lengths already match).
        :raises ValidationError: If a frequency bin is covered by no band.
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
        # Reference quirks, kept verbatim: force the DC bin into the first
        # band and the Nyquist bin into the last (their filter weights round
        # to zero on some platforms, which would leave bins uncovered).
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
            # 'f -> (f s)' with per-channel offsets: bin index in the
            # channel-interleaved frequency axis.
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
                    # Reference Mel ``MLP`` builds ``depth + 1`` Linear layers.
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

        # 'b s f t c -> b (f s) t c'
        stft_repr = (
            stft_repr.view(batch, channels, n_freqs, n_frames, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, n_freqs * channels, n_frames, 2)
        )

        # Gather the (overlapping) per-band bins in one indexed read.
        x = stft_repr.index_select(1, self.freq_indices)
        # 'b f t c -> b t (f c)'
        x = x.permute(0, 2, 1, 3).reshape(batch, n_frames, -1)

        x = x.type(self.band_split.to_features[0][1].weight.dtype)
        x = self.band_split(x)
        x = self._run_transformers(x)

        masks = torch.stack([head(x) for head in self.mask_estimators], dim=1)
        # 'b n t (f c) -> b n f t c'
        masks = masks.view(batch, self.num_stems, n_frames, -1, 2).permute(
            0, 1, 3, 2, 4
        )
        masks = masks.float()

        # Overlapping bands each predict a mask for their bins; scatter-add
        # the per-band masks back onto the bin axis and divide by the number
        # of covering bands. Scatter runs on the real view (works on every
        # backend; complex scatter_add is CUDA/CPU-only).
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

        # 'b n (f s) t -> (b n s) f t'
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
    Construct a RoFormer variant from registry metadata and (optionally)
    load a checkpoint into it.

    :param architecture: ``"bs_roformer"`` or ``"mel_band_roformer"``.
    :param config: Constructor kwargs, as stored in ``metadata.json``
        (mirrors the reference config-file ``model:`` section).
    :param sources: Output stem names (see
        ``_RoformerBase.configure_inference`` for the single-stem
        complement convention).
    :param samplerate: Sample rate the checkpoint operates at.
    :param segment_samples: Training chunk length in samples.
    :param state: Checkpoint state dict to load (strict), or ``None``.
    :return: The constructed (and loaded) model in eval mode.
    :raises ValidationError: For an unknown architecture name.
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
