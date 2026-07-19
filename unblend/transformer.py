# Copyright (c) 2019-present, Meta, Inc.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Simon Rouard.

import math
import random
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def create_sin_embedding(
    length: int,
    dim: int,
    shift: int = 0,
    device: str = "cpu",
    max_period: float = 10000,
) -> torch.Tensor:
    """
    Create sinusoidal positional embedding in TBC format.

    :param length: Sequence length
    :param dim: Embedding dimension (must be even)
    :param shift: Position offset
    :param device: Device to create tensor on
    :param max_period: Maximum period for sinusoidal encoding
    :return: Positional embedding tensor of shape (length, 1, dim)
    """
    # We aim for TBC format
    assert dim % 2 == 0
    # Force FP32 for numerical stability — exponentiation of max_period overflows in FP16
    with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
        pos = shift + torch.arange(length, device=device, dtype=torch.float32).view(
            -1, 1, 1
        )
        half_dim = dim // 2
        adim = torch.arange(dim // 2, device=device, dtype=torch.float32).view(1, 1, -1)
        phase = pos / (max_period ** (adim / (half_dim - 1)))
        return torch.cat(
            [
                torch.cos(phase),
                torch.sin(phase),
            ],
            dim=-1,
        )


@torch.no_grad()
def create_2d_sin_embedding(
    d_model: int,
    height: int,
    width: int,
    device: str = "cpu",
    max_period: float = 10000,
) -> torch.Tensor:
    """
    Create 2D sinusoidal positional embedding.

    :param d_model: Dimension of the model (must be divisible by 4)
    :param height: Height of the positions
    :param width: Width of the positions
    :param device: Device to create tensor on
    :param max_period: Maximum period for sinusoidal encoding
    :return: Positional embedding tensor of shape (1, d_model, height, width)
    :raises ValueError: If d_model is not divisible by 4
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    # Force FP32 for numerical stability — exp/sin/cos of large values overflow in FP16
    with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
        pe = torch.zeros(d_model, height, width, dtype=torch.float32, device=device)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2, dtype=torch.float32, device=device)
            * -(math.log(max_period) / d_model)
        )
        pos_w = torch.arange(0.0, width, dtype=torch.float32, device=device).unsqueeze(
            1
        )
        pos_h = torch.arange(0.0, height, dtype=torch.float32, device=device).unsqueeze(
            1
        )
        pe[0:d_model:2, :, :] = (
            torch.sin(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[1:d_model:2, :, :] = (
            torch.cos(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[d_model::2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )
        pe[d_model + 1 :: 2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )

        return pe[None, :]


@torch.no_grad()
def create_sin_embedding_cape(
    length: int,
    dim: int,
    batch_size: int,
    mean_normalize: bool,
    device: str = "cpu",
    max_period: float = 10000.0,
) -> torch.Tensor:
    """
    Create sinusoidal CAPE positional embedding. The training-time CAPE
    augmentation (global/local shifts, scaling) was removed with the rest of
    the training code; this is the inference (un-augmented) variant only.

    :param length: Sequence length
    :param dim: Embedding dimension (must be even)
    :param batch_size: Batch size
    :param mean_normalize: Whether to mean-normalize positions
    :param device: Device to create tensor on
    :param max_period: Maximum period for sinusoidal encoding
    :return: Positional embedding tensor of shape (length, batch_size, dim)
    """
    # We aim for TBC format
    assert dim % 2 == 0
    # Force FP32 for numerical stability
    with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
        pos = torch.arange(length, dtype=torch.float32).view(-1, 1, 1)
        pos = pos.repeat(1, batch_size, 1)  # (length, batch_size, 1)
        if mean_normalize:
            pos -= torch.nanmean(pos, dim=0, keepdim=True)

        pos = pos.to(device)

        half_dim = dim // 2
        adim = torch.arange(dim // 2, device=device, dtype=torch.float32).view(1, 1, -1)
        phase = pos / (max_period ** (adim / (half_dim - 1)))
        return torch.cat(
            [
                torch.cos(phase),
                torch.sin(phase),
            ],
            dim=-1,
        )


class ScaledEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
        boost: float = 3.0,
    ) -> None:
        """
        Embedding with a learnable scale factor applied via boost.

        :param num_embeddings: Size of the embedding dictionary
        :param embedding_dim: Size of each embedding vector
        :param scale: Initial scale for embedding weights
        :param boost: Multiplicative boost applied during forward pass
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data *= scale / boost
        self.boost = boost

    @property
    def weight(self) -> torch.Tensor:
        """
        Boost-scaled embedding matrix.

        :return: ``embedding.weight * boost``.
        """
        return self.embedding.weight * self.boost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings and apply boost scaling.

        :param x: Input indices tensor
        :return: Scaled embedding tensor
        """
        return self.embedding(x) * self.boost


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(
        self, channels: int, init: float = 0, channel_last: bool = False
    ) -> None:
        """
        Initialize learnable diagonal rescaling for residual outputs.

        :param channels: Number of channels to scale
        :param init: Initial value for scale parameters
        :param channel_last: If False, expects (B, C, T) tensors; if True, expects (B, T, C)
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned diagonal scaling to input.

        :param x: Input tensor
        :return: Scaled tensor
        """
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


class MyGroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group normalization on (B, T, C) input.

        :param x: Input tensor of shape (B, T, C)
        :return: Normalized tensor of shape (B, T, C)
        """
        x = x.transpose(1, 2)
        return super().forward(x).transpose(1, 2)


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Any = F.relu,
        group_norm: int = 0,
        norm_first: bool = False,
        norm_out: bool = False,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        device: Any = None,
        dtype: Any = None,
        batch_first: bool = False,
    ) -> None:
        """
        Transformer encoder layer with optional group norm, layer scale, and norm_out.

        :param d_model: Model dimension
        :param nhead: Number of attention heads
        :param dim_feedforward: Feedforward hidden dimension
        :param dropout: Dropout rate
        :param activation: Activation function
        :param group_norm: Number of groups for group norm (0 to disable)
        :param norm_first: If True, apply norm before attention/FF blocks
        :param norm_out: If True and norm_first, apply output normalization
        :param layer_norm_eps: Epsilon for layer normalization
        :param layer_scale: If True, use LayerScale on residual outputs
        :param init_values: Initial values for LayerScale
        :param device: Device for parameters
        :param dtype: Data type for parameters
        :param batch_first: If True, input is (B, T, C) instead of (T, B, C)
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        if group_norm:
            self.norm1 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm2 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )

        self.norm_out = None
        if self.norm_first and norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)
        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer encoder layer.

        :param src: Source tensor of shape (B, T, C) (all instances are built
            with ``batch_first=True``)
        :param src_mask: Attention mask tensor
        :param src_key_padding_mask: Key padding mask tensor
        :return: Transformed tensor of same shape as src
        """
        x = src

        if self.norm_first:
            x = x + self.gamma_1(
                self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            )
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))

            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(
                x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask))
            )
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Any = F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: bool = False,
        norm_out: bool = False,
        device: Any = None,
        dtype: Any = None,
        batch_first: bool = False,
    ) -> None:
        """
        Cross-attention transformer encoder layer with optional group norm and layer scale.

        :param d_model: Model dimension
        :param nhead: Number of attention heads
        :param dim_feedforward: Feedforward hidden dimension
        :param dropout: Dropout rate
        :param activation: Activation function or string name
        :param layer_norm_eps: Epsilon for layer normalization
        :param layer_scale: If True, use LayerScale on residual outputs
        :param init_values: Initial values for LayerScale
        :param norm_first: If True, apply norm before attention/FF blocks
        :param group_norm: If True, use group norm instead of layer norm
        :param norm_out: If True and norm_first, apply output normalization
        :param device: Device for parameters
        :param dtype: Data type for parameters
        :param batch_first: If True, input is (B, T, C) instead of (T, B, C)
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.cross_attn: nn.Module
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1: nn.Module
        self.norm2: nn.Module
        self.norm3: nn.Module
        if group_norm:
            self.norm1 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm2 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm3 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first and norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention between query and key sequences.

        :param q: Query tensor of shape (B, T, C)
        :param k: Key tensor of shape (B, S, C)
        :param mask: Attention mask tensor of shape (T, S)
        :return: Transformed tensor of same shape as q
        """
        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

    def _ca_block(
        self, q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Cross-attention block.

        :param q: Query tensor
        :param k: Key/value tensor
        :param attn_mask: Optional attention mask
        :return: Cross-attended tensor with dropout applied
        """
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward block.

        :param x: Input tensor
        :return: Transformed tensor with dropout applied
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_activation_fn(self, activation: str) -> Callable:
        """
        Return the activation function corresponding to the given name.

        :param activation: Name of activation function ("relu" or "gelu")
        :return: The activation function
        :raises RuntimeError: If activation name is not recognized
        """
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# ----------------- MULTI-BLOCKS MODELS: -----------------------


class CrossTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        emb: str = "sin",
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 6,
        cross_first: bool = False,
        dropout: float = 0.0,
        max_positions: int = 1000,
        norm_in: bool = True,
        norm_in_group: bool = False,
        group_norm: int = 0,
        norm_first: bool = False,
        norm_out: bool = False,
        max_period: float = 10000.0,
        weight_decay: float = 0.0,
        lr: float | None = None,
        layer_scale: bool = False,
        gelu: bool = True,
        sin_random_shift: int = 0,
        weight_pos_embed: float = 1.0,
        cape_mean_normalize: bool = True,
        cape_augment: bool = True,
        cape_glob_loc_scale: list[float] = [5000.0, 1.0, 1.4],
    ) -> None:
        """
        Cross-transformer encoder alternating self-attention and cross-attention layers.

        :param dim: Model dimension
        :param emb: Positional embedding type ("sin", "cape", or "scaled")
        :param hidden_scale: Feedforward hidden dim multiplier
        :param num_heads: Number of attention heads
        :param num_layers: Number of transformer layers
        :param cross_first: If True, start with cross-attention layer
        :param dropout: Dropout rate
        :param max_positions: Maximum sequence length for scaled embeddings
        :param norm_in: If True, apply LayerNorm to inputs
        :param norm_in_group: If True, use GroupNorm for input normalization
        :param group_norm: Number of groups for group norm (0 to disable)
        :param norm_first: If True, apply norm before attention/FF blocks
        :param norm_out: If True and norm_first, apply output normalization
        :param max_period: Maximum period for sinusoidal encoding
        :param weight_decay: Weight decay for optimizer
        :param lr: Learning rate override (None to use default)
        :param layer_scale: If True, use LayerScale on residual outputs
        :param gelu: If True, use GELU activation; otherwise ReLU
        :param sin_random_shift: Maximum random shift for sinusoidal embeddings
        :param weight_pos_embed: Weight for positional embedding contribution
        :param cape_mean_normalize: Whether to mean-normalize CAPE positions
        :param cape_augment: Whether to augment CAPE positions
        :param cape_glob_loc_scale: CAPE global/local scale parameters
        """
        super().__init__()
        assert dim % num_heads == 0

        hidden_dim = int(dim * hidden_scale)

        self.num_layers = num_layers
        # classic parity = 1 means that if idx%2 == 1 there is a
        # classical encoder else there is a cross encoder
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_decay = weight_decay
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift
        if emb == "cape":
            self.cape_mean_normalize = cape_mean_normalize
            self.cape_augment = cape_augment
            self.cape_glob_loc_scale = cape_glob_loc_scale
        if emb == "scaled":
            self.position_embeddings = ScaledEmbedding(max_positions, dim, scale=0.2)

        self.lr = lr

        activation: Any = F.gelu if gelu else F.relu

        self.norm_in: nn.Module
        self.norm_in_t: nn.Module
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        # spectrogram layers
        self.layers = nn.ModuleList()
        # temporal layers
        self.layers_t = nn.ModuleList()

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
            "batch_first": True,
        }

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(MyTransformerEncoderLayer(**kwargs_common))
                self.layers_t.append(MyTransformerEncoderLayer(**kwargs_common))

            else:
                self.layers.append(CrossTransformerEncoderLayer(**kwargs_common))

                self.layers_t.append(CrossTransformerEncoderLayer(**kwargs_common))

        # Positional embedding caches keyed by shape/device/dtype.
        self._pos_emb_2d_cache: dict[
            tuple[int, int, int, torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._pos_emb_t_cache: dict[
            tuple[int, int, torch.device, torch.dtype], torch.Tensor
        ] = {}

    def _cached_pos_emb_2d(
        self, C: int, Fr: int, T1: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return the 2D sin positional embedding pre-permuted to
        ``(1, T1*Fr, C)``, memoised by ``(C, Fr, T1, device, dtype)``.

        :param C: Channel dimension.
        :param Fr: Frequency dimension.
        :param T1: Time dimension.
        :param device: Device the embedding lives on.
        :param dtype: Dtype the embedding matches.
        :return: Pre-shaped embedding tensor.
        """
        key = (C, Fr, T1, device, dtype)
        emb = self._pos_emb_2d_cache.get(key)
        if emb is None:
            base = create_2d_sin_embedding(C, Fr, T1, device, self.max_period)
            emb = base.permute(0, 3, 2, 1).reshape(1, T1 * Fr, C).to(dtype)
            self._pos_emb_2d_cache[key] = emb
        return emb

    def _cached_pos_emb_t(
        self, T2: int, C: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """
        Return the 1D sin positional embedding pre-permuted to
        ``(1, T2, C)``, memoised by ``(T2, C, device, dtype)``. Only the
        deterministic ``sin`` mode without random shift is safe to cache;
        every other mode returns ``None`` so the caller falls back to
        recomputing per call.

        :param T2: Sequence length (time dimension).
        :param C: Channel dimension.
        :param device: Device the embedding lives on.
        :param dtype: Dtype the embedding matches.
        :return: Pre-shaped embedding tensor, or ``None`` for non-cacheable modes.
        """
        if self.emb != "sin" or self.sin_random_shift > 0:
            return None
        key = (T2, C, device, dtype)
        emb = self._pos_emb_t_cache.get(key)
        if emb is None:
            base = create_sin_embedding(
                T2, C, shift=0, device=device, max_period=self.max_period
            )
            emb = base.permute(1, 0, 2).contiguous().to(dtype)
            self._pos_emb_t_cache[key] = emb
        return emb

    def forward(
        self, x: torch.Tensor, xt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through alternating self-attention and cross-attention layers.

        :param x: Spectrogram tensor of shape (B, C, Fr, T1)
        :param xt: Temporal tensor of shape (B, C, T2)
        :return: Tuple of transformed (spectrogram, temporal) tensors
        """
        B, C, Fr, T1 = x.shape
        pos_emb_2d = self._cached_pos_emb_2d(C, Fr, T1, x.device, x.dtype)
        x = x.permute(0, 3, 2, 1).reshape(B, T1 * Fr, C)  # "b c fr t1 -> b (t1 fr) c"
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B, C, T2 = xt.shape
        xt = xt.permute(0, 2, 1)  # "b c t2 -> b t2 c"
        cached_t = self._cached_pos_emb_t(T2, C, x.device, xt.dtype)
        if cached_t is not None:
            xt = self.norm_in_t(xt)
            xt = xt + self.weight_pos_embed * cached_t
        else:
            pos_emb = self._get_pos_embedding(T2, B, C, x.device)
            pos_emb = pos_emb.permute(1, 0, 2)  # "t2 b c -> b t2 c"
            xt = self.norm_in_t(xt)
            xt = xt + self.weight_pos_embed * pos_emb.to(xt.dtype)

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = x.reshape(B, T1, Fr, C).permute(0, 3, 2, 1)  # "b (t1 fr) c -> b c fr t1"
        xt = xt.permute(0, 2, 1)  # "b t2 c -> b c t2"
        return x, xt

    def _get_pos_embedding(
        self, T: int, B: int, C: int, device: torch.device | str
    ) -> torch.Tensor:
        """
        Compute positional embedding based on the configured embedding type.

        :param T: Sequence length
        :param B: Batch size
        :param C: Embedding dimension
        :param device: Device to create tensor on
        :return: Positional embedding tensor
        """
        if self.emb == "sin":
            shift = random.randrange(self.sin_random_shift + 1)
            pos_emb = create_sin_embedding(
                T, C, shift=shift, device=device, max_period=self.max_period
            )
        elif self.emb == "cape":
            pos_emb = create_sin_embedding_cape(
                T,
                C,
                B,
                device=device,
                max_period=self.max_period,
                mean_normalize=self.cape_mean_normalize,
            )

        elif self.emb == "scaled":
            pos = torch.arange(T, device=device)
            pos_emb = self.position_embeddings(pos)[:, None]

        else:
            raise ValueError(
                f"Unknown positional embedding type '{self.emb}'. "
                "Expected one of: 'sin', 'cape', 'scaled'."
            )

        return pos_emb
