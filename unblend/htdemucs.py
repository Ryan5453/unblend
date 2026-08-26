# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backends import ASSModel
from .blocks import (
    HDecLayer,
    HEncLayer,
    MultiWrap,
    ScaledEmbedding,
    ispectro,
    pad1d,
    rescale_module,
    spectro,
)
from .exceptions import ValidationError
from .transformer import CrossTransformerEncoder


class HTDemucs(ASSModel):
    """
    Hybrid spectrogram/waveform Demucs.
    """

    def __init__(
        self,
        sources: list[str],
        audio_channels: int = 2,
        channels: int = 48,
        channels_time: int | None = None,
        growth: int = 2,
        nfft: int = 4096,
        cac: bool = True,
        depth: int = 4,
        rewrite: bool = True,
        multi_freqs: list[int] | None = None,
        multi_freqs_depth: int = 3,
        freq_emb: float = 0.2,
        emb_scale: int = 10,
        emb_smooth: bool = True,
        kernel_size: int = 8,
        time_stride: int = 2,
        stride: int = 4,
        context: int = 1,
        context_enc: int = 0,
        norm_starts: int = 4,
        norm_groups: int = 4,
        dconv_mode: int = 1,
        dconv_depth: int = 2,
        dconv_comp: int = 8,
        dconv_init: float = 1e-3,
        bottom_channels: int = 0,
        t_layers: int = 5,
        t_emb: str = "sin",
        t_hidden_scale: float = 4.0,
        t_heads: int = 8,
        t_dropout: float = 0.0,
        t_max_positions: int = 10000,
        t_norm_in: bool = True,
        t_norm_in_group: bool = False,
        t_group_norm: bool = False,
        t_norm_first: bool = True,
        t_norm_out: bool = True,
        t_max_period: float = 10000.0,
        t_weight_decay: float = 0.0,
        t_lr: float | None = None,
        t_layer_scale: bool = True,
        t_gelu: bool = True,
        t_weight_pos_embed: float = 1.0,
        t_sin_random_shift: int = 0,
        t_cape_mean_normalize: bool = True,
        t_cape_augment: bool = False,
        t_cape_glob_loc_scale: list[float] = [5000.0, 1.0, 1.4],
        t_cross_first: bool = False,
        rescale: float = 0.1,
        samplerate: int = 44100,
        segment: int = 10,
    ) -> None:
        """
        Initialize the model.

        :param sources: Output stem names.
        :param audio_channels: Input/output audio channels.
        :param channels: Initial hidden channels.
        :param channels_time: Separate channel count for the time branch.
        :param growth: Channel growth per layer.
        :param nfft: STFT size.
        :param cac: Complex-as-channels output decoding.
        :param depth: Encoder/decoder depth.
        :param rewrite: Add a 1x1 conv rewrite to each layer.
        :param multi_freqs: Frequency band ratios for MultiWrap.
        :param multi_freqs_depth: Outer layers wrapped with MultiWrap.
        :param freq_emb: Frequency embedding weight (0 disables).
        :param emb_scale: Embedding scale (learning-rate equivalent).
        :param emb_smooth: Initialize the embedding smoothly over frequency.
        :param kernel_size: Encoder/decoder kernel size.
        :param time_stride: Stride of the final time layer after the merge.
        :param stride: Encoder/decoder stride.
        :param context: 1x1 conv context in the decoder.
        :param context_enc: 1x1 conv context in the encoder.
        :param norm_starts: Layer index where GroupNorm starts.
        :param norm_groups: GroupNorm group count.
        :param dconv_mode: 1 encoder DConv, 2 decoder, 3 both.
        :param dconv_depth: DConv residual branch depth.
        :param dconv_comp: DConv branch channel compression.
        :param dconv_init: Initial LayerScale value for DConv.
        :param bottom_channels: 1x1 conv bottleneck width before the transformer.
        :param t_layers: Transformer layers per branch.
        :param t_emb: Positional embedding type ("sin", "cape", "scaled").
        :param t_hidden_scale: Transformer FFN hidden multiplier.
        :param t_heads: Attention heads.
        :param t_dropout: Transformer dropout.
        :param t_max_positions: Max positions for "scaled" embeddings.
        :param t_norm_in: Normalize before positional embedding.
        :param t_norm_in_group: Use GroupNorm for the input norm.
        :param t_group_norm: Encoder layers use GroupNorm over all timesteps.
        :param t_norm_first: Pre-norm transformer layout.
        :param t_norm_out: Normalize each layer output.
        :param t_max_period: Sinusoidal embedding period.
        :param t_weight_decay: Transformer weight decay (training only).
        :param t_lr: Transformer learning rate (training only).
        :param t_layer_scale: Enable LayerScale.
        :param t_gelu: GELU activation, else ReLU.
        :param t_weight_pos_embed: Positional embedding weight.
        :param t_sin_random_shift: Random shift of the sinusoidal embedding.
        :param t_cape_mean_normalize: CAPE position normalization.
        :param t_cape_augment: CAPE augmentation (always False at inference).
        :param t_cape_glob_loc_scale: CAPE scale parameters.
        :param t_cross_first: Cross-attention first in each layer pair.
        :param rescale: Conv weight rescale factor (0 disables).
        :param samplerate: Audio sample rate in Hz.
        :param segment: Training segment length in seconds.
        """
        super().__init__()
        self.cac = cac
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.max_allowed_segment = segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.freq_emb = None

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=dconv_mode & 1,
                    context=context_enc,
                    empty=last_freq,
                    **kwt,
                )
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec,
            )
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=dconv_mode & 2,
                    empty=last_freq,
                    last=index == 0,
                    context=context,
                    **kwt,
                )
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )

            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
            )
        else:
            self.crosstransformer = None

    def _spec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the STFT spectrogram of the input signal.

        :param x: Input waveform tensor
        :return: Complex spectrogram tensor
        """
        hl = self.hop_length
        nfft = self.nfft

        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z: torch.Tensor, length: int) -> torch.Tensor:
        """
        Inverse STFT to reconstruct waveform from spectrogram.

        :param z: Complex spectrogram tensor
        :param length: Desired output length in samples
        :return: Reconstructed waveform tensor
        """
        hl = self.hop_length
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad : pad + length]
        return x

    def _magnitude(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude of the spectrogram, or reshape complex to channels if CaC.

        :param z: Complex spectrogram tensor
        :return: Magnitude or CaC-reshaped tensor
        """

        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Convert CaC mask output back to complex spectrogram.

        :param z: Original complex spectrogram (ignored in CaC mode)
        :param m: Mask or full spectrogram in CaC format
        :return: Complex spectrogram tensor
        """
        if not self.cac:
            return z[:, None] * m

        B, S, _C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        return torch.view_as_complex(out.contiguous())

    def valid_length(self, length: int) -> int:
        """
        Return a length that is appropriate for evaluation.

        :param length: Requested input length in samples
        :return: Training length for consistent segment processing
        :raises ValidationError: If length exceeds the training length
        """

        training_length = int(round(self.max_allowed_segment * self.samplerate))
        if training_length < length:
            raise ValidationError(
                f"Given length {length} is longer than "
                f"training length {training_length}"
            )
        return training_length

    def _cached_freq_emb(
        self, num_freqs: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return the frequency-positional embedding pre-shaped for broadcast,
        memoised by ``(num_freqs, device, dtype)``.

        :param num_freqs: Number of frequency bins (``Fq``).
        :param device: Device the embedding should live on.
        :param dtype: Dtype the embedding should match.
        :return: Tensor of shape ``(1, C, Fq, 1)`` ready to add to the encoder input.
        """
        cache = getattr(self, "_freq_emb_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_freq_emb_cache", cache)
        key = (num_freqs, device, dtype)
        emb = cache.get(key)
        if emb is None:
            frs = torch.arange(num_freqs, device=device)
            emb = self.freq_emb(frs).t()[None, :, :, None]
            if emb.dtype != dtype:
                emb = emb.to(dtype)
            cache[key] = emb
        return emb

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        """
        Load weights and invalidate the memoised frequency embedding —
        otherwise a reload into an already-used instance keeps serving the
        old weights' embedding.

        :param args: Forwarded to ``nn.Module._load_from_state_dict``.
        :param kwargs: Forwarded to ``nn.Module._load_from_state_dict``.
        :return: None.
        """
        cache = getattr(self, "_freq_emb_cache", None)
        if cache:
            cache.clear()
        super()._load_from_state_dict(*args, **kwargs)

    def prefill_inference_caches(self) -> None:
        """
        Eagerly populate caches.
        """
        training_length = int(self.max_allowed_segment * self.samplerate)
        model_dtype = next(self.parameters()).dtype
        model_device = next(self.parameters()).device

        with torch.no_grad():
            mix = torch.zeros(
                1,
                self.audio_channels,
                training_length,
                device=model_device,
                dtype=torch.float32,
            )
            z = self._spec(mix)
            x = self._magnitude(z).to(mix.device)
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (1e-5 + std)

            xt = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)

            if model_dtype != torch.float32:
                x = x.to(model_dtype)
                xt = xt.to(model_dtype)

            self.forward_core(x, xt)

    def forward_core(
        self, x: torch.Tensor, xt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Core encoder-transformer-decoder processing.

        :param x: Normalized frequency branch input [B, C*2, Fq, T] (CaC format)
        :param xt: Normalized time branch input [B, C, samples]
        :return: Tuple of (frequency output [B, S*C*2, Fq, T], time output [B, S*C, samples])
        """
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                emb = self._cached_freq_emb(x.shape[-2], x.device, x.dtype)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.flatten(2)
                x = self.channel_upsampler(x)
                x = x.view(b, -1, f, t)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = x.flatten(2)
                x = self.channel_downsampler(x)
                x = x.view(b, -1, f, t)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        return x, xt

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """
        Separate the input mixture into individual sources.

        :param mix: Input mixture waveform [B, C, samples]
        :return: Separated sources tensor [B, S, C, samples]
        :raises ValidationError: If the input is longer than the training
            length; use ``apply_model`` to separate full-length audio.
        """
        length_pre_pad = None

        training_length = self.valid_length(mix.shape[-1])
        if mix.shape[-1] < training_length:
            length_pre_pad = mix.shape[-1]
            mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape

        var, mean = torch.var_mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(var)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        var_t, meant = torch.var_mean(xt, dim=(1, 2), keepdim=True)
        stdt = torch.sqrt(var_t)
        xt = (xt - meant) / (1e-5 + stdt)

        model_dtype = next(self.parameters()).dtype
        if model_dtype != torch.float32:
            x = x.to(model_dtype)
            xt = xt.to(model_dtype)
            x, xt = self.forward_core(x, xt)

        else:
            x, xt = self.forward_core(x, xt)

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(z, x)
        x = self._ispec(zout, training_length)

        xt = xt.view(B, S, -1, training_length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x
