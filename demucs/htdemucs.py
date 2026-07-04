# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
from torch import nn
from torch.nn import functional as F

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
from .states import capture_init
from .transformer import CrossTransformerEncoder


class HTDemucs(nn.Module):
    """
    Spectrogram and hybrid Demucs model.
    The spectrogram model has the same structure as Demucs, except the first few layers are over the
    frequency axis, until there is only 1 frequency, and then it moves to time convolutions.
    Frequency layers can still access information across time steps thanks to the DConv residual.

    Hybrid model have a parallel time branch. At some layer, the time branch has the same stride
    as the frequency branch and then the two are combined. The opposite happens in the decoder.

    Models can either use naive iSTFT from masking, Wiener filtering ([Ulhih et al. 2017]),
    or complex as channels (CaC) [Choi et al. 2020]. Wiener filtering is based on
    Open Unmix implementation [Stoter et al. 2019].

    The loss is always on the temporal domain, by backpropagating through the above
    output methods and iSTFT. This allows to define hybrid models nicely. However, this breaks
    a bit Wiener filtering, as doing more iteration at test time will change the spectrogram
    contribution, without changing the one from the waveform, which will lead to worse performance.
    I tried using the residual option in OpenUnmix Wiener implementation, but it didn't improve.
    CaC on the other hand provides similar performance for hybrid, and works naturally with
    hybrid models.

    This model also uses frequency embeddings are used to improve efficiency on convolutions
    over the freq. axis, following [Isik et al. 2020] (https://arxiv.org/pdf/2008.04470.pdf).

    Unlike classic Demucs, there is no resampling here, and normalization is always applied.
    """

    @capture_init
    def __init__(
        self,
        sources: list[str],
        # Channels
        audio_channels: int = 2,
        channels: int = 48,
        channels_time: int | None = None,
        growth: int = 2,
        # STFT
        nfft: int = 4096,
        cac: bool = True,
        # Main structure
        depth: int = 4,
        rewrite: bool = True,
        # Frequency branch
        multi_freqs: list[int] | None = None,
        multi_freqs_depth: int = 3,
        freq_emb: float = 0.2,
        emb_scale: int = 10,
        emb_smooth: bool = True,
        # Convolutions
        kernel_size: int = 8,
        time_stride: int = 2,
        stride: int = 4,
        context: int = 1,
        context_enc: int = 0,
        # Normalization
        norm_starts: int = 4,
        norm_groups: int = 4,
        # DConv residual branch
        dconv_mode: int = 1,
        dconv_depth: int = 2,
        dconv_comp: int = 8,
        dconv_init: float = 1e-3,
        # Before the Transformer
        bottom_channels: int = 0,
        # Transformer
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
        t_cape_augment: bool = False,  # Always False for inference
        t_cape_glob_loc_scale: list[float] = [5000.0, 1.0, 1.4],
        # ------ Particuliar parameters
        t_cross_first: bool = False,
        # Weight init
        rescale: float = 0.1,
        # Metadata
        samplerate: int = 44100,
        segment: int = 10,
    ) -> None:
        """
        Initialize the HTDemucs model.

        :param sources: List of source names
        :param audio_channels: Input/output audio channels
        :param channels: Initial number of hidden channels
        :param channels_time: If not None, use a different channels value for the time branch
        :param growth: Factor to increase hidden channels by at each layer
        :param nfft: Number of FFT bins
        :param cac: Use complex as channels (complex numbers become 2 channels each)
        :param depth: Number of layers in the encoder and decoder
        :param rewrite: Add 1x1 convolution to each layer
        :param multi_freqs: Frequency ratios for splitting bands with MultiWrap
        :param multi_freqs_depth: How many outermost layers to wrap with MultiWrap
        :param freq_emb: Frequency embedding weight after first freq layer (0 to disable)
        :param emb_scale: Equivalent to scaling the embedding learning rate
        :param emb_smooth: Initialize embedding smoothly with respect to frequencies
        :param kernel_size: Kernel size for encoder and decoder layers
        :param time_stride: Stride for the final time layer after the merge
        :param stride: Stride for encoder and decoder layers
        :param context: Context for 1x1 conv in the decoder
        :param context_enc: Context for 1x1 conv in the encoder
        :param norm_starts: Layer at which group norm starts being used
        :param norm_groups: Number of groups for group norm
        :param dconv_mode: 1: dconv in encoder only, 2: decoder only, 3: both
        :param dconv_depth: Depth of residual DConv branch
        :param dconv_comp: Compression of DConv branch
        :param dconv_init: Initial scale for the DConv branch LayerScale
        :param bottom_channels: If >0, adds a 1x1 Conv before and after the transformer
        :param t_layers: Number of transformer layers in each branch
        :param t_emb: Positional embedding type ("sin", "cape", or "scaled")
        :param t_hidden_scale: Hidden scale of the transformer feedforward layers
        :param t_heads: Number of transformer attention heads
        :param t_dropout: Dropout rate in the transformer
        :param t_max_positions: Max positions for "scaled" positional embedding
        :param t_norm_in: Norm before adding positional embedding
        :param t_norm_in_group: If True with t_norm_in, use GroupNorm over all timesteps
        :param t_group_norm: If True, encoder layer norms use GroupNorm over all timesteps
        :param t_norm_first: If True, norm before attention and FFN
        :param t_norm_out: If True, GroupNorm at the end of each layer
        :param t_max_period: Denominator in the sinusoidal embedding expression
        :param t_weight_decay: Weight decay for the transformer
        :param t_lr: Specific learning rate for the transformer
        :param t_layer_scale: Enable Layer Scale for the transformer
        :param t_gelu: Use GeLU activations if True, ReLU otherwise
        :param t_weight_pos_embed: Weighting of the positional embedding
        :param t_sin_random_shift: Random shift for sinusoidal embedding
        :param t_cape_mean_normalize: CAPE positional embedding normalization
        :param t_cape_augment: CAPE augmentation (True for training, False for inference)
        :param t_cape_glob_loc_scale: CAPE parameters (list of 3 floats)
        :param t_cross_first: If True, cross attention is the first transformer layer
        :param rescale: Weight rescaling trick factor
        :param samplerate: Audio sample rate in Hz
        :param segment: Training segment length in seconds
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
        # Contract with ``apply_model``: when True, every forward must run at
        # exactly ``chunk_batch_size`` (sub-full tail batches get zero-padded
        # up). Set by ``Separator``'s torch.compile path, whose CUDAGraphs
        # capture replays a single batch shape; eager models keep it False and
        # run tails at their natural size.
        self._fixed_batch_shape = False

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin  # number of channels for the freq branch
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

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
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
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
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
        # Convert CaC format back to complex.
        # With CaC, `m` is actually a full spectrogram and `z` is ignored.
        B, S, C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        out = torch.view_as_complex(out.contiguous())
        return out

    def valid_length(self, length: int) -> int:
        """
        Return a length that is appropriate for evaluation.

        :param length: Requested input length in samples
        :return: Training length for consistent segment processing
        :raises ValidationError: If length exceeds the training length
        """
        training_length = int(self.max_allowed_segment * self.samplerate)
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
        """
        length_pre_pad = None

        training_length = int(self.max_allowed_segment * self.samplerate)
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
            # No explicit ``.float()`` here. The next op multiplies by FP32
            # ``std``/``mean`` tensors, which promotes the low-precision
            # output to FP32 in a single fused kernel — strictly fewer device
            # copies than an explicit cast followed by a same-dtype mul.
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
