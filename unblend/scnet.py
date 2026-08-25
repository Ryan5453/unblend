# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SCNet: Sparse Compression Network for Music Source Separation.

Reference: Tong et al., https://arxiv.org/abs/2401.13276

A third architecture family alongside HTDemucs and the RoFormers, and unrelated
to both. Where a RoFormer splits the spectrum into bands and runs axial
attention over them, SCNet splits into three bands and applies a *different
compression ratio to each* — dense modelling where the signal lives, aggressive
compression where it does not — then runs either a dual-path LSTM or
transformer trunk over the compressed representation.

Module and parameter names are deliberately identical to the reference
implementation (``SDlayer``, ``conv_modules``, ``globalconv``, ``convtrs``,
``dp_modules``, …), including its capitalisation, so published checkpoints load
strictly without a key-rename table.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import backends
from .exceptions import ValidationError


class Swish(nn.Module):
    """SiLU written as ``x * sigmoid(x)``, matching the reference."""

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the activation.

        :param x: Input tensor.
        :return: Activated tensor of the same shape.
        """
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    """
    Residual 1-D convolution stack applied within an SD block.
    """

    def __init__(
        self, channels: int, depth: int = 2, compress: float = 4, kernel: int = 3
    ) -> None:
        """
        Residual 1-D convolution stack applied within an SD block.

        :param channels: Input/output channels.
        :param depth: Number of residual layers.
        :param compress: Channel compression factor inside each layer.
        :param kernel: Convolution kernel size; must be odd.
        """
        super().__init__()
        if kernel % 2 == 0:
            raise ValidationError(f"SCNet conv kernel must be odd, got {kernel}.")
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        padding = kernel // 2
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    nn.Conv1d(channels, hidden_size * 2, kernel, padding=padding),
                    nn.GLU(1),
                    nn.Conv1d(
                        hidden_size,
                        hidden_size,
                        kernel,
                        padding=padding,
                        groups=hidden_size,
                    ),
                    nn.GroupNorm(1, hidden_size),
                    Swish(),
                    nn.Conv1d(hidden_size, channels, 1),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the residual stack.

        :param x: Input of shape ``[batch, channels, time]``.
        :return: Output of the same shape.
        """
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    """
    Decoder fusion of a decoded tensor with its encoder skip.
    """

    def __init__(
        self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1
    ) -> None:
        """
        Decoder fusion of a decoded tensor with its encoder skip.

        :param channels: Channel count of the decoded tensor.
        :param kernel_size: Convolution kernel size.
        :param stride: Convolution stride.
        :param padding: Convolution padding.
        """
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            channels * 2, channels * 2, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        """
        Fuse ``x`` with ``skip`` and halve the channels through a GLU.

        :param x: Decoded tensor ``[batch, channels, freq, time]``.
        :param skip: Matching encoder skip, or ``None``.
        :return: Fused tensor of the same shape as ``x``.
        """
        if skip is not None:
            # Out-of-place: the reference mutates in place, which is unsafe
            # when the input aliases a view under inference_mode.
            x = x + skip
        # This convolves ``cat([x, x])``, which folds algebraically to a
        # half-width convolution — but profiling puts the whole decoder under
        # 2% of runtime (the dual-path LSTM trunk is 81-90%), so the fold was
        # not worth losing bit-exactness with the reference implementation.
        x = x.repeat(1, 2, 1, 1)
        x = self.conv(x)
        return F.glu(x, dim=1)


class SDlayer(nn.Module):
    """
    Sparse down-sample layer: split into bands and compress each differently.
    """

    def __init__(self, channels_in: int, channels_out: int, band_configs: dict) -> None:
        """
        Sparse down-sample layer: split into bands and compress each differently.

        :param channels_in: Input channels.
        :param channels_out: Output channels.
        :param band_configs: Per-band ``SR``/``stride``/``kernel`` settings,
            keyed by ``low``/``mid``/``high``.
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.strides: list[int] = []
        self.kernels: list[int] = []
        for config in band_configs.values():
            self.convs.append(
                nn.Conv2d(
                    channels_in,
                    channels_out,
                    (config["kernel"], 1),
                    (config["stride"], 1),
                    (0, 0),
                )
            )
            self.strides.append(config["stride"])
            self.kernels.append(config["kernel"])

        self.SR_low = band_configs["low"]["SR"]
        self.SR_mid = band_configs["mid"]["SR"]

    def forward(self, x: Tensor) -> tuple[list[Tensor], list[int]]:
        """
        Split the spectrogram into three bands and convolve each.

        :param x: Input of shape ``[batch, channels, freq, time]``.
        :return: The per-band outputs and their pre-convolution frequency
            lengths (needed to trim symmetrically on the way back up).
        """
        _, _, freq, _ = x.shape
        splits = [
            (0, math.ceil(freq * self.SR_low)),
            (
                math.ceil(freq * self.SR_low),
                math.ceil(freq * (self.SR_low + self.SR_mid)),
            ),
            (math.ceil(freq * (self.SR_low + self.SR_mid)), freq),
        ]

        outputs: list[Tensor] = []
        original_lengths: list[int] = []
        for conv, stride, kernel, (start, end) in zip(
            self.convs, self.strides, self.kernels, splits
        ):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[2]

            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            outputs.append(conv(F.pad(extracted, (0, 0, pad_left, pad_right))))

        return outputs, original_lengths


class SUlayer(nn.Module):
    """
    Sparse up-sample layer: the decoder counterpart of :class:`SDlayer`.
    """

    def __init__(self, channels_in: int, channels_out: int, band_configs: dict) -> None:
        """
        Sparse up-sample layer: the decoder counterpart of :class:`SDlayer`.

        :param channels_in: Input channels.
        :param channels_out: Output channels.
        :param band_configs: Per-band ``stride``/``kernel`` settings.
        """
        super().__init__()
        self.convtrs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels_in,
                    channels_out,
                    [config["kernel"], 1],
                    [config["stride"], 1],
                )
                for config in band_configs.values()
            ]
        )

    def forward(
        self, x: Tensor, lengths: list[int], origin_lengths: list[int]
    ) -> Tensor:
        """
        Up-sample each band and restore its original frequency extent.

        :param x: Input of shape ``[batch, channels, freq, time]``.
        :param lengths: Per-band frequency lengths within ``x``.
        :param origin_lengths: Per-band lengths to trim back to.
        :return: The re-concatenated tensor.
        """
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None),
        ]
        outputs: list[Tensor] = []
        for index, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            # Trim symmetrically: the transposed convolution overshoots by a
            # stride-dependent amount that is not generally even.
            distance = abs(origin_lengths[index] - out.shape[2]) // 2
            outputs.append(out[:, :, distance : distance + origin_lengths[index], :])
        return torch.cat(outputs, dim=2)


class SDblock(nn.Module):
    """
    One encoder stage: sparse down-sample, per-band convolution, global mix.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        band_configs: dict | None = None,
        conv_config: dict | None = None,
        depths: list[int] | None = None,
        kernel_size: int = 3,
    ) -> None:
        """
        One encoder stage: sparse down-sample, per-band convolution, global mix.

        :param channels_in: Input channels.
        :param channels_out: Output channels.
        :param band_configs: Band split configuration.
        :param conv_config: ``compress``/``kernel`` for the convolution modules.
        :param depths: Residual depth for the low/mid/high band respectively.
        :param kernel_size: Kernel for the global convolution; must be odd.
        """
        super().__init__()
        band_configs = band_configs or {}
        conv_config = conv_config or {}
        depths = depths if depths is not None else [3, 2, 1]
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)
        self.conv_modules = nn.ModuleList(
            [ConvolutionModule(channels_out, depth, **conv_config) for depth in depths]
        )
        self.globalconv = nn.Conv2d(
            channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, list[int], list[int]]:
        """
        Encode one stage.

        :param x: Input of shape ``[batch, channels, freq, time]``.
        :return: Stage output, the skip tensor for the decoder, per-band
            frequency lengths, and pre-convolution band lengths.
        """
        bands, original_lengths = self.SDlayer(x)
        # Each band is folded into the batch so the 1-D convolution module runs
        # per frequency bin, then unfolded back.
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)
        ]
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        return self.globalconv(full_band), full_band, lengths, original_lengths


class FeatureConversion(nn.Module):
    """
    Move between time and frequency representations inside the trunk.
    """

    def __init__(self, channels: int, inverse: bool) -> None:
        """
        Move between time and frequency representations inside the trunk.

        :param channels: Channel count of the packed real/imaginary tensor.
        :param inverse: Whether to apply the inverse transform.
        """
        super().__init__()
        self.inverse = inverse
        self.channels = channels
        # ``torch.fft.irfft`` lowers to an ONNX ``DFT`` node carrying both
        # ``inverse`` and ``onesided``, which the spec forbids and onnxruntime
        # rejects at load. ``SCNetONNXWrapper`` flips this on to take the
        # explicit real-valued DFT below instead: algebraically the same
        # transform expressed as two matmuls, so exports stay valid while
        # native inference keeps using the fast fused kernels.
        self.onnx_safe = False
        self._dft_cache: dict[tuple[int, torch.device, torch.dtype], tuple] = {}

    def _dft_matrices(
        self, frames: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """
        Real/imaginary DFT basis matrices for a given frame count.

        :param frames: Length of the time axis being transformed.
        :param device: Device to build the matrices on.
        :param dtype: Dtype to build the matrices in.
        :return: ``(cos_basis, sin_basis)`` for the requested direction.
        """
        key = (frames, device, dtype)
        cached = self._dft_cache.get(key)
        if cached is not None:
            return cached

        bins = frames // 2 + 1
        # Reduce k*t modulo `frames` in exact integer arithmetic before forming
        # the angle. Multiplying the raw indices gives values in the tens of
        # thousands, which needs float64 to phase-resolve; folding them into
        # one period first keeps the angle below 2*pi so float32 is ample.
        # That matters beyond tidiness: `frames` is dynamic, so the exporter
        # traces this arithmetic into the graph rather than folding it to a
        # constant, and a float64 subgraph is unrunnable on onnxruntime-web's
        # WebGPU backend.
        k = torch.arange(bins, device=device).unsqueeze(1)
        t = torch.arange(frames, device=device).unsqueeze(0)
        phase = torch.remainder(k * t, frames).to(torch.float32)
        angle = (2.0 * math.pi / frames) * phase
        scale = 1.0 / math.sqrt(frames)

        if self.inverse:
            # Hermitian reconstruction: every bin except DC (and Nyquist, when
            # the length is even) stands in for a conjugate pair, so it is
            # counted twice.
            weights = torch.full((bins, 1), 2.0, device=device, dtype=torch.float32)
            weights[0] = 1.0
            if frames % 2 == 0:
                weights[-1] = 1.0
            # Contracts over bins: ``[..., bins] @ [bins, frames]``.
            cos_basis = weights * angle.cos() * scale
            sin_basis = -weights * angle.sin() * scale
        else:
            cos_basis = (angle.cos() * scale).transpose(0, 1)
            sin_basis = (-angle.sin() * scale).transpose(0, 1)

        result = (cos_basis.to(dtype), sin_basis.to(dtype))
        self._dft_cache[key] = result
        return result

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the (inverse) real FFT along the time axis.

        :param x: Input of shape ``[batch, channels, freq, time]``.
        :return: Transformed tensor.
        """
        # The transform runs in float32 regardless of the model's dtype: cuFFT
        # rejects bfloat16 outright, and half-precision FFTs lose accuracy on
        # long frame axes. The result is cast back so the next LSTM sees its
        # own parameter dtype — without this, a half-precision model fails with
        # "mixed dtype: expect parameter to have scalar type Float".
        dtype = x.dtype
        x = x.float()
        if self.inverse:
            x_r = x[:, : self.channels // 2, :, :]
            x_i = x[:, self.channels // 2 :, :, :]
            if not self.onnx_safe:
                out = torch.fft.irfft(torch.complex(x_r, x_i), dim=3, norm="ortho")
            else:
                frames = (x_r.shape[-1] - 1) * 2
                cos_basis, sin_basis = self._dft_matrices(frames, x.device, x.dtype)
                out = x_r @ cos_basis + x_i @ sin_basis
            return out.to(dtype)

        if not self.onnx_safe:
            spectrum = torch.fft.rfft(x, dim=3, norm="ortho")
            out = torch.cat([spectrum.real, spectrum.imag], dim=1)
        else:
            cos_basis, sin_basis = self._dft_matrices(x.shape[-1], x.device, x.dtype)
            out = torch.cat([x @ cos_basis, x @ sin_basis], dim=1)
        return out.to(dtype)


class DualPathRNN(nn.Module):
    """
    Bidirectional LSTM applied along the frequency axis, then the time axis.
    """

    def __init__(self, d_model: int, expand: int, bidirectional: bool = True) -> None:
        """
        Bidirectional LSTM applied along the frequency axis, then the time axis.

        :param d_model: Feature width.
        :param expand: Hidden-size expansion factor.
        :param bidirectional: Whether each LSTM is bidirectional.
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        # batch_first=False on purpose. cuDNN's native layout is
        # [seq, batch, feature]; with batch_first=True PyTorch transposes both
        # the input and the output around every call. Profiling puts this
        # trunk at 81-90% of total runtime, so those four extra copies per
        # layer are worth avoiding — the tensors are reshaped into the native
        # layout directly below. The flag does not affect parameter names or
        # shapes, so checkpoints are unaffected.
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(
                    d_model,
                    self.hidden_size,
                    num_layers=1,
                    bidirectional=bidirectional,
                    batch_first=False,
                )
                for _ in range(2)
            ]
        )
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size * 2, d_model) for _ in range(2)]
        )
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the frequency path then the time path, each residually.

        :param x: Input of shape ``[batch, channels, freq, time]``.
        :return: Output of the same shape.
        """
        batch, channels, freq, time = x.shape

        # Frequency path: sequence over freq bins, one sequence per (batch,
        # time) pair. Built as [freq, batch * time, channels] so cuDNN gets its
        # native layout with no internal transpose.
        original = x
        y = self.norm_layers[0](x)
        y = y.permute(2, 0, 3, 1).reshape(freq, batch * time, channels)
        y, _ = self.lstm_layers[0](y)
        y = self.linear_layers[0](y)
        y = y.view(freq, batch, time, channels).permute(1, 3, 0, 2)
        x = y + original

        # Time path: sequence over frames, one sequence per (batch, freq) pair.
        original = x
        y = self.norm_layers[1](x)
        y = y.permute(3, 0, 2, 1).reshape(time, batch * freq, channels)
        y, _ = self.lstm_layers[1](y)
        y = self.linear_layers[1](y)
        y = y.view(time, batch, freq, channels).permute(1, 3, 2, 0)
        return y + original


class SeparationNet(nn.Module):
    """
    The dual-path trunk: alternating RNN blocks with FFT domain swaps.
    """

    def __init__(self, channels: int, expand: int = 1, num_layers: int = 6) -> None:
        """
        The dual-path trunk: alternating RNN blocks with FFT domain swaps.

        :param channels: Trunk width.
        :param expand: LSTM hidden expansion.
        :param num_layers: Number of dual-path layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dp_modules = nn.ModuleList(
            [
                DualPathRNN(channels * (2 if i % 2 == 1 else 1), expand)
                for i in range(num_layers)
            ]
        )
        self.feature_conversion = nn.ModuleList(
            [
                FeatureConversion(channels * 2, inverse=bool(i % 2))
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Run every dual-path layer.

        :param x: Input of shape ``[batch, channels, freq, time]``.
        :return: Output of the same shape.
        """
        for index in range(self.num_layers):
            x = self.dp_modules[index](x)
            x = self.feature_conversion[index](x)
        return x


def stft_padding(samples: int, hop_length: int) -> int:
    """
    Trailing zeros needed before SCNet's STFT.

    The trunk applies a real FFT across the time axis, which needs an even
    frame count, so the input is padded up to a hop boundary and then by one
    more hop if that landed on an odd count.

    An ONNX consumer has to reproduce this exactly: the exported graph is
    traced at the padded frame count and will reject a spectrogram computed
    from unpadded audio. Trim ``padding`` samples off the end after the inverse
    transform.

    :param samples: Length of the input audio in samples.
    :param hop_length: STFT hop length.
    :return: Number of samples to append.
    """
    padding = hop_length - samples % hop_length
    if (samples + padding) // hop_length % 2 == 0:
        padding += hop_length
    return padding


class SCNet(nn.Module):
    """
    Sparse Compression Network.

    Constructor defaults mirror the reference so a published config maps onto
    it directly.
    """

    # Most published SCNet configs disable track-level normalization. The two
    # starrytong checkpoints opt in through the constructor instead.
    external_normalization = False

    def __init__(
        self,
        sources: list[str] | None = None,
        audio_channels: int = 2,
        dims: list[int] | None = None,
        nfft: int = 4096,
        hop_size: int = 1024,
        win_size: int = 4096,
        normalized: bool = True,
        band_SR: list[float] | None = None,
        band_stride: list[int] | None = None,
        band_kernel: list[int] | None = None,
        conv_depths: list[int] | None = None,
        compress: int = 4,
        conv_kernel: int = 3,
        num_dplayer: int = 6,
        expand: int = 1,
        external_normalization: bool = False,
    ) -> None:
        """
        Sparse Compression Network.

        :param sources: Output stem names.
        :param audio_channels: Input/output audio channels.
        :param dims: Channel width per encoder stage.
        :param nfft: STFT size.
        :param hop_size: STFT hop.
        :param win_size: STFT window length.
        :param normalized: Whether the STFT is normalised.
        :param band_SR: Proportion of the spectrum in each band.
        :param band_stride: Down-sample ratio per band.
        :param band_kernel: Down-sample kernel per band.
        :param conv_depths: Residual depth per band.
        :param compress: Channel compression inside convolution modules.
        :param conv_kernel: Convolution module kernel size.
        :param num_dplayer: Number of dual-path layers.
        :param expand: LSTM hidden expansion factor.
        :param external_normalization: Whether the caller applies track-level
            mean/std normalization around inference.
        """
        super().__init__()
        sources = list(sources) if sources else ["drums", "bass", "other", "vocals"]
        dims = list(dims) if dims else [4, 32, 64, 128]
        band_SR = list(band_SR) if band_SR else [0.175, 0.392, 0.433]
        band_stride = list(band_stride) if band_stride else [1, 4, 16]
        band_kernel = list(band_kernel) if band_kernel else [3, 4, 16]
        conv_depths = list(conv_depths) if conv_depths else [3, 2, 1]

        self.sources = sources
        self.external_normalization = bool(external_normalization)
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ["low", "mid", "high"]
        self.band_configs = {
            band_keys[i]: {
                "SR": band_SR[i],
                "stride": band_stride[i],
                "kernel": band_kernel[i],
            }
            for i in range(len(band_keys))
        }
        self.hop_length = hop_size
        self.conv_config = {"compress": compress, "kernel": conv_kernel}
        self.stft_config = {
            "n_fft": nfft,
            "hop_length": hop_size,
            "win_length": win_size,
            "center": True,
            "normalized": normalized,
        }

        # Inference interface (see configure_inference); class-level defaults
        # keep a bare construction usable.
        self.samplerate = 44100
        self.max_allowed_segment = 11.0

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(len(dims) - 1):
            self.encoder.append(
                SDblock(
                    channels_in=dims[index],
                    channels_out=dims[index + 1],
                    band_configs=self.band_configs,
                    conv_config=self.conv_config,
                    depths=conv_depths,
                )
            )
            self.decoder.insert(
                0,
                nn.Sequential(
                    FusionLayer(channels=dims[index + 1]),
                    SUlayer(
                        channels_in=dims[index + 1],
                        channels_out=(
                            dims[index] if index != 0 else dims[index] * len(sources)
                        ),
                        band_configs=self.band_configs,
                    ),
                ),
            )

        self.separation_net = SeparationNet(
            channels=dims[-1], expand=expand, num_layers=num_dplayer
        )

    def configure_inference(
        self, *, sources: list[str], samplerate: int, segment_samples: int
    ) -> None:
        """
        Attach the checkpoint-specific inference interface.

        :param sources: Output stem names; must match the head count.
        :param samplerate: Sample rate the checkpoint operates at.
        :param segment_samples: Training chunk length in samples.
        :raises ValidationError: If ``sources`` does not match the decoder.
        """
        if len(sources) != len(self.sources):
            raise ValidationError(
                f"SCNet checkpoint emits {len(self.sources)} stems; got "
                f"{len(sources)} source names."
            )
        self.sources = list(sources)
        self.samplerate = int(samplerate)
        self.max_allowed_segment = segment_samples / float(samplerate)

    def _stft_kwargs(self, device: torch.device) -> dict:
        """
        Arguments for the boundary transforms.

        Plain SCNet applies **no window** — the reference passes none, so
        adding one would silently change the result. The masked variant
        overrides this to supply its Hann window.

        :param device: Device the transform runs on.
        :return: Keyword arguments for ``torch.stft``/``torch.istft``.
        """
        return dict(self.stft_config)

    def forward_core(self, x: Tensor) -> Tensor:
        """
        Encoder, dual-path trunk, and decoder — everything between the
        transforms.

        Kept separate from :meth:`forward` for the same reason HTDemucs and the
        RoFormers do it: STFT/iSTFT are a poor Inductor target and inflate
        compile time without improving steady-state throughput.

        :param x: Packed spectrogram ``[batch, channels, freq, time]``.
        :return: Decoded tensor before the inverse transform.
        """
        save_skip: deque[Tensor] = deque()
        save_lengths: deque[list[int]] = deque()
        save_original_lengths: deque[list[int]] = deque()

        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        x = self.separation_net(x)

        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
        return x

    def forward(self, mix: Tensor) -> Tensor:
        """
        Separate a batch of mixtures.

        :param mix: ``(batch, channels, samples)`` audio.
        :return: ``(batch, stems, channels, samples)`` estimates.
        """
        batch = mix.shape[0]
        model_dtype = next(self.parameters()).dtype
        padding = stft_padding(mix.shape[-1], self.hop_length)
        mix = F.pad(mix, (0, padding))

        length = mix.shape[-1]
        # cuFFT has no bfloat16 kernel and half-precision STFT is needlessly
        # lossy, so the transforms stay float32 and only the trunk runs in the
        # model's dtype.
        x = mix.float().reshape(-1, length)
        x = torch.stft(x, **self._stft_kwargs(x.device), return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(
            x.shape[0] // self.audio_channels,
            x.shape[3] * self.audio_channels,
            x.shape[1],
            x.shape[2],
        )
        _, _, freq, time = x.shape

        x = self.forward_core(x.to(model_dtype)).float()

        n = self.dims[0]
        x = x.view(batch, n, -1, freq, time)
        x = x.reshape(-1, 2, freq, time).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        x = torch.istft(x, **self._stft_kwargs(x.device))
        x = x.reshape(batch, len(self.sources), self.audio_channels, -1)
        return x[:, :, :, :-padding]

    def enable_compiled_core(self) -> None:
        """
        Compile the encoder/trunk/decoder, leaving the transforms eager.
        """
        if not hasattr(self, "_uncompiled_forward_core"):
            self._uncompiled_forward_core = self.forward_core
        self.forward_core = torch.compile(
            self._uncompiled_forward_core, mode="reduce-overhead"
        )
        self._fixed_batch_shape = True

    def disable_compiled_core(self) -> None:
        """
        Restore the eager core so a retry does not double-wrap it.
        """
        original = getattr(self, "_uncompiled_forward_core", None)
        if original is not None:
            self.forward_core = original
            del self._uncompiled_forward_core


class SCNetMasked(SCNet):
    """
    SCNet variant that predicts a complex mask instead of the spectrogram.

    Three additions over the plain network: a learned positional embedding over
    frequency, a Hann-windowed STFT (the plain variant uses none), and a small
    convolutional head whose tanh output multiplies the repeated mixture. The
    ``scnet_small`` checkpoint uses this variant; ``scnet_xl_wide_v5`` uses the
    plain network.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """
        Build the plain network, then the masking additions.

        :param args: Positional arguments forwarded to :class:`SCNet`.
        :param kwargs: Keyword arguments forwarded to :class:`SCNet`.
        """
        super().__init__(*args, **kwargs)
        self.embed_dim = self.dims[0]
        self.max_f = int(self.stft_config["n_fft"]) // 2 + 1
        self.pos_embed_f = nn.Parameter(torch.zeros(1, self.embed_dim, self.max_f, 1))
        nn.init.trunc_normal_(self.pos_embed_f, std=0.02)
        # Not persistent: the window is derived from n_fft, and the published
        # checkpoints do not carry it.
        self.register_buffer(
            "window",
            torch.hann_window(int(self.stft_config["n_fft"]), periodic=True),
            persistent=False,
        )
        stems = len(self.sources)
        self.mask_layer = nn.Sequential(
            nn.Conv2d(4 * stems, 64, kernel_size=3, padding="same"),
            nn.GELU(),
            nn.Conv2d(64, 4 * stems, kernel_size=1, padding="same"),
            nn.Tanh(),
        )

    def _stft_kwargs(self, device: torch.device) -> dict:
        """
        Transform arguments, including this variant's Hann window.

        :param device: Device the transform runs on.
        :return: Keyword arguments for ``torch.stft``/``torch.istft``.
        """
        kwargs = dict(self.stft_config)
        kwargs["window"] = self.window.to(device)
        return kwargs

    def forward(self, mix: Tensor) -> Tensor:
        """
        Separate a batch of mixtures by masking their spectrogram.

        :param mix: ``(batch, channels, samples)`` audio.
        :return: ``(batch, stems, channels, samples)`` estimates.
        """
        batch = mix.shape[0]
        model_dtype = next(self.parameters()).dtype
        padding = stft_padding(mix.shape[-1], self.hop_length)
        mix = F.pad(mix, (0, padding))

        length = mix.shape[-1]
        # See SCNet.forward: transforms stay float32.
        x = mix.float().reshape(-1, length)
        x = torch.stft(x, **self._stft_kwargs(x.device), return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(
            x.shape[0] // self.audio_channels,
            x.shape[3] * self.audio_channels,
            x.shape[1],
            x.shape[2],
        )
        _, channels, freq, frames = x.shape
        if channels != self.embed_dim:
            raise ValidationError(
                f"SCNet masked expects {self.embed_dim} packed channels after "
                f"the STFT, got {channels}."
            )

        stems = len(self.sources)
        mixture = x.repeat(1, stems, 1, 1)

        if freq > self.max_f:
            repeats = math.ceil(freq / self.max_f)
            pos_f = self.pos_embed_f.repeat(1, 1, repeats, 1)[:, :, :freq, :]
        else:
            pos_f = self.pos_embed_f[:, :, :freq, :]
        x = x + pos_f.float()

        mask = self.mask_layer(self.forward_core(x.to(model_dtype))).float()

        n = self.dims[0]
        mixture = mixture.view(batch, n, -1, freq, frames)
        mixture = mixture.reshape(-1, 2, freq, frames).permute(0, 2, 3, 1)
        mixture = torch.view_as_complex(mixture.contiguous())

        mask = mask.view(batch, n, -1, freq, frames)
        mask = mask.reshape(-1, 2, freq, frames).permute(0, 2, 3, 1)
        mask = torch.view_as_complex(mask.contiguous())

        x = torch.istft(mixture * mask, **self._stft_kwargs(mask.device))
        x = x.reshape(batch, stems, self.audio_channels, -1)
        return x[:, :, :, :-padding]


_ARCHITECTURES: dict[str, type[nn.Module]] = {
    "scnet": SCNet,
    "scnet_masked": SCNetMasked,
}


def build_scnet(
    architecture: str,
    config: dict,
    *,
    sources: list[str],
    samplerate: int,
    segment_samples: int,
    state: dict | None = None,
) -> SCNet:
    """
    Construct an SCNet variant from registry metadata and load a checkpoint.

    :param architecture: Registered SCNet architecture name.
    :param config: Constructor kwargs, as stored in ``metadata.yaml``.
    :param sources: Output stem names.
    :param samplerate: Sample rate the checkpoint operates at.
    :param segment_samples: Training chunk length in samples.
    :param state: Checkpoint state dict to load (strict), or ``None``.
    :return: The constructed (and loaded) model in eval mode.
    :raises ValidationError: For an unknown architecture name.
    """
    klass = _ARCHITECTURES.get(architecture)
    if klass is None:
        raise ValidationError(
            f"Unknown SCNet architecture {architecture!r}; expected one of "
            f"{sorted(_ARCHITECTURES)}."
        )
    model = klass(**config)
    model.configure_inference(
        sources=sources, samplerate=samplerate, segment_samples=segment_samples
    )
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model.eval()


backends.register_backend("scnet", build_scnet, _ARCHITECTURES)
