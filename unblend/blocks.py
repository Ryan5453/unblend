# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional

from .transformer import LayerScale


def center_trim(tensor: Tensor, reference: Tensor | int) -> Tensor:
    """
    Center trim ``tensor`` with respect to ``reference`` along the last dimension.

    :param tensor: Tensor to trim
    :param reference: Reference tensor or integer length to trim to
    :return: Trimmed tensor
    :raises ValueError: If tensor is smaller than reference
    """
    ref_size: int
    if isinstance(reference, Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor


_HANN_CACHE: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}


def _hann_window(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Return a cached Hann window — ``torch.hann_window`` allocates fresh
    on every call, which adds up inside the STFT hot loop.

    :param size: Window length in samples.
    :param device: Device to allocate the window on.
    :param dtype: Dtype of the window.
    :return: 1-D Hann window tensor of length ``size``.
    """
    key = (size, device, dtype)
    win = _HANN_CACHE.get(key)
    if win is None:
        win = torch.hann_window(size, device=device, dtype=dtype)
        _HANN_CACHE[key] = win
    return win


def spectro(
    x: Tensor, n_fft: int = 512, hop_length: int | None = None, pad: int = 0
) -> Tensor:
    """
    Compute the STFT of the input signal.

    :param x: Input tensor
    :param n_fft: FFT size
    :param hop_length: Hop length between frames, defaults to n_fft // 4
    :param pad: Padding multiplier for the FFT size
    :return: Complex STFT tensor
    """
    *other, length = x.shape
    x = x.reshape(-1, length)

    win_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    z = torch.stft(
        x,
        n_fft * (1 + pad),
        hop_length or n_fft // 4,
        window=_hann_window(n_fft, x.device, win_dtype),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def _istft_fold(
    z: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: Tensor,
    length: int | None,
) -> Tensor:
    """
    Custom centered, normalized=True iSTFT that bypasses ``torch.istft``'s
    NOLA check (which calls ``.item()`` and forces a device→host sync on
    MPS — see PyTorch issue #94718).

    :param z: Complex spectrogram of shape ``[B, freqs, frames]``.
    :param n_fft: FFT size.
    :param hop_length: Hop length between frames.
    :param win_length: Window length (must be ``<= n_fft``).
    :param window: Window tensor of length ``win_length``.
    :param length: Optional output length to trim/pad to.
    :return: Reconstructed signal of shape ``[B, L]``.
    """
    n_frames = z.shape[-1]
    frames = torch.fft.irfft(z, n=n_fft, dim=1)  # [B, n_fft, n_frames]

    if win_length < n_fft:
        pad_total = n_fft - win_length
        pad_left = pad_total // 2
        window = functional.pad(window, (pad_left, pad_total - pad_left))
    frames = frames * window[None, :, None]

    output_length = (n_frames - 1) * hop_length + n_fft
    out = (
        functional.fold(
            frames,
            output_size=(output_length, 1),
            kernel_size=(n_fft, 1),
            stride=(hop_length, 1),
        )
        .squeeze(-1)
        .squeeze(1)
    )

    win_sq_frames = (window * window)[None, :, None].expand(1, n_fft, n_frames)
    norm = (
        functional.fold(
            win_sq_frames,
            output_size=(output_length, 1),
            kernel_size=(n_fft, 1),
            stride=(hop_length, 1),
        )
        .squeeze(-1)
        .squeeze(1)
    )
    out = out / norm

    if length is None:
        out = out[..., n_fft // 2 : -(n_fft // 2)]
    else:
        out = out[..., n_fft // 2 : n_fft // 2 + length]
    return out * (n_fft**0.5)


def ispectro(
    z: Tensor, hop_length: int | None = None, length: int | None = None, pad: int = 0
) -> Tensor:
    """
    Compute the inverse STFT of a complex spectrogram.

    :param z: Complex STFT tensor
    :param hop_length: Hop length between frames
    :param length: Expected output length
    :param pad: Padding multiplier used in the forward STFT
    :return: Reconstructed time-domain signal
    """
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)

    if z.device.type == "mps":
        # Avoid torch.istft's NOLA-check host sync on MPS (issue #94718).
        x = _istft_fold(
            z,
            n_fft=n_fft,
            hop_length=hop_length if hop_length is not None else n_fft // 4,
            win_length=win_length,
            window=_hann_window(win_length, z.real.device, z.real.dtype),
            length=length,
        )
    else:
        x = torch.istft(
            z,
            n_fft,
            hop_length,
            window=_hann_window(win_length, z.real.device, z.real.dtype),
            win_length=win_length,
            normalized=True,
            length=length,
            center=True,
        )
    _, output_length = x.shape
    return x.view(*other, output_length)


def rescale_conv(
    conv: nn.Conv1d | nn.Conv2d | nn.ConvTranspose1d | nn.ConvTranspose2d,
    reference: float,
) -> None:
    """
    Rescale initial weight scale. It is unclear why it helps but it certainly does.

    :param conv: Convolution module whose weights will be rescaled
    :param reference: Reference standard deviation for rescaling
    """
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module: nn.Module, reference: float) -> None:
    """
    Rescale all convolution weights in a module.

    :param module: Module whose convolution submodules will be rescaled
    :param reference: Reference standard deviation for rescaling
    """
    for sub in module.modules():
        if isinstance(
            sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)
        ):
            rescale_conv(sub, reference)


class DConv(nn.Module):
    """
    New residual branches in each encoder layer.
    This alternates dilated convolutions.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.
    """

    def __init__(
        self,
        channels: int,
        compress: float = 4,
        depth: int = 2,
        init: float = 1e-4,
        norm: bool = True,
        gelu: bool = True,
        kernel: int = 3,
    ) -> None:
        """
        Initialize DConv residual branch.

        :param channels: Input/output channels for residual branch
        :param compress: Amount of channel compression inside the branch
        :param depth: Number of layers in the residual branch
        :param init: Initial scale for LayerNorm
        :param norm: Use GroupNorm
        :param gelu: Use GELU activation
        :param kernel: Kernel size for the (dilated) convolutions
        """

        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        # The sign of `depth` selects dilation: a positive depth dilates
        # (2**d per layer), a negative depth uses |depth| layers with no
        # dilation. (Vestigial upstream convention; the shipped configs all
        # pass positive depths.)
        dilate = depth > 0

        norm_fn: Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa

        hidden = int(channels / compress)

        act: type[nn.Module]
        if gelu:
            act = nn.GELU
        else:
            act = nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = 2**d if dilate else 1
            padding = dilation * (kernel // 2)
            mods = [
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden),
                act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels),
                nn.GLU(1),
                LayerScale(channels, init),
            ]
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply all residual dilated convolution layers.

        :param x: Input tensor
        :return: Output tensor with residual connections applied
        """
        for layer in self.layers:
            x = x + layer(x)
        return x


def pad1d(
    x: Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    """
    Wrapper around F.pad to allow reflect padding on small input.

    :param x: Input tensor
    :param paddings: Left and right padding amounts
    :param mode: Padding mode
    :param value: Fill value for constant padding
    :return: Padded tensor
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = functional.pad(x, (extra_pad_left, extra_pad_right))
    out = functional.pad(x, paddings, mode, value)
    # Shape-only check kept; the prior elementwise `(out[...] == x0).all()`
    # assertion forced a host-side sync on every STFT call inside the chunk
    # loop, with no functional benefit on a well-tested PyTorch primitive.
    assert out.shape[-1] == length + padding_left + padding_right
    return out


class ScaledEmbedding(nn.Module):
    """
    Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 10.0,
        smooth: bool = False,
    ) -> None:
        """
        Initialize ScaledEmbedding.

        :param num_embeddings: Number of embeddings
        :param embedding_dim: Dimension of each embedding
        :param scale: Learning rate boost factor
        :param smooth: If True, make embeddings continuous via cumulative sum
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = (
                weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            )
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self) -> Tensor:
        """
        Scaled embedding weight.

        :return: Embedding weights multiplied by scale factor
        """
        return self.embedding.weight * self.scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Look up and scale embeddings.

        :param x: Input indices tensor
        :return: Scaled embedding vectors
        """
        out = self.embedding(x) * self.scale
        return out


class HEncLayer(nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        dconv: bool = True,
        norm: bool = True,
        context: int = 0,
        dconv_kw: dict | None = None,
        pad: bool = True,
        rewrite: bool = True,
    ) -> None:
        """
        Encoder layer used by both time and frequency branches.

        :param chin: Number of input channels
        :param chout: Number of output channels
        :param kernel_size: Kernel size for the convolution
        :param stride: Stride for the convolution
        :param norm_groups: Number of groups for group norm
        :param empty: If True, only use the first conv (for branch merging)
        :param freq: If True, operate on frequencies (use Conv2d)
        :param dconv: If True, insert DConv residual branches
        :param norm: If True, use GroupNorm
        :param context: Context size for the 1x1 conv
        :param dconv_kw: Keyword arguments for the DConv class
        :param pad: If True, pad input so output size = input size / stride
        :param rewrite: If True, add 1x1 conv at the end of the layer
        """
        super().__init__()
        dconv_kw = dconv_kw or {}
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x: Tensor, inject: Tensor | None = None) -> Tensor:
        """
        Apply the encoder layer.

        :param x: Input tensor
        :param inject: Optional injection from the time branch into the frequency branch
        :return: Encoded output tensor
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = functional.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = functional.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = functional.glu(z, dim=1)
        else:
            z = y
        return z


class MultiWrap(nn.Module):
    """
    Takes one layer and replicate it N times. each replica will act
    on a frequency band. All is done so that if the N replica have the same weights,
    then this is exactly equivalent to applying the original module on all frequencies.

    This is a bit over-engineered to avoid edge artifacts when splitting
    the frequency bands, but it is possible the naive implementation would work as well...
    """

    def __init__(
        self, layer: "HEncLayer | HDecLayer", split_ratios: list[float]
    ) -> None:
        """
        Initialize MultiWrap.

        :param layer: Module to clone, must be either HEncLayer or HDecLayer
        :param split_ratios: Ratios indicating which fraction to keep for each band
        """
        super().__init__()
        self.split_ratios = split_ratios
        self.layers = nn.ModuleList()
        self.conv = isinstance(layer, HEncLayer)
        assert not layer.norm
        assert layer.freq
        assert layer.pad
        if not self.conv:
            assert not layer.context_freq
        for k in range(len(split_ratios) + 1):
            lay = deepcopy(layer)
            if self.conv:
                lay.conv.padding = (0, 0)
            else:
                lay.pad = False
            for m in lay.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            self.layers.append(lay)

    def forward(
        self, x: Tensor, skip: Tensor | None = None, length: int | None = None
    ) -> Tensor | tuple[Tensor, None]:
        """
        Apply wrapped layers across frequency bands.

        :param x: Input tensor of shape (B, C, Fr, T)
        :param skip: Optional skip connection tensor (for decoder layers)
        :param length: Optional output length (for decoder layers)
        :return: Output tensor, or tuple of (output, None) for decoder layers
        """
        B, C, Fr, T = x.shape

        ratios = list(self.split_ratios) + [1]
        start = 0
        outs = []
        for ratio, layer in zip(ratios, self.layers):
            if self.conv:
                pad = layer.kernel_size // 4
                if ratio == 1:
                    limit = Fr
                    frames = -1
                else:
                    limit = int(round(Fr * ratio))
                    le = limit - start
                    if start == 0:
                        le += pad
                    frames = round((le - layer.kernel_size) / layer.stride + 1)
                    limit = start + (frames - 1) * layer.stride + layer.kernel_size
                    if start == 0:
                        limit -= pad
                assert limit - start > 0, (limit, start)
                assert limit <= Fr, (limit, Fr)
                y = x[:, :, start:limit, :]
                if start == 0:
                    y = functional.pad(y, (0, 0, pad, 0))
                if ratio == 1:
                    y = functional.pad(y, (0, 0, 0, pad))
                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                if ratio == 1:
                    limit = Fr
                else:
                    limit = int(round(Fr * ratio))
                last = layer.last
                layer.last = True

                y = x[:, :, start:limit]
                s = skip[:, :, start:limit]
                out, _ = layer(y, s, None)
                if outs:
                    outs[-1][:, :, -layer.stride :] += out[
                        :, :, : layer.stride
                    ] - layer.conv_tr.bias.view(1, -1, 1, 1)
                    out = out[:, :, layer.stride :]
                if ratio == 1:
                    out = out[:, :, : -layer.stride // 2, :]
                if start == 0:
                    out = out[:, :, layer.stride // 2 :, :]
                outs.append(out)
                layer.last = last
                start = limit
        out = torch.cat(outs, dim=2)
        if not self.conv and not last:
            out = functional.gelu(out)
        if self.conv:
            return out
        else:
            return out, None


class HDecLayer(nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        last: bool = False,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        dconv: bool = True,
        norm: bool = True,
        context: int = 1,
        dconv_kw: dict | None = None,
        pad: bool = True,
        context_freq: bool = True,
        rewrite: bool = True,
    ) -> None:
        """
        Decoder layer, mirror of HEncLayer.

        :param chin: Number of input channels
        :param chout: Number of output channels
        :param last: If True, this is the last layer (skip final activation)
        :param kernel_size: Kernel size for the transposed convolution
        :param stride: Stride for the transposed convolution
        :param norm_groups: Number of groups for group norm
        :param empty: If True, only use the transposed conv
        :param freq: If True, operate on frequencies (use Conv2d)
        :param dconv: If True, insert DConv residual branches
        :param norm: If True, use GroupNorm
        :param context: Context size for the 1x1 conv
        :param dconv_kw: Keyword arguments for the DConv class
        :param pad: If True, trim padding from output
        :param context_freq: If True, apply context along frequency axis
        :param rewrite: If True, add 1x1 conv at the start of the layer
        """
        super().__init__()
        dconv_kw = dconv_kw or {}
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(
                    chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context]
                )
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(
        self, x: Tensor, skip: Tensor | None, length: int
    ) -> tuple[Tensor, Tensor]:
        """
        Apply the decoder layer.

        :param x: Input tensor
        :param skip: Skip connection tensor from the encoder
        :param length: Target output length for time-domain trimming
        :return: Tuple of (decoded output, pre-transposed-conv tensor)
        """
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            if self.rewrite:
                y = functional.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = functional.gelu(z)
        return z, y
