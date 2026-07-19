"""Unit tests for the pure tensor helpers in ``unblend.audio``."""

import pytest
import torch

from unblend.audio import convert_audio_channels, prevent_clip
from unblend.exceptions import ValidationError


def test_convert_channels_stereo_passthrough() -> None:
    """
    A tensor that already has the requested channel count is returned as-is.
    """
    wav = torch.randn(2, 100)
    assert convert_audio_channels(wav, 2) is wav


def test_convert_channels_mono_to_stereo_replicates() -> None:
    """
    Mono input is broadcast across the requested number of channels.
    """
    wav = torch.randn(1, 100)
    out = convert_audio_channels(wav, 2)
    assert out.shape == (2, 100)
    assert torch.equal(out[0], out[1])


def test_convert_channels_more_than_requested_takes_first_n() -> None:
    """
    When the source has more channels than requested, the first N are kept.

    This mirrors the browser pipeline, which uses channels 0 and 1 only.
    """
    wav = torch.randn(6, 100)
    out = convert_audio_channels(wav, 2)
    assert out.shape == (2, 100)
    assert torch.equal(out, wav[:2])


def test_convert_channels_downmix_to_mono() -> None:
    """
    Requesting a single channel averages all source channels.
    """
    wav = torch.randn(4, 100)
    out = convert_audio_channels(wav, 1)
    assert out.shape == (1, 100)
    assert torch.allclose(out, wav.mean(dim=0, keepdim=True))


def test_convert_channels_too_few_non_mono_raises() -> None:
    """
    Upmixing a non-mono source to more channels is unsupported.
    """
    wav = torch.randn(2, 100)
    with pytest.raises(ValidationError):
        convert_audio_channels(wav, 3)
    # Pre-v1 callers caught the builtin; ValidationError must stay one.
    assert issubclass(ValidationError, ValueError)


def test_prevent_clip_rescale_bounds_peak() -> None:
    """
    ``rescale`` brings the peak within [-1, 1] when the input exceeds it.
    """
    wav = torch.tensor([[2.0, -2.0, 1.0]])
    out = prevent_clip(wav, "rescale")
    assert out.abs().max() <= 1.0


def test_prevent_clip_rescale_all_zero_is_safe() -> None:
    """
    All-zero input must not divide by zero / produce NaNs under rescale.
    """
    wav = torch.zeros(2, 10)
    out = prevent_clip(wav, "rescale")
    assert torch.equal(out, wav)
    assert not torch.isnan(out).any()


def test_prevent_clip_clamp() -> None:
    """
    ``clamp`` hard-limits samples to [-0.99, 0.99].
    """
    wav = torch.tensor([[5.0, -5.0, 0.5]])
    out = prevent_clip(wav, "clamp")
    assert out.max() <= 0.99 and out.min() >= -0.99


def test_prevent_clip_tanh_and_none() -> None:
    """
    ``tanh`` applies the squashing function; ``None`` passes through.
    """
    wav = torch.randn(2, 10)
    assert torch.allclose(prevent_clip(wav, "tanh"), torch.tanh(wav))
    assert prevent_clip(wav, None) is wav


def test_prevent_clip_invalid_mode_raises() -> None:
    """
    An unknown clip mode is rejected with ``ValidationError``.
    """
    with pytest.raises(ValidationError):
        prevent_clip(torch.zeros(2, 10), "nonsense")
