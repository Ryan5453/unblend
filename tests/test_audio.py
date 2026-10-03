"""
Unit tests for the pure tensor helpers in ``unblend.audio``.
"""

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
    with pytest.warns(UserWarning, match="using the first 2") as record:
        out = convert_audio_channels(wav, 2)
    assert out.shape == (2, 100)
    assert torch.equal(out, wav[:2])
    # The warning points at the caller (this test), not at unblend.
    assert record[0].filename == __file__


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


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("out/vocals", "out/vocals.wav"),
        ("out/Song ft. Artist", "out/Song ft. Artist.wav"),
        ("out/song.v2_bass", "out/song.v2_bass.wav"),
        ("out/vocals.flac", "out/vocals.flac"),
        ("out/vocals.MP3", "out/vocals.MP3"),
    ],
)
def test_resolve_output_path(given: str, expected: str) -> None:
    """
    Only a real audio extension counts as the container.

    :param given: Requested path.
    :param expected: Path that will be written.
    """
    from pathlib import Path

    from unblend.audio import resolve_output_path

    assert resolve_output_path(Path(given), "wav") == Path(expected)


@pytest.mark.parametrize("fmt", ["opus", "webm"])
def test_opus_containers_export_at_a_supported_rate(tmp_path, fmt: str) -> None:
    """
    Opus can't encode 44.1 kHz, so those containers are written at 48 kHz.

    :param tmp_path: pytest temporary directory fixture
    :param fmt: Opus-based container
    """
    from torchcodec.decoders import AudioDecoder

    from unblend.api import SeparatedSources

    sources = SeparatedSources(
        {"v": torch.randn(2, 44100) * 0.1}, 44100, torch.zeros(2, 44100)
    )
    path = sources.export_stem("v", tmp_path / f"v.{fmt}")
    assert AudioDecoder(str(path)).metadata.sample_rate == 48000
    assert sources.export_stem("v", format=fmt)


@pytest.mark.parametrize("rate", [44099, 22051, 48000])
def test_resampling_odd_rates_is_cheap_and_exact_length(rate: int) -> None:
    """
    Rates whose ratio to 44.1 kHz reduces to huge terms (44099) are resampled
    through a close small-term ratio instead of a multi-GB filter bank, with
    the exact output length.

    :param rate: Input sample rate
    """
    from unblend.audio import convert_audio

    out = convert_audio(torch.randn(2, rate), rate, 44100, 2)
    assert out.shape == (2, 44100)


def test_resampling_keeps_both_ratio_terms_small() -> None:
    """
    A small source rate far below the target (997 -> 44100 Hz) is resampled
    through a nearby small-term ratio too, and rates that can't be resampled
    raise ``ValidationError`` instead of a raw error.
    """
    from unblend.audio import convert_audio
    from unblend.exceptions import ValidationError

    out = convert_audio(torch.randn(2, 997), 997, 44100, 2)
    assert out.shape == (2, 44100)
    for rates in ((0, 44100), (44100, -1), (10, 44100)):
        with pytest.raises(ValidationError):
            convert_audio(torch.randn(2, 100), *rates, 2)


def test_rates_exactly_1000_times_apart_resample_both_ways() -> None:
    """
    The 1000x bound is inclusive in both directions (an exact Fraction, not
    the float 1/1000, so 96000 -> 96 isn't refused while 96 -> 96000 passes).
    """
    from unblend.audio import convert_audio

    assert convert_audio(torch.randn(2, 96000), 96000, 96, 2).shape == (2, 96)
    assert convert_audio(torch.randn(2, 96), 96, 96000, 2).shape == (2, 96000)
