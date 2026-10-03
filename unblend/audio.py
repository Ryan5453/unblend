# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import torch
import torchaudio
from torch import Tensor

from .exceptions import ValidationError

AUDIO_SUFFIXES = frozenset(
    {
        ".aac",
        ".ac3",
        ".aif",
        ".aifc",
        ".aiff",
        ".alac",
        ".amr",
        ".ape",
        ".au",
        ".avi",
        ".caf",
        ".flac",
        ".m4a",
        ".m4b",
        ".m4p",
        ".m4r",
        ".m4v",
        ".mka",
        ".mkv",
        ".mov",
        ".mp2",
        ".mp3",
        ".mp4",
        ".oga",
        ".ogg",
        ".opus",
        ".spx",
        ".tta",
        ".w64",
        ".wav",
        ".webm",
        ".wma",
        ".wv",
    }
)


def resolve_output_path(path: Path, format: str) -> Path:
    """
    Append ``.{format}`` unless ``path`` already ends in an audio extension.

    A dot elsewhere in the name (``"Song ft. Artist"``) is not an extension.

    :param path: Requested output path.
    :param format: Container to use when the path has no audio extension.
    :return: The path that will be written.
    """
    if path.suffix.lower() in AUDIO_SUFFIXES:
        return path
    return path.with_name(f"{path.name}.{format}")


def _caller_stacklevel() -> int:
    """
    The ``warnings.warn`` stacklevel of the first frame outside this package,
    so a warning points at the caller's line however deep the call is.

    :return: Stack level for a warning raised by this function's caller.
    """
    package = os.path.dirname(os.path.abspath(__file__))
    frame = sys._getframe(2)  # the function that is about to warn, plus one
    level = 2
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        package + os.sep
    ):
        frame = frame.f_back
        level += 1
    return level


def convert_audio_channels(wav: Tensor, channels: int = 2) -> Tensor:
    """
    Convert audio to the given number of channels.

    :param wav: Audio tensor to convert
    :param channels: Target number of channels
    :return: Audio tensor with the target number of channels
    :raises ValidationError: If the audio has fewer channels than requested but is not mono
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        wav = wav.expand(*shape, channels, length).contiguous()
    elif src_channels >= channels:
        # Surround layouts put centre (often the vocals) third; keeping the
        # first two channels drops it, so say so.
        warnings.warn(
            f"Input has {src_channels} channels; using the first {channels}.",
            stacklevel=_caller_stacklevel(),
        )
        wav = wav[..., :channels, :]
    else:
        raise ValidationError(
            "The audio has fewer channels than requested but is not mono."
        )
    return wav


# Largest numerator/denominator of a resampling ratio resampled exactly.
_MAX_RESAMPLE_TERM = 1000


def convert_audio(
    wav: Tensor, from_samplerate: int, to_samplerate: int, channels: int
) -> Tensor:
    """
    Convert audio to a target sample rate and number of channels.

    :param wav: Audio tensor to convert
    :param from_samplerate: Source sample rate
    :param to_samplerate: Target sample rate
    :param channels: Target number of channels
    :return: Converted audio tensor
    :raises ValidationError: If a rate isn't positive, or the rates are more
        than ``_MAX_RESAMPLE_TERM`` times apart.
    """
    for rate in (from_samplerate, to_samplerate):
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
            raise ValidationError(
                f"Sample rates must be positive integers, got {rate!r}."
            )
    wav = convert_audio_channels(wav, channels)
    if from_samplerate == to_samplerate:
        return wav
    ratio = Fraction(to_samplerate, from_samplerate)
    if not Fraction(1, _MAX_RESAMPLE_TERM) <= ratio <= _MAX_RESAMPLE_TERM:
        raise ValidationError(
            f"Can't resample {from_samplerate} Hz audio to {to_samplerate} Hz: "
            f"the rates are more than {_MAX_RESAMPLE_TERM} times apart."
        )
    if max(ratio.numerator, ratio.denominator) <= _MAX_RESAMPLE_TERM:
        return torchaudio.functional.resample(wav, from_samplerate, to_samplerate)
    # torchaudio's filter bank grows with the reduced ratio's terms: 44099 ->
    # 44100 Hz would take ~15 GB. A nearby ratio with both terms small shifts
    # pitch by far less than anyone can hear; the length is then made exact.
    # Bounding the smaller side's term bounds the larger one too.
    if ratio < 1:
        approx = ratio.limit_denominator(_MAX_RESAMPLE_TERM)
    else:
        approx = 1 / (1 / ratio).limit_denominator(_MAX_RESAMPLE_TERM)
    out = torchaudio.functional.resample(wav, approx.denominator, approx.numerator)
    length = round(wav.shape[-1] * to_samplerate / from_samplerate)
    if out.shape[-1] >= length:
        return out[..., :length]
    return torch.nn.functional.pad(out, (0, length - out.shape[-1]))


def prevent_clip(audio: Tensor, mode: str | None = "rescale") -> Tensor:
    """
    Keep audio inside full scale before encoding.

    ``"rescale"`` divides by ``max(1, 1.01 * peak)``, so it only acts on
    peaks above about 0.99; ``"clamp"`` hard-clips at 0.99; ``"tanh"`` soft-clips, and ``None``
    leaves it unchanged.

    :param audio: The audio tensor to prevent clipping from
    :param mode: ``"rescale"``, ``"clamp"``, ``"tanh"``, or None
    :return: The audio tensor with clipping prevented
    :raises ValidationError: If the clipping mode is invalid
    """
    if mode == "rescale":
        return audio / (1.01 * audio.abs().max()).clamp(min=1.0)
    elif mode == "clamp":
        return audio.clamp(-0.99, 0.99)
    elif mode == "tanh":
        return torch.tanh(audio)
    elif mode is None:
        return audio
    else:
        raise ValidationError(
            f"Invalid clip mode '{mode}'. Must be one of: 'rescale', 'clamp', 'tanh', or None"
        )
