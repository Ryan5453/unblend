# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio
from torch import Tensor

from .exceptions import ValidationError


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
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        # ``.contiguous()`` materialises the broadcast: ``expand`` returns a
        # stride-0 view, which breaks downstream ops that require a contiguous
        # last dim.
        wav = wav.expand(*shape, channels, length).contiguous()
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValidationError(
            "The audio file has less channels than requested but is not mono."
        )
    return wav


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
    """
    wav = convert_audio_channels(wav, channels)
    if from_samplerate == to_samplerate:
        return wav
    return torchaudio.functional.resample(wav, from_samplerate, to_samplerate)


def prevent_clip(audio: Tensor, mode: str | None = "rescale") -> Tensor:
    """
    Different strategies for avoiding raw clipping.

    :param audio: The audio tensor to prevent clipping from
    :param mode: The mode to use for preventing clipping ("rescale", "clamp", "tanh", or None)
    :return: The audio tensor with clipping prevented
    :raises ValidationError: If the clipping mode is invalid
    """
    if mode == "rescale":
        return audio / (1.01 * audio.abs().max()).clamp(min=1.0)
    elif mode == "clamp":
        return audio.clamp(-0.99, 0.99)
    elif mode == "tanh":
        return torch.tanh(audio)
    elif not mode:
        return audio
    else:
        raise ValidationError(
            f"Invalid clip mode '{mode}'. Must be one of: 'rescale', 'clamp', 'tanh', or None"
        )
