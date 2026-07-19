"""
Unit tests for inference-precision resolution and the GPU-residency gate helpers.
"""

import pathlib

import pytest
import torch

from unblend.api import Separator, default_dtype
from unblend.apply import _gpu_accum_budget_bytes, _gpu_accum_bytes_needed
from unblend.exceptions import ValidationError


def test_default_dtype_cpu_is_fp32() -> None:
    """
    CPU has no faster-than-FP32 path, so auto resolves to None.
    """
    assert default_dtype("cpu") is None


def test_default_dtype_mps_is_fp16() -> None:
    """
    MPS auto picks FP16 (custom Metal kernels).
    """
    assert default_dtype("mps") is torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_default_dtype_cuda_matches_capability() -> None:
    """
    CUDA auto picks FP16 on tensor-core GPUs (cc >= 7.0), FP32 otherwise.
    """
    major, _ = torch.cuda.get_device_capability()
    expected = torch.float16 if major >= 7 else None
    assert default_dtype("cuda") is expected


def test_separator_rejects_unknown_dtype_string() -> None:
    """
    Only the literal 'auto' is accepted as a string dtype.
    """
    with pytest.raises(ValidationError):
        Separator(device="cpu", dtype="fp16")


def test_separator_rejects_reduced_precision_on_cpu() -> None:
    """
    Explicit FP16/BF16 on CPU is rejected before any model loading.
    """
    with pytest.raises(ValidationError):
        Separator(device="cpu", dtype=torch.float16)


def test_gpu_accum_bytes_needed_formula() -> None:
    """
    Bytes = fp32 mix + per-source fp32 accumulator + fp32 weight sum.
    """
    batch, sources, channels, length = 1, 4, 2, 1000
    expected = (batch * channels * (sources + 1) * length + length) * 4
    assert _gpu_accum_bytes_needed(batch, sources, channels, length) == expected


def test_gpu_accum_budget_is_zero_without_cuda() -> None:
    """
    Querying the budget must never raise, even with no usable CUDA device.
    """
    if torch.cuda.is_available():
        assert _gpu_accum_budget_bytes("cuda") >= 0
    else:
        assert _gpu_accum_budget_bytes("cuda") == 0


def test_read_pcm16_wav_matches_torchcodec(tmp_path: pathlib.Path) -> None:
    """
    The PCM16 WAV fast path decodes sample-exactly vs torchcodec.

    :param tmp_path: pytest temporary directory fixture
    """
    import wave

    import numpy as np
    from torchcodec.decoders import AudioDecoder

    rng = np.random.default_rng(0)
    samples = (rng.uniform(-0.5, 0.5, size=(2, 4410)) * 32767).astype("<i2")
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(samples.T.tobytes())

    fast = Separator._read_pcm16_wav(path)
    assert fast is not None
    wav, sr = fast
    assert sr == 44100 and wav.shape == (2, 4410)

    ref = AudioDecoder(str(path)).get_all_samples()
    assert torch.equal(wav, ref.data)


def test_read_pcm16_wav_rejects_non_wav(tmp_path: pathlib.Path) -> None:
    """
    Non-WAV input falls back (returns None) instead of raising.

    :param tmp_path: pytest temporary directory fixture
    """
    path = tmp_path / "notwav.mp3"
    path.write_bytes(b"\x00" * 64)
    assert Separator._read_pcm16_wav(path) is None
