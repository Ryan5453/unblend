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


def test_checkpoint_at_any_precision_loads_into_fp32_modules() -> None:
    """
    Storage precision is independent of compute: an fp8 state dict loads.

    Modules are built in fp32 and ``load_state_dict`` widens on the way in, so
    a model trained at a precision unblend cannot compute in still runs here —
    the same mechanism that already carries the fp16 HTDemucs checkpoint.
    """
    module = torch.nn.Sequential(torch.nn.Conv1d(4, 4, 3), torch.nn.LSTM(4, 4))
    for dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        state = {
            key: value.to(dtype) if value.is_floating_point() else value
            for key, value in module.state_dict().items()
        }
        module.load_state_dict(state, strict=True)
        assert next(module.parameters()).dtype is torch.float32


def test_reduced_precision_error_separates_storage_from_compute() -> None:
    """
    A dtype we cannot compute in says so without implying storage is limited.
    """
    with pytest.raises(ValidationError) as excinfo:
        Separator(device="cpu", dtype=torch.float8_e4m3fn)
    assert "stored at, which may be anything" in str(excinfo.value)


def test_export_native_precision_follows_the_checkpoint_header() -> None:
    """
    "native" is the dtype the Safetensors header declares, not an inference.

    HTDemucs ships fp16 because upstream rounded it at release; the RoFormer
    and SCNet checkpoints are genuinely fp32.
    """
    from unblend.onnx import _resolve_export_precision
    from unblend.repo import ModelRepository

    models = ModelRepository().list_models()
    for name, expected in (
        ("htdemucs", torch.float16),
        ("htdemucs_6s", torch.float16),
        ("scnet_small", torch.float32),
        ("bs_roformer_anvuew", torch.float32),
    ):
        assert _resolve_export_precision("native", models[name]) is expected


def test_export_precision_falls_back_to_fp32_when_header_is_unreadable() -> None:
    """
    A missing or unreadable checkpoint exports at full width, not a guess.
    """
    from unblend.onnx import _resolve_export_precision

    absent = {"checkpoint": {"path": "/nonexistent/model.safetensors"}}
    assert _resolve_export_precision("native", absent) is torch.float32
    # A spec naming neither a local path nor a digest has nothing to inspect.
    assert _resolve_export_precision("native", {"checkpoint": {}}) is torch.float32
