"""CUDA ``torch.compile`` coverage for both RoFormer architectures."""

from typing import Callable

import pytest
import torch

from unblend.api import Separator
from unblend.roformer import BSRoformer, MelBandRoformer, _RoformerBase

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _tiny_bs() -> BSRoformer:
    """
    Build a small BS-RoFormer with production-shaped attention blocks.

    :return: Configured model in evaluation mode.
    """
    model = BSRoformer(
        dim=32,
        depth=1,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=16,
        heads=2,
    ).eval()
    model.configure_inference(
        sources=["vocals", "other"], samplerate=44100, segment_samples=22050
    )
    return model


def _tiny_mel() -> MelBandRoformer:
    """
    Build a small Mel-Band RoFormer with production-shaped attention blocks.

    :return: Configured model in evaluation mode.
    """
    model = MelBandRoformer(
        dim=32,
        depth=1,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        num_bands=8,
        dim_head=16,
        heads=2,
    ).eval()
    model.configure_inference(
        sources=["vocals", "other"], samplerate=44100, segment_samples=22050
    )
    return model


@cuda_only
@pytest.mark.parametrize("builder", [_tiny_bs, _tiny_mel], ids=["bs", "mel"])
def test_cuda_compiled_transformer_core_matches_eager(
    builder: Callable[[], _RoformerBase],
) -> None:
    """
    The family-specific Inductor/CUDAGraph target compiles and preserves an
    FP16 end-to-end forward for both RoFormer variants.
    """
    torch.manual_seed(11)
    model: _RoformerBase = builder().to(device="cuda", dtype=torch.float16)
    audio = torch.randn(1, 2, 22050, device="cuda")
    state_keys = set(model.state_dict())
    with torch.inference_mode():
        expected = model(audio)

    Separator._compile_roformer_transformer_core(model)
    with torch.inference_mode():
        actual = model(audio)
        replay = model(audio)
    torch.cuda.synchronize()

    assert model._fixed_batch_shape is True
    assert set(model.state_dict()) == state_keys
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(replay, actual, atol=3e-3, rtol=3e-3)
