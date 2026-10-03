"""
CUDA ``torch.compile`` coverage for both RoFormer architectures.
"""

from typing import Callable

import pytest
import torch

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

    model.enable_compiled_core()
    with torch.inference_mode():
        actual = model(audio)
        replay = model(audio)
    torch.cuda.synchronize()

    assert model._fixed_batch_shape is True
    assert set(model.state_dict()) == state_keys
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(replay, actual, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("builder", [_tiny_bs, _tiny_mel], ids=["bs", "mel"])
def test_primed_rotary_tables_are_skipped_at_another_length(
    builder: Callable[[], _RoformerBase],
) -> None:
    """
    Rotary tables primed for the training segment are used under compile only
    when the sequence matches; a trace at another length (a direct call, or a
    copy that kept them) builds its own instead of failing to broadcast.
    """
    torch.manual_seed(3)
    model: _RoformerBase = builder()
    model.prefill_inference_caches()
    audio = torch.randn(1, 2, 11025)  # half the 22050-sample training segment
    with torch.inference_mode():
        expected = model(audio)
        actual = torch.compile(model, backend="eager")(audio)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_a_compiled_call_at_an_unprimed_length_does_not_cache_inference_tensors() -> (
    None
):
    """
    Tables built inside a traced graph aren't cached (there they would be
    inference tensors, and under CUDAGraphs replay-owned memory), so the cache
    doesn't grow and a later grad-enabled forward at that length still trains.
    """
    from unblend.roformer import RotaryEmbedding

    torch.manual_seed(4)
    model = _tiny_bs()
    model.prefill_inference_caches()
    rotaries = [m for m in model.modules() if isinstance(m, RotaryEmbedding)]
    before = [dict(r._cos_sin_cache) for r in rotaries]
    audio = torch.randn(1, 2, 11025)
    with torch.inference_mode():
        torch.compile(model, backend="aot_eager")(audio)
    assert [dict(r._cos_sin_cache) for r in rotaries] == before
    model(audio).sum().backward()
