"""Unit tests for ``demucs.apply`` (chunk views, routing, shifts, progress)."""

import pytest
import torch

from demucs.apply import TensorChunk, apply_model, tensor_chunk
from demucs.exceptions import ValidationError


def _ramp() -> torch.Tensor:
    """
    Build a deterministic ``[1, 10]`` ramp tensor for chunk assertions.

    :return: Tensor with values 0..9 along the last dimension.
    """
    return torch.arange(10, dtype=torch.float32)[None]


def test_full_chunk_shape_and_padded_identity() -> None:
    """
    A chunk over the whole tensor reports its shape and pads to a no-op.
    """
    t = _ramp()
    tc = TensorChunk(t)
    assert tc.shape == [1, 10]
    assert torch.equal(tc.padded(10), t)


def test_offset_and_length_clamp() -> None:
    """
    Length is clamped so a chunk never runs past the end of the tensor.
    """
    t = _ramp()
    assert TensorChunk(t, 8, 5).length == 2  # min(10 - 8, 5)
    assert TensorChunk(t, 2, 3).shape == [1, 3]


def test_padded_centers_and_zero_pads() -> None:
    """
    ``padded`` centers the chunk and zero-pads symmetrically.
    """
    t = _ramp()
    out = TensorChunk(t, 0, 10).padded(12)
    assert out.shape == (1, 12)
    # delta = 2 -> one zero on each side, original ramp in the middle.
    assert out[0, 0] == 0.0 and out[0, -1] == 0.0
    assert torch.equal(out[0, 1:11], t[0])


def test_negative_offset_rejected() -> None:
    """
    A negative offset is invalid.
    """
    with pytest.raises(ValidationError):
        TensorChunk(_ramp(), -1)


def test_empty_tensor_rejected() -> None:
    """
    A zero-length tensor cannot be wrapped (offset must be < total length).
    """
    with pytest.raises(ValidationError):
        TensorChunk(torch.zeros(1, 0))


def test_tensor_chunk_passthrough() -> None:
    """
    ``tensor_chunk`` wraps a raw tensor but passes an existing chunk through.
    """
    t = _ramp()
    tc = TensorChunk(t, 1, 4)
    assert tensor_chunk(tc) is tc
    assert isinstance(tensor_chunk(t), TensorChunk)


class _DoublingModel(torch.nn.Module):
    """
    Tiny stand-in model returning ``[x, 2x]`` stacked as two sources.

    Because it's pointwise, overlap-add and shift averaging must reproduce
    the input exactly — any chunk misrouting shows up as a mismatch.
    """

    sources = ["one", "two"]
    samplerate = 100
    audio_channels = 1
    max_allowed_segment = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stack ``x`` and ``2x`` along a new sources dimension.

        :param x: Input of shape ``[batch, channels, samples]``.
        :return: Output of shape ``[batch, 2, channels, samples]``.
        """
        return torch.stack([x, 2 * x], dim=1)


def test_apply_model_batched_mix_routes_rows_independently() -> None:
    """
    A mix with batch dim > 1 separates each row independently (this used
    to misroute: all rows got row 0's chunks broadcast onto them).
    """
    model = _DoublingModel()
    mix = torch.randn(3, 1, 250)

    out = apply_model(model, mix)

    assert out.shape == (3, 2, 1, 250)
    assert torch.allclose(out[:, 0], mix, atol=1e-5)
    assert torch.allclose(out[:, 1], 2 * mix, atol=1e-5)


def test_apply_model_2d_mix_lifted_to_batch_one() -> None:
    """
    A 2-D ``[channels, samples]`` mix behaves as batch 1.
    """
    model = _DoublingModel()
    mix = torch.randn(1, 250)

    out = apply_model(model, mix)

    assert out.shape == (1, 2, 1, 250)
    assert torch.allclose(out[0, 0], mix, atol=1e-5)


def test_apply_model_shifts_progress_single_monotonic_span() -> None:
    """
    With shifts > 1, progress is one continuous span: a single start
    event whose total covers all rounds, strictly increasing counts, and
    completed == total at the end (previously it restarted per round).
    """
    model = _DoublingModel()
    mix = torch.randn(1, 1, 250)
    events: list[tuple[str, dict]] = []

    out = apply_model(
        model,
        mix,
        shifts=3,
        progress_callback=lambda e, d: events.append((e, dict(d))),
    )
    assert torch.allclose(out[:, 0], mix, atol=1e-5)

    starts = [d for e, d in events if e == "processing_start"]
    completes = [d for e, d in events if e == "processing_complete"]
    chunks = [d for e, d in events if e == "chunk_complete"]
    assert len(starts) == 1
    assert len(completes) == 1

    total = starts[0]["total_chunks"]
    assert {d["total_chunks"] for d in chunks} == {total}
    assert [d["completed_chunks"] for d in chunks] == list(range(1, total + 1))
