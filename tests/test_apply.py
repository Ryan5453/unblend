"""Unit tests for ``unblend.apply`` (chunk views, routing, shifts, progress)."""

import pytest
import torch
from torch import nn

from unblend.apply import (
    ModelEnsemble,
    TensorChunk,
    _should_restore_submodel_device,
    apply_model,
    apply_model_multi,
    tensor_chunk,
)
from unblend.exceptions import ValidationError


def test_should_restore_submodel_device_same_device_is_noop() -> None:
    """
    No restore needed when the sub-model already lives on the inference device.
    """
    sub = nn.Linear(1, 1)
    device = torch.device("cpu")
    assert _should_restore_submodel_device(sub, device, device) is False


def test_should_restore_submodel_device_no_params_is_noop() -> None:
    """
    A sub-model without parameters has no original device to restore to, so
    nothing to do.
    """
    sub = nn.Linear(1, 1)
    assert _should_restore_submodel_device(sub, None, torch.device("cuda")) is False


def test_should_restore_submodel_device_uncompiled_returns_true() -> None:
    """
    Eager sub-models get restored — the classic BagOfModels behavior — so
    only the active member stays resident on the inference device.
    """
    sub = nn.Linear(1, 1)
    assert (
        _should_restore_submodel_device(sub, torch.device("cpu"), torch.device("cuda"))
        is True
    )


@pytest.mark.parametrize(
    "marker", ["_uncompiled_forward_core", "_uncompiled_run_transformers"]
)
def test_should_restore_submodel_device_compiled_skips_restore(marker: str) -> None:
    """
    Compiled HTDemucs and RoFormer sub-models stay on the inference device —
    bouncing them off invalidates the CUDAGraphs capture.

    :param marker: Family-specific attribute recording the eager callable.
    """
    sub = nn.Linear(1, 1)
    setattr(sub, marker, lambda *_args, **_kwargs: None)
    assert (
        _should_restore_submodel_device(sub, torch.device("cpu"), torch.device("cuda"))
        is False
    )


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


def test_model_ensemble_rejects_zero_weight_total() -> None:
    """A per-source zero weight total is rejected before inference."""
    with pytest.raises(ValidationError, match="non-zero total"):
        ModelEnsemble(
            [_DoublingModel(), _DoublingModel()],
            weights=[[1.0, 1.0], [-1.0, 1.0]],
        )


def test_model_ensemble_revalidates_mutated_weights() -> None:
    """Post-construction weight mutation cannot cause silent NaN output."""
    ensemble = ModelEnsemble([_DoublingModel()])
    ensemble.weights[0][0] = 0.0
    with pytest.raises(ValidationError, match="non-zero total"):
        apply_model(ensemble, torch.randn(1, 100))

    ensemble.weights[0] = [1.0, 0.0]
    with pytest.raises(ValidationError, match="non-zero total"):
        apply_model(
            ensemble,
            torch.randn(1, 100),
            use_only_stem="one",
        )


def test_specialist_shortcut_requires_exclusive_stem_weight() -> None:
    """A one-hot row cannot bypass another model contributing to that stem."""

    class DifferentModel(_DoublingModel):
        """Return distinguishable values for both sources."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``3x`` and ``4x`` as the two sources."""
            return torch.stack([3 * x, 4 * x], dim=1)

    ensemble = ModelEnsemble(
        [_DoublingModel(), DifferentModel()],
        weights=[[1.0, 0.0], [1.0, 1.0]],
    )
    mix = torch.randn(1, 100)

    expected = apply_model(ensemble, mix)
    actual = apply_model(ensemble, mix, use_only_stem="one")

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 0], 2 * mix[None])


def test_model_ensemble_propagates_contract_and_segment_cap() -> None:
    """Raw-audio ensembles preserve normalization and finite segment limits."""
    first = _DoublingModel()
    second = _DoublingModel()
    first.external_normalization = False
    second.external_normalization = False
    first.max_allowed_segment = 2.5
    second.max_allowed_segment = 3.0

    ensemble = ModelEnsemble([first, second], segment=4.0)

    assert ensemble.external_normalization is False
    assert ensemble.max_allowed_segment == 2.5
    assert first.max_allowed_segment == 2.5
    assert second.max_allowed_segment == 3.0


def test_model_ensemble_rejects_mixed_normalization_contracts() -> None:
    """Members requiring raw and externally-normalized audio cannot mix."""
    raw = _DoublingModel()
    raw.external_normalization = False
    with pytest.raises(ValidationError, match="external_normalization"):
        ModelEnsemble([_DoublingModel(), raw])


def test_htdemucs_mask_without_cac_applies_real_mask() -> None:
    """Non-CaC decoding applies a real mask while preserving mixture phase."""
    from unblend.htdemucs import HTDemucs

    model = object.__new__(HTDemucs)
    model.cac = False
    mixture = torch.randn(2, 2, 3, 4, dtype=torch.complex64)
    mask = torch.randn(2, 5, 2, 3, 4)

    actual = model._mask(mixture, mask)

    assert actual.shape == (2, 5, 2, 3, 4)
    assert torch.equal(actual, mixture[:, None] * mask)


def test_htdemucs_mask_with_cac_decodes_complex_channels() -> None:
    """CaC decoding still reconstructs adjacent real/imaginary channels."""
    from unblend.htdemucs import HTDemucs

    model = object.__new__(HTDemucs)
    model.cac = True
    target = torch.randn(2, 3, 2, 4, 5, dtype=torch.complex64)
    encoded = (
        torch.view_as_real(target).permute(0, 1, 2, 5, 3, 4).reshape(2, 3, 4, 4, 5)
    )

    actual = model._mask(torch.empty(0), encoded)

    assert torch.equal(actual, target)


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


def test_apply_model_multi_reports_aggregate_and_per_input_progress() -> None:
    """
    List-input chunk pooling emits one monotonic aggregate span plus enough
    input metadata for independent per-file progress displays.
    """
    model = _DoublingModel()
    mixes = [torch.randn(1, 1, 250), torch.randn(1, 1, 170)]
    events: list[tuple[str, dict]] = []

    outputs = apply_model_multi(
        model,
        mixes,
        shifts=2,
        chunk_batch_size=2,
        progress_callback=lambda event, data: events.append((event, dict(data))),
    )
    assert len(outputs) == 2

    starts = [data for event, data in events if event == "processing_start"]
    completes = [data for event, data in events if event == "processing_complete"]
    chunks = [data for event, data in events if event == "chunk_complete"]
    assert len(starts) == 1
    assert len(completes) == 1
    assert starts[0]["total_inputs"] == 2
    assert completes[0] == starts[0]

    total = starts[0]["total_chunks"]
    assert [data["completed_chunks"] for data in chunks] == list(range(1, total + 1))
    assert sum(starts[0]["input_total_chunks"]) == total
    for input_index, input_total in enumerate(starts[0]["input_total_chunks"]):
        input_events = [data for data in chunks if data["input_index"] == input_index]
        assert [data["input_completed_chunks"] for data in input_events] == list(
            range(1, input_total + 1)
        )
        assert {data["input_total_chunks"] for data in input_events} == {input_total}


def test_apply_model_rejects_out_of_range_overlap() -> None:
    """
    ``overlap`` outside ``[0, 1)`` is rejected up front — a negative overlap
    used to leave uncovered sample ranges and silently return NaN audio.
    """
    model = _DoublingModel()
    mix = torch.randn(1, 250)
    for overlap in (-1.0, 1.0, 1.5):
        with pytest.raises(ValidationError):
            apply_model(model, mix, overlap=overlap)


def test_htdemucs_forward_rejects_overlength_input() -> None:
    """
    ``HTDemucs.forward`` only supports inputs up to the training length —
    longer ones used to silently return wrong-shaped output because the
    time-branch ``view`` reinterpreted samples as channels. ``apply_model``
    is the supported path for full-length audio.
    """
    from unblend.htdemucs import HTDemucs

    model = HTDemucs(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )
    model.eval()
    with pytest.raises(ValidationError):
        with torch.no_grad():
            model(torch.randn(1, 2, 16000))


def test_htdemucs_freq_emb_cache_invalidated_on_weight_reload() -> None:
    """
    Reloading weights into an already-used ``HTDemucs`` must not keep serving
    the previous weights' memoised frequency embedding.
    """
    from unblend.htdemucs import HTDemucs

    kwargs = dict(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )
    torch.manual_seed(0)
    used = HTDemucs(**kwargs)
    torch.manual_seed(1)
    fresh = HTDemucs(**kwargs)
    used.eval()
    fresh.eval()

    x = torch.randn(1, 2, 4000)
    with torch.no_grad():
        used(x)  # populate the freq-emb cache with `used`'s weights
        used.load_state_dict(fresh.state_dict())
        assert torch.allclose(used(x), fresh(x), atol=1e-6)


class _FlakyOOMModel(_DoublingModel):
    """
    ``_DoublingModel`` that raises a CUDA-OOM-shaped RuntimeError whenever
    the batch is larger than ``fits`` — a GPU with room for ``fits`` chunks.
    """

    def __init__(self, fits: int) -> None:
        """
        :param fits: Largest batch dimension that "fits in memory".
        """
        super().__init__()
        self.fits = fits
        self.oom_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raise fake OOM above ``fits``, else behave like ``_DoublingModel``.

        :param x: Input of shape ``[batch, channels, samples]``.
        :return: Output of shape ``[batch, 2, channels, samples]``.
        """
        if x.shape[0] > self.fits:
            self.oom_count += 1
            raise RuntimeError("CUDA out of memory. (fake, for backoff test)")
        return super().forward(x)


def test_oom_backoff_halves_until_fit_and_output_is_exact() -> None:
    """
    Auto-sized runs degrade to a fitting batch size: 8 -> 4 -> 2 here, with
    the halvings recorded in the state dict and the output exact (the model
    is pointwise, so any dropped/duplicated chunk would show).
    """
    model = _FlakyOOMModel(fits=2)
    mix = torch.randn(1, 1, 250)
    state = {"chunk_batch_size": 8}

    out = apply_model(model, mix, chunk_batch_size=8, oom_backoff_state=state)

    assert state["chunk_batch_size"] == 2
    assert model.oom_count == 2
    assert torch.allclose(out[:, 0], mix, atol=1e-5)
    assert torch.allclose(out[:, 1], 2 * mix, atol=1e-5)


def test_oom_without_backoff_state_propagates() -> None:
    """
    No state dict (explicit sizing) means OOM raises untouched.
    """
    model = _FlakyOOMModel(fits=1)
    with pytest.raises(RuntimeError, match="out of memory"):
        apply_model(model, torch.randn(1, 1, 250), chunk_batch_size=4)


def test_oom_at_batch_one_raises_with_state_floored() -> None:
    """
    When even batch size 1 doesn't fit, the OOM propagates (the model
    genuinely doesn't fit) with the state floored at 1.
    """
    model = _FlakyOOMModel(fits=0)
    state = {"chunk_batch_size": 4}
    with pytest.raises(RuntimeError, match="out of memory"):
        apply_model(
            model, torch.randn(1, 1, 250), chunk_batch_size=4, oom_backoff_state=state
        )
    assert state["chunk_batch_size"] == 1


def test_non_oom_runtime_error_propagates_despite_backoff() -> None:
    """
    Backoff only rescues OOM-shaped failures; other RuntimeErrors raise
    with the state untouched.
    """

    class _Broken(_DoublingModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Always raise a non-OOM runtime error.

            :param x: Ignored.
            :return: Never returns.
            """
            raise RuntimeError("cuDNN launch failure (not memory)")

    state = {"chunk_batch_size": 4}
    with pytest.raises(RuntimeError, match="cuDNN"):
        apply_model(
            _Broken(),
            torch.randn(1, 1, 250),
            chunk_batch_size=4,
            oom_backoff_state=state,
        )
    assert state["chunk_batch_size"] == 4


def test_fixed_batch_shape_blocks_in_apply_backoff() -> None:
    """
    Compiled models (``_fixed_batch_shape``) can't change shape here — the
    OOM propagates so the Separator can recapture instead.
    """
    model = _FlakyOOMModel(fits=1)
    model._fixed_batch_shape = True
    state = {"chunk_batch_size": 4}
    with pytest.raises(RuntimeError, match="out of memory"):
        apply_model(
            model, torch.randn(1, 1, 250), chunk_batch_size=4, oom_backoff_state=state
        )
    assert state["chunk_batch_size"] == 4


def test_oom_during_accumulation_phase_is_retry_safe(monkeypatch) -> None:
    """
    An OOM raised after the forward but during the (allocating) contribution
    phase must not double-count already-processed chunks on retry: output
    stays exactly equal to a clean run and progress never overshoots. Uses a
    non-pointwise model — overlap contributions differ chunk to chunk, so
    any double accumulation breaks equality (a pointwise model would hide
    it: consistent out/sum_weight doubling cancels in the division).
    """
    import unblend.apply as apply_mod

    class _PositionalModel(torch.nn.Module):
        """
        Non-pointwise stand-in: output depends on position within the chunk.
        """

        sources = ["one", "two"]
        samplerate = 100
        audio_channels = 1
        max_allowed_segment = 1.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Stack ``x`` and its running cumsum along a sources dimension.

            :param x: Input of shape ``[batch, channels, samples]``.
            :return: Output of shape ``[batch, 2, channels, samples]``.
            """
            return torch.stack([x, x.cumsum(-1)], dim=1)

    model = _PositionalModel()
    mix = torch.randn(1, 1, 250)
    clean = apply_model(model, mix, chunk_batch_size=4)

    real_center_trim = apply_mod.center_trim
    calls = {"n": 0}

    def flaky_trim(tensor: torch.Tensor, reference) -> torch.Tensor:
        """
        Raise a fake OOM on the third contribution of the first attempt.

        :param tensor: Tensor to trim.
        :param reference: Trim reference.
        :return: The trimmed tensor.
        """
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("CUDA out of memory (fake, contribution phase)")
        return real_center_trim(tensor, reference)

    monkeypatch.setattr(apply_mod, "center_trim", flaky_trim)

    events: list[tuple[str, dict]] = []
    state = {"chunk_batch_size": 4}
    out = apply_model(
        model,
        mix,
        chunk_batch_size=4,
        oom_backoff_state=state,
        progress_callback=lambda e, d: events.append((e, dict(d))),
    )

    assert torch.allclose(out, clean, atol=1e-6)
    assert state["chunk_batch_size"] == 2
    chunk_events = [d for e, d in events if e == "chunk_complete"]
    total = chunk_events[-1]["total_chunks"]
    assert chunk_events[-1]["completed_chunks"] == total
    assert all(d["completed_chunks"] <= total for d in chunk_events)
