"""
Unit tests for ``unblend.apply`` (chunk views, routing, shifts, progress).
"""

from types import SimpleNamespace

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


def test_should_restore_submodel_device_compiled_skips_restore() -> None:
    """
    Compiled sub-models stay on the inference device — bouncing them off
    invalidates the CUDAGraphs capture.
    """
    sub = nn.Linear(1, 1)
    setattr(sub, "_eager_core", lambda *_args, **_kwargs: None)
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
    """
    A per-source zero weight total is rejected before inference.
    """
    with pytest.raises(ValidationError, match="non-zero total"):
        ModelEnsemble(
            [_DoublingModel(), _DoublingModel()],
            weights=[[1.0, 1.0], [-1.0, 1.0]],
        )


def test_model_ensemble_revalidates_mutated_weights() -> None:
    """
    Post-construction weight mutation cannot cause silent NaN output.
    """
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
    """
    A one-hot row cannot bypass another model contributing to that stem.
    """

    class DifferentModel(_DoublingModel):
        """
        Return distinguishable values for both sources.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Return ``3x`` and ``4x`` as the two sources.
            """
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
    """
    Raw-audio ensembles preserve normalization and finite segment limits.
    """
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


class _OffsetModel(torch.nn.Module):
    """
    Adds a constant to the input, which is what makes normalisation visible:
    an affine model run on normalised audio and scaled back returns
    ``x + (1e-5 + std)`` rather than ``x + 1``.
    """

    sources = ["one", "two"]
    samplerate = 100
    audio_channels = 1
    max_allowed_segment = 1.0

    def __init__(self, external_normalization: bool) -> None:
        """
        :param external_normalization: Whether this member expects the caller
            to have normalised its input.
        """
        super().__init__()
        self.external_normalization = external_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return ``x + 1`` as both sources.

        :param x: Input of shape ``[batch, channels, samples]``.
        :return: Output of shape ``[batch, 2, channels, samples]``.
        """
        return torch.stack([x + 1, x + 1], dim=1)


def test_members_with_different_normalization_contracts_can_ensemble() -> None:
    """
    HTDemucs wants track-level normalised audio and the other architectures
    want it raw; a mixed ensemble takes raw audio and normalises around the
    members that need it, so each sees what it would see running alone.
    """
    ensemble = ModelEnsemble(
        [_OffsetModel(True), _OffsetModel(False)],
        weights=[[1.0, 0.0], [0.0, 1.0]],
    )
    assert ensemble.member_normalization == [True, False]
    # The caller is handed raw audio, since one member could not use normalised.
    assert ensemble.external_normalization is False

    mix = torch.randn(1, 400)
    reference = mix.mean(dim=0)
    std = reference.std(correction=1)

    out = apply_model(ensemble, mix)

    torch.testing.assert_close(
        out[:, 0], mix[None] + (1e-5 + std), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(out[:, 1], mix[None] + 1.0, rtol=1e-5, atol=1e-5)


def test_uniform_normalization_contract_stays_with_the_caller() -> None:
    """
    When every member agrees, the contract is the ensemble's and normalisation
    happens once in ``Separator`` — unchanged for the shipped Demucs bags.
    """
    ensemble = ModelEnsemble([_OffsetModel(True), _OffsetModel(True)])
    assert ensemble.external_normalization is True

    mix = torch.randn(1, 400)
    out = apply_model(ensemble, mix)

    # Nothing was normalised inside the ensemble: the members saw the raw input.
    torch.testing.assert_close(out[:, 0], mix[None] + 1.0)


def test_isolated_member_is_normalized_like_the_full_ensemble() -> None:
    """
    The single-member shortcut must apply that member's normalisation too —
    skipping it would feed raw audio to a model expecting normalised.
    """
    ensemble = ModelEnsemble(
        [_OffsetModel(True), _OffsetModel(False)],
        weights=[[1.0, 0.0], [0.0, 1.0]],
    )
    mix = torch.randn(1, 400)
    std = mix.mean(dim=0).std(correction=1)

    out = apply_model(ensemble, mix, use_only_stem="one")

    torch.testing.assert_close(
        out[:, 0], mix[None] + (1e-5 + std), rtol=1e-5, atol=1e-5
    )


def test_htdemucs_valid_length_matches_rounded_apply_segment() -> None:
    """
    Fractional seconds use the same sample conversion in validation/chunking.
    """
    from unblend.htdemucs import HTDemucs

    model = SimpleNamespace(max_allowed_segment=1.0001, samplerate=8000)
    assert HTDemucs.valid_length(model, 8001) == 8001


def test_htdemucs_mask_without_cac_applies_real_mask() -> None:
    """
    Non-CaC decoding applies a real mask while preserving mixture phase.
    """
    from unblend.htdemucs import HTDemucs

    model = object.__new__(HTDemucs)
    model.cac = False
    mixture = torch.randn(2, 2, 3, 4, dtype=torch.complex64)
    mask = torch.randn(2, 5, 2, 3, 4)

    actual = model._mask(mixture, mask)

    assert actual.shape == (2, 5, 2, 3, 4)
    assert torch.equal(actual, mixture[:, None] * mask)


def test_htdemucs_mask_with_cac_decodes_complex_channels() -> None:
    """
    CaC decoding still reconstructs adjacent real/imaginary channels.
    """
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


def test_ensemble_shifts_share_offsets_and_report_exact_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Different segment lengths still produce one exact, non-clamped span.
    """

    class ShortSegmentModel(_DoublingModel):
        max_allowed_segment = 0.6

    ensemble = ModelEnsemble([_DoublingModel(), ShortSegmentModel()])
    mix = torch.randn(1, 1, 260)
    draws: list[int] = []
    values = iter([0, 17, 49])

    def fake_randint(_low: int, _high: int) -> int:
        value = next(values)
        draws.append(value)
        return value

    monkeypatch.setattr("unblend.apply.random.randint", fake_randint)
    events: list[tuple[str, dict]] = []
    apply_model(
        ensemble,
        mix,
        shifts=3,
        progress_callback=lambda event, data: events.append((event, dict(data))),
    )

    starts = [data for event, data in events if event == "processing_start"]
    chunks = [data for event, data in events if event == "chunk_complete"]
    completes = [data for event, data in events if event == "processing_complete"]
    assert draws == [0, 17, 49]  # one plan, not one set per ensemble member
    assert len(starts) == len(completes) == 1
    total = starts[0]["total_chunks"]
    assert completes[0] == starts[0]
    assert len(chunks) == total
    assert [event["completed_chunks"] for event in chunks] == list(range(1, total + 1))
    assert chunks[-1]["input_completed_chunks"] == starts[0]["input_total_chunks"][0]


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
    An OOM raised after the forward but before the in-place-weighted views are
    committed must not double-count already-processed chunks on retry: output
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


class _ScalingModel(torch.nn.Module):
    """
    Stand-in model returning ``[a*x, b*x]``.

    Pointwise and linear, so every combine mode's expected output is a known
    multiple of the input: an average is the weighted mean of the scales, and
    a magnitude-keyed pick is whichever scale has the smallest/largest
    magnitude — in the waveform *and* the STFT domain, since the transform is
    linear too.
    """

    sources = ["one", "two"]
    samplerate = 100
    audio_channels = 1
    max_allowed_segment = 1.0
    external_normalization = False

    def __init__(self, first: float, second: float) -> None:
        """
        :param first: Scale applied for source ``one``.
        :param second: Scale applied for source ``two``.
        """
        super().__init__()
        self.scales = (first, second)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale the input independently per source.

        :param x: Input of shape ``[batch, channels, samples]``.
        :return: Output of shape ``[batch, 2, channels, samples]``.
        """
        return torch.stack([self.scales[0] * x, self.scales[1] * x], dim=1)


@pytest.mark.parametrize(
    "combine, expected_scale",
    [
        ("weighted_mean", 3.0),
        ("avg_wave", 3.0),
        ("median_wave", 2.0),
        ("min_wave", 1.0),
        ("max_wave", 6.0),
        ("avg_fft", 3.0),
        ("median_fft", 2.0),
        ("min_fft", 1.0),
        ("max_fft", 6.0),
        ("uvr_min_spec", 1.0),
        ("uvr_max_spec", 6.0),
    ],
)
def test_combine_modes_reduce_members_as_specified(
    combine: str, expected_scale: float
) -> None:
    """
    Every mode combines three members exactly as its definition says.

    Scales 1, 2 and 6 make the four reductions distinguishable: mean 3,
    median 2, min 1, max 6.
    """
    members = [_ScalingModel(scale, scale) for scale in (1.0, 2.0, 6.0)]
    ensemble = ModelEnsemble(members, combine=combine)
    mix = torch.randn(1, 400)

    out = apply_model(ensemble, mix)

    torch.testing.assert_close(
        out[:, 0], expected_scale * mix[None], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        out[:, 1], expected_scale * mix[None], rtol=2e-5, atol=2e-5
    )


def test_selection_modes_reject_non_binary_weights() -> None:
    """
    Real-valued weights have nowhere to apply in a min or a median, so they
    are rejected rather than silently ignored (as upstream tools do).
    """
    with pytest.raises(ValidationError, match="participation mask"):
        ModelEnsemble(
            [_ScalingModel(1.0, 1.0), _ScalingModel(2.0, 2.0)],
            weights=[[0.5, 1.0], [1.0, 1.0]],
            combine="min_wave",
        )


def test_weighted_mean_still_accepts_real_weights() -> None:
    """
    The blending default keeps its per-source weighted average.
    """
    ensemble = ModelEnsemble(
        [_ScalingModel(1.0, 1.0), _ScalingModel(3.0, 3.0)],
        weights=[[3.0, 1.0], [1.0, 1.0]],
    )
    mix = torch.randn(1, 200)

    out = apply_model(ensemble, mix)

    # Source one: (3*1 + 1*3)/4 = 1.5. Source two: (1 + 3)/2 = 2.
    torch.testing.assert_close(out[:, 0], 1.5 * mix[None])
    torch.testing.assert_close(out[:, 1], 2.0 * mix[None])


@pytest.mark.parametrize("combine", ["min_wave", "max_fft", "median_wave"])
def test_zero_weight_excludes_a_member_per_stem(combine: str) -> None:
    """
    Contribution is per stem: a zero drops that member from that stem, so a
    stem with one contributor passes straight through under any mode.
    """
    ensemble = ModelEnsemble(
        [_ScalingModel(5.0, 5.0), _ScalingModel(9.0, 9.0)],
        weights=[[1.0, 0.0], [0.0, 1.0]],
        combine=combine,
    )
    mix = torch.randn(1, 300)

    out = apply_model(ensemble, mix)

    torch.testing.assert_close(out[:, 0], 5.0 * mix[None], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(out[:, 1], 9.0 * mix[None], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("combine", ["weighted_mean", "max_fft", "median_wave"])
def test_isolate_stem_runs_one_member_under_every_mode(combine: str) -> None:
    """
    The single-stem shortcut is about contribution, not linearity: with one
    contributor every mode reduces to that member, so only it runs.
    """
    calls: list[int] = []

    class Counting(_ScalingModel):
        """
        Records that its forward ran.
        """

        def __init__(self, scale: float, tag: int) -> None:
            super().__init__(scale, scale)
            self.tag = tag

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Record the call, then scale as usual.
            """
            calls.append(self.tag)
            return super().forward(x)

    ensemble = ModelEnsemble(
        [Counting(4.0, 0), Counting(7.0, 1)],
        weights=[[1.0, 0.0], [0.0, 1.0]],
        combine=combine,
    )
    mix = torch.randn(1, 200)

    out = apply_model(ensemble, mix, use_only_stem="two")

    assert set(calls) == {1}, "only the contributing member should run"
    torch.testing.assert_close(out[:, 1], 7.0 * mix[None], rtol=2e-5, atol=2e-5)


def test_spectral_combine_is_seamless_across_blocks() -> None:
    """
    The spectral modes transform in blocks to bound memory; a tiny geometry
    forces several blocks, and the result must still be exact — a misaligned
    frame grid or an undiscarded margin would show up as a seam.
    """
    ensemble = ModelEnsemble(
        [_ScalingModel(1.0, 1.0), _ScalingModel(4.0, 4.0)],
        combine="max_fft",
        combine_params={"n_fft": 32, "hop_length": 8},
    )
    mix = torch.randn(1, 200_000)

    out = apply_model(ensemble, mix)

    torch.testing.assert_close(out[:, 0], 4.0 * mix[None], rtol=2e-4, atol=2e-4)


def test_unknown_combine_mode_is_rejected() -> None:
    """
    An unimplemented mode fails at construction, naming the alternatives.
    """
    with pytest.raises(ValidationError, match="Unknown ensemble combine mode"):
        ModelEnsemble([_ScalingModel(1.0, 1.0)], combine="telepathy")


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"n_fft": 1000, "hop_length": 256}, "whole multiple"),
        ({"n_fft": 0, "hop_length": 256}, "positive integer"),
        ({"hop_length": 1.5}, "positive integer"),
    ],
)
def test_combine_params_are_validated(params: dict, expected: str) -> None:
    """
    STFT geometry has to be usable before any audio is processed.
    """
    with pytest.raises(ValidationError, match=expected):
        ModelEnsemble(
            [_ScalingModel(1.0, 1.0)], combine="min_fft", combine_params=params
        )


def test_compile_is_applied_to_every_ensemble_member() -> None:
    """
    ``torch.compile`` targets each member's own hot path, so an ensemble
    compiles member by member — including one whose members are different
    architectures, where each has its own compiled core.
    """
    from unblend.api import Separator

    compiled: list[int] = []

    class Compilable(_OffsetModel):
        """
        Records that its compiled core was swapped in and out.
        """

        def __init__(self, tag: int, external_normalization: bool) -> None:
            super().__init__(external_normalization)
            self.tag = tag

        def enable_compiled_core(self) -> None:
            """
            Stand in for swapping in the compiled hot path.
            """
            compiled.append(self.tag)

        def disable_compiled_core(self) -> None:
            """
            Stand in for restoring the eager hot path.
            """
            compiled.remove(self.tag)

    ensemble = ModelEnsemble([Compilable(0, True), Compilable(1, False)])
    separator = Separator(model=ensemble, device="cpu")

    separator._setup_compile()
    assert sorted(compiled) == [0, 1], "every member should be compiled"

    separator._teardown_compile_state()
    assert compiled == [], "teardown must restore every member"
