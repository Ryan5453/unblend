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
    with pytest.raises(ValidationError, match="non-zero total|no member contributing"):
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
    with pytest.raises(ValidationError, match="non-zero total|no member contributing"):
        apply_model(ensemble, torch.randn(1, 100))

    ensemble.weights[0] = [1.0, 0.0]
    with pytest.raises(ValidationError, match="non-zero total|no member contributing"):
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


def test_htdemucs_refuses_cac_false() -> None:
    """
    Upstream's magnitude-mask decoding needs Wiener filtering, which isn't
    implemented, so ``cac=False`` is refused rather than giving wrong output.
    """
    from unblend.htdemucs import HTDemucs

    with pytest.raises(ValidationError, match="cac=True"):
        HTDemucs(sources=["a", "b"], cac=False)


def test_htdemucs_mask_with_cac_decodes_complex_channels() -> None:
    """
    CaC decoding still reconstructs adjacent real/imaginary channels.
    """
    from unblend.htdemucs import HTDemucs

    model = object.__new__(HTDemucs)
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
    completed == total at the end, rather than restarting per round.
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
    ``overlap`` outside ``[0, 1)`` is rejected up front; a negative overlap
    would leave uncovered sample ranges and return NaN audio.
    """
    model = _DoublingModel()
    mix = torch.randn(1, 250)
    for overlap in (-1.0, 1.0, 1.5):
        with pytest.raises(ValidationError):
            apply_model(model, mix, overlap=overlap)


def test_htdemucs_forward_rejects_overlength_input() -> None:
    """
    ``HTDemucs.forward`` only supports inputs up to the training length and
    rejects longer ones, whose time-branch ``view`` would reinterpret samples
    as channels. ``apply_model`` is the supported path for full-length audio.
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

    # Replacing the parameter twice without a forward can hand the second
    # one the first one's freed id at the same version; still not stale.
    for _ in range(20):
        with torch.no_grad():
            used(x)
        for _ in range(2):
            used.freq_emb.embedding.weight = torch.nn.Parameter(
                torch.randn_like(used.freq_emb.embedding.weight)
            )
        with torch.no_grad():
            expected = used.freq_emb(
                torch.arange(used.freq_emb.embedding.num_embeddings)
            )
            got = used._cached_freq_emb(
                used.freq_emb.embedding.num_embeddings, x.device, x.dtype
            )
            assert torch.allclose(got[0, :, :, 0], expected.t().to(got.dtype))

    # An in-place update (an optimizer step, a copy_) must not be served the
    # old embedding either.
    torch.manual_seed(2)
    other = HTDemucs(**kwargs).eval()
    with torch.no_grad():
        used(x)
        for target, source in zip(used.parameters(), other.parameters()):
            target.copy_(source)
        assert torch.allclose(used(x), other(x), atol=1e-6)


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


def test_separator_combine_override_keeps_ensemble_stft_geometry() -> None:
    """
    Overriding only the mode, or only one STFT key, keeps the rest of the
    ensemble's own ``combine_params``, without changing the caller's ensemble.
    """
    from unblend.api import Separator

    ensemble = ModelEnsemble(
        [_ScalingModel(1.0, 1.0), _ScalingModel(1.0, 1.0)],
        combine="min_fft",
        combine_params={"n_fft": 2048, "hop_length": 512},
    )
    sep = Separator(model=ensemble, device="cpu", combine="max_fft")
    assert sep.model.combine_params == {"n_fft": 2048, "hop_length": 512}
    sep = Separator(model=ensemble, device="cpu", combine_params={"hop_length": 256})
    assert sep.model.combine_params == {"n_fft": 2048, "hop_length": 256}
    # Valid against the ensemble's n_fft (2048), not the 1024 default.
    sep = Separator(model=ensemble, device="cpu", combine_params={"hop_length": 1024})
    assert sep.model.combine_params == {"n_fft": 2048, "hop_length": 1024}
    with pytest.raises(ValidationError, match="must be a mapping"):
        Separator(model=ensemble, device="cpu", combine_params=5)  # type: ignore[arg-type]
    # With only_load leaving one member, the unused override is still checked
    # against the ensemble's own geometry, not the default.
    one_hot = ModelEnsemble(
        [_ScalingModel(1.0, 1.0), _ScalingModel(1.0, 1.0)],
        weights=[[1.0, 0.0], [0.0, 1.0]],
        combine="min_fft",
        combine_params={"n_fft": 2048, "hop_length": 512},
    )
    sources = one_hot.sources
    isolated = Separator(
        model=one_hot,
        device="cpu",
        only_load=sources[0],
        combine_params={"hop_length": 1024},
    )
    assert not isinstance(isolated.model, ModelEnsemble)
    with pytest.raises(ValidationError):
        Separator(
            model=one_hot,
            device="cpu",
            only_load=sources[0],
            combine_params={"hop_length": 4096},
        )
    # The override is the Separator's; the caller's ensemble is untouched.
    assert ensemble.combine == "min_fft"
    assert ensemble.combine_params == {"n_fft": 2048, "hop_length": 512}


def test_separator_only_load_reduces_passed_in_ensemble() -> None:
    """
    A passed-in ensemble whose weights give the requested stem to one member
    keeps only that member, as a registry ensemble does; ``combine`` on a
    single model is still rejected even with ``only_load``.
    """
    from unblend.api import Separator

    first, second = _ScalingModel(1.0, 1.0), _ScalingModel(2.0, 2.0)
    ensemble = ModelEnsemble([first, second], weights=[[1.0, 0.0], [0.0, 1.0]])
    sep = Separator(model=ensemble, device="cpu", only_load="two")
    assert sep.model is second
    sep = Separator(model=ensemble, device="cpu", only_load="two", combine="max_wave")
    assert sep.model is second
    with pytest.raises(ValidationError):
        Separator(model=ensemble, device="cpu", only_load="two", combine="bogus")
    with pytest.raises(ValidationError, match="single member"):
        Separator(model=first, device="cpu", only_load="two", combine="max_wave")


def test_compile_is_applied_to_every_compilable_ensemble_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``torch.compile`` targets each member's own hot path, member by member,
    and skips members whose family has no compile target (SCNet).

    :param monkeypatch: pytest monkeypatch fixture
    """
    import unblend.api as api
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

    class Supported(Compilable):
        """
        Stands in for a family with a compile target.
        """

    monkeypatch.setattr(api, "HTDemucs", Supported)
    ensemble = ModelEnsemble(
        [Supported(0, True), Supported(1, False), Compilable(2, False)]
    )
    separator = Separator(model=ensemble, device="cpu")

    separator._setup_compile()
    assert sorted(compiled) == [0, 1], "only compilable members are compiled"

    separator._teardown_compile_state()
    assert compiled == [], "teardown must restore every member"


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(chunk_batch_size=0), "chunk_batch_size"),
        (dict(shifts=-1), "shifts"),
    ],
)
def test_apply_model_rejects_bad_arguments(kwargs: dict, expected: str) -> None:
    """
    Out-of-range arguments raise ``ValidationError`` rather than a raw
    ``ZeroDivisionError``/``RuntimeError`` from deep inside chunking.

    :param kwargs: Bad argument under test
    :param expected: Text the error must mention
    """
    with pytest.raises(ValidationError, match=expected):
        apply_model(_ScalingModel(1.0, 1.0), torch.zeros(2, 100), **kwargs)


def test_non_numeric_selection_weights_raise_validation_error() -> None:
    """
    Weight types are checked before the selection-mode mask rule reads them.
    """
    with pytest.raises(ValidationError, match="numeric"):
        ModelEnsemble(
            [_ScalingModel(1.0, 1.0), _ScalingModel(1.0, 1.0)],
            weights=[["a", "b"], [1, 1]],
            combine="min_fft",
        )


@pytest.mark.parametrize("transition_power", [1.0, 12.0])
@pytest.mark.parametrize("shifts", [0, 2])
def test_identity_model_returns_its_input(transition_power: float, shifts: int) -> None:
    """
    Overlap-add, shifts and a steep transition power reconstruct an identity
    model's input exactly (a high power used to underflow the edge weights
    and give NaN).

    :param transition_power: Fade exponent.
    :param shifts: Shift rounds.
    """
    model = _ScalingModel(1.0, 1.0)
    model.max_allowed_segment = 400.0  # 40000 samples: (1/20000)^12 underflows
    mix = torch.randn(1, 1, 100_000)
    out = apply_model(
        model, mix, shifts=shifts, overlap=0.25, transition_power=transition_power
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0, 0], mix[0], atol=1e-5, rtol=1e-5)


def test_apply_model_rejects_mixes_of_the_wrong_rank() -> None:
    """
    1-D and 4-D input raise ``ValidationError`` rather than an indexing or
    broadcasting error from deep inside the chunk loop.
    """
    model = _ScalingModel(1.0, 1.0)
    for shape in ((250,), (1, 1, 1, 250)):
        with pytest.raises(ValidationError, match="shape"):
            apply_model(model, torch.randn(*shape))


def test_negligible_ensemble_weights_are_refused() -> None:
    """
    A source whose contributing weights are negligible or cancel to about
    zero is refused up front, instead of a spectral combine crashing or
    returning NaN, or a weighted mean blowing the stem up.
    """
    models = [_ScalingModel(1.0, 1.0) for _ in range(3)]
    for weights in (
        [[1e-10, 1.0], [0.0, 1.0], [0.0, 1.0]],  # only a negligible member
        [[1.0, 1.0], [-1.0, 1.0], [1e-10, 1.0]],  # cancels to ~1e-10
        [[0.1, 1.0], [0.2, 1.0], [-0.3, 1.0]],  # cancels to ~5e-17
        [[1.0, 1.0], [-0.99999999, 1.0], [0.0, 1.0]],  # 0 at float32 precision
        # Contributors sum to ~0 over all members though the used ones don't.
        [[1.0, 1.0], [-0.999999998, 1.0], [-1e-9, 1.0]],
        [[1e300, 1.0], [1e300, 1.0], [0.0, 1.0]],  # overflows float32
        [[1e-9, 1.0], [1e-9, 1.0], [2e-9, 1.0]],  # modes disagree near cutoff
    ):
        for combine in ("avg_fft", "weighted_mean"):
            with pytest.raises(
                ValidationError,
                match="non-zero total|no member contributing|at most|tiny",
            ):
                ModelEnsemble(models, weights=weights, combine=combine)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_steep_transition_keeps_track_edges_on_mps() -> None:
    """
    MPS flushes denormals to zero, so a steep ``transition_power`` must not
    push edge weights times audio into the denormal range and silence the
    start and end of the track.
    """
    model = _ScalingModel(1.0, 1.0)
    model.max_allowed_segment = 400.0
    mix = torch.randn(1, 1, 100_000)
    out = apply_model(model, mix, device="mps", shifts=0, transition_power=32.0)
    out = out[0, 0].cpu()
    assert not ((out == 0) & (mix[0] != 0)).any()
    torch.testing.assert_close(out, mix[0], atol=1e-5, rtol=1e-5)


def test_a_largest_weight_of_exactly_one_millionth_is_accepted() -> None:
    """
    The "all tiny" floor is 1e-6 itself, not 1000 * 1e-9 (a hair above it).
    """
    from unblend.apply import check_weight_totals

    check_weight_totals([[1e-6], [0.0]], ["a"])
    with pytest.raises(ValidationError, match="all tiny"):
        check_weight_totals([[9.99e-7], [0.0]], ["a"])


def test_ensemble_segment_bounds_are_clean_errors_and_nesting_is_capped() -> None:
    """
    A segment shorter than one sample is a ``ValidationError``, a huge one is
    no cap (not an ``OverflowError``), and a cap on an ensemble of ensembles reaches the
    innermost members.
    """
    with pytest.raises(ValidationError, match="shorter than one sample"):
        ModelEnsemble([_DoublingModel(), _DoublingModel()], segment=1e-9)
    # A huge cap is no cap: accepted, not an OverflowError.
    huge = ModelEnsemble([_DoublingModel(), _DoublingModel()], segment=1e304)
    assert huge.max_allowed_segment == _DoublingModel().max_allowed_segment

    inner_a, inner_b, outer_member = (
        _DoublingModel(),
        _DoublingModel(),
        _DoublingModel(),
    )
    for model in (inner_a, inner_b, outer_member):
        model.max_allowed_segment = 3.0
    nested = ModelEnsemble(
        [ModelEnsemble([inner_a, inner_b]), outer_member], segment=1.5
    )
    assert nested.max_allowed_segment == 1.5
    assert inner_a.max_allowed_segment == inner_b.max_allowed_segment == 1.5


def test_nested_ensemble_progress_counts_every_inner_member() -> None:
    """
    Progress over an ensemble of ensembles plans each inner member's chunks,
    so it rises steadily to the announced total instead of overshooting and
    jumping back.
    """
    short = _DoublingModel()
    short.max_allowed_segment = 0.5
    inner = ModelEnsemble([_DoublingModel(), short])
    outer = ModelEnsemble([inner, _DoublingModel()])
    events: list[tuple[str, dict]] = []
    apply_model(
        outer,
        torch.randn(1, 1, 1000),
        shifts=0,
        overlap=0.25,
        progress_callback=lambda kind, data: events.append((kind, data)),
    )
    total = events[0][1]["total_chunks"]
    done = [d["completed_chunks"] for k, d in events if k == "chunk_complete"]
    assert done == sorted(done) and done[-1] == total


def test_combine_params_must_be_a_mapping() -> None:
    """
    A non-mapping ``combine_params`` is a ``ValidationError``, as documented,
    not a ``TypeError``.
    """
    from unblend.apply import resolve_combine_params

    for bad in (5, 1.5, True, [1], "n_fft"):
        with pytest.raises(ValidationError, match="must be a mapping"):
            resolve_combine_params(bad)  # type: ignore[arg-type]


def test_separator_combine_overrides_are_checked_the_same_with_only_load() -> None:
    """
    ``only_load`` narrowing an ensemble doesn't change which overrides are
    valid (a selection mode needs 0/1 weights either way), and never moves
    the override onto a nested inner ensemble.
    """
    from unblend.api import Separator

    blended = ModelEnsemble(
        [_ScalingModel(1.0, 1.0), _ScalingModel(1.0, 1.0)],
        weights=[[1.0, 0.5], [0.0, 1.0]],
    )
    first = blended.sources[0]
    for only_load in (None, first):
        with pytest.raises(ValidationError, match="participation mask"):
            Separator(
                model=blended, device="cpu", combine="median_wave", only_load=only_load
            )

    inner = ModelEnsemble(
        [_ScalingModel(1.0, 1.0), _ScalingModel(1.0, 1.0)], combine="avg_wave"
    )
    outer = ModelEnsemble(
        [inner, _ScalingModel(1.0, 1.0)], weights=[[1.0, 0.0], [0.0, 1.0]]
    )
    isolated = Separator(
        model=outer, device="cpu", combine="max_wave", only_load=outer.sources[0]
    )
    assert isolated.model is inner
    assert inner.combine == "avg_wave"


def test_separator_refuses_a_bad_override_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A ``combine`` override on a single-model registry entry, an unknown mode,
    or a non-mapping ``combine_params`` fails before ``get_model`` downloads.
    """
    from unblend.api import Separator
    from unblend.repo import ModelRepository

    monkeypatch.setattr(
        ModelRepository,
        "get_model",
        lambda *a, **k: pytest.fail("loaded before checking the override"),
    )
    for model, kwargs in (
        ("htdemucs", {"combine": "max_fft"}),
        ("htdemucs_ft", {"combine": "typo"}),
        ("htdemucs_ft", {"combine_params": 5}),
    ):
        with pytest.raises(ValidationError):
            Separator(model=model, device="cpu", **kwargs)


def test_htdemucs_forward_traces_as_one_graph() -> None:
    """
    ``torch.compile`` captures HTDemucs's forward in a single graph: nothing
    in it (such as reading a version counter for a cache) breaks the trace.
    Runs on CPU, so CI covers what the CUDA compile path relies on.
    """
    import torch._dynamo as dynamo

    from unblend.htdemucs import HTDemucs

    model = HTDemucs(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    ).eval()
    x = torch.randn(1, 2, 8000)
    dynamo.reset()
    with torch.no_grad():
        model(x)  # the eager call fills the caches, as Separator's warmup does
        report = dynamo.explain(model)(x)
    dynamo.reset()
    assert report.graph_break_count == 0, [b.reason for b in report.break_reasons]


def test_isolating_a_specialists_stem_uses_the_mix_minus_the_stem() -> None:
    """
    When only one member runs for a stem (``only_load`` narrowing, or
    ``use_only_stem``), its other outputs are untrained heads: the complement
    is the mix minus the stem, and isolating any other stem is refused. A full
    ensemble run still sums the other stems.
    """
    from unblend.api import Separator

    specialist = _ScalingModel(0.25, 3.0)  # "one" is its stem; "two" is junk
    other = _ScalingModel(5.0, 0.75)
    ensemble = ModelEnsemble([specialist, other], weights=[[1.0, 0.0], [0.0, 1.0]])
    mix = torch.randn(1, 400)

    isolated = Separator(model=ensemble, device="cpu", only_load="one")
    result = isolated.separate((mix, 100), shifts=0)
    pair = result.isolate_stem("one")
    assert torch.allclose(pair.sources["no_one"], mix - pair.sources["one"], atol=1e-5)
    with pytest.raises(ValidationError, match="Only 'one' was separated"):
        result.isolate_stem("two")

    full = Separator(model=ensemble, device="cpu")
    run = full.separate((mix, 100), shifts=0, use_only_stem="one")
    assert torch.allclose(
        run.isolate_stem("one").sources["no_one"], mix - run.sources["one"], atol=1e-5
    )
    everything = full.separate((mix, 100), shifts=0)
    assert everything.reliable is None
    assert torch.allclose(
        everything.isolate_stem("one").sources["no_one"],
        everything.sources["two"],
        atol=1e-6,
    )


def test_use_only_stem_wins_over_only_load_inside_a_nested_ensemble() -> None:
    """
    ``only_load`` can keep a nested inner ensemble; ``use_only_stem`` then
    runs only that ensemble's member for its stem, so that stem is the
    reliable one.
    """
    from unblend.api import Separator

    inner = ModelEnsemble(
        [_ScalingModel(0.25, 9.0), _ScalingModel(7.0, 0.5)],
        weights=[[1.0, 0.0], [0.0, 1.0]],
    )
    outer = ModelEnsemble(
        [inner, _ScalingModel(5.0, 0.75)], weights=[[1.0, 1.0], [0.0, 0.0]]
    )
    separator = Separator(model=outer, device="cpu", only_load="one")
    mix = torch.randn(1, 400)
    # Narrowed to the inner ensemble, a complete model: every output is real.
    assert separator.separate((mix, 100), shifts=0).reliable is None
    result = separator.separate((mix, 100), shifts=0, use_only_stem="two")
    assert result.reliable == ("two",)
    pair = result.isolate_stem("two")
    assert torch.allclose(pair.sources["no_two"], mix - pair.sources["two"], atol=1e-5)


def test_htdemucs_pickles_after_a_forward() -> None:
    """
    ``torch.save`` of the whole module works after inference: the memoised
    embedding and its version record aren't pickled, and the
    loaded copy separates identically.
    """
    import io

    from unblend.htdemucs import HTDemucs

    model = HTDemucs(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    ).eval()
    x = torch.randn(1, 2, 8000)
    with torch.no_grad():
        expected = model(x)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    with torch.no_grad():
        assert torch.equal(loaded(x), expected)


def test_use_only_stem_routed_to_an_inner_ensemble_keeps_all_stems_real() -> None:
    """
    When ``use_only_stem``'s sole contributor is itself an ensemble, it runs
    whole, so every output is real: the same as narrowing with ``only_load``.
    """
    from unblend.api import Separator

    inner = ModelEnsemble(
        [_ScalingModel(0.25, 9.0), _ScalingModel(7.0, 0.5)],
        weights=[[1.0, 0.0], [0.0, 1.0]],
    )
    outer = ModelEnsemble(
        [inner, _ScalingModel(5.0, 0.75)], weights=[[1.0, 1.0], [0.0, 0.0]]
    )
    mix = torch.randn(1, 400)
    via_use = Separator(model=outer, device="cpu").separate(
        (mix, 100), shifts=0, use_only_stem="one"
    )
    via_load = Separator(model=outer, device="cpu", only_load="one").separate(
        (mix, 100), shifts=0
    )
    assert via_use.reliable is None and via_load.reliable is None
    assert torch.equal(
        via_use.isolate_stem("one").sources["no_one"],
        via_load.isolate_stem("one").sources["no_one"],
    )


def test_a_member_that_owns_several_stems_keeps_them_all() -> None:
    """
    When the member that ran is the ensemble's sole source of several stems,
    all of those outputs are real and any of them can be isolated; if it
    owns every stem, nothing is marked.
    """
    from unblend.api import Separator

    three = ModelEnsemble(
        [_ScalingModel(0.25, 3.0), _ScalingModel(5.0, 0.75)],
        weights=[[1.0, 1.0], [0.0, 0.0]],
    )
    result = Separator(model=three, device="cpu", only_load="one").separate(
        (torch.randn(1, 400), 100), shifts=0
    )
    assert result.reliable is None
    result.isolate_stem("two")


@pytest.mark.parametrize("how", ["deepcopy", "torch.save"])
def test_a_copied_htdemucs_is_not_served_the_originals_cache(how: str) -> None:
    """
    A copy's parameter versions restart, so a record carried along would match
    again after a few in-place edits and serve the original's stale embedding.
    Copies drop the cache and record instead.
    """
    import copy
    import io

    from unblend.htdemucs import HTDemucs

    model = HTDemucs(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    ).eval()
    weight = model.freq_emb.embedding.weight
    with torch.no_grad():
        for _ in range(3):
            weight.mul_(1.0)  # push the original's version counter up
        x = torch.randn(1, 2, 8000)
        model(x)
    if how == "deepcopy":
        clone = copy.deepcopy(model)
    else:
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        clone = torch.load(buffer, weights_only=False)
    # Edit the copy, with no forward between, until its restarted version
    # counter reaches the original's: the point where a carried-over record
    # would match again.
    target = weight._version
    clone_weight = clone.freq_emb.embedding.weight
    with torch.no_grad():
        while clone_weight._version < target:
            clone_weight.mul_(1.5)
        fresh = HTDemucs(
            sources=["a", "b"],
            samplerate=8000,
            segment=1.0,
            nfft=512,
            depth=2,
            channels=16,
            t_layers=1,
        ).eval()
        fresh.load_state_dict(clone.state_dict())
        assert torch.allclose(clone(x), fresh(x), atol=1e-5)


def test_copying_a_compiled_model_gives_an_eager_copy_of_its_own_weights() -> None:
    """
    ``torch.compile`` binds the compiled core to the original instance; a
    deepcopy or pickle comes back eager and runs its own weights (zeroing the
    copy's parameters changes its output), and pickling no longer fails.
    """
    import copy
    import io

    from unblend.htdemucs import HTDemucs

    model = HTDemucs(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    ).eval()
    x = torch.randn(1, 2, 8000)
    model.enable_compiled_core()  # wraps only; nothing here runs the compiled core
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    for clone in (copy.deepcopy(model), torch.load(buffer, weights_only=False)):
        assert model.core_name not in clone.__dict__
        assert not hasattr(clone, "_eager_core")
        assert clone._fixed_batch_shape is False
        with torch.no_grad():
            before = clone(x)
            for parameter in clone.parameters():
                parameter.zero_()
            assert not torch.equal(clone(x), before)


def test_split_weight_cache_stays_bounded() -> None:
    """
    A caller varying the segment length per request doesn't keep every
    overlap weight alive for the life of the process.
    """
    import unblend.apply as apply_mod

    for length in range(100, 140):
        apply_mod._split_weight(length, 1.0, torch.device("cpu"), torch.float32)
    assert len(apply_mod._SPLIT_WEIGHT_CACHE) <= 16


class _HalfWeightModel(_ScalingModel):
    """
    Like HTDemucs: FP16 weights, FP32 output. In FP16 it overflows on any
    input above 0.5 and adds 1 to everything (finite but wrong); in FP32 it
    is exact.
    """

    max_allowed_segment = 10.0  # 1000-sample chunks
    sparse_chunks_need_fp32 = True

    def __init__(self) -> None:
        super().__init__(2.0, 1.0)
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input of shape ``[batch, channels, samples]``.
        :return: FP32 output of shape ``[batch, 2, channels, samples]``.
        """
        out = super().forward(x.float())
        if self.anchor.dtype is torch.float16:
            out = out + 1.0
            out[(x.float() > 0.5).any(dim=-1).any(dim=-1)] = float("inf")
        return out


def test_fp16_chunks_that_overflow_are_recomputed_in_fp32() -> None:
    """
    A chunk whose output overflows in FP16 (from a model with FP16 weights
    and FP32 output, like HTDemucs) is redone in FP32.
    """
    mix = torch.rand(1, 1, 1000) * 0.4
    mix[..., 10] = 1.0
    out = apply_model(_HalfWeightModel(), mix, shifts=0, overlap=0.0)
    torch.testing.assert_close(out[0, 0, 0], 2 * mix[0, 0])


def test_sparse_fp16_chunks_are_recomputed_but_dense_ones_are_not() -> None:
    """
    A chunk that is mostly silence around a brief sound comes back finite but
    unreliable from FP16 HTDemucs, so it is redone in FP32; ordinary dense
    audio keeps its FP16 result.
    """
    sparse = torch.zeros(1, 1, 1000)
    sparse[..., 500] = 0.4
    out = apply_model(_HalfWeightModel(), sparse, shifts=0, overlap=0.25)
    torch.testing.assert_close(out[0, 0], 2 * sparse[0])

    dense = torch.rand(1, 1, 1000) * 0.4
    out = apply_model(_HalfWeightModel(), dense, shifts=0, overlap=0.25)
    torch.testing.assert_close(out[0, 0], 2 * dense[0] + 1.0)


def test_sparse_chunks_keep_fp16_for_models_that_dont_need_fp32() -> None:
    """
    The sparse-chunk rule is for models that normalize each chunk by its own
    spread; other FP16 models keep their result (only non-finite output is
    redone for them).
    """
    model = _HalfWeightModel()
    model.sparse_chunks_need_fp32 = False
    sparse = torch.zeros(1, 1, 1000)
    sparse[..., 500] = 0.4
    out = apply_model(model, sparse, shifts=0, overlap=0.25)
    torch.testing.assert_close(out[0, 0], 2 * sparse[0] + 1.0)


def test_the_fp32_copy_is_made_once_across_shift_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Every shift pass reuses one FP32 copy rather than building its own.
    """
    import unblend.apply as apply_mod

    made = []
    real = apply_mod._fp32_copy
    monkeypatch.setattr(apply_mod, "_fp32_copy", lambda *a: made.append(1) or real(*a))
    sparse = torch.zeros(1, 1, 3000)
    sparse[..., 1500] = 0.4
    apply_model(_HalfWeightModel(), sparse, shifts=3, overlap=0.25)
    assert len(made) == 1


def test_fp32_copy_leaves_the_model_alone() -> None:
    """
    The FP32 copy has FP32 parameters while the model keeps its FP16 ones.
    """
    import unblend.apply as apply_mod

    model = _HalfWeightModel()
    twin = apply_mod._fp32_copy(model, torch.device("cpu"))
    assert twin.anchor.dtype is torch.float32
    assert model.anchor.dtype is torch.float16
    assert twin.anchor is not model.anchor


def test_digital_silence_is_not_mistaken_for_a_sparse_chunk() -> None:
    """
    A constant chunk (digital silence once the track is normalized) with a
    rounding-sized residue keeps its FP16 result: the residue isn't a peak.
    """
    flat = torch.full((1, 1, 1000), 0.3)
    flat[..., 500] += 1e-6
    out = apply_model(_HalfWeightModel(), flat, shifts=0, overlap=0.25)
    torch.testing.assert_close(out[0, 0], 2 * flat[0] + 1.0)


def test_the_fp32_copy_moves_to_the_cpu_when_the_device_runs_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An FP32 copy that runs out of memory on the model's device is rebuilt on
    the CPU, and the result is still correct (the OOM never reaches the
    batch backoff).
    """
    import unblend.apply as apply_mod

    model = _HalfWeightModel()

    class _OutOfMemory(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("MPS backend out of memory (simulated)")

    # The "device" is meta, which every torch build has (moving a chunk to
    # mps fails outright where torch lacks MPS, and that isn't an OOM).
    real_device = apply_mod._param_device
    real_copy = apply_mod._fp32_copy
    monkeypatch.setattr(
        apply_mod,
        "_param_device",
        lambda m: (
            torch.device("meta")
            if m is model or isinstance(m, _OutOfMemory)
            else real_device(m)
        ),
    )
    monkeypatch.setattr(
        apply_mod,
        "_fp32_copy",
        lambda m, d: _OutOfMemory() if d.type == "meta" else real_copy(m, d),
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    twin: dict = {}
    chunk = torch.rand(1, 1, 1000) * 0.4
    out = apply_mod._fp32_forward(model, chunk, twin)
    torch.testing.assert_close(out[0, 0], 2 * chunk[0])
    assert next(twin["model"].parameters()).device.type == "cpu"


def test_the_device_copy_is_freed_before_falling_back_to_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the device copy runs out of memory, no reference to it survives
    into the cache flush, so the flush can return its memory before the CPU
    copy is built.
    """
    import weakref

    import unblend.apply as apply_mod

    model = torch.nn.Linear(4, 4).half()
    made: list[weakref.ReferenceType] = []

    class _OutOfMemory(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("MPS backend out of memory (simulated)")

    def fake_copy(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        if device.type == "meta":
            copy = _OutOfMemory()
            made.append(weakref.ref(copy))
            return copy
        return real_copy(module, device)

    # The "device" is meta, which every torch build has (moving a chunk to
    # mps fails outright where torch lacks MPS, and that isn't an OOM).
    real_device = apply_mod._param_device
    real_copy = apply_mod._fp32_copy
    monkeypatch.setattr(
        apply_mod,
        "_param_device",
        lambda m: (
            torch.device("meta")
            if m is model or isinstance(m, _OutOfMemory)
            else real_device(m)
        ),
    )
    monkeypatch.setattr(apply_mod, "_fp32_copy", fake_copy)
    alive_at_flush: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.mps, "empty_cache", lambda: alive_at_flush.append(made[0]() is not None)
    )
    apply_mod._fp32_forward(model, torch.rand(1, 3, 4), {})
    assert alive_at_flush == [False]
