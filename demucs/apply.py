# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from typing import (
    Any,
    Callable,
    TypeAlias,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .blocks import center_trim
from .htdemucs import HTDemucs

Model: TypeAlias = HTDemucs


class ModelEnsemble(nn.Module):
    def __init__(
        self,
        models: list[Model],
        weights: list[list[float]] | None = None,
        segment: float | None = None,
    ):
        """
        Represents a model ensemble with specific weights.
        You should call ``apply_model`` rather than calling the forward directly
        for optimal performance.

        :param models: List of Demucs models.
        :param weights: List of per-model weight lists. If ``None``, assumed to
            be all ones, otherwise a list of N lists (N number of models),
            each containing S floats (S number of sources).
        :param segment: Overrides the ``segment`` attribute of each model
            (performed in-place, be careful if you reuse the models passed).
        """
        super().__init__()
        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if segment is not None:
                if (
                    not isinstance(other, HTDemucs)
                    or segment <= other.max_allowed_segment
                ):
                    other.max_allowed_segment = segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [[1.0 for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)
            for weight in weights:
                assert len(weight) == len(first.sources)
        self.weights = weights

    @property
    def max_allowed_segment(self) -> float:
        """
        Return the minimum ``max_allowed_segment`` across all models in the ensemble.

        :return: Maximum allowed segment length in seconds.
        """
        max_allowed_segment = float("inf")
        for model in self.models:
            if isinstance(model, HTDemucs):
                max_allowed_segment = min(
                    max_allowed_segment, float(model.max_allowed_segment)
                )
        return max_allowed_segment

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass is not supported; use ``apply_model`` instead.

        :param x: Input tensor.
        :return: Never returns.
        :raises NotImplementedError: Always raised.
        """
        raise NotImplementedError("Call `apply_model` on this.")


class TensorChunk:
    def __init__(
        self, tensor: Tensor | "TensorChunk", offset: int = 0, length: int | None = None
    ) -> None:
        """
        A lazy view into a tensor along the last dimension.

        :param tensor: Source tensor or another ``TensorChunk`` to wrap.
        :param offset: Start offset along the last dimension.
        :param length: Number of frames to include. If ``None``, extends to the end.
        """
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self) -> list[int]:
        """
        Return the virtual shape with the last dimension reflecting the chunk length.

        :return: Shape as a list of ints.
        """
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length: int) -> Tensor:
        """
        Return the chunk padded (or trimmed) to ``target_length``, centered on the chunk.

        :param target_length: Desired length of the last dimension.
        :return: Padded tensor of the requested length.
        """
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        # Common case: target_length matches the chunk's natural length and the
        # chunk lies fully inside the underlying tensor. Skip the F.pad call,
        # which on MPS still allocates a fresh contiguous tensor even when the
        # padding amount is (0, 0).
        if pad_left == 0 and pad_right == 0:
            out = self.tensor[..., correct_start:correct_end]
        else:
            out = F.pad(
                self.tensor[..., correct_start:correct_end],
                (pad_left, pad_right),
            )
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk: Tensor | TensorChunk) -> TensorChunk:
    """
    Wrap a tensor or pass through an existing ``TensorChunk``.

    :param tensor_or_chunk: A raw tensor or an existing ``TensorChunk``.
    :return: A ``TensorChunk`` instance.
    """
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, Tensor)
        return TensorChunk(tensor_or_chunk)


_SPLIT_WEIGHT_CACHE: dict[
    tuple[int, float, torch.device, torch.dtype], Tensor
] = {}


def _split_weight(
    segment_length: int,
    transition_power: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Build (or look up) the triangular cross-fade weight applied to each
    chunk during overlap-add.

    :param segment_length: Length of one segment in samples.
    :param transition_power: Exponent applied to the normalised triangle.
    :param device: Device to allocate the weight tensor on.
    :param dtype: Dtype for the weight tensor.
    :return: 1-D weight tensor of length ``segment_length``.
    """
    key = (segment_length, transition_power, device, dtype)
    cached = _SPLIT_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    half = segment_length // 2
    rising = torch.arange(1, half + 1, device=device, dtype=dtype)
    falling = torch.arange(segment_length - half, 0, -1, device=device, dtype=dtype)
    weight = torch.cat([rising, falling])
    weight = (weight / weight.max()) ** transition_power
    _SPLIT_WEIGHT_CACHE[key] = weight
    return weight


def apply_model(
    model: ModelEnsemble | Model,
    mix: Tensor | TensorChunk,
    device: torch.device | str | None = None,
    shifts: int = 0,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    use_only_stem: str | None = None,
    chunk_batch_size: int = 1,
) -> Tensor:
    """
    Apply model to a given mixture, tiling into segments of the model's
    training length (``model.max_allowed_segment * model.samplerate``).

    :param model: Model or ensemble to apply.
    :param mix: Input mixture tensor or chunk.
    :param device: Device for local computation; if ``None``, ``mix.device``.
    :param shifts: If > 0, average over *shifts* random sub-second shifts.
    :param overlap: Overlap ratio between consecutive segments.
    :param transition_power: Exponent on the triangular crossfade weight.
    :param progress_callback: Optional ``callback(event_type, data)``; events:
        ``processing_start``, ``chunk_complete``, ``processing_complete``.
    :param use_only_stem: For ``ModelEnsemble``, only run the sub-model
        specialised for this stem.
    :param chunk_batch_size: Chunks processed in parallel.
    :return: Separated sources tensor.
    :raises ValueError: If ``overlap`` produces a non-positive segment stride.
    """
    # Single-mix separation is just the one-element case of the multi-mix
    # path, so we delegate rather than keep a second copy of the chunking /
    # bounded-GPU staging / tail-padding / shift-averaging / ensemble-
    # weighting machinery. For a single mix the cross-mix tail pool degenerates
    # to that mix's own tail batch, the random-shift offsets are drawn in the
    # same order (one ``randint`` per shift round), and the result is moved
    # back to ``mix.device`` identically — so output is unchanged.
    return apply_model_multi(
        model,
        [mix],
        device=device,
        shifts=shifts,
        overlap=overlap,
        transition_power=transition_power,
        progress_callback=progress_callback,
        use_only_stem=use_only_stem,
        chunk_batch_size=chunk_batch_size,
    )[0]


def apply_model_multi(
    model: ModelEnsemble | Model,
    mixes: list[Tensor],
    device: torch.device | str | None = None,
    shifts: int = 0,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    use_only_stem: str | None = None,
    chunk_batch_size: int = 1,
) -> list[Tensor]:
    """
    Apply model to multiple mixes simultaneously, pooling tail chunks across
    mixes so every forward pass is exactly ``chunk_batch_size`` items.

    Each mix is chunked independently into segments of the model's training
    length. Per-mix "full" batches (those that already fill ``chunk_batch_size``)
    run as today. The leftover chunks across *all* mixes — which would each be
    a sub-full tail batch under the single-mix path — are collected into a
    global pool and drained in full-size batches. Outputs are routed back to
    their source mix's accumulator. Compared with calling ``apply_model`` per
    mix, this eliminates the per-mix tail-pad overhead which can be
    significant on short audio with large ``chunk_batch_size``.

    Same semantics as ``apply_model`` otherwise: shifts averaging applies
    per-mix (each mix gets its own random offsets per shift round), and
    ``ModelEnsemble`` is handled by running every sub-model with the same
    pooling.

    :param model: Model or ensemble to apply.
    :param mixes: List of input mixtures, each of shape ``[batch, channels, samples]``.
    :param device: Device for local computation; if ``None``, ``mixes[0].device``.
    :param shifts: If > 0, average over ``shifts`` random sub-second shifts per mix.
    :param overlap: Overlap ratio between consecutive segments.
    :param transition_power: Exponent on the triangular crossfade weight.
    :param progress_callback: Optional ``callback(event_type, data)``; events:
        ``processing_start``, ``chunk_complete``, ``processing_complete``.
        Chunk counts are summed across all mixes.
    :param use_only_stem: For ``ModelEnsemble``, only run the sub-model
        specialised for this stem.
    :param chunk_batch_size: Chunks processed in parallel per forward pass.
    :return: One separated-sources tensor per input mix, same shape as
        ``apply_model`` would have produced.
    :raises ValueError: If ``overlap`` produces a non-positive segment stride.
    """
    if not mixes:
        return []

    if device is None:
        device = mixes[0].device
    else:
        device = torch.device(device)

    if isinstance(model, ModelEnsemble):
        # Same specialisation shortcut as apply_model: when use_only_stem points
        # at a model that has a 1.0 weight for that stem (and zeros elsewhere),
        # we can run that sub-model alone.
        if use_only_stem:
            try:
                stem_index = model.sources.index(use_only_stem)
            except ValueError:
                stem_index = None
            if stem_index is not None:
                model_index: int | None = None
                for i, weights in enumerate(model.weights):
                    if (
                        len(weights) > stem_index
                        and abs(weights[stem_index] - 1.0) < 1e-6
                        and all(
                            abs(w) < 1e-6
                            for j, w in enumerate(weights)
                            if j != stem_index
                        )
                    ):
                        model_index = i
                        break
                if model_index is not None:
                    return apply_model_multi(
                        model.models[model_index],
                        mixes,
                        device=device,
                        shifts=shifts,
                        overlap=overlap,
                        transition_power=transition_power,
                        progress_callback=progress_callback,
                        chunk_batch_size=chunk_batch_size,
                    )

        # Run every sub-model with the same pooling, then weighted-average.
        results: list[Tensor] | None = None
        totals = [0.0] * len(model.sources)
        for sub_model, model_weights in zip(model.models, model.weights):
            sub_outs = apply_model_multi(
                sub_model,
                mixes,
                device=device,
                shifts=shifts,
                overlap=overlap,
                transition_power=transition_power,
                progress_callback=progress_callback,
                chunk_batch_size=chunk_batch_size,
            )
            for k, inst_weight in enumerate(model_weights):
                totals[k] += inst_weight
                for sub_out in sub_outs:
                    sub_out[:, k, :, :] *= inst_weight
            if results is None:
                results = sub_outs
            else:
                for acc, sub_out in zip(results, sub_outs):
                    acc += sub_out
        assert results is not None
        for acc in results:
            for k in range(acc.shape[1]):
                acc[:, k, :, :] /= totals[k]
        return results

    # Move/eval the model only when needed.
    first_param = next(model.parameters(), None)
    if first_param is not None and first_param.device != device:
        model.to(device)
    if model.training:
        model.eval()
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        accumulators: list[Tensor | None] = [None] * len(mixes)
        for _shift_idx in range(shifts):
            # Per-mix random offset, mirroring apply_model.
            offsets_per_mix = [random.randint(0, max_shift) for _ in mixes]
            shifted_inputs: list[Tensor | TensorChunk] = []
            for mix, offset in zip(mixes, offsets_per_mix):
                length = mix.shape[-1]
                tc = tensor_chunk(mix)
                padded = tc.padded(length + 2 * max_shift)
                shifted_inputs.append(
                    TensorChunk(padded, offset, length + max_shift - offset)
                )
            partials = _apply_model_multi_unshifted(
                model,
                shifted_inputs,
                device=device,
                overlap=overlap,
                transition_power=transition_power,
                progress_callback=progress_callback,
                chunk_batch_size=chunk_batch_size,
            )
            for i, (partial, offset) in enumerate(zip(partials, offsets_per_mix)):
                trimmed = partial[..., max_shift - offset :]
                trimmed = trimmed[..., : mixes[i].shape[-1]]
                if accumulators[i] is None:
                    accumulators[i] = trimmed
                else:
                    accumulators[i] = accumulators[i] + trimmed
        assert all(a is not None for a in accumulators)
        return [a / shifts for a in accumulators]  # type: ignore[operator]

    return _apply_model_multi_unshifted(
        model,
        mixes,
        device=device,
        overlap=overlap,
        transition_power=transition_power,
        progress_callback=progress_callback,
        chunk_batch_size=chunk_batch_size,
    )


def _apply_model_multi_unshifted(
    model: Model,
    mixes: list[Tensor | TensorChunk],
    *,
    device: torch.device,
    overlap: float,
    transition_power: float,
    chunk_batch_size: int,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[Tensor]:
    """
    Multi-mix forward without shift averaging — pools tail chunks across
    mixes so every forward pass has exactly ``chunk_batch_size`` items.

    Internal helper for ``apply_model_multi``. Each mix's full-size batches
    are processed first; whatever's left across all mixes is collected into
    a single pool and drained as full-size batches (with a final tail-pad
    on the last drain batch if needed).
    """
    segment = model.max_allowed_segment
    assert segment > 0.0
    segment_length: int = int(model.samplerate * segment)
    stride = int((1 - overlap) * segment_length)
    if stride < 1:
        raise ValueError(
            f"split overlap {overlap} produces an invalid stride for segment length {segment_length}"
        )
    # See ``apply_model`` for the bounded-GPU / unified-memory split.
    bounded_gpu = str(device).startswith("cuda")
    accum_device = torch.device("cpu") if bounded_gpu else device
    weight = _split_weight(segment_length, transition_power, accum_device, torch.float32)

    if isinstance(model, HTDemucs):
        chunk_valid_length: int = segment_length
    elif hasattr(model, "valid_length"):
        chunk_valid_length = model.valid_length(segment_length)  # type: ignore[attr-defined]
    else:
        chunk_valid_length = segment_length

    mix_states: list[dict[str, Any]] = []
    full_pool: list[tuple[int, int, TensorChunk]] = []  # (mix_idx, offset, chunk)
    tail_pool: list[tuple[int, int, TensorChunk]] = []

    for mix_idx, mix in enumerate(mixes):
        # CUDA bounded path keeps mix on CPU; MPS/CPU move to inference
        # device once (unified memory on MPS = essentially free).
        if isinstance(mix, TensorChunk):
            length = mix.length
            channels = mix.tensor.shape[-2]
            batch_dim = mix.tensor.shape[0] if mix.tensor.dim() > 2 else 1
            original_device = mix.tensor.device
            if bounded_gpu:
                mix_dev: Tensor | TensorChunk = mix
            else:
                inner = mix.tensor
                if inner.device != device:
                    inner = inner.to(device)
                mix_dev = TensorChunk(inner, mix.offset, mix.length)
        else:
            length = mix.shape[-1]
            channels = mix.shape[-2]
            batch_dim = mix.shape[0] if mix.dim() > 2 else 1
            original_device = mix.device
            if bounded_gpu:
                mix_dev = mix
            else:
                mix_dev = mix if mix.device == device else mix.to(device)

        out_acc = torch.zeros(
            batch_dim,
            len(model.sources),
            channels,
            length,
            device=accum_device,
        )
        sum_weight_mix = torch.zeros(length, device=accum_device)
        mix_states.append(
            {
                "out": out_acc,
                "sum_weight": sum_weight_mix,
                "original_device": original_device,
            }
        )

        offsets = range(0, length, stride)
        chunks_for_mix = [
            (offset, TensorChunk(mix_dev, offset, segment_length)) for offset in offsets
        ]
        n = len(chunks_for_mix)
        n_full = (n // chunk_batch_size) * chunk_batch_size
        for offset, chunk in chunks_for_mix[:n_full]:
            full_pool.append((mix_idx, offset, chunk))
        for offset, chunk in chunks_for_mix[n_full:]:
            tail_pool.append((mix_idx, offset, chunk))

    # Progress is reported across every mix's chunks (for a single mix this is
    # just that mix's chunk count, matching the old single-mix ``apply_model``).
    total_chunks = len(full_pool) + len(tail_pool)
    completed_chunks = 0
    if progress_callback:
        progress_callback("processing_start", {"total_chunks": total_chunks})

    def run_batch(batch_items: list[tuple[int, int, TensorChunk]]) -> None:
        """Run one forward pass for ``batch_items`` and accumulate into per-mix state."""
        nonlocal completed_chunks
        padded = torch.cat(
            [chunk.padded(chunk_valid_length) for _, _, chunk in batch_items],
            dim=0,
        )
        n_actual = padded.shape[0]
        if n_actual < chunk_batch_size:
            pad_count = chunk_batch_size - n_actual
            zero_pad = padded.new_zeros((pad_count, *padded.shape[1:]))
            padded = torch.cat([padded, zero_pad], dim=0)

        if bounded_gpu:
            padded = padded.to(device)

        with torch.inference_mode():
            batch_out = model(padded)

        if bounded_gpu:
            batch_out = batch_out.cpu()

        for i, (mix_idx, offset, chunk) in enumerate(batch_items):
            state = mix_states[mix_idx]
            chunk_out = center_trim(batch_out[i : i + 1], chunk.length)
            chunk_length = chunk_out.shape[-1]
            w = weight[:chunk_length]
            state["out"][..., offset : offset + segment_length] += w * chunk_out
            state["sum_weight"][offset : offset + segment_length] += w

            completed_chunks += 1
            if progress_callback:
                progress_callback(
                    "chunk_complete",
                    {
                        "completed_chunks": completed_chunks,
                        "total_chunks": total_chunks,
                    },
                )

    # Full-size batches first — by construction each block has exactly
    # ``chunk_batch_size`` items, so no padding is needed.
    for batch_start in range(0, len(full_pool), chunk_batch_size):
        run_batch(full_pool[batch_start : batch_start + chunk_batch_size])

    # Drain the cross-mix tail pool. The final batch tail-pads if the pool's
    # length isn't a clean multiple of ``chunk_batch_size``.
    for batch_start in range(0, len(tail_pool), chunk_batch_size):
        run_batch(tail_pool[batch_start : batch_start + chunk_batch_size])

    if progress_callback:
        progress_callback("processing_complete", {"total_chunks": total_chunks})

    results: list[Tensor] = []
    for state in mix_states:
        out_acc = state["out"] / state["sum_weight"]
        if out_acc.device != state["original_device"]:
            out_acc = out_acc.to(state["original_device"])
        results.append(out_acc)
    return results
