# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
import random
from numbers import Real
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
from .exceptions import ValidationError
from .htdemucs import HTDemucs
from .roformer import _RoformerBase

logger = logging.getLogger(__name__)


def _looks_like_cuda_oom(exc: BaseException) -> bool:
    """
    Whether an exception is a CUDA out-of-memory failure.

    CUDA OOM doesn't always surface as ``torch.cuda.OutOfMemoryError`` —
    graph capture and cuBLAS workspace failures under memory pressure raise
    plain RuntimeErrors.

    :param exc: The exception to classify.
    :return: True if the exception indicates CUDA memory exhaustion.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cudaerrormemoryallocation",
            "cublas_status_alloc_failed",
        )
    )


# Both backends satisfy the same inference contract used below — ``sources``,
# ``samplerate``, ``audio_channels``, ``max_allowed_segment``, and a
# ``[B, S, C, T]`` forward — so the chunking / overlap-add / tail-pooling
# machinery is shared verbatim across Demucs and RoFormer.
Model: TypeAlias = HTDemucs | _RoformerBase


class ModelEnsemble(nn.Module):
    def __init__(
        self,
        models: list[Model],
        weights: list[list[float]] | None = None,
        segment: float | None = None,
    ) -> None:
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
        if not models:
            raise ValidationError("ModelEnsemble requires at least one model.")
        if segment is not None and (
            isinstance(segment, bool)
            or not isinstance(segment, Real)
            or not math.isfinite(float(segment))
            or segment <= 0
        ):
            raise ValidationError(
                f"segment must be a finite positive number, got {segment}"
            )

        first = models[0]
        normalization = getattr(first, "external_normalization", True)
        for index, other in enumerate(models):
            if other.sources != first.sources:
                raise ValidationError(
                    f"Ensemble model {index} has sources {other.sources}, "
                    f"expected {first.sources}."
                )
            if other.samplerate != first.samplerate:
                raise ValidationError(
                    f"Ensemble model {index} has samplerate {other.samplerate}, "
                    f"expected {first.samplerate}."
                )
            if other.audio_channels != first.audio_channels:
                raise ValidationError(
                    f"Ensemble model {index} has {other.audio_channels} channels, "
                    f"expected {first.audio_channels}."
                )
            if getattr(other, "external_normalization", True) != normalization:
                raise ValidationError(
                    "Ensemble members must share the same external_normalization "
                    "contract."
                )
            maximum = float(other.max_allowed_segment)
            if not math.isfinite(maximum) or maximum <= 0:
                raise ValidationError(
                    f"Ensemble model {index} has invalid max_allowed_segment "
                    f"{other.max_allowed_segment}."
                )
            if segment is not None:
                other.max_allowed_segment = min(float(segment), maximum)

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.external_normalization = normalization
        self.models = nn.ModuleList(models)

        if weights is None:
            normalized_weights = [[1.0 for _ in first.sources] for _ in models]
        else:
            if len(weights) != len(models):
                raise ValidationError(
                    f"weights must have one row per model ({len(models)}), "
                    f"got {len(weights)}."
                )
            normalized_weights = []
            for model_index, row in enumerate(weights):
                if len(row) != len(first.sources):
                    raise ValidationError(
                        f"weights row {model_index} must contain "
                        f"{len(first.sources)} source weights, got {len(row)}."
                    )
                converted = []
                for source_index, value in enumerate(row):
                    if isinstance(value, bool) or not isinstance(value, Real):
                        raise ValidationError(
                            f"weights[{model_index}][{source_index}] must be numeric."
                        )
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValidationError(
                            f"weights[{model_index}][{source_index}] must be finite."
                        )
                    converted.append(value)
                normalized_weights.append(converted)

        # Copy caller-owned lists and validate through the same defensive path
        # used before every inference (the public attribute remains mutable).
        self.weights = [list(row) for row in normalized_weights]
        self.validated_weight_totals()

    @property
    def max_allowed_segment(self) -> float:
        """
        Return the minimum ``max_allowed_segment`` across all models in the ensemble.

        :return: Maximum allowed segment length in seconds.
        """
        values = [float(model.max_allowed_segment) for model in self.models]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValidationError(
                "Every ensemble member must have a finite, positive "
                "max_allowed_segment."
            )
        return min(values)

    def validated_weight_totals(self) -> list[float]:
        """
        Validate the mutable weight matrix and return per-source totals.

        :return: Finite, non-zero total weight for every source.
        :raises ValidationError: If dimensions or values are invalid.
        """
        if len(self.weights) != len(self.models):
            raise ValidationError(
                f"weights must have one row per model ({len(self.models)}), "
                f"got {len(self.weights)}."
            )
        for model_index, row in enumerate(self.weights):
            if len(row) != len(self.sources):
                raise ValidationError(
                    f"weights row {model_index} must contain {len(self.sources)} "
                    f"source weights, got {len(row)}."
                )
            for source_index, value in enumerate(row):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                ):
                    raise ValidationError(
                        f"weights[{model_index}][{source_index}] must be a "
                        "finite number."
                    )
        totals = [
            sum(float(row[source]) for row in self.weights)
            for source in range(len(self.sources))
        ]
        for source, total in zip(self.sources, totals):
            if not math.isfinite(total) or total == 0:
                raise ValidationError(
                    f"Ensemble weights for source '{source}' must have a finite, "
                    "non-zero total."
                )
        return totals

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
        if offset < 0:
            raise ValidationError(f"offset must be >= 0, got {offset}")
        if offset >= total_length:
            raise ValidationError(
                f"offset ({offset}) must be < total length ({total_length}); "
                "cannot wrap an empty tensor"
            )

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
        Return the chunk padded to ``target_length``, centered on the chunk.

        :param target_length: Desired length of the last dimension; must be
            >= the chunk length (chunks are never trimmed).
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


_SPLIT_WEIGHT_CACHE: dict[tuple[int, float, torch.device, torch.dtype], Tensor] = {}

# Sizing for the GPU-resident accumulation fast path. The CUDA pipeline keeps
# each mix and its overlap-add accumulators on the GPU when they fit within
# this fraction of the currently-free VRAM (after a fixed reserve for the
# active chunk batch + STFT scratch, which scale with chunk_batch_size, not
# input length). Inputs too long for the budget fall back to the bounded
# CPU-accumulation path, preserving the "GPU usage bounded by model +
# active batch" property for arbitrarily long audio.
_GPU_ACCUM_VRAM_FRACTION = 0.3
_GPU_ACCUM_VRAM_RESERVE_BYTES = 2 * 1024**3


def _require_cuda_available() -> None:
    """
    Raise unless CUDA is usable. Shared with ``Separator.__init__`` so the
    two entry points can't drift on wording.

    :raises ValidationError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise ValidationError(
            "Device 'cuda' requested but CUDA is not available in this "
            "PyTorch build/environment."
        )


def _gpu_accum_budget_bytes(
    device: torch.device | str, forward_reserve_bytes: int | None = None
) -> int:
    """
    VRAM budget (bytes) available for keeping mixes and overlap-add
    accumulators resident on the GPU.

    :param device: CUDA device to query.
    :param forward_reserve_bytes: Measured per-batch working set of the eager
        forward (Separator plumbs ``model._forward_reserve_bytes``); the
        reserve carved out of the budget is the larger of this and the flat
        default, so batch-scaled eager activations (e.g. the iSTFT of a full
        chunk batch) always have room. ``None`` keeps the flat default.
    :return: Usable byte budget; 0 if free memory cannot be determined.
    """
    try:
        free_bytes, _total = torch.cuda.mem_get_info(
            torch.device(device) if not isinstance(device, torch.device) else device
        )
    except Exception:
        return 0
    # Allocator-reserved-but-unallocated blocks are reusable by us on top of
    # the driver-reported free memory.
    reserved_slack = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
        device
    )
    usable = free_bytes + max(0, reserved_slack)
    reserve = max(_GPU_ACCUM_VRAM_RESERVE_BYTES, forward_reserve_bytes or 0)
    return max(0, int(_GPU_ACCUM_VRAM_FRACTION * (usable - reserve)))


def _gpu_accum_bytes_needed(
    batch_dim: int, n_sources: int, channels: int, length: int
) -> int:
    """
    Bytes needed to keep one mix's GPU-resident state for the fast path:
    the staged fp32 mix, the fp32 output accumulator, and the weight sum.

    Shared by ``_apply_model_multi_unshifted`` (per-mix gate) and
    ``Separator`` (deciding whether to stage the waveform on the GPU up
    front) so the two can't disagree about what fits.

    :param batch_dim: Leading batch dimension of the mix.
    :param n_sources: Number of output sources.
    :param channels: Audio channels.
    :param length: Mix length in samples.
    :return: Estimated bytes of GPU memory required.
    """
    return (batch_dim * channels * (n_sources + 1) * length + length) * 4


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


def _should_restore_submodel_device(
    sub_model: nn.Module,
    sub_device: torch.device | None,
    device: torch.device,
) -> bool:
    """
    Whether the ensemble loop should move ``sub_model`` back to ``sub_device``.

    Compiled sub-models keep a CUDAGraphs capture tied to their current
    device; bouncing them off the inference device throws that capture away
    and forces a re-compile on the next forward. Family-specific compile
    setup records either ``_uncompiled_forward_core`` (HTDemucs) or
    ``_uncompiled_run_transformers`` (RoFormer).

    :param sub_model: The just-run ensemble member.
    :param sub_device: Device the sub-model was on before the ensemble call,
        or ``None`` if no parameters were present.
    :param device: Inference device the ensemble call ran on.
    :return: ``True`` to restore via ``.to(sub_device)``, ``False`` to leave
        the sub-model where it is.
    """
    if sub_device is None or sub_device == device:
        return False
    return not (
        hasattr(sub_model, "_uncompiled_forward_core")
        or hasattr(sub_model, "_uncompiled_run_transformers")
    )


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
    oom_backoff_state: dict[str, int] | None = None,
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
        Payloads include aggregate totals and per-input chunk fields. For a
        ``ModelEnsemble`` that runs all sub-models, the reported totals span the
        whole ensemble (``total_chunks = per_model_chunks * num_sub_models``), so
        progress advances monotonically instead of restarting per sub-model.
    :param use_only_stem: Performance optimisation for a ``ModelEnsemble`` of
        fine-tuned specialists (e.g. ``htdemucs_ft``): run only the sub-model
        whose weights select this stem with a clean one-hot (1.0/0.0) row,
        skipping the others. The returned tensor still contains **all** of the
        model's sources (the lone specialist produces them all — only the named
        stem is high quality); it does *not* filter the output to one stem (use
        ``SeparatedSources.isolate_stem`` for that). Silently has no effect when
        the model is not such an ensemble or no sub-model matches one-hot.
    :param chunk_batch_size: Chunks processed in parallel.
    :param oom_backoff_state: Mutable ``{"chunk_batch_size": n}`` opting into
        runtime CUDA-OOM halving for auto-sized eager runs (Separator
        internal). Halvings persist in the dict for the caller; ``None``
        disables backoff (OOM propagates).
    :return: Separated sources tensor.
    :raises ValidationError: If ``overlap`` is outside ``[0, 1)``, if the
        device is invalid, out of range, or is CUDA/MPS without that backend
        available, or if the overlap produces a non-positive segment stride.
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
        oom_backoff_state=oom_backoff_state,
    )[0]


def apply_model_multi(
    model: ModelEnsemble | Model,
    mixes: list[Tensor | TensorChunk],
    device: torch.device | str | None = None,
    shifts: int = 0,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    use_only_stem: str | None = None,
    chunk_batch_size: int = 1,
    oom_backoff_state: dict[str, int] | None = None,
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
    :param mixes: List of input mixtures (tensors or ``TensorChunk`` views),
        each ``[batch, channels, samples]`` or ``[channels, samples]``.
    :param device: Device for local computation; if ``None``, ``mixes[0].device``.
    :param shifts: If > 0, average over ``shifts`` random sub-second shifts per mix.
    :param overlap: Overlap ratio between consecutive segments.
    :param transition_power: Exponent on the triangular crossfade weight.
    :param progress_callback: Optional ``callback(event_type, data)``; events:
        ``processing_start``, ``chunk_complete``, ``processing_complete``.
        Chunk counts are summed across all mixes; start/complete payloads include
        ``input_total_chunks``, while chunk events identify ``input_index`` and
        that input's completed/total counts. For a ``ModelEnsemble`` that
        runs all sub-models, the totals also span every sub-model
        (``total_chunks = per_model_chunks * num_sub_models``), so the progress
        bar advances continuously rather than restarting once per sub-model.
    :param use_only_stem: Performance optimisation for a ``ModelEnsemble`` of
        fine-tuned specialists (e.g. ``htdemucs_ft``): run only the sub-model
        whose weights select this stem with a clean one-hot (1.0/0.0) row,
        skipping the others. The returned tensor still contains **all** of the
        model's sources (the lone specialist produces them all — only the named
        stem is high quality); it does *not* filter the output to one stem (use
        ``SeparatedSources.isolate_stem`` for that). Silently has no effect when
        the model is not such an ensemble or no sub-model matches one-hot.
    :param chunk_batch_size: Chunks processed in parallel per forward pass.
    :param oom_backoff_state: Mutable ``{"chunk_batch_size": n}`` opting into
        runtime CUDA-OOM halving for auto-sized eager runs (Separator
        internal); halvings persist in the dict for the caller. ``None``
        disables backoff (OOM propagates).
    :return: One separated-sources tensor per input mix, same shape as
        ``apply_model`` would have produced.
    :raises ValidationError: If ``overlap`` is outside ``[0, 1)``, if the
        device is invalid, out of range, or is CUDA/MPS without that backend
        available, or if the overlap produces a non-positive segment stride.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValidationError(f"overlap must be in [0, 1), got {overlap}")

    # Validate an explicit device before the empty-input early return, so
    # bad arguments raise regardless of input.
    if device is not None:
        try:
            device = torch.device(device)
        except (TypeError, RuntimeError, ValueError) as e:
            raise ValidationError(f"Invalid device {device!r}: {e}") from e
        if device.type == "cuda":
            _require_cuda_available()
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValidationError(
                    f"Device 'cuda:{device.index}' requested but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are available."
                )
            if device.index is None:
                # An indexless "cuda" never compares equal to a tensor's
                # "cuda:0", which would defeat every device comparison below
                # (most visibly the progress-sync branch, which keys off
                # tensors already living on ``accum_device``).
                device = torch.device("cuda", torch.cuda.current_device())
        elif device.type == "mps" and not torch.backends.mps.is_available():
            raise ValidationError(
                "Device 'mps' requested but MPS is not available on this system."
            )

    if not mixes:
        return []

    if device is None:
        # A CUDA tensor's device is always indexed ("cuda:N"), so no
        # normalization is needed on this path (and a CUDA tensor existing
        # proves CUDA is available).
        device = mixes[0].device

    # The pooled chunk batching below assumes one accumulator row per chunk,
    # so a mix with batch dim > 1 is split into per-row mixes here and the
    # outputs re-stacked afterwards (2-D mixes are lifted to batch 1). For
    # the common batch-1 case this is a no-op passthrough.
    flat_mixes: list[Tensor | TensorChunk] = []
    spans: list[int] = []
    needs_restack = False
    for mix in mixes:
        inner = mix.tensor if isinstance(mix, TensorChunk) else mix
        if inner.dim() == 2:
            if isinstance(mix, TensorChunk):
                flat_mixes.append(TensorChunk(inner[None], mix.offset, mix.length))
            else:
                flat_mixes.append(mix[None])
            spans.append(1)
            needs_restack = True
        elif inner.dim() == 3 and inner.shape[0] > 1:
            for b in range(inner.shape[0]):
                row_mix = inner[b : b + 1]
                if isinstance(mix, TensorChunk):
                    flat_mixes.append(TensorChunk(row_mix, mix.offset, mix.length))
                else:
                    flat_mixes.append(row_mix)
            spans.append(inner.shape[0])
            needs_restack = True
        else:
            flat_mixes.append(mix)
            spans.append(1)
    if needs_restack:
        flat_results = apply_model_multi(
            model,
            flat_mixes,
            device=device,
            shifts=shifts,
            overlap=overlap,
            transition_power=transition_power,
            progress_callback=progress_callback,
            use_only_stem=use_only_stem,
            chunk_batch_size=chunk_batch_size,
            oom_backoff_state=oom_backoff_state,
        )
        results: list[Tensor] = []
        row = 0
        for span in spans:
            results.append(torch.cat(flat_results[row : row + span], dim=0))
            row += span
        return results

    if isinstance(model, ModelEnsemble):
        totals = model.validated_weight_totals()
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
                        and all(
                            other_index == i or abs(other_weights[stem_index]) < 1e-6
                            for other_index, other_weights in enumerate(model.weights)
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
                        oom_backoff_state=oom_backoff_state,
                    )

        # Run every sub-model with the same pooling, then weighted-average.
        # Progress is reported as one continuous span across the whole
        # ensemble: each sub-model processes the same chunks, so the aggregate
        # total is ``per_model_chunks * num_sub_models``. Without this wrapper
        # each sub-model would emit its own start/complete cycle and the bar
        # would restart N times.
        num_sub_models = len(model.models)
        sub_models_done = 0
        sub_total_chunks: int | None = None
        sub_input_total_chunks: list[int] | None = None

        def ensemble_progress(event_type: str, data: dict[str, Any]) -> None:
            """
            Re-scale per-sub-model progress events into one continuous span
            across the whole ensemble before forwarding to ``progress_callback``.

            :param event_type: Progress event name (``"processing_start"`` or
                ``"chunk_complete"``); per-sub-model ``"processing_complete"``
                events are swallowed.
            :param data: Aggregate and per-input progress payload.
            :return: None.
            """
            nonlocal sub_input_total_chunks, sub_total_chunks
            assert progress_callback is not None
            if event_type == "processing_start":
                if sub_total_chunks is None:
                    sub_total_chunks = int(data.get("total_chunks", 0))
                    sub_input_total_chunks = [
                        int(value) for value in data.get("input_total_chunks", [])
                    ]
                    progress_callback(
                        "processing_start",
                        {
                            "total_chunks": sub_total_chunks * num_sub_models,
                            "total_inputs": len(mixes),
                            "input_total_chunks": [
                                value * num_sub_models
                                for value in sub_input_total_chunks
                            ],
                        },
                    )
            elif event_type == "chunk_complete":
                # Use the latched first sub-model total for the span math and
                # clamp the per-sub-model progress to it: with shifts > 1 each
                # sub-model run draws its own random offsets, so totals can
                # differ by a few chunks between sub-models. Clamping keeps
                # the bar monotonic and bounded by the declared total.
                per_model = sub_total_chunks or int(data.get("total_chunks", 0))
                completed = sub_models_done * per_model + min(
                    int(data.get("completed_chunks", 0)), per_model
                )
                mix_index = int(data["input_index"])
                assert sub_input_total_chunks is not None
                per_input = sub_input_total_chunks[mix_index]
                input_completed = sub_models_done * per_input + min(
                    int(data.get("input_completed_chunks", 0)), per_input
                )
                progress_callback(
                    "chunk_complete",
                    {
                        "completed_chunks": completed,
                        "total_chunks": per_model * num_sub_models,
                        "input_index": mix_index,
                        "input_completed_chunks": input_completed,
                        "input_total_chunks": per_input * num_sub_models,
                    },
                )
            # Swallow per-sub-model "processing_complete"; one is emitted for
            # the whole ensemble after the loop.

        sub_callback = ensemble_progress if progress_callback else None

        results: list[Tensor] | None = None
        for sub_model, model_weights in zip(model.models, model.weights):
            sub_param = next(sub_model.parameters(), None)
            sub_device = sub_param.device if sub_param is not None else None
            sub_outs = apply_model_multi(
                sub_model,
                mixes,
                device=device,
                shifts=shifts,
                overlap=overlap,
                transition_power=transition_power,
                progress_callback=sub_callback,
                chunk_batch_size=chunk_batch_size,
                oom_backoff_state=oom_backoff_state,
            )
            sub_models_done += 1
            if _should_restore_submodel_device(sub_model, sub_device, device):
                sub_model.to(sub_device)
            for k, inst_weight in enumerate(model_weights):
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
        if (
            progress_callback
            and sub_total_chunks is not None
            and sub_input_total_chunks is not None
        ):
            progress_callback(
                "processing_complete",
                {
                    "total_chunks": sub_total_chunks * num_sub_models,
                    "total_inputs": len(mixes),
                    "input_total_chunks": [
                        value * num_sub_models for value in sub_input_total_chunks
                    ],
                },
            )
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
        # Pre-draw every round's offsets — same RNG draw order as drawing them
        # round by round (round-major, one randint per mix) — so the exact
        # chunk total across all rounds is known up front. Without this each
        # round would emit its own processing_start/complete cycle and the
        # progress bar would restart per shift.
        all_offsets = [
            [random.randint(0, max_shift) for _ in mixes] for _ in range(shifts)
        ]

        # Same validation as the unshifted helper, but raised before any
        # progress events fire — otherwise a doomed run emits a
        # ``processing_start`` with ``total_chunks: 0`` first.
        segment_length = int(round(model.samplerate * model.max_allowed_segment))
        stride = int((1 - overlap) * segment_length)
        if stride < 1:
            raise ValidationError(
                f"split overlap {overlap} produces an invalid stride for segment length {segment_length}"
            )

        inner_callback = progress_callback
        if progress_callback is not None:
            input_total_chunks = [0] * len(mixes)
            # Mirrors ``range(0, length, stride)`` in the unshifted helper.
            for offsets_per_mix in all_offsets:
                for mix_index, (mix, offset) in enumerate(zip(mixes, offsets_per_mix)):
                    shifted_length = mix.shape[-1] + max_shift - offset
                    input_total_chunks[mix_index] += -(-shifted_length // stride)
            total_chunks = sum(input_total_chunks)
            completed_total = 0
            input_completed_chunks = [0] * len(mixes)

            def shift_progress(event_type: str, data: dict[str, Any]) -> None:
                """
                Aggregate per-round progress into one continuous span across
                all shift rounds; per-round start/complete events are
                swallowed (one spanning pair is emitted around the loop).

                :param event_type: Progress event name.
                :param data: Event payload.
                :return: None.
                """
                nonlocal completed_total
                assert progress_callback is not None
                if event_type == "chunk_complete":
                    mix_index = int(data["input_index"])
                    completed_total += 1
                    input_completed_chunks[mix_index] += 1
                    progress_callback(
                        "chunk_complete",
                        {
                            "completed_chunks": completed_total,
                            "total_chunks": total_chunks,
                            "input_index": mix_index,
                            "input_completed_chunks": input_completed_chunks[mix_index],
                            "input_total_chunks": input_total_chunks[mix_index],
                        },
                    )

            inner_callback = shift_progress
            progress_callback(
                "processing_start",
                {
                    "total_chunks": total_chunks,
                    "total_inputs": len(mixes),
                    "input_total_chunks": input_total_chunks,
                },
            )

        accumulators: list[Tensor | None] = [None] * len(mixes)
        for offsets_per_mix in all_offsets:
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
                progress_callback=inner_callback,
                chunk_batch_size=chunk_batch_size,
                oom_backoff_state=oom_backoff_state,
            )
            for i, (partial, offset) in enumerate(zip(partials, offsets_per_mix)):
                trimmed = partial[..., max_shift - offset :]
                trimmed = trimmed[..., : mixes[i].shape[-1]]
                # Accumulate in-place to avoid a per-round full-output
                # allocation. The first round clones because ``trimmed`` is a
                # view into the (otherwise-discarded) round's partial, and a
                # subsequent ``add_`` would mutate that backing storage rather
                # than build an accumulator.
                if accumulators[i] is None:
                    accumulators[i] = trimmed.clone()
                else:
                    accumulators[i].add_(trimmed)
        if progress_callback is not None:
            progress_callback(
                "processing_complete",
                {
                    "total_chunks": total_chunks,
                    "total_inputs": len(mixes),
                    "input_total_chunks": input_total_chunks,
                },
            )
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
        oom_backoff_state=oom_backoff_state,
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
    oom_backoff_state: dict[str, int] | None = None,
) -> list[Tensor]:
    """
    Multi-mix forward without shift averaging — pools tail chunks across
    mixes so every forward pass has exactly ``chunk_batch_size`` items.

    Internal helper for ``apply_model_multi``. Each mix's full-size batches
    are processed first; whatever's left across all mixes is collected into
    a single pool and drained as full-size batches (with a final tail-pad
    on the last drain batch if needed).

    :param model: Model to run on each chunk.
    :param mixes: Input mixes as tensors or ``TensorChunk`` views.
    :param device: Inference device.
    :param overlap: Overlap between segments, used to derive the chunk stride.
    :param transition_power: Exponent for the triangular overlap-add weighting.
    :param chunk_batch_size: Number of chunks per forward pass.
    :param progress_callback: Optional aggregate/per-input progress callback.
    :param oom_backoff_state: Mutable ``{"chunk_batch_size": n}`` opting into
        runtime CUDA-OOM halving for auto-sized eager runs (Separator
        internal); halvings persist in the dict for the caller. ``None``
        disables backoff (OOM propagates).
    :return: One separated-sources tensor per input mix, in input order.
    :raises ValidationError: If ``overlap`` produces a non-positive segment
        stride (range validation happens in ``apply_model_multi``).
    """
    # The accum-device comparisons below need an indexed CUDA device;
    # apply_model_multi (the only caller) normalizes it (and rejects "cuda"
    # when CUDA is unavailable, so normalization always happened).
    assert device.type != "cuda" or device.index is not None
    segment = model.max_allowed_segment
    assert segment > 0.0
    segment_length: int = int(round(model.samplerate * segment))
    stride = int((1 - overlap) * segment_length)
    if stride < 1:
        raise ValidationError(
            f"split overlap {overlap} produces an invalid stride for segment length {segment_length}"
        )
    # CUDA path: accumulate on the GPU when every mix's accumulators fit the
    # VRAM budget (single D2H at the end, no per-batch CPU round-trip);
    # otherwise fall back to CPU accumulation with per-batch staging, which
    # bounds GPU usage by ``model + active_batch`` for arbitrarily long audio.
    # MPS/CPU accumulate on ``device`` as before (unified memory on MPS).
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        bytes_needed = 0
        for mix in mixes:
            inner = mix.tensor if isinstance(mix, TensorChunk) else mix
            mix_length = mix.length if isinstance(mix, TensorChunk) else mix.shape[-1]
            batch_dim = inner.shape[0] if inner.dim() > 2 else 1
            bytes_needed += _gpu_accum_bytes_needed(
                batch_dim,
                len(model.sources),
                inner.shape[-2],
                mix_length,
            )
            if isinstance(mix, TensorChunk) and inner.shape[-1] > mix_length:
                # The GPU-resident path stages the chunk's entire backing
                # tensor, not just the viewed span — budget the difference.
                bytes_needed += (
                    batch_dim * inner.shape[-2] * (inner.shape[-1] - mix_length) * 4
                )
        gpu_resident = bytes_needed <= _gpu_accum_budget_bytes(
            device, getattr(model, "_forward_reserve_bytes", None)
        )
    else:
        gpu_resident = True
    accum_device = device if gpu_resident else torch.device("cpu")
    weight = _split_weight(
        segment_length, transition_power, accum_device, torch.float32
    )
    # CUDAGraphs capture (Separator's compile path) requires every forward to
    # have the captured batch shape, so sub-full tail batches are zero-padded
    # up to ``chunk_batch_size``. Eager execution has no such constraint, and
    # padding would just burn forward compute on zero chunks.
    fixed_batch_shape = bool(getattr(model, "_fixed_batch_shape", False))

    chunk_valid_length: int = segment_length

    mix_states: list[dict[str, Any]] = []
    full_pool: list[tuple[int, int, TensorChunk]] = []  # (mix_idx, offset, chunk)
    tail_pool: list[tuple[int, int, TensorChunk]] = []
    input_total_chunks: list[int] = []

    for mix_idx, mix in enumerate(mixes):
        # GPU-resident path moves each mix to the inference device once and
        # slices chunks there; the bounded CPU path keeps the mix on CPU and
        # stages each batch instead. MPS/CPU always move once (unified memory
        # on MPS = essentially free).
        if isinstance(mix, TensorChunk):
            length = mix.length
            channels = mix.tensor.shape[-2]
            batch_dim = mix.tensor.shape[0] if mix.tensor.dim() > 2 else 1
            original_device = mix.tensor.device
            if not gpu_resident:
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
            if not gpu_resident:
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
        input_total_chunks.append(n)
        n_full = (n // chunk_batch_size) * chunk_batch_size
        for offset, chunk in chunks_for_mix[:n_full]:
            full_pool.append((mix_idx, offset, chunk))
        for offset, chunk in chunks_for_mix[n_full:]:
            tail_pool.append((mix_idx, offset, chunk))

    # Progress is reported across every mix's chunks (for a single mix this is
    # just that mix's chunk count, matching the old single-mix ``apply_model``).
    total_chunks = len(full_pool) + len(tail_pool)
    completed_chunks = 0
    input_completed_chunks = [0] * len(mixes)
    if progress_callback:
        progress_callback(
            "processing_start",
            {
                "total_chunks": total_chunks,
                "total_inputs": len(mixes),
                "input_total_chunks": input_total_chunks,
            },
        )

    def run_batch(
        batch_items: list[tuple[int, int, TensorChunk]],
    ) -> list[dict[str, int]]:
        """
        Run one forward pass for ``batch_items`` and accumulate into per-mix state.

        :param batch_items: List of ``(mix_idx, offset, chunk)`` tuples to run
            together as a single batch.
        :return: Progress payloads to emit once the batch is out of the
            OOM-retry boundary.
        """
        nonlocal completed_chunks
        padded = torch.cat(
            [chunk.padded(chunk_valid_length) for _, _, chunk in batch_items],
            dim=0,
        )
        n_actual = padded.shape[0]
        if n_actual < chunk_batch_size and fixed_batch_shape:
            # CUDAGraphs replay (Separator's compile path) requires the
            # captured batch shape, so tails pad all the way up. Eager models
            # run tails at their natural size — padding would burn forward
            # compute on zero chunks. (Natural tail shapes are only viable
            # because eager mode leaves ``cudnn.benchmark`` off; with it on,
            # every distinct shape costs a multi-second exhaustive algorithm
            # search, which measured far worse than any padding waste.)
            pad_count = chunk_batch_size - n_actual
            zero_pad = padded.new_zeros((pad_count, *padded.shape[1:]))
            padded = torch.cat([padded, zero_pad], dim=0)

        if padded.device != device:
            padded = padded.to(device)

        with torch.inference_mode():
            batch_out = model(padded)

        if batch_out.device != accum_device:
            batch_out = batch_out.to(accum_device)
        elif progress_callback is not None and is_cuda:
            # The GPU-resident path has no host sync per batch (that's the
            # point), so without this the per-chunk events below would fire
            # at kernel-enqueue time — racing ahead of the actual compute and
            # rendering progress meaningless. Sync only when someone is
            # watching; unobserved runs keep the free-running pipeline.
            torch.cuda.synchronize(device)

        # Phase 1 — everything that allocates (trims are views, but the
        # weighted product materialises a tensor per chunk). Kept strictly
        # before the commits below so an OOM anywhere in this batch leaves
        # the accumulators untouched: _drain can then re-run the whole batch
        # without double-counting chunks that had already been added.
        contributions: list[tuple[int, int, Tensor, Tensor]] = []
        for i, (mix_idx, offset, chunk) in enumerate(batch_items):
            chunk_out = center_trim(batch_out[i : i + 1], chunk.length)
            chunk_length = chunk_out.shape[-1]
            w = weight[:chunk_length]
            contributions.append((mix_idx, offset, w * chunk_out, w))

        # Phase 2 — pure in-place adds into preallocated accumulators: no
        # allocation, so this phase cannot OOM partway through. Progress
        # payloads are returned rather than emitted here: a user callback
        # raising an OOM-shaped error inside the retry boundary would
        # otherwise re-run a batch whose commits already landed.
        pending_events: list[dict[str, int]] = []
        for mix_idx, offset, weighted, w in contributions:
            state = mix_states[mix_idx]
            state["out"][..., offset : offset + segment_length] += weighted
            state["sum_weight"][offset : offset + segment_length] += w

            completed_chunks += 1
            input_completed_chunks[mix_idx] += 1
            pending_events.append(
                {
                    "completed_chunks": completed_chunks,
                    "total_chunks": total_chunks,
                    "input_index": mix_idx,
                    "input_completed_chunks": input_completed_chunks[mix_idx],
                    "input_total_chunks": input_total_chunks[mix_idx],
                }
            )
        return pending_events

    def _effective_cbs() -> int:
        """
        The batch size currently in force: the backoff dict's live value when
        the caller opted in, else the fixed argument.

        :return: Current chunks-per-forward batch size (>= 1).
        """
        if oom_backoff_state is not None:
            return max(1, int(oom_backoff_state["chunk_batch_size"]))
        return chunk_batch_size

    def _drain(pool: list[tuple[int, int, TensorChunk]]) -> None:
        """
        Run ``pool`` in batches, halving on CUDA OOM when the caller opted
        into backoff. Backoff never applies to compiled models
        (``fixed_batch_shape`` — the captured shape can't change here; the
        Separator recaptures instead) and re-raises at batch size 1, where
        the model genuinely doesn't fit.

        :param pool: ``(mix_idx, offset, chunk)`` tuples to run.
        """
        idx = 0
        while idx < len(pool):
            batch = pool[idx : idx + _effective_cbs()]
            try:
                pending_events = run_batch(batch)
            except RuntimeError as exc:
                current = _effective_cbs()
                if (
                    oom_backoff_state is None
                    or fixed_batch_shape
                    or current <= 1
                    or not _looks_like_cuda_oom(exc)
                ):
                    raise
                new_cbs = max(1, current // 2)
                oom_backoff_state["chunk_batch_size"] = new_cbs
                logger.warning(
                    "GPU OOM at chunk_batch_size=%d; retrying the failed "
                    "batch at %d (sticky for the rest of this run).",
                    current,
                    new_cbs,
                )
                if is_cuda:
                    torch.cuda.empty_cache()
                elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                continue
            idx += len(batch)
            # Outside the try: a raising progress callback propagates as the
            # caller's own error instead of masquerading as a batch OOM.
            if progress_callback:
                for payload in pending_events:
                    progress_callback("chunk_complete", payload)

    # Full-size batches first — blocks were built at the original
    # ``chunk_batch_size``, so no padding is needed even if backoff later
    # shrinks the slice size.
    _drain(full_pool)

    # Drain the cross-mix tail pool. The final batch tail-pads if the pool's
    # length isn't a clean multiple of ``chunk_batch_size``.
    _drain(tail_pool)

    if progress_callback:
        progress_callback(
            "processing_complete",
            {
                "total_chunks": total_chunks,
                "total_inputs": len(mixes),
                "input_total_chunks": input_total_chunks,
            },
        )

    results: list[Tensor] = []
    for state in mix_states:
        out_acc = state["out"] / state["sum_weight"]
        if out_acc.device != state["original_device"]:
            out_acc = out_acc.to(state["original_device"])
        results.append(out_acc)
    return results
