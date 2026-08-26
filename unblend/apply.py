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

from .backends import ASSModel
from .blocks import center_trim
from .exceptions import ValidationError

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


Model: TypeAlias = ASSModel

COMBINE_DEFAULT = "weighted_mean"

COMBINE_ALIASES: dict[str, str] = {
    "avg_wave": COMBINE_DEFAULT,
    "uvr_min_spec": "min_fft",
    "uvr_max_spec": "max_fft",
}

COMBINE_SELECTION_MODES = frozenset(
    {"median_wave", "min_wave", "max_wave", "median_fft", "min_fft", "max_fft"}
)

COMBINE_SPECTRAL_MODES = frozenset({"avg_fft", "median_fft", "min_fft", "max_fft"})

COMBINE_MODES = (
    frozenset({COMBINE_DEFAULT})
    | COMBINE_SELECTION_MODES
    | COMBINE_SPECTRAL_MODES
    | frozenset(COMBINE_ALIASES)
)

DEFAULT_COMBINE_STFT: dict[str, int] = {"n_fft": 1024, "hop_length": 256}


def canonical_combine(mode: str) -> str:
    """
    Resolve a combine mode to the implementation that runs it.

    :param mode: Mode name as written in metadata.
    :return: The canonical mode name.
    :raises ValidationError: If the mode is not one Unblend implements.
    """
    if not isinstance(mode, str) or mode not in COMBINE_MODES:
        raise ValidationError(
            f"Unknown ensemble combine mode {mode!r}; expected one of "
            f"{', '.join(sorted(COMBINE_MODES))}."
        )
    return COMBINE_ALIASES.get(mode, mode)


def validate_combine_weights(mode: str, weights: list[list[float]] | None) -> None:
    """
    Enforce the weight contract for a combine mode.

    :param mode: Mode name, alias or canonical.
    :param weights: The per-member, per-source weight matrix, or ``None``.
    :raises ValidationError: If a selection mode was given weights that are not
        a 0/1 participation mask.
    """
    if weights is None or canonical_combine(mode) not in COMBINE_SELECTION_MODES:
        return
    for row_index, row in enumerate(weights):
        for column, value in enumerate(row):
            if float(value) not in (0.0, 1.0):
                raise ValidationError(
                    f"combine={mode!r} selects among members rather than "
                    "blending them, so its weights must be a 0/1 participation "
                    f"mask; weights[{row_index}][{column}] is {value}."
                )


def resolve_combine_params(params: dict | None) -> dict[str, int]:
    """
    Fill in and check the STFT geometry used by the spectral modes.

    :param params: Caller-supplied overrides, or ``None`` for the defaults.
    :return: Complete ``{"n_fft", "hop_length"}`` geometry.
    :raises ValidationError: If a value is not a positive integer, or if the
        two are not commensurate.
    """
    resolved = dict(DEFAULT_COMBINE_STFT)
    for key in ("n_fft", "hop_length"):
        if params is None or key not in params:
            continue
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(
                f"combine_params[{key!r}] must be a positive integer, got {value!r}."
            )
        resolved[key] = value
    if resolved["n_fft"] % resolved["hop_length"]:
        raise ValidationError(
            "combine_params: n_fft must be a whole multiple of hop_length so "
            "that block boundaries land on the same frame grid a whole-track "
            "transform would use."
        )
    return resolved


NORMALIZATION_EPSILON = 1e-5


def normalization_stats(mix: Tensor) -> tuple[Tensor, Tensor]:
    """
    Track-level mean/std as in Demucs.

        :param mix: ``[batch, channels, samples]`` audio.
        :return: ``(mean, std)`` per batch entry.
    """
    reference = mix.mean(dim=-2)
    mean = reference.mean(dim=-1)
    correction = 1 if reference.shape[-1] > 1 else 0
    std = reference.std(dim=-1, correction=correction)
    return mean, std


def _normalize_mix(mix: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Apply track-level normalisation to a batch of mixtures.

    :param mix: ``[batch, channels, samples]`` audio.
    :param mean: Per-batch means from :func:`normalization_stats`.
    :param std: Per-batch deviations from :func:`normalization_stats`.
    :return: The normalised audio.
    """
    shape = (-1,) + (1,) * (mix.dim() - 1)
    return (mix - mean.reshape(shape)) / (NORMALIZATION_EPSILON + std.reshape(shape))


def _denormalize_sources(sources: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Undo :func:`_normalize_mix` on a member's separated sources.

    :param sources: ``[batch, stems, channels, samples]`` estimates.
    :param mean: The means used to normalise.
    :param std: The deviations used to normalise.
    :return: Sources back in the input's scale.
    """
    shape = (-1,) + (1,) * (sources.dim() - 1)
    return sources * (NORMALIZATION_EPSILON + std.reshape(shape)) + mean.reshape(shape)


def sole_contributor(weights: list[list[float]], stem_index: int) -> int | None:
    """
    Member that sole-contributes for a stem, if any.

        :param weights: Per-member, per-source weight matrix.
        :param stem_index: Stem index.
        :return: Member index or None.
    """
    contributors = [
        index
        for index, row in enumerate(weights)
        if stem_index < len(row) and abs(float(row[stem_index])) > 1e-9
    ]
    return contributors[0] if len(contributors) == 1 else None


def _select_by_magnitude(stacked: Tensor, mode: str) -> Tensor:
    """
    Reduce over dim 0 by magnitude.

        :param stacked: Tensor whose dim 0 indexes members.
        :param mode: ``"min"``, ``"max"`` or ``"median"``.
        :return: Reduced tensor without dim 0.
    """
    magnitude = stacked.abs()
    count = stacked.shape[0]
    middle = count // 2
    if mode == "min":
        index = magnitude.argmin(dim=0, keepdim=True)
    elif mode == "max":
        index = magnitude.argmax(dim=0, keepdim=True)
    else:
        order = magnitude.argsort(dim=0)
        if count % 2 == 0:
            lower = torch.gather(stacked, 0, order.narrow(0, middle - 1, 1))
            upper = torch.gather(stacked, 0, order.narrow(0, middle, 1))
            return ((lower + upper) / 2).squeeze(0)
        index = order.narrow(0, middle, 1)
    return torch.gather(stacked, 0, index).squeeze(0)


def _median_signed(stacked: Tensor) -> Tensor:
    """
    Element-wise median of signed values over dim 0.

    Averages the two middles for an even member count, as numpy's ``median``
    does -- ``torch.median`` would return the lower of the two.

    :param stacked: Real tensor whose dim 0 indexes members.
    :return: The median, without dim 0.
    """
    ordered, _ = stacked.sort(dim=0)
    count = stacked.shape[0]
    middle = count // 2
    if count % 2:
        return ordered.narrow(0, middle, 1).squeeze(0)
    pair = ordered.narrow(0, middle - 1, 1) + ordered.narrow(0, middle, 1)
    return pair.squeeze(0) / 2


def _reduce_waveforms(parts: list[Tensor], mode: str) -> Tensor:
    """
    Combine one stem's per-member waveforms.

    :param parts: One ``[B, C, T]`` tensor per contributing member.
    :param mode: Canonical waveform mode.
    :return: The combined ``[B, C, T]`` waveform.
    """
    stacked = torch.stack(parts)
    if mode == "median_wave":
        return _median_signed(stacked)
    return _select_by_magnitude(stacked, "min" if mode == "min_wave" else "max")


def _reduce_spectra(stacked: Tensor, mode: str, weights: Tensor) -> Tensor:
    """
    Combine stacked complex spectrograms.

    :param stacked: ``[members, ...]`` complex spectrograms.
    :param mode: Canonical spectral mode.
    :param weights: Per-member weights, used only by ``avg_fft``.
    :return: The combined spectrogram, without dim 0.
    """
    if mode == "avg_fft":
        shape = (-1,) + (1,) * (stacked.dim() - 1)
        scaled = stacked * weights.reshape(shape).to(stacked.dtype)
        return scaled.sum(dim=0) / weights.sum()
    if mode == "median_fft":
        return _select_by_magnitude(stacked, "median")
    return _select_by_magnitude(stacked, "min" if mode == "min_fft" else "max")


def _reduce_spectral(
    parts: list[Tensor], mode: str, weights: list[float], stft: dict[str, int]
) -> Tensor:
    """
    Combine waveforms in STFT domain in blocks.

        :param parts: One ``[B, C, T]`` per member.
        :param mode: Canonical spectral mode.
        :param weights: Per-member weights for this stem.
        :param stft: ``{"n_fft", "hop_length"}`` geometry.
        :return: Combined ``[B, C, T]`` waveform.
    """
    n_fft = stft["n_fft"]
    hop = stft["hop_length"]
    reference = parts[0]
    total = reference.shape[-1]
    window = torch.hann_window(n_fft, device=reference.device, dtype=torch.float32)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=reference.device)
    margin = n_fft * 4
    block = max(hop * 8192, margin * 4)

    combined = torch.empty_like(reference)
    start = 0
    while start < total:
        stop = min(start + block, total)
        left = min(margin, start)
        right = min(margin, total - stop)
        length = left + (stop - start) + right
        spectra = torch.stack(
            [
                torch.stft(
                    part[..., start - left : stop + right].reshape(-1, length).float(),
                    n_fft,
                    hop,
                    window=window,
                    return_complex=True,
                    center=True,
                    pad_mode="constant",
                )
                for part in parts
            ]
        )
        waveform = torch.istft(
            _reduce_spectra(spectra, mode, weight_tensor),
            n_fft,
            hop,
            window=window,
            length=length,
            center=True,
        ).reshape(reference.shape[:-1] + (length,))
        combined[..., start:stop] = waveform[..., left : left + (stop - start)].to(
            combined.dtype
        )
        start = stop
    return combined


def combine_member_outputs(
    members: list[Tensor],
    weights: list[list[float]],
    mode: str,
    stft: dict[str, int],
) -> Tensor:
    """
    Combine per-member outputs into one result.

        :param members: One ``[B, S, C, T]`` per member.
        :param weights: Per-member, per-source weight matrix.
        :param mode: Canonical combine mode.
        :param stft: STFT geometry.
        :return: Combined ``[B, S, C, T]`` output.
    """
    combined = torch.empty_like(members[0])
    for stem in range(combined.shape[1]):
        contributing = [
            index
            for index in range(len(members))
            if stem < len(weights[index]) and abs(float(weights[index][stem])) > 1e-9
        ]
        parts = [members[index][:, stem] for index in contributing]
        if len(parts) == 1:
            combined[:, stem] = parts[0]
        elif mode in COMBINE_SPECTRAL_MODES:
            combined[:, stem] = _reduce_spectral(
                parts,
                mode,
                [float(weights[index][stem]) for index in contributing],
                stft,
            )
        else:
            combined[:, stem] = _reduce_waveforms(parts, mode)
    return combined


class ModelEnsemble(nn.Module):
    def __init__(
        self,
        models: list[Model],
        weights: list[list[float]] | None = None,
        segment: float | None = None,
        combine: str = COMBINE_DEFAULT,
        combine_params: dict | None = None,
    ) -> None:
        """
        Ensemble of models with weights.

            :param models: Ensemble members.
            :param weights: Per-model weight lists, or None for ones.
            :param segment: Override segment length.
            :param combine: Combine mode.
            :param combine_params: STFT geometry for spectral modes.
            :raises ValidationError: If args invalid.
        """
        super().__init__()
        self.combine = combine
        self.combine_mode = canonical_combine(combine)
        self.combine_params = resolve_combine_params(combine_params)
        validate_combine_weights(combine, weights)
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
        member_normalization = [
            bool(getattr(other, "external_normalization", True)) for other in models
        ]
        normalization = all(member_normalization)
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
        self.member_normalization = member_normalization
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

        self.weights = [list(row) for row in normalized_weights]
        self.validated_weight_totals()

    def set_combine(self, combine: str, combine_params: dict | None = None) -> None:
        """
        Change ensemble combine mode.

            :param combine: Mode name.
            :param combine_params: STFT geometry, or None.
            :raises ValidationError: If mode invalid.
        """
        mode = canonical_combine(combine)
        validate_combine_weights(combine, self.weights)
        self.combine = combine
        self.combine_mode = mode
        self.combine_params = resolve_combine_params(combine_params)

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
    VRAM budget for GPU-resident mixes/accumulators.

        :param device: CUDA device.
        :param forward_reserve_bytes: Per-batch working set reserve.
        :return: Usable byte budget.
    """
    try:
        free_bytes, _total = torch.cuda.mem_get_info(
            torch.device(device) if not isinstance(device, torch.device) else device
        )
    except Exception:
        return 0
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
    Bytes to keep one mix's GPU state resident.

        :param batch_dim: Batch dimension.
        :param n_sources: Number of sources.
        :param channels: Audio channels.
        :param length: Mix length.
        :return: Estimated bytes.
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


def _planned_input_chunks(
    model: Model,
    mixes: list[Tensor | TensorChunk],
    shifts: int,
    overlap: float,
    shift_offsets: list[list[int]] | None,
) -> list[int]:
    """
    Return exact per-input chunk counts for a pre-drawn shift plan.

    :param model: Ensemble member whose segment length determines the stride.
    :param mixes: Input mixtures to count.
    :param shifts: Number of shift rounds, or zero for unshifted inference.
    :param overlap: Segment overlap ratio.
    :param shift_offsets: Pre-drawn offsets for every round/input.
    :return: One exact chunk count per input mixture.
    """
    segment_length = int(round(model.samplerate * model.max_allowed_segment))
    stride = int((1 - overlap) * segment_length)
    if stride < 1:
        raise ValidationError(
            f"split overlap {overlap} produces an invalid stride for segment "
            f"length {segment_length}"
        )
    if not shifts:
        return [-(-tensor_chunk(mix).length // stride) for mix in mixes]
    assert shift_offsets is not None
    max_shift = int(0.5 * model.samplerate)
    totals = [0] * len(mixes)
    for offsets_per_mix in shift_offsets:
        for index, (mix, offset) in enumerate(zip(mixes, offsets_per_mix)):
            shifted_length = mix.shape[-1] + max_shift - offset
            totals[index] += -(-shifted_length // stride)
    return totals


def _should_restore_submodel_device(
    sub_model: nn.Module,
    sub_device: torch.device | None,
    device: torch.device,
) -> bool:
    """
    Whether to move sub-model back to its original device.

        :param sub_model: Just-run member.
        :param sub_device: Device before call.
        :param device: Inference device.
        :return: True to restore.
    """
    if sub_device is None or sub_device == device:
        return False
    return not hasattr(sub_model, "_eager_core")


def _run_ensemble_member(
    sub_model: Model,
    mixes: list[Tensor | TensorChunk],
    *,
    normalize: bool,
    stats: list[tuple[Tensor, Tensor]] | None,
    **kwargs: Any,
) -> list[Tensor]:
    """
    Run one ensemble member with optional normalization.

        :param sub_model: Member to run.
        :param mixes: Input mixtures.
        :param normalize: Whether member needs normalization.
        :param stats: Per-mix ``(mean, std)``.
        :param kwargs: Forwarded to ``apply_model_multi``.
        :return: Per-mix estimates.
    """
    if not normalize:
        return apply_model_multi(sub_model, mixes, **kwargs)

    assert stats is not None
    normalized = [
        _normalize_mix(mix, mean, std) for mix, (mean, std) in zip(mixes, stats)
    ]
    outputs = apply_model_multi(sub_model, normalized, **kwargs)
    return [
        _denormalize_sources(output, mean, std)
        for output, (mean, std) in zip(outputs, stats)
    ]


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
    Apply model to a mixture tiled into segments.

        :param model: Model or ensemble.
        :param mix: Input mixture.
        :param device: Device; defaults to ``mix.device``.
        :param shifts: Shifts to average, or 0.
        :param overlap: Overlap ratio.
        :param transition_power: Crossfade exponent.
        :param progress_callback: Progress callback.
        :param use_only_stem: One-hot specialist shortcut.
        :param chunk_batch_size: Chunks per forward.
        :param oom_backoff_state: Mutable backoff dict or None.
        :return: Separated sources tensor.
        :raises ValidationError: If args invalid.
    """
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
    *,
    _shift_offsets: list[list[int]] | None = None,
) -> list[Tensor]:
    """
    Apply model to multiple mixes pooling tail chunks.

        :param model: Model or ensemble.
        :param mixes: List of input mixtures.
        :param device: Device; defaults to ``mixes[0].device``.
        :param shifts: Shifts per mix.
        :param overlap: Overlap ratio.
        :param transition_power: Crossfade exponent.
        :param progress_callback: Progress callback.
        :param use_only_stem: Specialist shortcut.
        :param chunk_batch_size: Chunks per forward.
        :param oom_backoff_state: Mutable backoff dict or None.
        :param _shift_offsets: Internal pre-drawn offsets.
        :return: One tensor per input mix.
        :raises ValidationError: If args invalid.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValidationError(f"overlap must be in [0, 1), got {overlap}")

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
                device = torch.device("cuda", torch.cuda.current_device())
        elif device.type == "mps" and not torch.backends.mps.is_available():
            raise ValidationError(
                "Device 'mps' requested but MPS is not available on this system."
            )

    if not mixes:
        return []

    if device is None:
        device = mixes[0].device

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
            _shift_offsets=_shift_offsets,
        )
        results: list[Tensor] = []
        row = 0
        for span in spans:
            results.append(torch.cat(flat_results[row : row + span], dim=0))
            row += span
        return results

    if shifts and _shift_offsets is None:
        max_shift = int(0.5 * model.samplerate)
        _shift_offsets = [
            [random.randint(0, max_shift) for _ in mixes] for _ in range(shifts)
        ]

    if isinstance(model, ModelEnsemble):
        totals = model.validated_weight_totals()
        combine = getattr(model, "combine_mode", COMBINE_DEFAULT)

        member_normalization = list(
            getattr(model, "member_normalization", [False] * len(model.models))
        )
        if getattr(model, "external_normalization", True):
            member_normalization = [False] * len(model.models)
        mix_stats: list[tuple[Tensor, Tensor]] | None = None
        if any(member_normalization):
            materialized = [
                mix.padded(mix.length) if isinstance(mix, TensorChunk) else mix
                for mix in mixes
            ]
            mix_stats = [normalization_stats(mix) for mix in materialized]
            mixes = materialized
        if use_only_stem:
            try:
                stem_index = model.sources.index(use_only_stem)
            except ValueError:
                stem_index = None
            if stem_index is not None:
                model_index = sole_contributor(model.weights, stem_index)
                if model_index is not None:
                    return _run_ensemble_member(
                        model.models[model_index],
                        mixes,
                        normalize=member_normalization[model_index],
                        stats=mix_stats,
                        device=device,
                        shifts=shifts,
                        overlap=overlap,
                        transition_power=transition_power,
                        progress_callback=progress_callback,
                        chunk_batch_size=chunk_batch_size,
                        oom_backoff_state=oom_backoff_state,
                        _shift_offsets=_shift_offsets,
                    )

        sub_models_done = 0
        model_input_totals = [
            _planned_input_chunks(sub_model, mixes, shifts, overlap, _shift_offsets)
            for sub_model in model.models
        ]
        aggregate_input_totals = [
            sum(per_model[index] for per_model in model_input_totals)
            for index in range(len(mixes))
        ]
        aggregate_total = sum(aggregate_input_totals)
        if progress_callback:
            progress_callback(
                "processing_start",
                {
                    "total_chunks": aggregate_total,
                    "total_inputs": len(mixes),
                    "input_total_chunks": aggregate_input_totals,
                },
            )

        def ensemble_progress(event_type: str, data: dict[str, Any]) -> None:
            """
            Map one member's chunk event into the aggregate exact span.

            :param event_type: Child progress event name.
            :param data: Child progress payload.
            :return: None.
            """
            assert progress_callback is not None
            if event_type != "chunk_complete":
                return
            mix_index = int(data["input_index"])
            prior_total = sum(
                sum(values) for values in model_input_totals[:sub_models_done]
            )
            prior_input = sum(
                values[mix_index] for values in model_input_totals[:sub_models_done]
            )
            progress_callback(
                "chunk_complete",
                {
                    "completed_chunks": prior_total + int(data["completed_chunks"]),
                    "total_chunks": aggregate_total,
                    "input_index": mix_index,
                    "input_completed_chunks": prior_input
                    + int(data["input_completed_chunks"]),
                    "input_total_chunks": aggregate_input_totals[mix_index],
                },
            )

        sub_callback = ensemble_progress if progress_callback else None

        results: list[Tensor] | None = None
        buffered: list[list[Tensor]] = []
        for member_index, (sub_model, model_weights) in enumerate(
            zip(model.models, model.weights)
        ):
            sub_param = next(sub_model.parameters(), None)
            sub_device = sub_param.device if sub_param is not None else None
            sub_outs = _run_ensemble_member(
                sub_model,
                mixes,
                normalize=member_normalization[member_index],
                stats=mix_stats,
                device=device,
                shifts=shifts,
                overlap=overlap,
                transition_power=transition_power,
                progress_callback=sub_callback,
                chunk_batch_size=chunk_batch_size,
                oom_backoff_state=oom_backoff_state,
                _shift_offsets=_shift_offsets,
            )
            sub_models_done += 1
            if _should_restore_submodel_device(sub_model, sub_device, device):
                sub_model.to(sub_device)
            if combine == COMBINE_DEFAULT:
                for k, inst_weight in enumerate(model_weights):
                    for sub_out in sub_outs:
                        sub_out[:, k, :, :] *= inst_weight
                if results is None:
                    results = sub_outs
                else:
                    for acc, sub_out in zip(results, sub_outs):
                        acc += sub_out
            else:
                buffered.append(sub_outs)

        if combine == COMBINE_DEFAULT:
            assert results is not None
            for acc in results:
                for k in range(acc.shape[1]):
                    acc[:, k, :, :] /= totals[k]
        else:
            results = [
                combine_member_outputs(
                    [member[index] for member in buffered],
                    model.weights,
                    combine,
                    model.combine_params,
                )
                for index in range(len(mixes))
            ]
        if progress_callback:
            progress_callback(
                "processing_complete",
                {
                    "total_chunks": aggregate_total,
                    "total_inputs": len(mixes),
                    "input_total_chunks": aggregate_input_totals,
                },
            )
        return results

    first_param = next(model.parameters(), None)
    if first_param is not None and first_param.device != device:
        model.to(device)
    if model.training:
        model.eval()
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        all_offsets = _shift_offsets
        if all_offsets is None:
            all_offsets = [
                [random.randint(0, max_shift) for _ in mixes] for _ in range(shifts)
            ]
        if len(all_offsets) != shifts or any(
            len(offsets) != len(mixes) for offsets in all_offsets
        ):
            raise RuntimeError("Internal shift-offset plan does not match inputs.")

        segment_length = int(round(model.samplerate * model.max_allowed_segment))
        stride = int((1 - overlap) * segment_length)
        if stride < 1:
            raise ValidationError(
                f"split overlap {overlap} produces an invalid stride for segment length {segment_length}"
            )

        inner_callback = progress_callback
        if progress_callback is not None:
            input_total_chunks = [0] * len(mixes)
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
    Multi-mix forward without shift averaging.

        :param model: Model to run.
        :param mixes: Input mixes.
        :param device: Inference device.
        :param overlap: Overlap ratio.
        :param transition_power: Crossfade exponent.
        :param chunk_batch_size: Chunks per forward.
        :param progress_callback: Progress callback.
        :param oom_backoff_state: Backoff dict or None.
        :return: One tensor per input.
        :raises ValidationError: If stride invalid.
    """
    assert device.type != "cuda" or device.index is not None
    segment = model.max_allowed_segment
    assert segment > 0.0
    segment_length: int = int(round(model.samplerate * segment))
    stride = int((1 - overlap) * segment_length)
    if stride < 1:
        raise ValidationError(
            f"split overlap {overlap} produces an invalid stride for segment length {segment_length}"
        )
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
    fixed_batch_shape = bool(getattr(model, "_fixed_batch_shape", False))

    chunk_valid_length: int = segment_length

    mix_states: list[dict[str, Any]] = []
    full_pool: list[tuple[int, int, TensorChunk]] = []  # (mix_idx, offset, chunk)
    tail_pool: list[tuple[int, int, TensorChunk]] = []
    input_total_chunks: list[int] = []

    for mix_idx, mix in enumerate(mixes):
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
            torch.cuda.synchronize(device)

        contributions: list[tuple[int, int, Tensor, Tensor]] = []
        for i, (mix_idx, offset, chunk) in enumerate(batch_items):
            chunk_out = center_trim(batch_out[i : i + 1], chunk.length)
            chunk_length = chunk_out.shape[-1]
            w = weight[:chunk_length]
            with torch.inference_mode():
                chunk_out.mul_(w)
            contributions.append((mix_idx, offset, chunk_out, w))

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
            if progress_callback:
                for payload in pending_events:
                    progress_callback("chunk_complete", payload)

    _drain(full_pool)

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
