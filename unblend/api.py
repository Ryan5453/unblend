# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import errno
import functools
import gc
import logging
import os
import random
import re
import tempfile
import uuid
import warnings
import wave
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
import torchaudio
from torch import Tensor

from . import __version__
from .apply import (
    COMBINE_DEFAULT,
    NORMALIZATION_EPSILON,
    Model,
    ModelEnsemble,
    _gpu_accum_budget_bytes,
    _gpu_accum_bytes_needed,
    _looks_like_cuda_oom,
    _require_cuda_available,
    apply_model_multi,
    canonical_combine,
    normalization_stats,
    resolve_combine_params,
    sole_contributor,
    validate_combine_weights,
)
from .audio import convert_audio, prevent_clip, resolve_output_path
from .backends import ASSModel, disable_custom_kernels
from .exceptions import (
    LoadAudioError,
    ModelLoadingError,
    ValidationError,
)
from .htdemucs import HTDemucs
from .repo import ModelRepository
from .roformer import _RoformerBase

logger = logging.getLogger(__name__)


class SeparatedSources:
    """
    Container for storing and processing separated audio sources.
    """

    def __init__(
        self,
        sources: dict[str, Tensor],
        sample_rate: int,
        original: Tensor,
        reliable: tuple[str, ...] | None = None,
    ) -> None:
        """
        Initialize a SeparatedSources object.

        :param sources: Mapping of stem names to audio tensors.
        :param sample_rate: Sample rate of the audio (the model's sample rate).
        :param original: The input mixture, as separated: resampled to the
            model's rate and converted to its channel count.
        :param reliable: When only one ensemble member ran (``only_load`` or
            ``use_only_stem``), the stems it is the ensemble's sole source
            of; its other outputs are a specialist's untrained heads, not real
            stems.
        """
        self.sources = sources
        self.sample_rate = sample_rate
        self.original = original
        self.reliable = reliable

    def isolate_stem(self, name: str) -> "SeparatedSources":
        """
        Isolate one stem and form its complement, ``no_<name>``: the sum of
        the other stems, or the mix minus the stem when ``reliable`` is set
        (one ensemble member ran and the other stems aren't all real).

        :param name: Name of the stem to isolate.
        :return: New SeparatedSources holding the stem and its complement.
        :raises ValidationError: If the requested stem isn't found in the
            sources, or only one ensemble member ran (``reliable``) and it was
            for a different stem.
        """
        if name not in self.sources:
            raise ValidationError(
                f"Stem '{name}' not found in sources. Available stems: {list(self.sources.keys())}"
            )
        if self.reliable is not None and name not in self.reliable:
            names = [repr(stem) for stem in self.reliable]
            listed = (
                names[0]
                if len(names) == 1
                else f"{', '.join(names[:-1])} and {names[-1]}"
            )
            raise ValidationError(
                f"Only {listed} {'was' if len(names) == 1 else 'were'} separated "
                f"(one ensemble member ran); separate again for {name!r}, or "
                "without only_load / use_only_stem."
            )

        if self.reliable is not None:
            # The specialist's other heads aren't trained stems; the mix minus
            # the stem is the faithful complement.
            complement = self.original.to(self.sources[name]) - self.sources[name]
        else:
            complement = torch.zeros_like(self.sources[name])
            for source, audio in self.sources.items():
                if source != name:
                    complement += audio

        return SeparatedSources(
            sources={name: self.sources[name], f"no_{name}": complement},
            sample_rate=self.sample_rate,
            original=self.original,
        )

    def export_stem(
        self,
        stem_name: str,
        path: Path | str | None = None,
        format: str = "wav",
        clip: str | None = "rescale",
    ) -> Path | bytes:
        """
        Export a stem to a file or return as bytes.

        :param stem_name: Name of the stem to export.
        :param path: Path to save to; if None, returns bytes. A path that
            doesn't end in an audio extension gets ``.<format>`` appended.
        :param format: Container format, e.g. ``"wav"``, ``"flac"``, ``"mp3"``.
        :param clip: Clipping mode passed to :func:`prevent_clip`, or ``None``.
        :return: The written path, or the encoded bytes when ``path`` is None.
        :raises ValidationError: If the stem isn't found or the format
            can't be encoded.
        :raises LoadAudioError: If FFmpeg can't be loaded.
        :raises OSError: If the target can't be written (``PermissionError``
            for an unwritable folder or read-only file, ``IsADirectoryError``
            for a folder at the target path).
        """
        if stem_name not in self.sources:
            raise ValidationError(
                f"Stem '{stem_name}' not found. Available stems: {list(self.sources.keys())}"
            )

        tensor = self.sources[stem_name]

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        tensor = prevent_clip(tensor, mode=clip)

        format = format.lstrip(".").lower()
        if path is not None and (
            not str(path).strip()
            or Path(path).name in ("", ".", "..")
            # The raw string too: Path("out/.") drops the "." and the slash.
            or os.path.basename(os.fspath(path)) in ("", ".", "..")
        ):
            raise ValidationError(
                f"path {os.fspath(path)!r} doesn't name a file; pass a file path or "
                "None for bytes."
            )
        file_path = None if path is None else resolve_output_path(Path(path), format)
        container = (file_path.suffix.lstrip(".") if file_path else format).lower()
        sample_rate = self.sample_rate
        if container in _OPUS_CONTAINERS and sample_rate not in _OPUS_RATES:
            # Opus only encodes at these rates; the models run at 44.1 kHz.
            tensor = torchaudio.functional.resample(tensor, sample_rate, 48000)
            sample_rate = 48000
        encoder = _torchcodec("encoder")(samples=tensor, sample_rate=sample_rate)
        if file_path is None:
            try:
                with warnings.catch_warnings():
                    # torchcodec wraps its read-only output buffer and warns.
                    warnings.filterwarnings(
                        "ignore", message="The given buffer is not writable"
                    )
                    encoded = encoder.to_tensor(format=format)
                return encoded.numpy().tobytes()
            except RuntimeError as stream_error:
                # Some formats (m4a, or muxers that need a seekable output)
                # only work through a file.
                with tempfile.TemporaryDirectory() as tmp:
                    staged = Path(tmp) / f"stem.{format}"
                    try:
                        encoder.to_file(staged)
                    except RuntimeError:
                        raise ValidationError(
                            f"Could not encode as '{format}': {stream_error}"
                        ) from stream_error
                    return staged.read_bytes()

        created = []
        missing = file_path.parent
        while not missing.exists():
            created.append(missing)
            missing = missing.parent
        file_path.parent.mkdir(exist_ok=True, parents=True)
        # Written through a symlink, as a direct write would be (a hard link
        # is replaced, since the new file is renamed into place).
        target = Path(os.path.realpath(file_path))
        if file_path.is_symlink():
            try:
                file_path.stat()
            except FileNotFoundError:
                pass  # dangling: written through below, like a direct write
            # A link loop raises ELOOP here, as a direct write would.
        if not target.parent.is_dir():
            # A dangling link into a folder that doesn't exist.
            raise FileNotFoundError(errno.ENOENT, "No such folder", str(target.parent))
        if target.is_dir():
            raise IsADirectoryError(f"{file_path} is a folder.")
        if not os.access(target.parent, os.W_OK) or (
            target.exists() and not os.access(target, os.W_OK)
        ):
            raise PermissionError(f"Can't write {file_path}.")
        # An overwritten file keeps its permissions.
        mode = target.stat().st_mode & 0o7777 if target.exists() else None
        # Encoded beside the target and renamed on success: FFmpeg truncates
        # its output before a codec can fail, which would destroy an existing
        # file. The staging name is short (so any legal target name works)
        # and keeps the suffix FFmpeg picks the muxer from.
        staging = target.with_name(f".unblend-{uuid.uuid4().hex[:8]}{file_path.suffix}")
        try:
            encoder.to_file(staging)
            if mode is not None:
                os.chmod(staging, mode)
            os.replace(staging, target)
        except BaseException as e:
            # Any failure, Ctrl-C included, must not leave the staging file.
            staging.unlink(missing_ok=True)
            # Don't leave empty directories behind for a failed export.
            for directory in created:
                try:
                    directory.rmdir()
                except OSError:
                    break
            if isinstance(e, RuntimeError):
                raise ValidationError(f"Could not encode {file_path}: {e}") from e
            raise
        return file_path


def _is_url(audio: "str | Path") -> bool:
    """
    Decide whether an audio location is a URL for FFmpeg.

    :param audio: Audio location as given by the caller.
    :return: True when the input should be decoded as a URL.
    """
    return isinstance(audio, str) and "://" in audio and not os.path.exists(audio)


def _torchcodec(kind: str) -> Any:
    """
    torchcodec's ``AudioDecoder`` or ``AudioEncoder``, imported on first use
    so a missing or unsupported FFmpeg only breaks decoding and encoding, not
    ``import unblend`` or tensor input.

    :param kind: ``"decoder"`` or ``"encoder"``.
    :return: The class.
    :raises LoadAudioError: If torchcodec can't load FFmpeg.
    """
    # torchcodec's macOS wheel for Python 3.10 bundles a profiling-enabled
    # libpython that writes default.profraw into the working folder at exit;
    # it reads this variable when the library loads.
    os.environ.setdefault("LLVM_PROFILE_FILE", os.devnull)
    try:
        if kind == "decoder":
            from torchcodec.decoders import AudioDecoder

            return AudioDecoder
        from torchcodec.encoders import AudioEncoder

        return AudioEncoder
    except (ImportError, OSError, RuntimeError) as exc:
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise LoadAudioError(
            "FFmpeg couldn't be loaded, so audio can't be read or written. "
            f"Unblend needs FFmpeg 4-8 (see the readme). Details: {detail}"
        ) from exc


# Containers whose default codec is Opus, and the only rates Opus encodes at.
_OPUS_CONTAINERS = frozenset({"opus", "webm"})
_OPUS_RATES = frozenset({8000, 12000, 16000, 24000, 48000})

# More leading-axis entries than this means time-major input, not channels.
_MAX_INPUT_CHANNELS = 32

_CUSTOM_KERNELS_ENV = "UNBLEND_CUSTOM_KERNELS"
_OFF_STRINGS = frozenset({"0", "off", "false", "no"})


def custom_kernels_enabled(setting: bool | None) -> bool:
    """
    Resolve the fused-kernel switch from an explicit setting or the environment.

    :param setting: Caller-supplied setting, or ``None`` to defer to
        ``UNBLEND_CUSTOM_KERNELS`` (on unless set to 0/off/false/no).
    :return: Whether fused kernels may be used.
    :raises ValidationError: If the setting isn't a bool or None.
    """
    if setting is not None:
        if not isinstance(setting, bool):
            raise ValidationError(
                f"custom_kernels must be True, False or None, got {setting!r}."
            )
        return setting
    return os.environ.get(_CUSTOM_KERNELS_ENV, "").strip().lower() not in _OFF_STRINGS


def default_device() -> str:
    """
    Pick the best available inference device: cuda > mps > cpu.

    :return: Device string suitable for ``Separator(device=...)``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype | None:
    """
    Pick the default compute dtype for a device: FP16 where it's fast and
    matches FP32 output closely, FP32 elsewhere.

    :param device: ``"cuda"``, ``"cuda:N"``, ``"mps"``, ``"cpu"``, or a
        ``torch.device``.
    :return: Dtype to cast weights to, or ``None`` for FP32.
    :raises ValidationError: If device is invalid or CUDA unavailable.
    """
    device = str(device)
    if device == "cuda" or re.fullmatch(r"cuda:\d+", device):
        _require_cuda_available(device)
        if device != "cuda" and int(device[5:]) >= torch.cuda.device_count():
            raise ValidationError(
                f"Device '{device}' requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        major, _minor = torch.cuda.get_device_capability(
            None if device == "cuda" else torch.device(device)
        )
        return torch.float16 if major >= 7 else None
    if device in ("mps", "mps:0"):
        return torch.float16
    if device == "cpu":
        return None
    raise ValidationError(
        f"Invalid device '{device}'. Must be one of: cpu, cuda, cuda:N, mps"
    )


def _validate_chunk_batch_size(value: object) -> int:
    """
    Validate a chunk_batch_size value (init param or per-call override).

    The upper bound is a sanity guard against typos like ``10000`` that no
    real workload needs on these models.

    :param value: Candidate chunk_batch_size.
    :return: The value as a plain ``int`` (NumPy integers are accepted).
    :raises ValidationError: If not a positive int <= 1024 (bools rejected).
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValidationError(
            f"chunk_batch_size must be a positive integer, got {value!r}"
        )
    if value > 1024:
        raise ValidationError(f"chunk_batch_size must be <= 1024, got {value}")
    return int(value)


#: Registry models whose CUDA throughput depends on the chunk batch size being
#: a multiple of some value. This is measured per model, not an architecture or
#: tensor-core rule: other models are unaffected or faster unaligned, so
#: rounding globally would be a regression.
_PREFERRED_BATCH_MULTIPLE: dict[str, int] = {"scnet_small": 8}


def _align_batch_size(model_name: str | None, estimate: int) -> int:
    """
    Round a memory-derived batch estimate onto a model's preferred multiple.

    Rounds down, never up: the estimate is a memory bound, so exceeding it to
    gain alignment would trade a throughput win for an OOM. Left alone below
    the multiple, so a card tight enough to want a small batch keeps it.

    :param model_name: Registry model name, or ``None`` for a custom model.
    :param estimate: Raw estimate derived from available memory.
    :return: The estimate, aligned if this model asks for it.
    """
    multiple = _PREFERRED_BATCH_MULTIPLE.get(model_name or "")
    if multiple is None or estimate < multiple:
        return estimate
    return estimate - (estimate % multiple)


def _on_selected_gpu(method: Callable) -> Callable:
    """
    Run a ``Separator`` method with its chosen GPU current.

    :param method: Method to wrap.
    :return: The wrapped method.
    """

    @functools.wraps(method)
    def wrapper(self: "Separator", *args: Any, **kwargs: Any) -> Any:
        with self._selected_gpu():
            return method(self, *args, **kwargs)

    return wrapper


def _owned_stems(
    weights: list[list[float]], sources: list[str], member: int | None
) -> tuple[str, ...] | None:
    """
    The stems an ensemble takes from ``member`` alone.

    :param weights: The ensemble's weight matrix.
    :param sources: Its stem names.
    :param member: The member that ran.
    :return: Those stems, or None if it is the sole source of every stem
        (then all its outputs are real) or there is no such member.
    """
    if member is None:
        return None
    owned = tuple(
        stem
        for index, stem in enumerate(sources)
        if sole_contributor(weights, index) == member
    )
    return None if len(owned) == len(sources) else owned


def _check_combine_override(
    combine: str | None,
    combine_params: Mapping | None,
    *,
    label: str,
    is_ensemble: bool,
    own_combine: str,
    own_params: Mapping | None,
    weights: list[list[float]] | None,
) -> None:
    """
    Validate a ``combine``/``combine_params`` override against the ensemble it
    would apply to, before any model is downloaded or built.

    Checked the same way with or without ``only_load``, even when the stem
    needs only one member and the override then has nothing to do.

    :param combine: Requested mode, or ``None`` to keep the ensemble's.
    :param combine_params: Requested STFT keys, merged over the ensemble's.
    :param label: How to name the model in errors.
    :param is_ensemble: Whether the model has more than one member.
    :param own_combine: The ensemble's own mode.
    :param own_params: The ensemble's own ``combine_params``.
    :param weights: The ensemble's weight matrix, if any.
    :raises ValidationError: If the override can't apply.
    """
    if combine is None and combine_params is None:
        return
    if not is_ensemble:
        given = "combine" if combine is not None else "combine_params"
        raise ValidationError(
            f"{given} only applies to an ensemble; {label} has a single member."
        )
    if combine_params is not None and not isinstance(combine_params, Mapping):
        resolve_combine_params(combine_params)  # raises: not a mapping
    mode = combine if combine is not None else own_combine
    canonical_combine(mode)
    resolve_combine_params({**(own_params or {}), **(combine_params or {})})
    validate_combine_weights(mode, weights)


class Separator:
    """
    Separates music into stems with a registry model (HTDemucs, BS-RoFormer,
    Mel-Band RoFormer, or SCNet), a custom model, or an ensemble.

    Construction loads the model onto the device and sizes the chunk batch;
    call :meth:`separate` to run it.
    """

    _CUDAGRAPH_RESERVATION_FACTOR: float = 5.0
    _EAGER_RESERVATION_FACTOR: float = 2.5
    _CUDA_VRAM_SAFETY_BYTES: int = 1 * 1024**3
    _CHUNK_BATCH_MAX_ATTEMPTS: int = 4
    _COMPILE_ROFORMER_CBS_CANDIDATES: tuple[int, ...] = (4, 8, 16, 32)

    def _members(self) -> "list[Model]":
        """
        The concrete models this separator runs, ensemble or not.

        :return: One entry for a single model, one per member for an ensemble.
        """
        if isinstance(self.model, ModelEnsemble):
            return list(self.model.models)
        return [self.model]

    def _sizing_references(self) -> "list[Model]":
        """
        The models to probe for memory sizing: every member that is an
        :class:`ASSModel`, which guarantees ``samplerate``, ``audio_channels``,
        ``max_allowed_segment`` and a ``(batch, channels, samples)`` forward.

        :return: The runnable members, possibly empty.
        """
        return [m for m in self._members() if isinstance(m, ASSModel)]

    def _measure_per_chunk_steady_bytes(self) -> int | None:
        """
        Measure per-chunk steady VRAM with an eager batch-1 forward per member.

        Members run one after another, so the largest member's footprint is
        what a chunk needs; the probe time is the sum, since every member runs
        each chunk.

        :return: Peak per-chunk delta in bytes, or ``None`` if unavailable.
        """
        if self.device != "cuda":
            return None
        refs = self._sizing_references()
        if not refs:
            return None
        measured = 0
        probe_seconds = 0.0
        try:
            for ref in refs:
                training_length = int(round(ref.max_allowed_segment * ref.samplerate))
                device_obj = next(ref.parameters()).device
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                resident_before = torch.cuda.memory_allocated()
                cudnn_benchmark_saved = torch.backends.cudnn.benchmark
                torch.backends.cudnn.benchmark = False
                try:
                    dummy = torch.zeros(
                        1,
                        ref.audio_channels,
                        training_length,
                        device=device_obj,
                        dtype=torch.float32,
                    )
                    with torch.inference_mode():
                        _ = ref(dummy)
                    torch.cuda.synchronize()
                    probe_started = perf_counter()
                    with torch.inference_mode():
                        _ = ref(dummy)
                    torch.cuda.synchronize()
                    probe_seconds += perf_counter() - probe_started
                finally:
                    torch.backends.cudnn.benchmark = cudnn_benchmark_saved
                peak = torch.cuda.max_memory_allocated()
                measured = max(measured, peak - resident_before)
        except Exception:
            return None
        self._eager_probe_seconds = probe_seconds
        self._per_chunk_steady_bytes = max(1, measured)
        return self._per_chunk_steady_bytes

    def _prewarm_allocator(self) -> None:
        """
        Run one discarded forward per member at the chosen batch size.

        The sizing probe runs at batch 1 after ``empty_cache()``, leaving the
        caching allocator nearly empty; without this, the first tracks pay for
        growing it to the real batch size.

        Best effort: a failure here must not stop construction, and a batch
        that does not fit is caught by the OOM backoff on the real call.
        """
        if self.device != "cuda" or self._compile_enabled:
            return
        # Every CUDA call, including the sync, stays inside the guard: this
        # also runs on stubs and CUDA-less builds where ``synchronize()`` raises.
        try:
            warmed = False
            for member in self._members():
                if not isinstance(member, ASSModel):
                    continue
                parameter = next(member.parameters(), None)
                if parameter is None:
                    continue
                segment_length = int(
                    round(member.samplerate * member.max_allowed_segment)
                )
                dummy = torch.zeros(
                    self.chunk_batch_size,
                    member.audio_channels,
                    segment_length,
                    device=parameter.device,
                    dtype=torch.float32,
                )
                with torch.inference_mode():
                    member(dummy)
                del dummy
                warmed = True
            if warmed:
                torch.cuda.synchronize()
        except Exception:
            # Includes OOM: fall through to the real call's backoff.
            return

    def _initial_chunk_batch_size_estimate(self) -> int:
        """
        Estimate the initial batch size from free VRAM and per-chunk cost.

        :return: Initial chunks per batch; 1 on CPU and MPS.
        """
        if self.device == "cpu":
            return 1
        if self.device == "mps":
            # MPS gains little from batching while the working set grows with
            # the batch, and some models slow sharply at larger batches. Batch
            # 1 is close to every model's optimum and uses the least memory; a
            # per-run timing sweep costs more than it saves. Callers can still
            # pass ``chunk_batch_size=``.
            return 1

        per_chunk_steady = getattr(self, "_per_chunk_steady_bytes", None)
        if per_chunk_steady is None:
            per_chunk_steady = self._measure_per_chunk_steady_bytes()
        if per_chunk_steady is None:
            return 4

        try:
            free_bytes, _total = torch.cuda.mem_get_info()
        except Exception:
            return 4

        available = max(0, free_bytes - self._CUDA_VRAM_SAFETY_BYTES)
        reservation_factor = (
            self._CUDAGRAPH_RESERVATION_FACTOR
            if self._compile_enabled
            else self._EAGER_RESERVATION_FACTOR
        )
        transient_per_chunk = reservation_factor * per_chunk_steady
        if transient_per_chunk <= 0:
            return 4
        estimate = max(1, min(1024, int(available // transient_per_chunk)))

        wants_power_of_two = any(
            bool(getattr(model, "prefers_power_of_two_batch", False))
            for model in self._members()
        )
        if self._compile_enabled and wants_power_of_two:
            # Already a power of two, hence already aligned to any multiple
            # _PREFERRED_BATCH_MULTIPLE asks for.
            return max(1, 1 << (estimate.bit_length() - 1))

        return _align_batch_size(self._model_name, estimate)

    def _setup_compile(self) -> None:
        """
        Apply the family-specific CUDA compile target to every model.
        """
        for model in self._members():
            # Only the families with a compile target; an ensemble's other
            # members stay eager.
            if not isinstance(model, (HTDemucs, _RoformerBase)):
                continue
            hook = getattr(model, "enable_compiled_core", None)
            if hook is not None:
                hook()

    def _teardown_compile_state(self) -> None:
        """
        Reverse :meth:`_setup_compile` and release compile state.
        """
        for model in self._members():
            if not isinstance(model, (HTDemucs, _RoformerBase)):
                continue
            hook = getattr(model, "disable_compiled_core", None)
            if hook is not None:
                hook()
            model._fixed_batch_shape = False
        torch._dynamo.reset()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _calibrate_chunk_batch_size(
        self, initial_guess: int, compile_enabled: bool
    ) -> int:
        """
        Verify a batch size by CUDA graph capture when compiling.

        Compiled RoFormers sweep candidate sizes for the fastest; other models
        halve on OOM until capture fits. Without compile this is a no-op.

        :param initial_guess: Starting batch size to verify.
        :param compile_enabled: Whether torch.compile is active.
        :return: Verified batch size.
        :raises ModelLoadingError: If no size fits.
        """
        if self.device != "cuda":
            return initial_guess
        if not compile_enabled:
            return initial_guess

        if isinstance(self.model, _RoformerBase):
            return self._sweep_compiled_roformer_cbs(initial_guess, self.model)
        return self._calibrate_by_halving(initial_guess)

    def _calibrate_by_halving(self, initial_guess: int) -> int:
        """
        Capture-verify ``initial_guess`` and halve on CUDA OOM until it fits.

        :param initial_guess: Starting chunk_batch_size to capture at.
        :return: The largest verified chunk_batch_size at or below the guess.
        :raises ModelLoadingError: If even batch size 1 OOMs during capture.
        """
        candidate = max(1, initial_guess)
        last_error: BaseException | None = None
        tried: list[int] = []
        for attempt in range(self._CHUNK_BATCH_MAX_ATTEMPTS):
            tried.append(candidate)
            self.chunk_batch_size = candidate
            try:
                self._setup_compile()
                self._warmup_via_inference()
                self._calibration_attempts = tried
                return candidate
            except RuntimeError as exc:
                if not _looks_like_cuda_oom(exc):
                    raise
                last_error = exc
                self._teardown_compile_state()
                if candidate <= 1:
                    break
                candidate = max(1, candidate // 2)
        self._calibration_attempts = tried
        raise ModelLoadingError(
            f"chunk_batch_size calibration exhausted "
            f"{len(tried)} attempts (tried {tried}). "
            f"Last error: {last_error}"
        )

    def _time_forward_seconds_per_chunk(self, ref: Model) -> float:
        """
        Time a full forward at the current batch size, best of five.

        :param ref: Model to time.
        :return: Best seconds per chunk.
        """
        segment_length = int(round(ref.samplerate * ref.max_allowed_segment))
        device = next(ref.parameters()).device
        dummy = torch.zeros(
            self.chunk_batch_size,
            ref.audio_channels,
            segment_length,
            device=device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            ref(dummy)
            ref(dummy)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(5):
            started = perf_counter()
            with torch.inference_mode():
                ref(dummy)
            torch.cuda.synchronize()
            best = min(best, perf_counter() - started)
        return best / self.chunk_batch_size

    def _sweep_compiled_roformer_cbs(self, ceiling: int, ref: Model) -> int:
        """
        Sweep candidate batch sizes for a compiled RoFormer and keep the fastest.

        The sweep stops at the first candidate that is not at least 2% faster
        per chunk than the best so far, or that OOMs.

        :param ceiling: Largest batch estimated to fit in VRAM.
        :param ref: RoFormer being compiled.
        :return: Selected batch size.
        :raises ModelLoadingError: If no size fits.
        """
        candidates = [c for c in self._COMPILE_ROFORMER_CBS_CANDIDATES if c <= ceiling]
        if not candidates:
            candidates = [max(1, ceiling)]

        best_cbs: int | None = None
        best_seconds_per_chunk = float("inf")
        tried: list[int] = []
        for cbs in candidates:
            self.chunk_batch_size = cbs
            tried.append(cbs)
            try:
                self._setup_compile()
                self._warmup_via_inference()
            except RuntimeError as exc:
                if not _looks_like_cuda_oom(exc):
                    raise
                self._teardown_compile_state()
                if best_cbs is None:
                    self._calibration_attempts = tried
                    return self._calibrate_by_halving(max(1, cbs // 2))
                break
            seconds_per_chunk = self._time_forward_seconds_per_chunk(ref)
            self._teardown_compile_state()
            if seconds_per_chunk < best_seconds_per_chunk * 0.98:
                best_seconds_per_chunk = seconds_per_chunk
                best_cbs = cbs
            else:
                break

        assert best_cbs is not None
        self.chunk_batch_size = best_cbs
        self._setup_compile()
        try:
            self._warmup_via_inference()
        except RuntimeError as exc:
            if not _looks_like_cuda_oom(exc):
                raise
            self._teardown_compile_state()
            raise ModelLoadingError(
                f"Compiled capture at chunk_batch_size={best_cbs} ran out of GPU "
                f"memory. Pass a smaller chunk_batch_size. Original error: {exc}"
            ) from exc
        self._calibration_attempts = tried
        return best_cbs

    def _warmup_via_inference(self) -> None:
        """
        Capture CUDAGraphs via dummy inference through :meth:`separate`.
        """
        supported = (HTDemucs, _RoformerBase)
        ref = next((m for m in self._members() if isinstance(m, supported)), None)
        if ref is None:
            return

        samplerate = ref.samplerate
        channels = ref.audio_channels
        segment_length = int(round(ref.max_allowed_segment * samplerate))
        dummy = torch.zeros(channels, segment_length, dtype=torch.float32)

        self.separate(
            audio=(dummy, samplerate),
            shifts=1,
            split_overlap=0.25,
            chunk_batch_size=self.chunk_batch_size,
        )

        self.separate(
            audio=[(dummy, samplerate), (dummy, samplerate)],
            shifts=1,
            split_overlap=0.25,
            chunk_batch_size=self.chunk_batch_size,
        )

    def __init__(
        self,
        model: str | Model | ModelEnsemble = "htdemucs",
        device: str | None = None,
        only_load: str | None = None,
        dtype: torch.dtype | str | None = "auto",
        compile: bool = False,
        chunk_batch_size: int | None = None,
        custom_kernels: bool | None = None,
        combine: str | None = None,
        combine_params: dict | None = None,
    ) -> None:
        """
        Load a model and prepare it for inference on a device.

        :param model: Registry model name, or a loaded model or ensemble.
        :param device: ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or ``"mps"``;
            auto-selects if None.
        :param only_load: Stem to isolate; an ensemble whose weights give that
            stem to one member loads only that member.
        :param dtype: Compute dtype: ``"auto"`` picks per device, ``None`` or
            ``torch.float32`` is FP32, or ``torch.float16``/``torch.bfloat16``
            on CUDA or MPS.
        :param compile: Apply torch.compile with CUDA graphs. CUDA only (ignored
            elsewhere); HTDemucs and RoFormer only (``ValidationError`` for
            SCNet).
        :param chunk_batch_size: Chunks per forward; sized automatically if None.
        :param custom_kernels: Use fused CUDA/Metal kernels where available;
            ``None`` defers to ``UNBLEND_CUSTOM_KERNELS``.
        :param combine: Ensemble combine mode override.
        :param combine_params: STFT geometry for spectral combine modes.
        :raises ValidationError: If args invalid.
        :raises ModelLoadingError: If model fails to load.
        """
        if device is None:
            device = default_device()
        elif isinstance(device, torch.device):
            device = str(device)

        self._cuda_index: int | None = None
        match = re.fullmatch(r"cuda:(\d+)", device) if isinstance(device, str) else None
        if match:
            _require_cuda_available(device)
            index = int(match.group(1))
            if index >= torch.cuda.device_count():
                raise ValidationError(
                    f"Device '{device}' requested but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are visible."
                )
            self._cuda_index = index
            device = "cuda"
        elif device == "mps:0":
            # str(torch.device("mps")) after a tensor lands there; there is
            # only ever one MPS device.
            device = "mps"

        valid_devices = {"cpu", "cuda", "mps"}
        if device not in valid_devices:
            raise ValidationError(
                f"Invalid device '{device}'. Must be one of: cpu, cuda, cuda:N, mps"
            )
        if device == "cuda":
            _require_cuda_available()
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValidationError(
                "Device 'mps' requested but MPS is not available on this system."
            )

        if isinstance(dtype, str):
            if dtype != "auto":
                raise ValidationError(
                    f"Invalid dtype '{dtype}'. Use 'auto', None, or a torch.dtype "
                    "(torch.float32, torch.float16, torch.bfloat16)."
                )
            with self._selected_gpu():
                dtype = default_dtype(device)
        elif dtype == torch.float32:
            dtype = None
        if dtype is not None:
            if dtype not in (torch.float16, torch.bfloat16):
                raise ValidationError(
                    f"Invalid dtype '{dtype}'. Only torch.float16 and torch.bfloat16 "
                    "are supported for compute. This is separate from the precision a "
                    "checkpoint is stored at, which may be anything and is widened "
                    "on load."
                )
            if device == "cpu":
                raise ValidationError(
                    f"{dtype} inference is not supported on CPU. Use cuda or mps."
                )

        if chunk_batch_size is not None:
            chunk_batch_size = _validate_chunk_batch_size(chunk_batch_size)

        self.device = device
        self.dtype = dtype
        use_custom_kernels = custom_kernels_enabled(custom_kernels)

        model_info: dict | None = None
        if isinstance(model, str):
            model_repo = ModelRepository()
            model_info = model_repo.list_models().get(model)
            if (
                model_info is not None
                and only_load is not None
                and only_load not in model_info["sources"]
            ):
                raise ValidationError(
                    f"Stem {only_load!r} not found in model. Available stems: "
                    f"{', '.join(model_info['sources'])}"
                )
            if (
                device == "cuda"
                and dtype in (torch.float16, torch.bfloat16)
                and not compile
                and use_custom_kernels
                and model_info is not None
            ):
                from .cuda import swappable_backends, warmup_async

                if model_info.get("backend") in swappable_backends():
                    warmup_async(self._cuda_index)
            if model_info is not None:
                # An unknown name falls through to get_model's own error.
                _check_combine_override(
                    combine,
                    combine_params,
                    label=model,
                    is_ensemble="members" in model_info,
                    own_combine=model_info.get("combine", COMBINE_DEFAULT),
                    own_params=model_info.get("combine_params"),
                    weights=model_info.get("weights"),
                )
            self.model = model_repo.get_model(name=model, only_load=only_load)
        elif isinstance(model, torch.nn.Module):
            missing = [
                attr
                for attr in (
                    "sources",
                    "samplerate",
                    "audio_channels",
                    "max_allowed_segment",
                )
                if not hasattr(model, attr)
            ]
            if missing:
                raise ValidationError(
                    f"{type(model).__name__} isn't a separation model: it has no "
                    f"{', '.join(missing)}. Pass a registry name, a Model or a "
                    "ModelEnsemble."
                )
            self.model = model
            is_ensemble = isinstance(model, ModelEnsemble)
            _check_combine_override(
                combine,
                combine_params,
                label=type(model).__name__,
                is_ensemble=is_ensemble,
                own_combine=model.combine if is_ensemble else COMBINE_DEFAULT,
                own_params=model.combine_params if is_ensemble else None,
                weights=model.weights if is_ensemble else None,
            )
            # As for registry ensembles: a stem only one member produces
            # needs only that member.
            if (
                isinstance(model, ModelEnsemble)
                and only_load is not None
                and only_load in model.sources
            ):
                index = sole_contributor(model.weights, model.sources.index(only_load))
                if index is not None:
                    self.model = model.models[index]
        else:
            raise ValidationError(
                "model must be a registry name, a Model or a ModelEnsemble, got "
                f"{type(model).__name__}."
            )

        if self.model is None:
            raise ModelLoadingError("Failed to load model")

        # The override (validated above) applies to the ensemble requested.
        # only_load may have narrowed that to one member — or, for a nested
        # ensemble passed in, to an inner ensemble — through which the stem
        # passes unchanged, so the override has nothing to do.
        narrowed = (
            self.model is not model
            if isinstance(model, torch.nn.Module)
            else not isinstance(self.model, ModelEnsemble)
        )
        requested_ensemble = isinstance(model, ModelEnsemble) or (
            isinstance(model, str)
            and model_info is not None
            and "members" in model_info
        )
        # The stems of the single member only_load narrowed an ensemble to:
        # its other outputs aren't trained stems (see SeparatedSources.reliable).
        # A nested inner ensemble kept instead is a complete model, all of whose
        # outputs are real.
        self._specialist_stems: tuple[str, ...] | None = None
        if (
            only_load is not None
            and narrowed
            and requested_ensemble
            and not isinstance(self.model, ModelEnsemble)
        ):
            if isinstance(model, ModelEnsemble):
                weights, sources = model.weights, list(model.sources)
            else:
                weights, sources = model_info.get("weights"), model_info["sources"]
            self._specialist_stems = _owned_stems(
                weights, sources, sole_contributor(weights, sources.index(only_load))
            )
        if (combine is not None or combine_params is not None) and not narrowed:
            # A shallow copy: the override is this Separator's, and the
            # caller's ensemble (members and weights shared) keeps its mode.
            self.model = copy.copy(self.model)
            self.model.set_combine(
                combine if combine is not None else self.model.combine,
                {**self.model.combine_params, **(combine_params or {})},
            )

        self.model.eval()

        if only_load is not None and only_load not in self.model.sources:
            raise ValidationError(
                f"Stem {only_load!r} not found in model. "
                f"Available stems: {', '.join(self.model.sources)}"
            )

        self.audio_channels = self.model.audio_channels
        self.sample_rate = self.model.samplerate

        prev_cudnn_benchmark = (
            torch.backends.cudnn.benchmark if self.device == "cuda" else None
        )
        prev_matmul_precision = (
            torch.get_float32_matmul_precision() if self.device == "cuda" else None
        )
        gpu = self._selected_gpu()
        gpu.__enter__()
        try:
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = compile
                torch.set_float32_matmul_precision("high")

            # Also for CPU: a model passed in on a GPU must come back.
            self.model.to(self.device)

            # None means FP32, including for a model passed in at lower
            # precision.
            compute = self.dtype if self.dtype is not None else torch.float32
            if isinstance(self.model, ModelEnsemble):
                for m in self.model.models:
                    m.to(dtype=compute)
            else:
                self.model.to(dtype=compute)

            if not use_custom_kernels:
                disable_custom_kernels(self.model)
                swapped = [
                    m
                    for m in self.model.modules()
                    if type(m).__module__ in ("unblend.metal", "unblend.cuda")
                ]
                if swapped:
                    # Fused replacements from an earlier Separator can't be
                    # swapped back; only a fresh model is guaranteed unfused.
                    raise ValidationError(
                        "custom_kernels=False, but this model instance already "
                        "has fused-kernel modules swapped in by an earlier "
                        "Separator. Load a fresh model for an unfused run."
                    )

            if (
                use_custom_kernels
                and self.dtype in (torch.float16, torch.bfloat16)
                and self.device == "mps"
            ):
                from .metal import apply_metal_optimizations, has_swappable_modules

                for member in self._members():
                    if has_swappable_modules(member):
                        apply_metal_optimizations(member)

            if (
                use_custom_kernels
                and self.dtype in (torch.float16, torch.bfloat16)
                and self.device == "cuda"
                and not compile
            ):
                from .cuda import apply_cuda_optimizations, has_swappable_modules

                for member in self._members():
                    if has_swappable_modules(member):
                        apply_cuda_optimizations(member)

            self._model_name = model if isinstance(model, str) else None
            if (
                compile
                and self.device == "cuda"
                and not any(
                    isinstance(member, (HTDemucs, _RoformerBase))
                    for member in self._members()
                )
            ):
                raise ValidationError(
                    "compile=True is only supported for HTDemucs and RoFormer models."
                )
            self._compile_enabled = compile and self.device == "cuda"
            self._eager_probe_seconds: float | None = None
            self._per_chunk_steady_bytes: int | None = None
            self._calibration_attempts: list[int] = []
            self._chunk_batch_size_auto = chunk_batch_size is None
            if chunk_batch_size is not None:
                self.chunk_batch_size = chunk_batch_size
                if self._compile_enabled:
                    try:
                        self._setup_compile()
                        self._warmup_via_inference()
                    except RuntimeError as exc:
                        if not _looks_like_cuda_oom(exc):
                            raise
                        self._teardown_compile_state()
                        raise ModelLoadingError(
                            f"Explicit chunk_batch_size={chunk_batch_size} does "
                            f"not fit on this GPU under compile (OOM during "
                            f"capture). Lower it, or omit it for auto-sizing. "
                            f"Original error: {exc}"
                        ) from exc
            else:
                initial_cbs = self._initial_chunk_batch_size_estimate()
                self.chunk_batch_size = self._calibrate_chunk_batch_size(
                    initial_guess=initial_cbs,
                    compile_enabled=self._compile_enabled,
                )
                per_chunk = getattr(self, "_per_chunk_steady_bytes", None)
                if per_chunk is not None and self.device == "cuda":
                    reserve = int(1.5 * per_chunk * self.chunk_batch_size)
                    targets = (
                        list(self.model.models) + [self.model]
                        if isinstance(self.model, ModelEnsemble)
                        else [self.model]
                    )
                    for target in targets:
                        target._forward_reserve_bytes = reserve
            self._prewarm_allocator()
        finally:
            gpu.__exit__(None, None, None)
            if prev_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = prev_cudnn_benchmark
            if prev_matmul_precision is not None:
                torch.set_float32_matmul_precision(prev_matmul_precision)

    def _selected_gpu(self) -> contextlib.AbstractContextManager:
        """
        Make the GPU chosen with ``device="cuda:N"`` current, so every
        ``"cuda"`` allocation and kernel launch lands on it.

        :return: A ``torch.cuda.device`` context, or a no-op one.
        """
        if getattr(self, "_cuda_index", None) is None:
            return contextlib.nullcontext()
        return torch.cuda.device(self._cuda_index)

    @_on_selected_gpu
    def enable_compile(self) -> None:
        """
        Compile an eager CUDA separator in place, recalibrating the batch size
        if it was sized automatically. A no-op if already compiled.

        :raises ValidationError: If not CUDA or unsupported model.
        :raises ModelLoadingError: If capture OOMs.
        """
        if self._compile_enabled:
            return
        if self.device != "cuda":
            raise ValidationError(
                "enable_compile() is only supported for CUDA separators."
            )
        supported = (HTDemucs, _RoformerBase)
        if not any(isinstance(model, supported) for model in self._members()):
            raise ValidationError(
                "enable_compile() is only supported for HTDemucs and RoFormer models."
            )

        previous_batch_size = self.chunk_batch_size
        previous_cudnn_benchmark = torch.backends.cudnn.benchmark
        self._compile_enabled = True
        torch.backends.cudnn.benchmark = True
        try:
            if getattr(self, "_chunk_batch_size_auto", True):
                initial_cbs = self._initial_chunk_batch_size_estimate()
                self.chunk_batch_size = self._calibrate_chunk_batch_size(
                    initial_guess=initial_cbs,
                    compile_enabled=True,
                )
            else:
                self._setup_compile()
                self._warmup_via_inference()
        except Exception as exc:
            self._teardown_compile_state()
            self._compile_enabled = False
            self.chunk_batch_size = previous_batch_size
            if isinstance(exc, RuntimeError) and _looks_like_cuda_oom(exc):
                raise ModelLoadingError(
                    f"chunk_batch_size={previous_batch_size} does not fit on this "
                    "GPU under compile (OOM during capture). Lower it, or leave "
                    f"it unset for auto-sizing. Original error: {exc}"
                ) from exc
            raise
        finally:
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark

        per_chunk = getattr(self, "_per_chunk_steady_bytes", None)
        if per_chunk is not None:
            reserve = int(1.5 * per_chunk * self.chunk_batch_size)
            targets = (
                list(self.model.models) + [self.model]
                if isinstance(self.model, ModelEnsemble)
                else [self.model]
            )
            for target in targets:
                target._forward_reserve_bytes = reserve

    @_on_selected_gpu
    def warmup(self) -> None:
        """
        Pay compile and capture cost up front, before the first real call.

        :raises ValidationError: If not CUDA or unsupported model.
        """
        if self.device != "cuda":
            raise ValidationError("warmup() is only supported for CUDA separators.")
        supported = (HTDemucs, _RoformerBase)
        if not any(isinstance(model, supported) for model in self._members()):
            raise ValidationError(
                "warmup() is only supported for HTDemucs and RoFormer models."
            )
        self._warmup_via_inference()

    @staticmethod
    def _read_pcm16_wav(path: Path | str) -> tuple[Tensor, int] | None:
        """
        Decode a 16-bit PCM WAV directly, bypassing FFmpeg.

        :param path: Candidate file path.
        :return: ``(waveform, sample_rate)`` if it is 16-bit PCM WAV, else
            ``None`` so the caller falls back to torchcodec.
        """
        try:
            with wave.open(str(path), "rb") as w:
                if w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                    return None
                num_frames = w.getnframes()
                channels = w.getnchannels()
                sample_rate = w.getframerate()
                raw = w.readframes(num_frames)
        except (wave.Error, EOFError, OSError, ValueError):
            return None
        if channels <= 0 or sample_rate <= 0 or not raw:
            return None
        if len(raw) % (2 * channels) != 0:
            return None
        samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
        wav = torch.from_numpy(
            np.ascontiguousarray(samples.astype(np.float32).T / 32768.0)
        )
        return wav, sample_rate

    def _to_tensor(self, audio: tuple[Tensor, int] | Path | str | bytes) -> Tensor:
        """
        Decode or validate an input into a float32 waveform at the model's
        sample rate and channel count.

        :param audio: ``(Tensor, sample_rate)`` tuple, path or URL, or bytes.
        :return: ``[channels, samples]`` waveform.
        :raises LoadAudioError: If decoding fails.
        :raises ValidationError: If input type invalid or empty.
        """
        wav: Tensor
        input_sr: int | None = None

        if isinstance(audio, tuple):
            if len(audio) != 2:
                raise ValidationError(
                    f"Expected a (Tensor, sample_rate) tuple, got {len(audio)} "
                    "elements."
                )
            wav, input_sr = audio
            if not isinstance(wav, Tensor):
                raise ValidationError(
                    "Expected a torch.Tensor as the first tuple element, got "
                    f"{type(wav).__name__}."
                )
            if wav.dim() not in (1, 2):
                raise ValidationError(
                    f"Expected a 1-D or 2-D waveform tensor, got {wav.dim()} "
                    "dimensions."
                )
            if wav.dim() == 2 and wav.shape[0] > _MAX_INPUT_CHANNELS:
                raise ValidationError(
                    f"Expected a [channels, samples] tensor, got shape "
                    f"{tuple(wav.shape)}; transpose time-major audio with .T."
                )
            if not wav.is_floating_point():
                raise ValidationError(
                    "Waveform tensor must use a floating-point dtype with "
                    "samples already normalized to audio amplitude range; got "
                    f"{wav.dtype}."
                )
            if isinstance(input_sr, bool):
                raise ValidationError("Sample rate must be an int, got bool.")
            if isinstance(input_sr, (int, np.integer)) or (
                isinstance(input_sr, (float, np.floating))
                and float(input_sr).is_integer()
            ):
                input_sr = int(input_sr)
            else:
                raise ValidationError(
                    f"Sample rate must be an int, got {type(input_sr).__name__}."
                )
            if input_sr <= 0:
                raise ValidationError(f"Sample rate must be positive, got {input_sr}.")
        elif isinstance(audio, (str, Path)):
            is_url = _is_url(audio)
            if not is_url:
                try:
                    Path(audio).stat()
                except FileNotFoundError:
                    raise LoadAudioError(f"File not found: {audio}") from None
                except (OSError, ValueError):
                    pass
            pcm = None if is_url else self._read_pcm16_wav(audio)
            if pcm is None:
                AudioDecoder = _torchcodec("decoder")
            if pcm is not None:
                wav, input_sr = pcm
            elif is_url:
                try:
                    decoder = AudioDecoder(audio)
                    audio_samples = decoder.get_all_samples()
                    wav = audio_samples.data
                    input_sr = audio_samples.sample_rate
                except Exception as e:
                    raise LoadAudioError(
                        f"Could not load {audio} using torchcodec: {e}"
                    ) from e
            else:
                try:
                    decoder = AudioDecoder(str(Path(audio)))
                    audio_samples = decoder.get_all_samples()
                    wav = audio_samples.data
                    input_sr = audio_samples.sample_rate
                except Exception as e:
                    raise LoadAudioError(
                        f"Could not load file {audio} using torchcodec: {e}. "
                        "Make sure the file format is supported."
                    ) from e
        elif isinstance(audio, bytes):
            AudioDecoder = _torchcodec("decoder")
            audio_buffer = BytesIO(audio)
            try:
                decoder = AudioDecoder(audio_buffer)
                audio_samples = decoder.get_all_samples()
                wav = audio_samples.data
                input_sr = audio_samples.sample_rate
            except Exception as e:
                raise LoadAudioError(
                    f"Could not load audio from bytes using torchcodec: {e}. "
                    "Make sure the audio format is supported."
                ) from e
            finally:
                audio_buffer.close()
        else:
            raise ValidationError(
                f"Unsupported audio input type: {type(audio)}. "
                "Expected tuple of (Tensor, sample_rate), file path (str/Path), or bytes."
            )

        wav = wav.detach()
        if wav.dim() == 1:
            wav = wav[None]
        if wav.dtype != torch.float32:
            wav = wav.float()

        if wav.shape[-1] == 0:
            raise ValidationError("Audio input is empty (zero samples).")
        if not torch.isfinite(wav).all():
            raise ValidationError("Audio input contains NaN or infinite samples.")

        if input_sr is not None and input_sr != self.sample_rate:
            wav = convert_audio(wav, input_sr, self.sample_rate, self.audio_channels)
        elif wav.shape[0] != self.audio_channels:
            wav = convert_audio(
                wav, self.sample_rate, self.sample_rate, self.audio_channels
            )

        return wav

    @_on_selected_gpu
    def separate(
        self,
        audio: tuple[Tensor, int]
        | Path
        | str
        | bytes
        | list[tuple[Tensor, int] | Path | str | bytes],
        shifts: int = 1,
        split_overlap: float = 0.25,
        seed: int | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        use_only_stem: str | None = None,
        chunk_batch_size: int | None = None,
    ) -> "SeparatedSources | list[SeparatedSources]":
        """
        Separate audio into stems.

        :param audio: A ``(Tensor, sample_rate)`` tuple, a file path or URL,
            encoded bytes, or a list of these to separate as one batch.
        :param shifts: Random time shifts to average over (0-20); more is
            slower and slightly better. 0 is deterministic without a seed.
        :param split_overlap: Fractional overlap between segments, in [0, 1).
        :param seed: Seed for the shift offsets, making output reproducible.
            Uses a private RNG; the global ``random``/``torch`` state is not
            touched.
        :param progress_callback: Called as ``callback(event, payload)``.
        :param use_only_stem: For an ensemble, run only the member that owns
            this stem.
        :param chunk_batch_size: Chunks per forward for this call; defaults to
            the separator's. Must match it under compile.
        :return: A ``SeparatedSources``, or a list of them for list input.
        :raises ValidationError: If a parameter or input is invalid.
        :raises LoadAudioError: If an input cannot be decoded.
        """
        if (
            isinstance(shifts, bool)
            or not isinstance(shifts, (int, np.integer))
            or not 0 <= shifts <= 20
        ):
            raise ValidationError(
                f"shifts must be an integer between 0 and 20 (inclusive), got {shifts}"
            )

        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
        ):
            raise ValidationError(
                f"seed must be an integer if provided, got {type(seed)}"
            )
        shifts = int(shifts)
        seed = None if seed is None else int(seed)

        if (
            isinstance(split_overlap, bool)
            or not isinstance(split_overlap, (int, float, np.floating))
            or not 0.0 <= split_overlap < 1.0
        ):
            raise ValidationError(
                f"split_overlap must be a float between 0.0 (inclusive) and 1.0 (exclusive), got {split_overlap}"
            )

        per_call_chunk_batch_size = chunk_batch_size is not None
        if chunk_batch_size is None:
            chunk_batch_size = self.chunk_batch_size
        else:
            chunk_batch_size = _validate_chunk_batch_size(chunk_batch_size)
            if self._compile_enabled and chunk_batch_size != self.chunk_batch_size:
                raise ValidationError(
                    f"This separator is compiled with a fixed "
                    f"chunk_batch_size={self.chunk_batch_size}; per-call "
                    f"overrides are not supported under compile. Pass "
                    f"chunk_batch_size to Separator(...) instead."
                )

        allow_oom_backoff = (
            getattr(self, "_chunk_batch_size_auto", True)
            and not per_call_chunk_batch_size
        )

        if use_only_stem is not None and use_only_stem not in self.model.sources:
            raise ValidationError(
                f"use_only_stem '{use_only_stem}' is not a source of this model. "
                f"Available stems: {', '.join(self.model.sources)}"
            )

        if progress_callback is not None and not callable(progress_callback):
            raise ValidationError(
                f"progress_callback must be callable if provided, got {type(progress_callback)}"
            )

        try:
            if isinstance(audio, list):
                if not audio:
                    return []
                return self._run_with_oom_backoff(
                    lambda cbs, state: self._separate_batch(
                        audio,
                        shifts=shifts,
                        split_overlap=split_overlap,
                        seed=seed,
                        progress_callback=progress_callback,
                        use_only_stem=use_only_stem,
                        chunk_batch_size=cbs,
                        oom_backoff_state=state,
                    ),
                    chunk_batch_size=chunk_batch_size,
                    allow=allow_oom_backoff,
                )

            return self._run_with_oom_backoff(
                lambda cbs, state: self._separate_one(
                    audio,
                    shifts=shifts,
                    split_overlap=split_overlap,
                    seed=seed,
                    progress_callback=progress_callback,
                    use_only_stem=use_only_stem,
                    chunk_batch_size=cbs,
                    oom_backoff_state=state,
                ),
                chunk_batch_size=chunk_batch_size,
                allow=allow_oom_backoff,
            )
        finally:
            self._release_mps_cache()

    def _release_mps_cache(self) -> None:
        """
        Return cached Metal buffers to the OS.
        """
        if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _run_with_oom_backoff(
        self,
        call: "Callable[[int, dict[str, int] | None], Any]",
        *,
        chunk_batch_size: int,
        allow: bool,
    ) -> Any:
        """
        Run a separation call, recapturing at half the batch size on CUDA OOM.

        A lowered batch size sticks for the rest of this separator's life.

        :param call: Closure taking ``(chunk_batch_size, backoff_state)``.
        :param chunk_batch_size: Batch size for the first attempt.
        :param allow: Whether backoff applies (auto-sized batches only).
        :return: The call's result.
        """
        state = {"chunk_batch_size": chunk_batch_size} if allow else None
        current = chunk_batch_size
        attempts = 0
        while True:
            try:
                result = call(current, state)
            except RuntimeError as exc:
                if (
                    not allow
                    or not self._compile_enabled
                    or current <= 1
                    or attempts >= self._CHUNK_BATCH_MAX_ATTEMPTS
                    or not _looks_like_cuda_oom(exc)
                ):
                    raise
                attempts += 1
                previous = current
                self._teardown_compile_state()
                self.chunk_batch_size = self._calibrate_chunk_batch_size(
                    initial_guess=max(1, previous // 2),
                    compile_enabled=True,
                )
                current = self.chunk_batch_size
                if state is not None:
                    state["chunk_batch_size"] = current
                logger.warning(
                    "CUDA OOM mid-run at chunk_batch_size=%d (compiled); "
                    "recaptured at %d and retrying the request from the "
                    "start (progress restarts).",
                    previous,
                    current,
                )
                continue
            if state is not None and state["chunk_batch_size"] < self.chunk_batch_size:
                logger.warning(
                    "chunk_batch_size lowered %d -> %d after CUDA OOM "
                    "backoff (sticky for this separator).",
                    self.chunk_batch_size,
                    state["chunk_batch_size"],
                )
                self.chunk_batch_size = state["chunk_batch_size"]
            return result

    def _stage_for_inference(self, wavs: list[Tensor], shifts: int) -> list[Tensor]:
        """
        Move waveforms to the GPU when the GPU-resident pipeline fits.

        :param wavs: Decoded waveforms.
        :param shifts: Number of shift rounds.
        :return: The waveforms, on CUDA if they fit the accumulation budget.
        """
        if self.device != "cuda":
            return wavs
        n_sources = len(self.model.sources)
        max_shift = int(0.5 * self.model.samplerate)
        total_needed = 0
        for w in wavs:
            channels, length = w.shape[-2], w.shape[-1]
            needed = _gpu_accum_bytes_needed(1, n_sources, channels, length + max_shift)
            if shifts:
                needed += channels * (length + 2 * max_shift) * 4
                needed += n_sources * channels * length * 4
            total_needed += needed
        if total_needed <= _gpu_accum_budget_bytes(
            self.device, getattr(self.model, "_forward_reserve_bytes", None)
        ):
            return [w.to(self.device) for w in wavs]
        return wavs

    def _cpu_sources(self, sources_tensor: Tensor) -> dict[str, Tensor]:
        """
        Move model output to the CPU and split it into a per-stem dict.

        :param sources_tensor: ``[sources, channels, samples]`` output.
        :return: Stem to waveform mapping.
        """
        if sources_tensor.device.type == "cuda" and sources_tensor.numel():
            pinned = torch.empty(
                tuple(sources_tensor.shape),
                dtype=sources_tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            pinned.copy_(sources_tensor.detach(), non_blocking=True)
            torch.cuda.synchronize()
            sources_tensor = pinned
        elif sources_tensor.device.type != "cpu":
            sources_tensor = sources_tensor.cpu()
        return {
            name: sources_tensor[idx].clone()
            for idx, name in enumerate(self.model.sources)
        }

    def _unnormalized_cpu_sources(
        self, sources_tensor: Tensor, mean: Tensor, std: Tensor
    ) -> dict[str, Tensor]:
        """
        Undo normalization and return a per-stem dict on the CPU.

        :param sources_tensor: Model output.
        :param mean: Normalization mean.
        :param std: Normalization std.
        :return: Stem to waveform mapping.
        """
        return self._cpu_sources(sources_tensor * (NORMALIZATION_EPSILON + std) + mean)

    @staticmethod
    def _normalize(wav: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Normalize a waveform by the mean and std of its channel average.

        :param wav: ``[channels, samples]`` waveform.
        :return: ``(normalized, mean, std)``.
        """
        mean, std = normalization_stats(wav[None])
        mean, std = mean[0], std[0]
        return (wav - mean) / (NORMALIZATION_EPSILON + std), mean, std

    def _shift_offsets(
        self, seed: int | None, shifts: int, count: int
    ) -> list[list[int]] | None:
        """
        Draw the shift offsets from a private RNG, so a seed makes output
        reproducible without touching the process-global RNGs.

        Every input gets the same offsets, so a seeded input separates
        identically alone or batched with others.

        :param seed: Random seed, or ``None`` to let ``apply`` draw offsets.
        :param shifts: Number of shift rounds.
        :param count: Number of inputs.
        :return: ``offsets[shift][input]``, or ``None`` without a seed.
        """
        if seed is None or not shifts:
            return None
        rng = random.Random(seed)
        max_shift = int(0.5 * self.model.samplerate)
        return [[rng.randint(0, max_shift)] * count for _ in range(shifts)]

    def _separate_one(
        self,
        audio: tuple[Tensor, int] | Path | str | bytes,
        *,
        shifts: int,
        split_overlap: float,
        seed: int | None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
        use_only_stem: str | None,
        chunk_batch_size: int,
        oom_backoff_state: dict[str, int] | None = None,
    ) -> SeparatedSources:
        """
        Separate one input.

        :param audio: Single audio input.
        :param shifts: Number of random shifts.
        :param split_overlap: Overlap between segments.
        :param seed: RNG seed.
        :param progress_callback: Progress callback.
        :param use_only_stem: Ensemble stem whose member alone runs, or None.
        :param chunk_batch_size: Chunks per forward.
        :param oom_backoff_state: Shared OOM backoff state, or None.
        :return: The separated sources.
        """
        wav = self._to_tensor(audio)
        original = wav.cpu().clone()

        wav = self._stage_for_inference([wav], shifts)[0]
        external_norm = getattr(self.model, "external_normalization", True)
        if external_norm:
            wav, mean, std = self._normalize(wav)

        sources_tensor = apply_model_multi(
            self.model,
            [wav[None]],
            device=self.device,
            shifts=shifts,
            overlap=split_overlap,
            progress_callback=progress_callback,
            use_only_stem=use_only_stem,
            chunk_batch_size=chunk_batch_size,
            oom_backoff_state=oom_backoff_state,
            _shift_offsets=self._shift_offsets(seed, shifts, 1),
        )[0][0]

        if external_norm:
            sources = self._unnormalized_cpu_sources(sources_tensor, mean, std)
        else:
            sources = self._cpu_sources(sources_tensor)
        return SeparatedSources(
            sources,
            self.sample_rate,
            original=original,
            reliable=self._reliable_stem(use_only_stem),
        )

    def _reliable_stem(self, use_only_stem: str | None) -> tuple[str, ...] | None:
        """
        The stems a run really separated, when only one member runs.

        :param use_only_stem: The run's ``use_only_stem``, if any.
        :return: That member's stems, or None when every output is a real stem.
        """
        # use_only_stem narrows further when the loaded model is still an
        # ensemble (only_load may have kept a nested inner one): then only its
        # member for that stem runs, whatever only_load chose.
        if use_only_stem is not None and isinstance(self.model, ModelEnsemble):
            index = self.model.sources.index(use_only_stem)
            member = sole_contributor(self.model.weights, index)
            # An inner ensemble that runs whole (use_only_stem isn't passed
            # down) is a complete model: its outputs are all real stems.
            if member is not None and not isinstance(
                self.model.models[member], ModelEnsemble
            ):
                return _owned_stems(
                    self.model.weights, list(self.model.sources), member
                )
        return getattr(self, "_specialist_stems", None)

    def _separate_batch(
        self,
        audios: list,
        *,
        shifts: int,
        split_overlap: float,
        seed: int | None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
        use_only_stem: str | None,
        chunk_batch_size: int,
        oom_backoff_state: dict[str, int] | None = None,
    ) -> "list[SeparatedSources]":
        """
        Separate several inputs, pooling their chunks into shared batches.

        :param audios: List of audio inputs.
        :param shifts: Number of random shifts.
        :param split_overlap: Overlap between segments.
        :param seed: RNG seed.
        :param progress_callback: Progress callback.
        :param use_only_stem: Ensemble stem whose member alone runs, or None.
        :param chunk_batch_size: Chunks per forward.
        :param oom_backoff_state: Shared OOM backoff state, or None.
        :return: One SeparatedSources per input, in order.
        """
        wavs = [self._to_tensor(a) for a in audios]
        originals = [w.cpu().clone() for w in wavs]

        wavs = self._stage_for_inference(wavs, shifts)

        external_norm = getattr(self.model, "external_normalization", True)
        staged: list[Tensor] = []
        stats: list[tuple[Tensor, Tensor] | None] = []
        for w in wavs:
            if external_norm:
                normed_w, mean, std = self._normalize(w)
                staged.append(normed_w[None])
                stats.append((mean, std))
            else:
                staged.append(w[None])
                stats.append(None)

        outputs = apply_model_multi(
            self.model,
            staged,
            device=self.device,
            shifts=shifts,
            overlap=split_overlap,
            progress_callback=progress_callback,
            use_only_stem=use_only_stem,
            chunk_batch_size=chunk_batch_size,
            oom_backoff_state=oom_backoff_state,
            _shift_offsets=self._shift_offsets(seed, shifts, len(staged)),
        )

        results: list[SeparatedSources] = []
        for out, stat, original in zip(outputs, stats, originals):
            if stat is not None:
                sources = self._unnormalized_cpu_sources(out[0], stat[0], stat[1])
            else:
                sources = self._cpu_sources(out[0])
            results.append(
                SeparatedSources(
                    sources,
                    self.sample_rate,
                    original=original,
                    reliable=self._reliable_stem(use_only_stem),
                )
            )
        return results


def select_model(
    isolate_stem: str | None = None,
) -> tuple[str, str | None]:
    """
    Pick the best HTDemucs model for a stem.

    Routes only among the ``htdemucs`` family; other architectures are never
    chosen here.

    :param isolate_stem: Stem to isolate, or None for a full separation.
    :return: ``(model_name, only_load_stem)``.
    """
    if isolate_stem:
        isolate_stem = isolate_stem.lower()
        # htdemucs_6s has the best drums of the family (9.57 dB SDR on
        # MUSDB18-HQ, against 9.33 for htdemucs_ft's specialist).
        if isolate_stem in ["guitar", "piano", "drums"]:
            return ("htdemucs_6s", None)
        if isolate_stem in ["bass", "other", "vocals"]:
            return ("htdemucs_ft", isolate_stem)

    return ("htdemucs", None)


def get_version() -> str:
    """
    Get the installed unblend version.

    :return: Version string.
    """
    return __version__
