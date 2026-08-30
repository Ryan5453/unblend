# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import os
import random
import wave
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from . import __version__
from .apply import (
    Model,
    ModelEnsemble,
    _gpu_accum_budget_bytes,
    _gpu_accum_bytes_needed,
    _looks_like_cuda_oom,
    _require_cuda_available,
    apply_model,
    apply_model_multi,
)
from .audio import convert_audio, prevent_clip
from .backends import disable_custom_kernels
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
    ) -> None:
        """
        Initialize a SeparatedSources object.

        :param sources: Mapping of stem names to audio tensors
        :param sample_rate: Sample rate of the audio - comes from the model's sample rate
        :param original: Original unseparated audio
        """
        self.sources = sources
        self.sample_rate = sample_rate
        self.original = original

    def isolate_stem(self, name: str) -> "SeparatedSources":
        """
        Isolate a stem from the separated sources.
        This creates a new SeparatedSources object with the isolated stem and the accompanying complement stem (no_{STEM})

        :param name: Name of the stem to isolate
        :return: New SeparatedSources object with the isolated stem and the accompanying complement stem
        :raises ValidationError: If the requested stem isn't found in the sources
        """
        if name not in self.sources:
            raise ValidationError(
                f"Stem '{name}' not found in sources. Available stems: {list(self.sources.keys())}"
            )

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
            :param path: Path to save to; if None, returns bytes.
            :param format: Container format.
            :param clip: Clipping mode.
            :return: Path or raw bytes.
            :raises ValidationError: If stem not found.
        """
        if stem_name not in self.sources:
            raise ValidationError(
                f"Stem '{stem_name}' not found. Available stems: {list(self.sources.keys())}"
            )

        tensor = self.sources[stem_name]

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        tensor = prevent_clip(tensor, mode=clip)

        if path is not None:
            path = Path(path)

            if not path.suffix:
                file_path = path.with_suffix(f".{format}")
            else:
                file_path = path

            file_path.parent.mkdir(exist_ok=True, parents=True)

            encoder = AudioEncoder(samples=tensor, sample_rate=self.sample_rate)
            encoder.to_file(file_path)

            return file_path
        else:
            encoder = AudioEncoder(samples=tensor, sample_rate=self.sample_rate)
            encoded_tensor = encoder.to_tensor(format=format)
            return encoded_tensor.numpy().tobytes()


def _is_url(audio: "str | Path") -> bool:
    """
    Decide whether an audio location is a URL for FFmpeg.

        :param audio: Audio location as given by the caller.
        :return: True when the input should be decoded as a URL.
    """
    return isinstance(audio, str) and "://" in audio and not os.path.exists(audio)


_CUSTOM_KERNELS_ENV = "UNBLEND_CUSTOM_KERNELS"
_OFF_STRINGS = frozenset({"0", "off", "false", "no"})


def custom_kernels_enabled(setting: bool | None) -> bool:
    """
    Resolve the fused-kernel switch from explicit setting plus env.

        :param setting: Caller-supplied setting, or ``None`` for auto.
        :return: Whether fused kernels may be used.
    """
    if setting is not None:
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
    Pick the fastest dtype that keeps quality at FP32.

        :param device: ``"cuda"``, ``"mps"``, or ``"cpu"``.
        :return: Dtype to cast weights to, or ``None`` for FP32.
        :raises ValidationError: If device is invalid or CUDA unavailable.
    """
    if device == "cuda":
        _require_cuda_available()
        major, _minor = torch.cuda.get_device_capability()
        return torch.float16 if major >= 7 else None
    if device == "mps":
        return torch.float16
    if device == "cpu":
        return None
    raise ValidationError(f"Invalid device '{device}'. Must be one of: cpu, cuda, mps")


def _validate_chunk_batch_size(value: object) -> None:
    """
    Validate a chunk_batch_size value (init param or per-call override).

    The upper bound is a sanity guard against typos like ``10000`` that no
    real workload needs on these models.

    :param value: Candidate chunk_batch_size.
    :raises ValidationError: If not a positive int <= 1024 (bools rejected).
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(
            f"chunk_batch_size must be a positive integer, got {value}"
        )
    if value > 1024:
        raise ValidationError(f"chunk_batch_size must be <= 1024, got {value}")


def _contains_htdemucs(model: "Model | ModelEnsemble") -> bool:
    """
    Whether ``model`` is (or contains) an HTDemucs, used to gate the
    HTDemucs-specific MPS module-replacement pass.

    :param model: A loaded model or ensemble.
    :return: True for HTDemucs / ensembles with an HTDemucs member.
    """
    if isinstance(model, HTDemucs):
        return True
    return isinstance(model, ModelEnsemble) and any(
        isinstance(m, HTDemucs) for m in model.models
    )


class Separator:
    """
    Audio source separation using Demucs models.
    """

    _CUDAGRAPH_RESERVATION_FACTOR: float = 5.0
    _EAGER_RESERVATION_FACTOR: float = 2.5
    _CUDA_VRAM_SAFETY_BYTES: int = 1 * 1024**3
    _CHUNK_BATCH_MAX_ATTEMPTS: int = 4
    _COMPILE_ROFORMER_CBS_CANDIDATES: tuple[int, ...] = (4, 8, 16, 32)

    def _measure_per_chunk_steady_bytes(self) -> int | None:
        """
        Measure per-chunk steady VRAM via eager batch-1 forward.

            :return: Peak per-chunk delta in bytes, or ``None`` if unavailable.
        """
        if self.device != "cuda":
            return None
        supported = (HTDemucs, _RoformerBase)
        if isinstance(self.model, ModelEnsemble):
            ref = next((m for m in self.model.models if isinstance(m, supported)), None)
        elif isinstance(self.model, supported):
            ref = self.model
        else:
            ref = None
        if ref is None:
            return None
        try:
            training_length = int(ref.max_allowed_segment * ref.samplerate)
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
                self._eager_probe_seconds = perf_counter() - probe_started
            finally:
                torch.backends.cudnn.benchmark = cudnn_benchmark_saved
            peak = torch.cuda.max_memory_allocated()
            measured = max(1, peak - resident_before)
            self._per_chunk_steady_bytes = measured
            return measured
        except Exception:
            return None

    def _initial_chunk_batch_size_estimate(self) -> int:
        """
        Estimate initial batch size from free VRAM and per-chunk cost.

            :return: Initial chunks per batch.
        """
        if self.device == "cpu":
            return 1
        if self.device == "mps":
            try:
                budget = torch.mps.recommended_max_memory()
            except Exception:
                return 2
            if budget >= 20e9:
                return 8
            if budget >= 10e9:
                return 4
            return 2

        per_chunk_steady = getattr(self, "_per_chunk_steady_bytes", None)
        if per_chunk_steady is None:
            per_chunk_steady = self._measure_per_chunk_steady_bytes()
        if per_chunk_steady is None:
            return 4  # fallback for unsupported models / measurement failure

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
            for model in (
                self.model.models
                if isinstance(self.model, ModelEnsemble)
                else [self.model]
            )
        )
        if self._compile_enabled and wants_power_of_two:
            return max(1, 1 << (estimate.bit_length() - 1))
        return estimate

    def _setup_compile(self) -> None:
        """
        Apply the family-specific CUDA compile target to every model.
        """
        models = (
            list(self.model.models)
            if isinstance(self.model, ModelEnsemble)
            else [self.model]
        )
        for model in models:
            hook = getattr(model, "enable_compiled_core", None)
            if hook is not None:
                hook()

    def _teardown_compile_state(self) -> None:
        """
        Reverse :meth:`_setup_compile` and release compile state.
        """
        models = (
            list(self.model.models)
            if isinstance(self.model, ModelEnsemble)
            else [self.model]
        )
        for model in models:
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
        Verify batch size via capture; halve on OOM for compile.

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
        Time one full forward at current batch size.

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
        Sweep candidate batch sizes for compiled RoFormer and keep fastest.

            :param ceiling: Largest batch that fits in VRAM.
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
        self._warmup_via_inference()
        self._calibration_attempts = tried
        return best_cbs

    def _warmup_via_inference(self) -> None:
        """
        Capture CUDAGraphs via dummy inference through :meth:`separate`.
        """
        supported = (HTDemucs, _RoformerBase)
        if isinstance(self.model, ModelEnsemble):
            ref = next((m for m in self.model.models if isinstance(m, supported)), None)
        elif isinstance(self.model, supported):
            ref = self.model
        else:
            ref = None
        if ref is None:
            return

        samplerate = ref.samplerate
        channels = ref.audio_channels
        segment_length = int(ref.max_allowed_segment * samplerate)
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
        Initialize Separator.

            :param model: Model name or instance.
            :param device: ``"cpu"``, ``"cuda"``, or ``"mps"``; auto-selects if None.
            :param only_load: Load only one stem for ensembles.
            :param dtype: Inference dtype; ``"auto"`` picks best per device.
            :param compile: Apply torch.compile on CUDA.
            :param chunk_batch_size: Chunks per forward; auto if None.
            :param custom_kernels: Enable fused kernels.
            :param combine: Ensemble combine mode override.
            :param combine_params: STFT geometry for spectral modes.
            :raises ValidationError: If args invalid.
            :raises ModelLoadingError: If model fails to load.
        """
        if device is None:
            device = default_device()

        valid_devices = {"cpu", "cuda", "mps"}
        if device not in valid_devices:
            raise ValidationError(
                f"Invalid device '{device}'. Must be one of: {', '.join(sorted(valid_devices))}"
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
            _validate_chunk_batch_size(chunk_batch_size)

        self.device = device
        self.dtype = dtype
        use_custom_kernels = custom_kernels_enabled(custom_kernels)

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
                    warmup_async()
            self.model = model_repo.get_model(name=model, only_load=only_load)
        else:
            self.model = model

        if self.model is None:
            raise ModelLoadingError("Failed to load model")

        if combine is not None or combine_params is not None:
            if not isinstance(self.model, ModelEnsemble):
                raise ValidationError(
                    "combine only applies to an ensemble; "
                    f"{model if isinstance(model, str) else type(self.model).__name__} "
                    "has a single member."
                )
            self.model.set_combine(
                combine if combine is not None else self.model.combine,
                combine_params,
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
        try:
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = compile
                torch.set_float32_matmul_precision("high")

            if self.device in {"cuda", "mps"}:
                self.model.to(self.device)

            if self.dtype is not None:
                if isinstance(self.model, ModelEnsemble):
                    for m in self.model.models:
                        m.to(dtype=self.dtype)
                else:
                    self.model.to(dtype=self.dtype)

            if not use_custom_kernels:
                disable_custom_kernels(self.model)

            if (
                use_custom_kernels
                and self.dtype in (torch.float16, torch.bfloat16)
                and self.device == "mps"
            ):
                from .metal import apply_metal_optimizations, has_swappable_modules

                members = (
                    list(self.model.models)
                    if isinstance(self.model, ModelEnsemble)
                    else [self.model]
                )
                for member in members:
                    if has_swappable_modules(member):
                        apply_metal_optimizations(member)

            if (
                use_custom_kernels
                and self.dtype in (torch.float16, torch.bfloat16)
                and self.device == "cuda"
                and not compile
            ):
                from .cuda import apply_cuda_optimizations, has_swappable_modules

                members = (
                    list(self.model.models)
                    if isinstance(self.model, ModelEnsemble)
                    else [self.model]
                )
                for member in members:
                    if has_swappable_modules(member):
                        apply_cuda_optimizations(member)

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
        finally:
            if prev_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = prev_cudnn_benchmark
            if prev_matmul_precision is not None:
                torch.set_float32_matmul_precision(prev_matmul_precision)

    def enable_compile(self) -> None:
        """
        Compile an eager CUDA separator in place.

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
        is_supported = isinstance(self.model, supported) or (
            isinstance(self.model, ModelEnsemble)
            and any(isinstance(model, supported) for model in self.model.models)
        )
        if not is_supported:
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
        except Exception:
            self._teardown_compile_state()
            self._compile_enabled = False
            self.chunk_batch_size = previous_batch_size
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

    def warmup(self) -> None:
        """
        Pay compile and capture cost up front.

            :raises ValidationError: If not CUDA or unsupported model.
        """
        if self.device != "cuda":
            raise ValidationError("warmup() is only supported for CUDA separators.")
        supported = (HTDemucs, _RoformerBase)
        is_supported = isinstance(self.model, supported) or (
            isinstance(self.model, ModelEnsemble)
            and any(isinstance(model, supported) for model in self.model.models)
        )
        if not is_supported:
            raise ValidationError(
                "warmup() is only supported for HTDemucs and RoFormer models."
            )
        self._warmup_via_inference()

    @staticmethod
    def _read_pcm16_wav(path: Path | str) -> tuple[Tensor, int] | None:
        """
        Fast 16-bit PCM WAV path via header parse.

            :param path: Candidate file path.
            :return: ``(waveform, sr)`` if PCM WAV, else ``None``.
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
        wav = torch.from_numpy(samples.astype(np.float32).T / 32768.0)
        return wav, sample_rate

    def _to_tensor(self, audio: tuple[Tensor, int] | Path | str | bytes) -> Tensor:
        """
        Convert input to 2-D float32 tensor at model sr/channels.

            :param audio: ``(Tensor, sr)`` tuple, path, or bytes.
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
                    raise LoadAudioError(f"File not found: {audio}")
                except (OSError, ValueError):
                    pass
            pcm = None if is_url else self._read_pcm16_wav(audio)
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
                    )
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
                    )
        elif isinstance(audio, bytes):
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
                )
            finally:
                audio_buffer.close()
        else:
            raise ValidationError(
                f"Unsupported audio input type: {type(audio)}. "
                "Expected tuple of (Tensor, sample_rate), file path (str/Path), or bytes."
            )

        if wav.dim() == 1:
            wav = wav[None]
        if wav.dtype != torch.float32:
            wav = wav.float()

        if input_sr is not None and input_sr != self.sample_rate:
            wav = convert_audio(wav, input_sr, self.sample_rate, self.audio_channels)
        elif wav.shape[0] != self.audio_channels:
            wav = convert_audio(
                wav, self.sample_rate, self.sample_rate, self.audio_channels
            )

        if wav.shape[-1] == 0:
            raise ValidationError("Audio input is empty (zero samples).")

        return wav

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

            :param audio: Single input or list of inputs.
            :param shifts: Random shifts (1-20).
            :param split_overlap: Overlap between segments.
            :param seed: RNG seed, or None.
            :param progress_callback: Progress callback.
            :param use_only_stem: Run only one specialist member.
            :param chunk_batch_size: Chunks per forward; defaults to auto.
            :return: ``SeparatedSources`` or list thereof.
            :raises ValidationError: If param invalid.
        """
        if (
            isinstance(shifts, bool)
            or not isinstance(shifts, int)
            or not 1 <= shifts <= 20
        ):
            raise ValidationError(
                f"shifts must be an integer between 1 and 20 (inclusive), got {shifts}"
            )

        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValidationError(
                f"seed must be an integer if provided, got {type(seed)}"
            )

        if (
            isinstance(split_overlap, bool)
            or not isinstance(split_overlap, (int, float))
            or split_overlap < 0.0
            or split_overlap >= 1.0
        ):
            raise ValidationError(
                f"split_overlap must be a float between 0.0 (inclusive) and 1.0 (exclusive), got {split_overlap}"
            )

        per_call_chunk_batch_size = chunk_batch_size is not None
        if chunk_batch_size is None:
            chunk_batch_size = self.chunk_batch_size
        else:
            _validate_chunk_batch_size(chunk_batch_size)
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
        Run dispatch with auto-size OOM backoff.

            :param call: Closure ``(cbs, state)``.
            :param chunk_batch_size: Batch size for first attempt.
            :param allow: Whether backoff applies.
            :return: Dispatch result.
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
        Move waveforms to GPU when GPU-resident pipeline fits.

            :param wavs: Decoded waveforms.
            :param shifts: Number of shift rounds.
            :return: Waveforms possibly moved to CUDA.
        """
        if self.device != "cuda":
            return wavs
        n_sources = len(self.model.sources)
        max_shift = int(0.5 * self.model.samplerate) if shifts else 0
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
        Move model output to CPU and split into per-stem dict.

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
        Undo normalization and return per-stem dict on CPU.

            :param sources_tensor: Model output.
            :param mean: Normalization mean.
            :param std: Normalization std.
            :return: Stem to waveform mapping.
        """
        return self._cpu_sources(sources_tensor * (1e-5 + std) + mean)

    @staticmethod
    def _normalize(wav: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Channel-mean/std normalize waveform.

            :param wav: ``[channels, samples]`` waveform.
            :return: ``(normalized, mean, std)``.
        """
        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std(correction=1 if ref.numel() > 1 else 0)
        return (wav - mean) / (1e-5 + std), mean, std

    @staticmethod
    def _seed_rngs(seed: int | None) -> None:
        """
        Seed the RNGs that drive shift offsets, mirroring both paths.

        :param seed: Random seed to apply, or ``None`` to leave RNGs untouched.
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

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
        Single-input separation.

            :param audio: Single audio input.
            :param shifts: Number of random shifts.
            :param split_overlap: Overlap between segments.
            :param seed: RNG seed.
            :param progress_callback: Progress callback.
            :param use_only_stem: Stem to specialize.
            :param chunk_batch_size: Chunks per forward.
            :param oom_backoff_state: Backoff state dict.
            :return: SeparatedSources.
        """
        wav = self._to_tensor(audio)
        original = wav.clone()

        wav = self._stage_for_inference([wav], shifts)[0]
        self._seed_rngs(seed)
        external_norm = getattr(self.model, "external_normalization", True)
        if external_norm:
            wav, mean, std = self._normalize(wav)

        sources_tensor = apply_model(
            self.model,
            wav[None],
            device=self.device,
            shifts=shifts,
            overlap=split_overlap,
            progress_callback=progress_callback,
            use_only_stem=use_only_stem,
            chunk_batch_size=chunk_batch_size,
            oom_backoff_state=oom_backoff_state,
        )[0]

        if external_norm:
            sources = self._unnormalized_cpu_sources(sources_tensor, mean, std)
        else:
            sources = self._cpu_sources(sources_tensor)
        return SeparatedSources(sources, self.sample_rate, original=original)

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
        Batched separation pooling tail chunks.

            :param audios: List of audio inputs.
            :param shifts: Number of random shifts.
            :param split_overlap: Overlap between segments.
            :param seed: RNG seed.
            :param progress_callback: Progress callback.
            :param use_only_stem: Stem to specialize.
            :param chunk_batch_size: Chunks per forward.
            :param oom_backoff_state: Backoff state dict.
            :return: List of SeparatedSources.
        """
        wavs = [self._to_tensor(a) for a in audios]
        originals = [w.clone() for w in wavs]

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

        self._seed_rngs(seed)

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
        )

        results: list[SeparatedSources] = []
        for out, stat, original in zip(outputs, stats, originals):
            if stat is not None:
                sources = self._unnormalized_cpu_sources(out[0], stat[0], stat[1])
            else:
                sources = self._cpu_sources(out[0])
            results.append(
                SeparatedSources(sources, self.sample_rate, original=original)
            )
        return results


def select_model(
    isolate_stem: str | None = None,
) -> tuple[str, str | None]:
    """
    Select optimal Demucs model for a stem.

        :param isolate_stem: Stem to isolate, or None.
        :return: ``(model_name, only_load_stem)``.
    """
    if isolate_stem:
        if isolate_stem in ["guitar", "piano"]:
            return ("htdemucs_6s", None)
        if isolate_stem == "drums":
            return ("htdemucs", None)
        if isolate_stem in ["bass", "other", "vocals"]:
            return ("htdemucs_ft", isolate_stem)

    return ("htdemucs", None)


def get_version() -> str:
    """
    Get the version of unblend you have installed.

    :return: Version string
    """
    return __version__
