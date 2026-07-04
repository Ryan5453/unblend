# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import random
import wave
from io import BytesIO
from pathlib import Path
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
    apply_model,
    apply_model_multi,
)
from .audio import convert_audio, prevent_clip
from .exceptions import (
    LoadAudioError,
    ModelLoadingError,
    ValidationError,
)
from .htdemucs import HTDemucs
from .repo import ModelRepository


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
        Export a stem to either a file path or return as bytes.

        :param stem_name: Name of the stem to export
        :param path: Path to save the stem to. If None, returns raw audio bytes
        :param format: Format to export the stem to, anything supported by FFmpeg.
            Only used when returning bytes or when ``path`` has no extension;
            a ``path`` with an extension determines the container itself
        :param clip: Clipping mode to prevent audio distortion ("rescale", "clamp", "tanh", or None)
        :return: Path to saved file if path provided, otherwise raw audio bytes
        :raises ValidationError: If the stem name is not found
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
            # ``bytes(storage)`` iterates the storage one byte at a time in
            # Python (~90 s for a 35 MB stem) AND includes uninitialised
            # allocator padding past the real encoded length. ``.numpy()``
            # exposes the tensor as a numpy view; ``.tobytes()`` is a C-level
            # memcpy of exactly ``encoded_tensor.numel()`` bytes — ~5800x
            # faster and produces the correct length.
            return encoded_tensor.numpy().tobytes()


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
    Pick the fastest inference dtype for a device that keeps separation
    quality at FP32 level.

    CUDA on Volta-or-newer (compute capability >= 7.0) picks FP16: tensor-core
    convolutions/GEMMs run ~1.7x faster than FP32 and measure SDR-identical to
    FP32 on MUSDB18 (within 0.001 dB — see the readme benchmarks). All
    reduction-heavy ops (GroupNorm stats, attention softmax, conv accumulation)
    accumulate in FP32 internally on CUDA, and HTDemucs normalises its inputs,
    so FP16's range is not a hazard here. Older CUDA GPUs without FP16 tensor
    cores stay at FP32. MPS picks FP16 (custom Metal kernels in
    ``demucs.metal``); CPU stays FP32 (no faster path).

    :param device: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    :return: Dtype to cast model weights to, or ``None`` to stay at FP32.
    """
    if device == "cuda":
        major, _minor = torch.cuda.get_device_capability()
        return torch.float16 if major >= 7 else None
    if device == "mps":
        return torch.float16
    return None


class Separator:
    """
    Audio source separation using Demucs models.
    """

    # Behavioural constants for the measurement-based chunk_batch_size
    # calibration. Not per-GPU thresholds — these encode general
    # hardware/PyTorch behaviour observed across the GPUs we've tested.
    #
    # Multiplier on the measured batch=1 working set to estimate total
    # per-chunk VRAM demand when serving. It has to cover two things the
    # raw measurement doesn't: the CUDAGraphs private pool reserved at
    # capture (~2-3× the steady working set), AND the uncompiled STFT/iSTFT
    # scratch the batched ``apply_model_multi`` path allocates outside that
    # pool. 5× lands the initial estimate at a cbs that passes the batched
    # warmup on the first try on the GPUs we've tested (no wasted
    # compile+halve cycle — which costs ~10s of init each).
    #
    # Erring high here is nearly free: the throughput-vs-cbs curve is flat
    # under compile=True, so a conservative (smaller) cbs costs no steady-
    # state speed. If it still over-shoots, the capture-verification loop in
    # ``_calibrate_chunk_batch_size`` halves until the batched warmup fits.
    _CUDAGRAPH_RESERVATION_FACTOR: float = 5.0
    # Eager (compile=False) has no CUDAGraphs private pool, so the per-chunk
    # demand is just the steady working set plus allocator fragmentation /
    # STFT scratch — 2.5x covers that with margin. Unlike compile=True, the
    # eager throughput-vs-cbs curve is NOT flat: ~8% more throughput going
    # from cbs 9 to 16 on an RTX A4000 at FP16, so under-sizing here costs
    # real speed. There is no capture-verify loop on this path (the estimate
    # is trusted), hence still conservative rather than exact.
    _EAGER_RESERVATION_FACTOR: float = 2.5
    # Headroom for non-cudagraph GPU allocations during inference: the active
    # chunk batch plus STFT scratch — both bounded by chunk_batch_size, not by
    # input length. 1 GiB covers this with margin to spare. (GPU-resident
    # mixes/accumulators are gated separately against free VRAM at separate()
    # time — see ``apply._gpu_accum_budget_bytes``.)
    _CUDA_VRAM_SAFETY_BYTES: int = 1 * 1024**3
    # Max halving attempts before giving up — bounds init time so that even
    # in pathological cases (wildly wrong initial estimate) we don't spin
    # for minutes. Each attempt is one full compile+capture, ~15-30s.
    _CHUNK_BATCH_MAX_ATTEMPTS: int = 4

    @staticmethod
    def _prefill_htdemucs_caches(model: HTDemucs) -> None:
        """
        Eagerly populate HTDemucs's positional/frequency-embedding caches via a
        single dummy ``forward_core`` pass.

        ``mode="reduce-overhead"`` wraps the graph in CUDAGraphs, which reuses
        internal allocation slots across replays. If the first call into the
        compiled ``forward_core`` is also what fills these caches, the cached
        tensors end up pointing into CUDAGraphs-managed memory that gets
        overwritten on the next replay (see ``_cached_freq_emb`` at
        ``htdemucs.py:467`` and the transformer's ``_cached_pos_emb_*``). By
        running one eager pass first we anchor each cached embedding in the
        regular allocator so CUDAGraphs treats it as a stable external input.

        :param model: HTDemucs model whose embedding caches to prefill.
        """
        training_length = int(model.max_allowed_segment * model.samplerate)
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device

        with torch.no_grad():
            mix = torch.zeros(
                1,
                model.audio_channels,
                training_length,
                device=model_device,
                dtype=torch.float32,
            )
            z = model._spec(mix)
            x = model._magnitude(z).to(mix.device)
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (1e-5 + std)

            xt = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)

            if model_dtype != torch.float32:
                x = x.to(model_dtype)
                xt = xt.to(model_dtype)

            model.forward_core(x, xt)

    @staticmethod
    def _compile_htdemucs_forward_core(model: HTDemucs) -> None:
        """
        Compile only the heavy neural network core of HTDemucs.

        Avoids pulling STFT/iSTFT into TorchInductor — those are a poor fit
        for Inductor and significantly inflate compile time without helping
        steady-state throughput.

        :param model: HTDemucs model to compile
        """
        # Snapshot the original forward_core once so the calibration retry
        # loop can re-call this safely without nesting torch.compile wrappers
        # (compiling an already-compiled function breaks dynamo).
        if not hasattr(model, "_uncompiled_forward_core"):
            model._uncompiled_forward_core = model.forward_core
        # Caches must be populated BEFORE compile — see _prefill_htdemucs_caches.
        Separator._prefill_htdemucs_caches(model)
        model.forward_core = torch.compile(
            model._uncompiled_forward_core, mode="reduce-overhead"
        )
        # CUDAGraphs replay requires the captured batch shape, so apply_model
        # must zero-pad sub-full tail batches up to chunk_batch_size for this
        # model (eager models run tails at their natural size instead).
        model._fixed_batch_shape = True

    def _measure_per_chunk_steady_bytes(self) -> int | None:
        """
        Run one eager forward at batch=1 and return the peak VRAM delta
        in bytes. This is the per-chunk steady-state working set, NOT the
        CUDAGraphs private-pool reservation — the cudagraph factor is
        applied separately.

        Returns ``None`` for non-HTDemucs models / non-CUDA devices, in
        which case the caller should pick a conservative default.

        :return: Peak per-chunk VRAM delta in bytes, or ``None`` for
            non-HTDemucs models / non-CUDA devices.
        """
        if self.device != "cuda":
            return None
        if isinstance(self.model, ModelEnsemble):
            ref = next((m for m in self.model.models if isinstance(m, HTDemucs)), None)
        elif isinstance(self.model, HTDemucs):
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
            # Measure under heuristic algorithm selection even when the
            # compile path has turned cudnn.benchmark on: the exhaustive
            # autotune sweep allocates large transient workspaces that land
            # in the peak reading and would halve the batch-size estimate.
            # We want the steady per-chunk working set, not autotune scratch.
            cudnn_benchmark_saved = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = False
            try:
                with torch.inference_mode():
                    dummy = torch.zeros(
                        1,
                        ref.audio_channels,
                        training_length,
                        device=device_obj,
                        dtype=torch.float32,
                    )
                    # Full model forward — covers STFT + core + iSTFT, matching
                    # what a real chunk through ``apply_model`` exercises.
                    _ = ref(dummy)
                torch.cuda.synchronize()
            finally:
                torch.backends.cudnn.benchmark = cudnn_benchmark_saved
            peak = torch.cuda.max_memory_allocated()
            return max(1, peak - resident_before)
        except Exception:
            return None

    def _initial_chunk_batch_size_estimate(self) -> int:
        """
        Math-based initial estimate from free VRAM and per-chunk measurement.

        No per-GPU thresholds: ``free_bytes`` and the eager-measured
        per-chunk cost are the only inputs. Behavioural constants
        (``_CUDAGRAPH_RESERVATION_FACTOR``, ``_CUDA_VRAM_SAFETY_BYTES``)
        encode general PyTorch CUDA behaviour, not specific GPU IDs.

        Intentionally conservative — the capture-verification loop in
        ``_calibrate_chunk_batch_size`` halves on OOM, so an over-estimate
        only costs a few extra setup attempts; an under-estimate is silent.

        :return: Initial number of chunks to process per batch.
        """
        if self.device == "cpu":
            return 1
        if self.device == "mps":
            return 2

        per_chunk_steady = self._measure_per_chunk_steady_bytes()
        if per_chunk_steady is None:
            return 4  # fallback for non-HTDemucs / measurement failure

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
        return max(1, int(available // transient_per_chunk))

    def _setup_compile(self) -> None:
        """
        Apply (or re-apply) torch.compile to every HTDemucs in the model.
        """
        if isinstance(self.model, ModelEnsemble):
            for sub in self.model.models:
                if isinstance(sub, HTDemucs):
                    self._compile_htdemucs_forward_core(sub)
        elif isinstance(self.model, HTDemucs):
            self._compile_htdemucs_forward_core(self.model)

    def _teardown_compile_state(self) -> None:
        """
        Reverse ``_setup_compile`` and release CUDAGraphs / Inductor state
        so the next attempt starts clean.

        Restores ``forward_core`` to the saved uncompiled version (so the
        next ``_setup_compile`` doesn't double-wrap), resets Dynamo, and
        empties the CUDA caching allocator. ``torch._dynamo.reset()``
        drops Inductor's compile-cache references plus the cudagraph tree
        manager state, which is what frees the cudagraph private pool
        once the wrapper is released.
        """
        if isinstance(self.model, ModelEnsemble):
            models = [m for m in self.model.models if isinstance(m, HTDemucs)]
        elif isinstance(self.model, HTDemucs):
            models = [self.model]
        else:
            models = []
        for m in models:
            original = getattr(m, "_uncompiled_forward_core", None)
            if original is not None:
                m.forward_core = original
            m._fixed_batch_shape = False
        torch._dynamo.reset()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _calibrate_chunk_batch_size(
        self, initial_guess: int, compile_enabled: bool
    ) -> int:
        """
        Pin down the actual chunk_batch_size by capture-verifying it.

        For ``compile=False`` we trust the math estimate directly and do NOT
        verify it or keep a runtime OOM-retry. The estimate assumes Demucs
        owns the whole GPU: ``mem_get_info`` at init reflects all the VRAM we
        will ever have, the output accumulator lives on CPU with chunks staged
        per-batch (so runtime GPU usage is bounded by ``model + active_batch``,
        not audio length), and the 5x ``_CUDAGRAPH_RESERVATION_FACTOR`` leaves
        a wide margin. If another process is sharing the GPU and we OOM, that's
        on the caller — pass an explicit ``chunk_batch_size`` to size for the
        VRAM you actually have.

        For ``compile=True`` we can't reason about VRAM up front: the cudagraph
        private pool is an opaque reservation only knowable by capturing it. So
        we actually compile + warm up at ``initial_guess`` and, on CUDA OOM,
        halve and retry up to ``_CHUNK_BATCH_MAX_ATTEMPTS`` times. Each retry
        tears down the prior attempt's compile state so the released cudagraph
        pool is reclaimable.

        :param initial_guess: Starting chunk_batch_size to verify (the math
            estimate from ``_initial_chunk_batch_size_estimate``).
        :param compile_enabled: Whether ``torch.compile`` is active; when
            ``False`` (or on non-CUDA) the guess is trusted without capture.
        :return: The verified chunk_batch_size to use.
        :raises ModelLoadingError: If calibration runs out of halving attempts
            (or batch size 1 still OOMs) without fitting.
        """
        if self.device != "cuda":
            return initial_guess
        if not compile_enabled:
            return initial_guess

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
                # CUDA OOM during capture doesn't always surface as
                # torch.cuda.OutOfMemoryError — graph capture and cuBLAS
                # workspace failures under memory pressure raise plain
                # RuntimeErrors. Treat those as OOM too; anything else is a
                # real bug and propagates.
                message = str(exc).lower()
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or any(
                    marker in message
                    for marker in (
                        "out of memory",
                        "cudaerrormemoryallocation",
                        "cublas_status_alloc_failed",
                    )
                )
                if not is_oom:
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

    def _warmup_via_inference(self) -> None:
        """
        Trigger CUDAGraphs capture AND realize the serving-path working set by
        running dummy inferences through the real ``self.separate()`` code
        paths — both single-input (``apply_model``) and batched-list
        (``apply_model_multi``).

        An earlier implementation called ``model.forward_core`` directly with
        hand-rolled normalised tensors. That captures a *different* CUDAGraphs
        tree from the one ``HTDemucs.forward`` actually invokes during real
        inference: the upstream normalisation in ``forward`` uses
        ``torch.var_mean`` while the warmup used separate ``.mean()``/``.std()``
        calls, and Inductor's tree-manager treats those as distinct dataflow
        graphs. Each tree reserves its own private memory pool (~5–8 GiB at
        ``chunk_batch_size=16``), so the warmup ended up *adding* VRAM
        pressure instead of removing it, and OOMed multi-separator setups
        like the Cog predictor.

        Running ``self.separate(...)`` on zero audio uses the exact wiring
        real requests go through, so the cudagraph captured is the one every
        subsequent request reuses (single and batched share it — same
        ``[chunk_batch_size, channels, segment]`` forward shape via
        tail-padding).

        We warm BOTH paths because ``apply_model_multi`` (the batched/list
        path used by ``separate([...])`` and the Cog coalescer) allocates a
        serving working set — the uncompiled STFT/iSTFT scratch outside the
        cudagraph pool — that ``apply_model`` does not. If we only warmed the
        single path, capture-verify would pass at a ``chunk_batch_size`` whose
        cudagraph pool nearly fills the GPU, and the first *batched* request
        would then OOM trying to allocate that scratch with no headroom left.
        By running a batched warmup here, that allocation happens during
        calibration, so an OOM correctly triggers the halving loop in
        ``_calibrate_chunk_batch_size`` and we settle on a cbs that leaves
        room to actually serve.
        """
        # Pick a reference HTDemucs to size the dummy (all sub-models in an
        # ensemble share architecture and samplerate).
        if isinstance(self.model, ModelEnsemble):
            ref = next((m for m in self.model.models if isinstance(m, HTDemucs)), None)
        elif isinstance(self.model, HTDemucs):
            ref = self.model
        else:
            ref = None
        if ref is None:
            return

        samplerate = ref.samplerate
        channels = ref.audio_channels
        segment_length = int(ref.max_allowed_segment * samplerate)
        dummy = torch.zeros(channels, segment_length, dtype=torch.float32)

        # Single-input path (apply_model): captures the shared cudagraph.
        # Pass tensor + samplerate so _to_tensor takes the dummy as-is
        # instead of going through audio decoding. shifts=1 / overlap=0.25
        # match the defaults real callers use.
        self.separate(
            audio=(dummy, samplerate),
            shifts=1,
            split_overlap=0.25,
            chunk_batch_size=self.chunk_batch_size,
        )

        # Batched-list path (apply_model_multi): realizes the heavier serving
        # working set so calibration verifies it fits. Two inputs exercise the
        # cross-input tail pooling; the per-forward shape is identical, so this
        # reuses the cudagraph captured above (no second pool).
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
    ) -> None:
        """
        Initialize a Separator with the specified model and device.

        :param model: Model to use for separation (name or model instance)
        :param device: Device to use for processing (must be "cpu", "cuda", or "mps").
                       If ``None`` (the default), auto-selects cuda > mps > cpu based
                       on availability at construction time.
        :param only_load: Construction-time optimisation for bag-of-models like
                         ``htdemucs_ft``: download and load *only* the sub-model
                         specialised for this stem instead of the full ensemble,
                         saving download time and memory. The loaded model still
                         outputs all of its sources (only the named stem is
                         high quality). This is the startup-time counterpart to
                         ``separate(..., use_only_stem=...)``, which makes the
                         same choice per-call on an already-loaded full ensemble.
                         Ignored for single (non-ensemble) models.
        :param dtype: Inference precision. The default ``"auto"`` picks the
                     fastest dtype that keeps SDR at FP32 level for the device
                     (FP16 on CUDA with tensor cores and on MPS; FP32 on CPU
                     and older CUDA GPUs — see ``default_dtype``). Pass
                     ``torch.float16`` or ``torch.bfloat16`` explicitly for
                     reduced precision (weights are cast at init; CPU is
                     rejected — no faster path in PyTorch), or ``None`` /
                     ``torch.float32`` to force FP32. On MPS, BF16 is supported
                     but, as of PyTorch 2.11, ~22% slower than FP16 (BF16
                     native ops are not well-optimised yet); use it when you
                     want BF16's FP32 exponent range — e.g. to skip the
                     FP16-overflow FP32-fallback cast in MyGroupNorm.
        :param compile: If True, apply ``torch.compile`` for faster inference after an
                       initial warmup. Best for API servers or batch processing; adds
                       significant latency to the first forward pass.
        :raises ValidationError: If device is not valid or only_load stem doesn't exist
        :raises ModelLoadingError: If model fails to load
        """
        # Resolve the device at call time (not import time): auto-select the
        # best available backend when the caller didn't specify one.
        if device is None:
            device = default_device()

        # Validate device
        valid_devices = {"cpu", "cuda", "mps"}
        if device not in valid_devices:
            raise ValidationError(
                f"Invalid device '{device}'. Must be one of: {', '.join(sorted(valid_devices))}"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValidationError(
                "Device 'cuda' requested but CUDA is not available in this "
                "PyTorch build/environment."
            )
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
                    f"Invalid dtype '{dtype}'. Only torch.float16 and torch.bfloat16 are supported."
                )
            if device == "cpu":
                raise ValidationError(
                    f"{dtype} inference is not supported on CPU. Use cuda or mps."
                )

        self.device = device
        self.dtype = dtype

        # Handle both string model names and model instances
        if isinstance(model, str):
            model_repo = ModelRepository()
            self.model = model_repo.get_model(name=model, only_load=only_load)
        else:
            # model is already a Model instance
            self.model = model

        if self.model is None:
            raise ModelLoadingError("Failed to load model")
        self.model.eval()

        # Validate only_load stem exists in the loaded model
        if only_load and only_load not in self.model.sources:
            raise ValidationError(
                f"Stem '{only_load}' not found in model. "
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
                # cudnn autotune (``benchmark=True``) only for the compile path:
                # there every forward has one fixed shape (tails are padded to the
                # captured batch), so the exhaustive algorithm search is paid once.
                # Eager runs see a different tail-batch shape on nearly every
                # input; each new shape would re-trigger a multi-second search,
                # while heuristic selection measures within ~1% of the tuned
                # algorithms for these conv shapes — so eager leaves it off.
                torch.backends.cudnn.benchmark = compile
                torch.set_float32_matmul_precision("high")

            if self.device in {"cuda", "mps"}:
                self.model.to(self.device)

            # Cast weights to the requested dtype.
            if self.dtype is not None:
                if isinstance(self.model, ModelEnsemble):
                    for m in self.model.models:
                        m.to(dtype=self.dtype)
                else:
                    self.model.to(dtype=self.dtype)

            # MPS-specific low-precision optimisations: PyTorch 2.11's MPS backend
            # has slow paths for FP16/BF16 GroupNorm and SDPA. We swap in custom
            # Metal kernels and a wrapped attention module that route around those.
            # The SCALAR_T-templated kernels compile for either ``half`` or
            # ``bfloat`` and dispatch by tensor dtype at call time. No-op for CUDA
            # (handled by tensor cores) and CPU (no low-precision path).
            if self.dtype in (torch.float16, torch.bfloat16) and self.device == "mps":
                from .metal import apply_metal_optimizations

                if isinstance(self.model, ModelEnsemble):
                    for m in self.model.models:
                        apply_metal_optimizations(m)
                else:
                    apply_metal_optimizations(self.model)

            # Compute an initial chunk_batch_size from a single eager forward
            # measurement. No per-GPU table — the math uses ``mem_get_info`` and
            # the measured per-chunk cost. For compile=True we then *verify* this
            # estimate by actually capturing a cudagraph at that batch size and
            # halving on OOM (see ``_calibrate_chunk_batch_size``); this is what
            # makes the path work on arbitrary GPUs without hardcoded thresholds.
            self._compile_enabled = compile and self.device == "cuda"
            # Records the chunk_batch_size values tried during calibration. Only the
            # CUDA+compile path actually iterates; initialise it here so the
            # attribute always exists (CPU/MPS/compile-disabled return early).
            self._calibration_attempts: list[int] = []
            initial_cbs = self._initial_chunk_batch_size_estimate()
            self.chunk_batch_size = self._calibrate_chunk_batch_size(
                initial_guess=initial_cbs,
                compile_enabled=self._compile_enabled,
            )
        except Exception:
            # Don't leave global cuDNN / matmul precision state flipped if
            # init failed.
            if prev_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = prev_cudnn_benchmark
            if prev_matmul_precision is not None:
                torch.set_float32_matmul_precision(prev_matmul_precision)
            raise

    def warmup(self) -> None:
        """
        Pay the compile + CUDAGraphs capture cost up front instead of on the
        first live request.

        Called automatically at the end of ``__init__`` when ``compile=True``;
        calling it again later re-runs the dummy inferences (a no-op for
        correctness, occasionally useful to re-realize the working set). With
        tail-padding (every batch is exactly ``self.chunk_batch_size``) there
        is only one batch shape to warm, so no ``batch_sizes`` argument is
        needed any more.

        :raises ValidationError: If called on CPU/MPS or on a non-HTDemucs model.
        """
        if self.device != "cuda":
            raise ValidationError("warmup() is only supported for CUDA separators.")
        is_htdemucs_like = isinstance(self.model, HTDemucs) or (
            isinstance(self.model, ModelEnsemble)
            and any(isinstance(m, HTDemucs) for m in self.model.models)
        )
        if not is_htdemucs_like:
            raise ValidationError(
                "warmup() is only supported for HTDemucs-based models."
            )
        self._warmup_via_inference()

    @staticmethod
    def _read_pcm16_wav(path: Path | str) -> tuple[Tensor, int] | None:
        """
        Fast path for plain 16-bit PCM WAV files: header parse + one memcpy +
        vectorised int16→float32, ~2x faster than going through the FFmpeg
        demux pipeline in torchcodec. Sample-exact with torchcodec's output
        (both normalise as ``int16 / 32768``).

        :param path: Path to the candidate file.
        :return: ``(waveform [channels, samples], sample_rate)`` if the file
            is readable 16-bit PCM WAV, else ``None`` to fall back to
            torchcodec (which handles every other format and codec).
        """
        try:
            with wave.open(str(path), "rb") as w:
                if w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                    return None
                num_frames = w.getnframes()
                channels = w.getnchannels()
                sample_rate = w.getframerate()
                raw = w.readframes(num_frames)
        except (wave.Error, EOFError, OSError):
            return None
        if channels <= 0 or sample_rate <= 0 or not raw:
            return None
        # A truncated data chunk (byte count not a whole number of frames)
        # can't be reshaped; fall back to torchcodec rather than crash.
        if len(raw) % (2 * channels) != 0:
            return None
        samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
        wav = torch.from_numpy(samples.astype(np.float32).T / 32768.0)
        return wav, sample_rate

    def _to_tensor(self, audio: tuple[Tensor, int] | Path | str | bytes) -> Tensor:
        """
        Convert various input types (tuple of Tensor and sample rate, path, bytes)
        to a 2D float32 tensor matching the model's sample rate and channels
        when possible. Device staging happens later, inside ``apply_model``.

        :param audio: Audio input as a ``(Tensor, sample_rate)`` tuple, a file
            path (str/Path), or raw audio bytes.
        :return: A 2D ``[channels, samples]`` float32 waveform tensor.
        :raises LoadAudioError: If a path or bytes input cannot be decoded.
        :raises ValidationError: If the input is not a supported type or the
            decoded audio is empty (zero samples).
        """
        wav: Tensor
        input_sr: int | None = None

        if isinstance(audio, tuple):
            wav, input_sr = audio
        elif isinstance(audio, (str, Path)):
            pcm = self._read_pcm16_wav(audio)
            if pcm is not None:
                wav, input_sr = pcm
            else:
                try:
                    # Use native torchcodec AudioDecoder for better performance
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
                # Use native torchcodec AudioDecoder for better performance
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

        # Minimal shape/dtype normalization
        if wav.dim() == 1:
            wav = wav[None]
        if wav.dtype != torch.float32:
            wav = wav.float()

        # Try to match expected sample rate/channels when we know input_sr, or channels mismatch
        if input_sr is not None and input_sr != self.sample_rate:
            wav = convert_audio(wav, input_sr, self.sample_rate, self.audio_channels)
        elif wav.shape[0] != self.audio_channels:
            # Adjust channels without resampling
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
        Separate audio into stems. Accepts a (tensor, sample_rate) pair, a file
        path, raw bytes, or a list of any of those for batched separation.

        Single input → returns one ``SeparatedSources``.
        List input → returns ``list[SeparatedSources]`` in the same order as
        the input. The batched path pools tail chunks across inputs so every
        forward pass is a full-size batch, which lifts throughput on multi-file
        workloads (CLI batch, Cog request coalescing) but means the per-input
        shift offsets advance naturally from a single seed rather than being
        reseeded per input — outputs are reproducible across runs but won't
        be bit-identical to calling ``separate()`` per file with the same seed.

        :param audio: Single audio input or a list of audio inputs.
        :param shifts: Number of random shifts for equivariant stabilization (1-20).
        :param split_overlap: Overlap between segments (0.0 to 1.0).
        :param seed: Optional random seed for reproducible shift-based inference.
            Note: this seeds the process-global ``random`` and ``torch`` RNGs as
            a side effect, affecting other code in the host process.
        :param progress_callback: Optional callback for progress updates. Only
            applies to single-input mode; the batched path doesn't surface
            per-input progress yet.
        :param use_only_stem: Performance optimisation for a ``ModelEnsemble`` of
            fine-tuned specialists (e.g. ``htdemucs_ft``): run only the sub-model
            specialised for this stem, skipping the others. The result still
            contains **all** of the model's sources (only the named stem is high
            quality) — it does *not* filter the output to one stem; use
            ``SeparatedSources.isolate_stem`` for that. Must name one of the
            model's sources (raises ``ValidationError`` otherwise); a valid stem
            on a model that can't specialise (single model, or no one-hot
            sub-model) just runs the full model normally.
        :param chunk_batch_size: Chunks processed in parallel. Defaults to
            ``self.chunk_batch_size`` (auto-detected at init from VRAM on CUDA,
            2 on MPS, 1 on CPU).
        :return: ``SeparatedSources`` for a single input, ``list[SeparatedSources]``
            for a list input.
        :raises ValidationError: If any parameter value is invalid.
        """
        # Validate shifts parameter
        if not isinstance(shifts, int) or shifts < 1 or shifts > 20:
            raise ValidationError(
                f"shifts must be an integer between 1 and 20 (inclusive), got {shifts}"
            )

        if seed is not None and not isinstance(seed, int):
            raise ValidationError(
                f"seed must be an integer if provided, got {type(seed)}"
            )

        # Validate split_overlap parameter
        if (
            not isinstance(split_overlap, (int, float))
            or split_overlap < 0.0
            or split_overlap >= 1.0
        ):
            raise ValidationError(
                f"split_overlap must be a float between 0.0 (inclusive) and 1.0 (exclusive), got {split_overlap}"
            )

        if chunk_batch_size is None:
            chunk_batch_size = self.chunk_batch_size

        # Validate chunk_batch_size parameter. The upper bound is a sanity
        # guard against typos like ``chunk_batch_size=10000`` that would OOM
        # before the calibration loop has a chance to intervene; real workloads
        # never need a four-figure batch on these models.
        if not isinstance(chunk_batch_size, int) or chunk_batch_size < 1:
            raise ValidationError(
                f"chunk_batch_size must be a positive integer, got {chunk_batch_size}"
            )
        if chunk_batch_size > 1024:
            raise ValidationError(
                f"chunk_batch_size must be <= 1024, got {chunk_batch_size}"
            )

        # An unknown use_only_stem is a caller mistake (e.g. a typo) — fail
        # loudly rather than silently running the full model. A *valid* stem on
        # a model that can't specialise (plain htdemucs, or no one-hot sub-model)
        # is fine and just runs normally; that's handled in apply_model.
        if use_only_stem is not None and use_only_stem not in self.model.sources:
            raise ValidationError(
                f"use_only_stem '{use_only_stem}' is not a source of this model. "
                f"Available stems: {', '.join(self.model.sources)}"
            )

        if isinstance(audio, list):
            if not audio:
                return []
            if progress_callback is not None:
                raise ValidationError(
                    "progress_callback is not supported for list-input batched separation."
                )

            return self._separate_batch(
                audio,
                shifts=shifts,
                split_overlap=split_overlap,
                seed=seed,
                use_only_stem=use_only_stem,
                chunk_batch_size=chunk_batch_size,
            )

        # Validate progress_callback parameter
        if progress_callback is not None and not callable(progress_callback):
            raise ValidationError(
                f"progress_callback must be callable if provided, got {type(progress_callback)}"
            )

        return self._separate_one(
            audio,
            shifts=shifts,
            split_overlap=split_overlap,
            seed=seed,
            progress_callback=progress_callback,
            use_only_stem=use_only_stem,
            chunk_batch_size=chunk_batch_size,
        )

    # (Runtime OOM-retry used to live here. Inputs and accumulators are kept
    # on the GPU only when they fit a conservative fraction of the VRAM that
    # is actually free (see ``apply._gpu_accum_budget_bytes``); anything
    # bigger falls back to the CPU-accumulation path whose GPU usage is
    # bounded by ``model + cudagraph_pool + active_batch`` regardless of
    # input length — so an OOM at separate() time would mean the at-init
    # capture verification was wrong, which we'd rather see as a loud
    # failure than silently halve and continue.)

    def _stage_for_inference(self, wavs: list[Tensor], shifts: int) -> list[Tensor]:
        """
        Move decoded waveforms to the GPU up front when the GPU-resident
        separation pipeline will be used for them.

        Staging before normalisation lets the normalise / un-normalise passes
        and every chunk slice run on the GPU, and ``apply_model`` then returns
        the separated sources still on the GPU (results come back on the
        mix's device) so the stems make exactly one device→host trip. Uses
        the same byte estimate as ``apply_model``'s own gate, summed over the
        whole call (``apply_model_multi`` gates its accumulators on the call's
        total, so staging must use the same all-or-nothing scope) — waveforms
        are only staged when the accumulators will also fit. CPU/MPS inputs
        are returned unchanged (``apply_model`` stages MPS itself).

        The shift path (``shifts >= 1``) keeps more resident on the mix's
        device than the unshifted accumulator alone: each round pads the mix to
        ``length + 2 * max_shift`` and the cross-round accumulator holds a full
        ``[sources, channels, length]`` output. The unshifted helper re-gates
        its *own* accumulators against live VRAM and falls back to CPU, but
        those two tensors are allocated unconditionally on the mix's device, so
        the staging estimate must cover them or staging could OOM where the
        helper's self-gating wouldn't have.

        :param wavs: Decoded ``[channels, samples]`` waveforms.
        :param shifts: Number of shift rounds the forward pass will run.
        :return: The waveforms, possibly moved to the CUDA device.
        """
        if self.device != "cuda":
            return wavs
        n_sources = len(self.model.sources)
        max_shift = int(0.5 * self.model.samplerate) if shifts else 0
        total_needed = 0
        for w in wavs:
            channels, length = w.shape[-2], w.shape[-1]
            # The helper accumulates over the shifted view (length + max_shift).
            needed = _gpu_accum_bytes_needed(
                1, n_sources, channels, length + max_shift
            )
            if shifts:
                # Per-round padded copy + the full-length cross-round accumulator.
                needed += channels * (length + 2 * max_shift) * 4
                needed += n_sources * channels * length * 4
            total_needed += needed
        if total_needed <= _gpu_accum_budget_bytes(self.device):
            return [w.to(self.device) for w in wavs]
        return wavs

    def _unnormalized_cpu_sources(
        self, sources_tensor: Tensor, mean: Tensor, std: Tensor
    ) -> dict[str, Tensor]:
        """
        Reverse the track-level normalisation and return the per-stem dict.

        Un-normalises on whatever device the sources came back on (GPU when
        the input was staged), then makes the single device→host move; the
        public ``SeparatedSources`` tensors are always CPU.

        :param sources_tensor: ``[sources, channels, samples]`` model output.
        :param mean: Normalisation mean recorded by ``_normalize``.
        :param std: Normalisation std recorded by ``_normalize``.
        :return: Mapping of stem name to CPU waveform.
        """
        unnormed = sources_tensor * (1e-5 + std) + mean
        if unnormed.device.type != "cpu":
            unnormed = unnormed.cpu()
        # Per-stem clone so user mutation (e.g. ``sources["vocals"][:] = 0``)
        # doesn't alias across stems via the shared underlying buffer.
        return {
            name: unnormed[idx].clone()
            for idx, name in enumerate(self.model.sources)
        }

    @staticmethod
    def _normalize(wav: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Channel-mean/std normalise a waveform the way Demucs expects.

        Shared by the single- and batched-input paths so the two can't drift
        (the ``1e-5`` floor and the reference-channel reduction must stay
        identical for ``separate(x)`` and ``separate([x])[0]`` to match).

        :param wav: ``[channels, samples]`` waveform.
        :return: ``(normalised_wav, mean, std)``; reverse with
            ``out * (1e-5 + std) + mean``.
        """
        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std()
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
    ) -> SeparatedSources:
        """
        Single-input separation. See ``separate`` for parameter semantics.

        :param audio: Single audio input ((Tensor, sample_rate) tuple, path,
            or bytes).
        :param shifts: Number of random shifts for equivariant stabilization.
        :param split_overlap: Overlap between segments (0.0 to 1.0).
        :param seed: Optional random seed for reproducible shift-based inference.
        :param progress_callback: Optional callback for progress updates.
        :param use_only_stem: Optional stem name to run only its specialised
            sub-model in a ``ModelEnsemble``; the result still contains all
            sources.
        :param chunk_batch_size: Number of chunks processed in parallel.
        :return: A ``SeparatedSources`` for the input.
        """
        wav = self._to_tensor(audio)
        original = wav.clone()

        wav = self._stage_for_inference([wav], shifts)[0]
        self._seed_rngs(seed)
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
        )[0]

        sources = self._unnormalized_cpu_sources(sources_tensor, mean, std)
        return SeparatedSources(sources, self.sample_rate, original=original)

    def _separate_batch(
        self,
        audios: list,
        *,
        shifts: int,
        split_overlap: float,
        seed: int | None,
        use_only_stem: str | None,
        chunk_batch_size: int,
    ) -> "list[SeparatedSources]":
        """
        Batched separation: pools tail chunks across inputs to keep every
        forward pass full-batch. Returns one ``SeparatedSources`` per input.

        See ``separate`` for parameter semantics.

        :param audios: List of audio inputs ((Tensor, sample_rate) tuples,
            paths, or bytes).
        :param shifts: Number of random shifts for equivariant stabilization.
        :param split_overlap: Overlap between segments (0.0 to 1.0).
        :param seed: Optional random seed for reproducible shift-based inference.
        :param use_only_stem: Optional stem name to run only its specialised
            sub-model in a ``ModelEnsemble``; each result still contains all
            sources.
        :param chunk_batch_size: Number of chunks processed in parallel.
        :return: A ``list[SeparatedSources]`` in the same order as ``audios``.
        """
        wavs = [self._to_tensor(a) for a in audios]
        originals = [w.clone() for w in wavs]

        wavs = self._stage_for_inference(wavs, shifts)

        # Per-input normalisation stats — applied locally and reversed after
        # the forward, via the same helper ``_separate_one`` uses.
        normed: list[Tensor] = []
        stats: list[tuple[Tensor, Tensor]] = []
        for w in wavs:
            normed_w, mean, std = self._normalize(w)
            normed.append(normed_w[None])
            stats.append((mean, std))

        self._seed_rngs(seed)

        outputs = apply_model_multi(
            self.model,
            normed,
            device=self.device,
            shifts=shifts,
            overlap=split_overlap,
            use_only_stem=use_only_stem,
            chunk_batch_size=chunk_batch_size,
        )

        results: list[SeparatedSources] = []
        for out, (mean, std), original in zip(outputs, stats, originals):
            sources = self._unnormalized_cpu_sources(out[0], mean, std)
            results.append(
                SeparatedSources(sources, self.sample_rate, original=original)
            )
        return results


def select_model(
    isolate_stem: str | None = None,
) -> tuple[str, str | None]:
    """
    Select optimal Demucs model for audio separation.

    This function automatically chooses the best model based on the use case,
    optimizing for quality and performance:

    - For specific stems: Uses specialized fine-tuned models
      - vocals, bass, other -> htdemucs_ft (fine-tuned, better quality)
      - guitar, piano -> htdemucs_6s (6-stem model)
      - drums -> htdemucs (already optimal)

    - Default: htdemucs (balanced quality/performance)

    :param isolate_stem: Specific stem to isolate.
    :return: Tuple of (model_name, only_load_stem)
        - model_name: Name of the recommended model
        - only_load_stem: Stem to load exclusively from ModelEnsemble (for htdemucs_ft),
                         or None to load all stems

    """
    # Stem-specific model selection (quality optimization)
    if isolate_stem:
        if isolate_stem in ["guitar", "piano"]:
            # htdemucs_6s is specialized for 6-stem separation
            return ("htdemucs_6s", None)
        if isolate_stem == "drums":
            # htdemucs already performs best for drums
            return ("htdemucs", None)
        if isolate_stem in ["bass", "other", "vocals"]:
            # htdemucs_ft has fine-tuned models for these stems
            return ("htdemucs_ft", isolate_stem)

    # Default: htdemucs is the best balanced model
    return ("htdemucs", None)


def get_version() -> str:
    """
    Get the version of Demucs you have installed.

    :return: Version string
    """
    return __version__
