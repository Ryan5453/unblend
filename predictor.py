# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path as PathlibPath
from typing import TypedDict

import torch
from cog import BasePredictor, File, Input, Path

from unblend import Separator


class Output(TypedDict, total=False):
    """
    Stems returned by a prediction, keyed by stem name.

    Every key is optional because which ones appear depends on
    ``isolate_stem``: the default returns htdemucs's four sources, while
    isolating returns just ``{stem}`` and ``no_{stem}`` (see
    SeparatedSources.isolate_stem in unblend/api.py).

    This is a TypedDict rather than a ``cog.BaseModel`` on purpose. Cog 0.17.0
    replaced pydantic with coglet, whose ``BaseModel`` is an empty marker class
    -- it has no field handling and no ``__init__``, so the old
    ``class Config: extra = "allow"`` silently became a no-op and every
    prediction failed with "Output.__init__() got an unexpected keyword
    argument 'drums'". Cog's static schema generator accepts BaseModel or
    TypedDict, and a TypedDict is a plain dict at runtime, so it both declares
    a real schema and constructs without relying on any SDK machinery.
    Keys must be declared: the generator reads only declared fields, so a
    dynamically-keyed output cannot be expressed here.
    """

    drums: File
    bass: File
    other: File
    vocals: File
    no_drums: File
    no_bass: File
    no_other: File
    no_vocals: File


@dataclass
class _Request:
    """One coalescer-bound request, with its result future."""

    audio_path: PathlibPath
    model_name: str
    isolate_stem: str
    format: str
    clip_mode: str
    future: asyncio.Future = field(default_factory=asyncio.Future)


# Window inside which we wait for additional requests before flushing a batch.
# 50 ms is the default starting point — short enough that single-file
# latency only sees ~50 ms of added queueing on idle traffic, long enough to
# collect a real batch under any meaningful concurrency. Tune via the
# ``COG_DEMUCS_BATCH_WINDOW_MS`` env var if real traffic shows it short.
_BATCH_WINDOW_MS_DEFAULT = 50


class Predictor(BasePredictor):
    """
    Cog predictor for Demucs audio source separation, with in-process request
    coalescing so concurrent calls share full-batch forward passes on the GPU.

    Architecture:

    * ``setup()`` loads each served model and initializes the coalescer
      registry. ``predict()`` lazily starts one finite-lived worker per active
      parameter partition on the same asyncio event loop. Cog's Rust
      orchestrator runs every async ``predict()`` on that loop (verified at
      event loop initialised in ``setup()`` (verified at
      ``crates/coglet-python/src/worker_bridge.rs:138-176`` in the cog
      source — ``new_event_loop`` + ``run_forever`` on a dedicated thread),
      so each partition worker is visible to every concurrent ``predict()``.

    * Each ``predict()`` writes its input + result future into the queue for
      its requested model and awaits the future. The coalescer drains the
      queue: takes the first request, then either fills up to the model's
      ``chunk_batch_size`` or hits the window deadline, whichever comes
      first. It then runs ``separator.separate(list)`` directly on the
      event loop's thread — not via ``asyncio.to_thread`` — because
      ``torch.compile`` with ``mode="reduce-overhead"`` binds its CUDAGraph
      manager to thread-local storage of the thread that first ran the
      compiled function (``torch/_inductor/cudagraph_trees.py:332``).
      Running the forward on a different thread asserts on a missing TLS
      key. This blocks the event loop for the full duration of the GPU
      work (seconds per batch) — no other coroutine runs until it returns.
      That's acceptable on a single GPU, which serializes the work anyway:
      Cog's Rust HTTP layer keeps accepting requests at the socket while we
      block, and they get enqueued in a burst the moment we yield, so the
      next batch coalesces normally.

    Concurrency is controlled by ``concurrency.max`` in ``cog.yaml`` (which
    Replicate's runtime translates to ``COG_MAX_CONCURRENCY`` at start) — it
    needs to be at least as large as the coalescer's max batch for batching
    to ever engage. The cog.yaml in this repo ships with ``concurrency.max:
    32``, comfortably above the auto-detected ``chunk_batch_size`` on the
    GPUs we target.

    Requests with different ``shifts`` / ``split_overlap`` / ``isolate_stem``
    can't share a forward pass (each sub-model and each shift count has a
    different compute footprint), so we partition the in-flight queue by
    ``(model, shifts, split_overlap, isolate_stem)``. In practice 99% of
    traffic uses defaults, so the default queue carries everything.
    """

    async def setup(self) -> None:
        """
        Load the served model and initialize lazy coalescer state.
        """
        self.separators: dict[str, Separator] = {}
        use_cuda = torch.cuda.is_available()
        # cog.yaml declares ``gpu: true``, so a CPU fallback here is always a
        # deployment fault, never a valid configuration.
        #
        # A driver too old for the installed wheel is only loud if you touch
        # the right API: ``torch.cuda.get_device_properties`` raises
        # "The NVIDIA driver on your system is too old", but
        # ``is_available()`` swallows that and returns False (verified on a
        # driver-570 host against cu130 wheels). Since we branch on
        # ``is_available()``, we get the quiet path -- setup would succeed and
        # the endpoint would serve correct output at a fraction of the speed
        # forever, with compile and batching both silently off. Hence the
        # explicit raise. Set UNBLEND_ALLOW_CPU=1 to run without a GPU.
        if not use_cuda and os.environ.get("UNBLEND_ALLOW_CPU") != "1":
            raise RuntimeError(
                "CUDA is unavailable but cog.yaml declares gpu: true. "
                f"torch {torch.__version__} was built against CUDA "
                f"{torch.version.cuda}; check that the host driver is new "
                "enough for it. Set UNBLEND_ALLOW_CPU=1 to run on CPU anyway."
            )
        # This Cog serves htdemucs only. A ``compile=True`` Separator reserves
        # a CUDAGraphs private memory pool sized to its auto-detected
        # ``chunk_batch_size``; with a single resident model that pool +
        # weights + activations fit comfortably even on a 16 GB GPU.
        #
        # We deliberately do NOT load htdemucs_ft / htdemucs_6s here. auto-cbs
        # sizes each compiled model's pool against the free VRAM *at the moment
        # it's constructed* — load a second compiled model afterward and its
        # weights land on top of the first's already-locked pool, OOMing
        # smaller GPUs at inference time. One model, one pool, no surprises.
        self.separators["htdemucs"] = Separator(
            model="htdemucs",
            device="cuda" if use_cuda else "cpu",
            dtype=torch.float16 if use_cuda else None,
            compile=use_cuda,
        )

        # Report what the accelerated path actually negotiated. Every one of
        # these can silently degrade without any error: compile can fall back
        # to eager, auto-calibration can settle on chunk_batch_size=1 after
        # repeated OOM halving, and a mismatched driver can strand us on CPU.
        # The symptom in all three cases is only "predictions are slow", which
        # is indistinguishable from normal cold-start cost without these
        # numbers in the log.
        # Wrapped because this reads private Separator attributes: a rename
        # upstream must degrade the log line, never fail the boot. Diagnostics
        # that can take down setup are worse than no diagnostics.
        try:
            if use_cuda:
                props = torch.cuda.get_device_properties(0)
                driver = getattr(torch._C, "_cuda_getDriverVersion", lambda: None)()
                print(
                    f"[predictor] gpu={props.name} "
                    f"sm={props.major}.{props.minor} "
                    f"vram={props.total_memory / 1024**3:.1f}GiB "
                    f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
                    f"driver={driver}",
                    flush=True,
                )
            for name, sep in self.separators.items():
                stems = getattr(sep.model, "sources", None)
                print(
                    f"[predictor] {name}: device={sep.device} dtype={sep.dtype} "
                    f"compile_enabled={getattr(sep, '_compile_enabled', '?')} "
                    f"chunk_batch_size={sep.chunk_batch_size} "
                    f"(auto={getattr(sep, '_chunk_batch_size_auto', '?')}, "
                    f"calibration_attempts="
                    f"{getattr(sep, '_calibration_attempts', '?')}, "
                    f"per_chunk_steady="
                    f"{getattr(sep, '_per_chunk_steady_bytes', '?')}) "
                    f"stems={list(stems) if stems else '?'}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not break setup
            print(f"[predictor] diagnostics unavailable: {exc!r}", flush=True)

        # Sanity check: cog.yaml's ``concurrency.max`` (read by the Rust
        # orchestrator at startup; surfaced to Python via the
        # ``COG_MAX_CONCURRENCY`` env var) must be ≥ each separator's
        # auto-detected ``chunk_batch_size`` for the coalescer to fully fill
        # batches under concurrent load. If the deployer underprovisioned
        # cog.yaml relative to the GPU's actual batch capacity, batching
        # silently undersizes — each forward still completes correctly but
        # the last ``cbs - max_concurrency`` slots get tail-padded zeros
        # instead of real requests, costing throughput. We can't *change*
        # the orchestrator setting from here (the Rust layer locked it in
        # before we ran), so we just warn.
        try:
            max_concurrency = int(os.environ.get("COG_MAX_CONCURRENCY", "1"))
        except ValueError:
            max_concurrency = 1
        for name, sep in self.separators.items():
            if sep.chunk_batch_size > max_concurrency:
                print(
                    f"[predictor] WARNING: {name} auto-detected "
                    f"chunk_batch_size={sep.chunk_batch_size} but cog.yaml "
                    f"concurrency.max={max_concurrency}; batching will only "
                    f"fill {max_concurrency}/{sep.chunk_batch_size} slots per "
                    f"forward under concurrent load. Bump ``concurrency.max`` "
                    f"in cog.yaml (or override at runtime with "
                    f"``-e COG_MAX_CONCURRENCY=N``) to ≥ "
                    f"{sep.chunk_batch_size}.",
                    flush=True,
                )

        # Per-queue-key coalescer state. Keys are
        # (model, shifts, split_overlap, isolate_stem); each gets its own
        # asyncio.Queue and a background coalescer task drained by the same
        # event loop.
        self._queues: dict[tuple, asyncio.Queue] = {}
        self._coalescers: dict[tuple, asyncio.Task] = {}

        # Window length is overridable via env so deployments can tune
        # without rebuilding the image.
        try:
            self._batch_window_s = (
                float(
                    os.environ.get(
                        "COG_DEMUCS_BATCH_WINDOW_MS", _BATCH_WINDOW_MS_DEFAULT
                    )
                )
                / 1000.0
            )
            if not math.isfinite(self._batch_window_s) or self._batch_window_s < 0:
                raise ValueError
        except ValueError:
            self._batch_window_s = _BATCH_WINDOW_MS_DEFAULT / 1000.0

    def _queue_key(
        self,
        model: str,
        shifts: int,
        split_overlap: float,
        isolate_stem: str,
    ) -> tuple:
        """
        Build the partition key for the coalescer queue.

        ``split_overlap`` is a client-supplied float and would otherwise make
        near-identical requests incompatible for batching. We quantise to 3
        decimals: requests in one batch are separated at this rounded overlap,
        the ≤0.0005 deviation is numerically irrelevant to quality, and workers
        now retire immediately when their queue drains.

        :param model: model name to separate with
        :param shifts: number of random shifts
        :param split_overlap: overlap between segments
        :param isolate_stem: stem to isolate, or "none"
        :return: the quantised partition key tuple
        """
        return (model, shifts, round(split_overlap, 3), isolate_stem)

    def _enqueue_request(self, key: tuple, request: _Request) -> None:
        """
        Atomically enqueue a request, creating a finite-lived worker if needed.

        There is no ``await`` between registry lookup, worker creation, and
        ``put_nowait``. A drained worker can therefore retire with an atomic
        empty-check without racing a request onto an orphaned queue.

        :param key: Partition key for compatible inference parameters.
        :param request: Request to enqueue.
        """
        queue = self._queues.get(key)
        task = self._coalescers.get(key)
        if queue is None or task is None or task.done():
            queue = asyncio.Queue()
            self._queues[key] = queue
            task = asyncio.create_task(
                self._coalesce(key, queue),
                name=f"demucs-coalescer-{'|'.join(map(str, key))}",
            )
            self._coalescers[key] = task
        queue.put_nowait(request)

    def _process_batch(
        self,
        separator: Separator,
        batch: list[_Request],
        *,
        shifts: int,
        split_overlap: float,
        isolate_stem: str,
    ) -> None:
        """
        Resolve one batch, falling back per request when the batch fails.

        Keeping result tensors in this synchronous helper ensures its frame is
        gone before the worker yields and starts another memory-heavy batch.

        :param separator: Loaded model separator.
        :param batch: Compatible live requests.
        :param shifts: Number of shift rounds.
        :param split_overlap: Segment overlap.
        :param isolate_stem: Stem specialization name or ``"none"``.
        """
        audio_paths = [request.audio_path for request in batch]
        stem_kwarg = isolate_stem if isolate_stem and isolate_stem != "none" else None
        try:
            results = separator.separate(
                audio_paths,
                shifts=shifts,
                split_overlap=split_overlap,
                use_only_stem=stem_kwarg,
            )
            if not isinstance(results, list) or len(results) != len(batch):
                raise RuntimeError(
                    "Batched separation returned an unexpected number of results."
                )
        except Exception:
            for request in batch:
                if request.future.done():
                    continue
                try:
                    single = separator.separate(
                        request.audio_path,
                        shifts=shifts,
                        split_overlap=split_overlap,
                        use_only_stem=stem_kwarg,
                    )
                except Exception as single_exc:
                    request.future.set_exception(single_exc)
                else:
                    request.future.set_result(single)
        else:
            for request, separated in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(separated)

    async def _coalesce(self, key: tuple, queue: asyncio.Queue) -> None:
        """
        Drain one parameter partition into full-batch separation calls.

        The worker retires as soon as its queue drains instead of waiting
        forever, so client-controlled parameter combinations do not accumulate
        permanent tasks and queues. Inference remains synchronous on the event
        loop thread to preserve the compiled CUDAGraph TLS contract.

        :param key: Partition key for compatible inference parameters.
        :param queue: Queue owned by this worker.
        """
        active_batch: list[_Request] = []

        try:
            model_name, shifts, split_overlap, isolate_stem = key
            separator = self.separators[model_name]
            max_batch = separator.chunk_batch_size
            while True:
                try:
                    first = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                batch: list[_Request] = [first]
                active_batch = batch
                next_request: _Request | None = None
                deadline = asyncio.get_running_loop().time() + self._batch_window_s
                while len(batch) < max_batch:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        next_request = await asyncio.wait_for(queue.get(), remaining)
                    except asyncio.TimeoutError:
                        break
                    batch.append(next_request)

                # Cog unlinks a cancelled request's temporary input path, so
                # discard cancelled futures before touching their paths.
                batch = [request for request in batch if not request.future.done()]
                active_batch = batch
                if batch:
                    self._process_batch(
                        separator,
                        batch,
                        shifts=shifts,
                        split_overlap=split_overlap,
                        isolate_stem=isolate_stem,
                    )

                active_batch = []
                batch.clear()
                first = None
                next_request = None
                # Let completed predict() calls encode and release their stem
                # tensors before starting another memory-heavy batch.
                await asyncio.sleep(0)
                if queue.empty():
                    return
        except asyncio.CancelledError:
            for request in active_batch:
                if not request.future.done():
                    request.future.cancel()
            while True:
                try:
                    request = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not request.future.done():
                    request.future.cancel()
            raise
        except Exception as exc:
            for request in active_batch:
                if not request.future.done():
                    request.future.set_exception(exc)
            while True:
                try:
                    request = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not request.future.done():
                    request.future.set_exception(exc)
        finally:
            current = asyncio.current_task()
            if self._queues.get(key) is queue and self._coalescers.get(key) is current:
                self._queues.pop(key, None)
                self._coalescers.pop(key, None)

    async def predict(
        self,
        audio: Path = Input(description="The audio file to separate"),
        model: str = Input(
            description="Model to use for separation (this deployment serves htdemucs)",
            default="htdemucs",
            choices=["htdemucs"],
        ),
        format: str = Input(
            description="Output audio format, anything supported by FFmpeg",
            default="wav",
        ),
        isolate_stem: str = Input(
            description="Only creates a {stem} and no_{stem} stem/file",
            default="none",
            # htdemucs produces drums/bass/other/vocals (guitar/piano are
            # htdemucs_6s-only, which this deployment doesn't serve).
            choices=[
                "none",
                "drums",
                "bass",
                "other",
                "vocals",
            ],
        ),
        shifts: int = Input(
            description="Number of random shifts for equivariant stabilization, more increases quality but increases processing time linearly",
            default=1,
            ge=1,
            le=20,
        ),
        split_overlap: float = Input(
            description="Overlap between segments; higher values improve quality at segment boundaries",
            default=0.25,
            ge=0.0,
            # cog's Input only supports inclusive bounds (ge/le), not lt. The
            # true contract is [0.0, 1.0); 0.99 is the practical ceiling
            # (higher overlaps are pathological) and Separator.separate still
            # enforces < 1.0 server-side.
            le=0.99,
        ),
        clip_mode: str = Input(
            description="Method to prevent audio clipping in output, or None for no clipping prevention",
            default="rescale",
            choices=["none", "rescale", "clamp", "tanh"],
        ),
    ) -> Output:
        """
        Run separation on one file. Compatible concurrent requests share a
        lazily-created forward batch; its worker retires when the queue drains.

        :param audio: the audio file to separate
        :param model: model to use for separation
        :param format: output audio format
        :param isolate_stem: stem to isolate, or "none" for all stems
        :param shifts: number of random shifts for equivariant stabilization
        :param split_overlap: overlap between segments
        :param clip_mode: method to prevent audio clipping in output
        :return: the separated stems as output files
        """
        key = self._queue_key(model, shifts, split_overlap, isolate_stem)
        request = _Request(
            audio_path=PathlibPath(str(audio)),
            model_name=model,
            isolate_stem=isolate_stem,
            format=format,
            clip_mode=clip_mode,
        )
        self._enqueue_request(key, request)

        # Split the timing at the separation boundary. A compiled model pays
        # its graph capture on the first forward of each new shape, so a slow
        # first prediction followed by fast ones is expected warm-up, while
        # uniformly slow separations mean compile or batching is not engaging.
        # Reporting one number cannot tell those apart; reporting separate
        # and encode can.
        started = time.perf_counter()
        try:
            separated = await request.future
        except asyncio.CancelledError:
            # Caller disconnected. Mark our future done so the coalescer
            # discards the result when it lands rather than logging a
            # missing-listener warning.
            if not request.future.done():
                request.future.cancel()
            raise

        separate_s = time.perf_counter() - started

        if isolate_stem != "none":
            separated = separated.isolate_stem(isolate_stem)

        output_data: dict[str, BytesIO] = {}
        clip = None if clip_mode == "none" else clip_mode
        for stem in separated.sources:
            audio_bytes = separated.export_stem(stem, format=format, clip=clip)
            buf = BytesIO(audio_bytes)
            buf.name = f"{stem}.{format}"
            output_data[stem] = buf

        encode_s = time.perf_counter() - started - separate_s
        sep = self.separators.get(model)
        audio_s = None
        if sep is not None and getattr(separated, "sources", None):
            first = next(iter(separated.sources.values()))
            audio_s = first.shape[-1] / sep.sample_rate
        # compile_enabled/chunk_batch_size are repeated here, not just in
        # setup(), because Replicate exposes setup output on a separate log
        # stream that the prediction API does not return. Without them on this
        # line there is no way to tell a compile fallback from a cold cache
        # when all you have is a slow prediction.
        timing = f"[predictor] separate={separate_s:.2f}s encode={encode_s:.2f}s"
        if audio_s:
            timing += f" audio={audio_s:.1f}s realtime={audio_s / separate_s:.2f}x"
        if sep is not None:
            timing += (
                f" compile_enabled={getattr(sep, '_compile_enabled', '?')}"
                f" cbs={sep.chunk_batch_size}"
                f" device={sep.device} dtype={sep.dtype}"
            )
        timing += f" shifts={shifts} overlap={split_overlap} format={format}"
        print(timing, flush=True)

        return Output(**output_data)
