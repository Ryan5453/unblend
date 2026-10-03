# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import functools
import math
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path as PathlibPath
from typing import TypedDict

import torch
from cog import BasePredictor, Input, Path

from unblend import Separator, ValidationError


class Output(TypedDict, total=False):
    """
    Stems returned by a prediction, keyed by stem name.

    Keys are optional because ``isolate_stem`` returns only ``{stem}`` and
    ``no_{stem}``. A TypedDict, because Cog reads declared keys for its schema;
    ``Path`` rather than ``File`` so stems are uploaded and returned as URLs
    instead of inlined as base64.
    """

    drums: Path
    bass: Path
    other: Path
    vocals: Path
    no_drums: Path
    no_bass: Path
    no_other: Path
    no_vocals: Path


@dataclass
class _Request:
    """One coalescer-bound request, with its result future."""

    audio_path: PathlibPath
    future: asyncio.Future = field(default_factory=asyncio.Future)
    #: Duration measured by ``predict``, reused for batch budgeting.
    seconds: float | None = None


# How long to wait for more requests before flushing a batch: short enough to
# add little latency on idle traffic, long enough to collect a batch under
# concurrency. Overridable via ``UNBLEND_BATCH_WINDOW_MS``.
_BATCH_WINDOW_MS_DEFAULT = 50

# Stems are written here for Cog to upload, one directory per prediction. Cog
# uploads only after ``predict()`` returns, so a prediction cannot delete its
# own files; instead each prediction sweeps directories left by earlier ones,
# bounding disk use on a long-lived container. The TTL must outlast an upload.
_OUTPUT_ROOT = PathlibPath("/tmp/unblend-outputs")
_OUTPUT_TTL_S = 900


def _max_request_seconds(default: float = 1800.0) -> float:
    """
    The per-request audio limit, from ``UNBLEND_MAX_AUDIO_SECONDS``.

    :param default: Limit used when the variable is unset or invalid.
    :return: Longest single request, in seconds.
    """
    raw = os.environ.get("UNBLEND_MAX_AUDIO_SECONDS")
    if raw is None:
        return default
    try:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError
    except ValueError:
        print(
            f"[predictor] ignoring invalid UNBLEND_MAX_AUDIO_SECONDS={raw!r}",
            flush=True,
        )
        return default
    return value


# Longest single request, in seconds (override with UNBLEND_MAX_AUDIO_SECONDS).
_MAX_REQUEST_AUDIO_SECONDS = _max_request_seconds()

# Total audio per batched separate() call. Inputs and stems for ten minutes of
# 44.1 kHz stereo are ~1 GB of float32; beyond that, batching saves nothing.
_MAX_BATCH_AUDIO_SECONDS = 600.0


def _duration_seconds(path: PathlibPath) -> float:
    """
    Duration for the request limit and batch budgeting.

    The container's duration when it reports one; otherwise the audio is
    decoded a minute at a time, stopping once it passes the request limit, so
    a file without a duration can't slip past the limit.

    :param path: Input audio path.
    :return: Duration in seconds, or 0 for an unreadable file (which fails
        on its own in separation).
    """
    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(str(path))
        duration = decoder.metadata.duration_seconds
    except Exception:
        return 0.0
    if duration is not None and math.isfinite(duration) and duration > 0:
        return float(duration)
    decoded = 0.0
    while decoded <= _MAX_REQUEST_AUDIO_SECONDS:
        try:
            samples = decoder.get_samples_played_in_range(decoded, decoded + 60.0)
        except Exception:
            # Raised past the end of the stream.
            break
        count = samples.data.shape[-1]
        if count == 0:
            break
        decoded += count / samples.sample_rate
    return decoded


def _request_seconds(request: _Request) -> float:
    """
    A request's duration, measuring it only if ``predict`` didn't.

    :param request: Queued request.
    :return: Duration in seconds.
    """
    if request.seconds is not None:
        return request.seconds
    return _duration_seconds(request.audio_path)


@functools.lru_cache(maxsize=32)
def _check_output_format(format: str) -> None:
    """
    Reject a format that isn't a bare extension FFmpeg can encode, before any
    GPU work. The format becomes a filename suffix, so it must not carry a path.

    :param format: Requested output format.
    :raises ValidationError: If the format is unusable.
    """
    if not re.fullmatch(r"[A-Za-z0-9]{1,16}", format):
        raise ValidationError(f"Unsupported output format {format!r}.")
    from unblend import SeparatedSources

    probe = SeparatedSources({"probe": torch.zeros(2, 441)}, 44100, torch.zeros(2, 441))
    try:
        probe.export_stem("probe", format=format, clip=None)
    except ValidationError as error:
        raise ValidationError(f"Unsupported output format {format!r}.") from error


def _prune_stale_outputs() -> None:
    """
    Delete per-prediction output directories older than ``_OUTPUT_TTL_S``.

    Best-effort by design: losing a sweep wastes disk, but raising here would
    fail a prediction whose audio is already separated.
    """
    try:
        cutoff = time.time() - _OUTPUT_TTL_S
        for entry in _OUTPUT_ROOT.iterdir():
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue
    except OSError:
        # Root missing on the first prediction is the normal case.
        pass


class Predictor(BasePredictor):
    """
    Cog predictor for Unblend's HTDemucs separation, with in-process request
    coalescing so concurrent calls share full-batch forward passes on the GPU.

    Cog runs every async ``predict()`` on one event loop. Each ``predict()``
    puts its input and a result future on the queue for its parameter
    partition, and a finite-lived worker per partition drains that queue:
    it takes the first request, then fills up to the model's
    ``chunk_batch_size`` or waits out the batch window, whichever comes
    first.

    The worker runs ``separator.separate`` directly on the event loop thread,
    not via ``asyncio.to_thread``: ``torch.compile`` with
    ``mode="reduce-overhead"`` binds its CUDAGraph manager to thread-local
    storage of the thread that first ran the compiled function, and a forward
    on another thread fails. Blocking the loop is acceptable on one GPU, which
    serializes the work anyway; Cog keeps accepting requests at the socket
    and they enqueue as soon as the loop yields.

    ``concurrency.max`` in ``cog.yaml`` must be at least the coalescer's max
    batch for batching to fill. Requests with different ``shifts``,
    ``split_overlap``, ``isolate_stem`` or ``seed`` cannot share a forward
    pass, so the queue is partitioned by those and the model.
    """

    async def setup(self) -> None:
        """
        Load the served model and initialize lazy coalescer state.
        """
        self.separators: dict[str, Separator] = {}
        use_cuda = torch.cuda.is_available()
        # cog.yaml declares ``gpu: true``, so a CPU fallback is a deployment
        # fault. ``is_available()`` returns False (rather than raising) when
        # the driver is too old for the wheel, which would otherwise serve
        # slowly on CPU forever. UNBLEND_ALLOW_CPU=1 opts out.
        if not use_cuda and os.environ.get("UNBLEND_ALLOW_CPU") != "1":
            raise RuntimeError(
                "CUDA is unavailable but cog.yaml declares gpu: true. "
                f"torch {torch.__version__} was built against CUDA "
                f"{torch.version.cuda}; check that the host driver is new "
                "enough for it. Set UNBLEND_ALLOW_CPU=1 to run on CPU anyway."
            )
        # One model only: a compiled Separator sizes its CUDAGraphs memory
        # pool from the free VRAM when it is constructed, so a second
        # compiled model would land on top of the first's pool and OOM
        # smaller GPUs.
        self.separators["htdemucs"] = Separator(
            model="htdemucs",
            device="cuda" if use_cuda else "cpu",
            dtype=torch.float16 if use_cuda else None,
            compile=use_cuda,
        )

        # Log what the accelerated path negotiated: compile, batch sizing and
        # device can each degrade silently, and the only symptom is slow
        # predictions. This reads private Separator attributes, so it must
        # never fail setup.
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
                import unblend.cuda as cuda_kernels

                # Needs nvcc + ninja matching torch's CUDA major; without them
                # the model runs on plain PyTorch ops.
                print(
                    "[predictor] custom_kernels_loaded="
                    f"{cuda_kernels._extension is not None} "
                    f"build_error={cuda_kernels._extension_error!r}",
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

        # Capture the CUDAGraph now rather than on the first real prediction.
        # One warmup covers every audio length, because the compiled path pads
        # every forward to the captured ``chunk_batch_size`` shape (see
        # ``_apply_model_multi_unshifted`` in unblend/apply.py). A failed
        # warmup only makes the first prediction slow, so it must not fail
        # setup.
        if use_cuda:
            for name, sep in self.separators.items():
                try:
                    started = time.perf_counter()
                    samplerate = int(getattr(sep.model, "samplerate", 44100))
                    warmup_audio = torch.rand(2, samplerate * 3) * 0.2
                    sep.separate((warmup_audio, samplerate))
                    print(
                        f"[predictor] {name}: warmup took "
                        f"{time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - must not break boot
                    print(
                        f"[predictor] {name}: warmup failed, the first "
                        f"prediction will pay compilation instead: {exc!r}",
                        flush=True,
                    )

        # One queue and one worker task per partition key.
        self._queues: dict[tuple, asyncio.Queue] = {}
        self._coalescers: dict[tuple, asyncio.Task] = {}

        try:
            self._batch_window_s = (
                float(
                    os.environ.get("UNBLEND_BATCH_WINDOW_MS", _BATCH_WINDOW_MS_DEFAULT)
                )
                / 1000.0
            )
            if not math.isfinite(self._batch_window_s) or self._batch_window_s < 0:
                raise ValueError
        except ValueError:
            print(
                "[predictor] ignoring invalid UNBLEND_BATCH_WINDOW_MS="
                f"{os.environ.get('UNBLEND_BATCH_WINDOW_MS')!r}",
                flush=True,
            )
            self._batch_window_s = _BATCH_WINDOW_MS_DEFAULT / 1000.0

    def _queue_key(
        self,
        model: str,
        shifts: int,
        split_overlap: float,
        isolate_stem: str,
        seed: int | None = None,
    ) -> tuple:
        """
        Build the partition key for the coalescer queue.

        ``split_overlap`` is quantised to 3 decimals so near-identical
        client-supplied floats can share a batch; requests are separated at
        the rounded overlap, and a deviation of at most 0.0005 does not affect
        quality.

        :param model: model name to separate with
        :param shifts: number of random shifts
        :param split_overlap: overlap between segments
        :param isolate_stem: stem to isolate, or "none"
        :param seed: shift seed; only same-seed requests share a batch
        :return: the quantised partition key tuple
        """
        return (model, shifts, round(split_overlap, 3), isolate_stem, seed)

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
                name=f"unblend-coalescer-{'|'.join(map(str, key))}",
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
        seed: int | None = None,
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
        :param seed: Shift seed, or None for random offsets.
        """
        audio_paths = [request.audio_path for request in batch]
        stem_kwarg = isolate_stem if isolate_stem and isolate_stem != "none" else None
        try:
            results = separator.separate(
                audio_paths,
                shifts=shifts,
                split_overlap=split_overlap,
                use_only_stem=stem_kwarg,
                seed=seed,
            )
            if not isinstance(results, list) or len(results) != len(batch):
                raise RuntimeError(
                    "Batched separation returned an unexpected number of results."
                )
        except Exception as exc:
            if len(batch) == 1:
                # Nothing to isolate: retrying alone would repeat the work.
                if not batch[0].future.done():
                    batch[0].future.set_exception(exc)
                return
            for request in batch:
                if request.future.done():
                    continue
                try:
                    single = separator.separate(
                        request.audio_path,
                        shifts=shifts,
                        split_overlap=split_overlap,
                        use_only_stem=stem_kwarg,
                        seed=seed,
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
        carry: _Request | None = None

        try:
            model_name, shifts, split_overlap, isolate_stem, seed = key
            separator = self.separators[model_name]
            max_batch = separator.chunk_batch_size
            while True:
                if carry is not None:
                    first, carry = carry, None
                else:
                    try:
                        first = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

                batch: list[_Request] = [first]
                active_batch = batch
                # Cap the batch by audio length too: one separate() call holds
                # every input and its stems in memory at once.
                seconds = _request_seconds(first)
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
                    added = _request_seconds(next_request)
                    if seconds + added > _MAX_BATCH_AUDIO_SECONDS:
                        carry = next_request
                        break
                    seconds += added
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
                        seed=seed,
                    )

                active_batch = []
                batch.clear()
                first = None
                next_request = None
                # Let completed predict() calls encode and release their stem
                # tensors before starting another memory-heavy batch.
                await asyncio.sleep(0)
                if queue.empty() and carry is None:
                    return
        except asyncio.CancelledError:
            if carry is not None:
                active_batch.append(carry)
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
            if carry is not None:
                active_batch.append(carry)
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
            description="Random time shifts to average; more is slower and slightly better. 0 disables them",
            default=1,
            ge=0,
            le=20,
        ),
        split_overlap: float = Input(
            description="Overlap between segments; higher values improve quality at segment boundaries",
            default=0.25,
            ge=0.0,
            # Input has no exclusive bound; the real contract is [0.0, 1.0),
            # which Separator.separate enforces.
            le=0.99,
        ),
        seed: int = Input(
            description="Seed for the random shifts, for reproducible output; -1 for random",
            default=-1,
        ),
        clip_mode: str = Input(
            description='How to keep output within [-1, 1]; "none" leaves it unchanged',
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
        :param seed: shift seed, or -1 for random
        :param clip_mode: method to prevent audio clipping in output
        :return: the separated stems as output files
        """
        format = format.lower()
        _check_output_format(format)
        # Off the event loop: a file without a duration is decoded to measure it.
        seconds = await asyncio.to_thread(_duration_seconds, PathlibPath(str(audio)))
        if seconds > _MAX_REQUEST_AUDIO_SECONDS:
            # Decoding plus stems cost ~5 GB of RAM per hour of audio, and one
            # oversized request would take concurrent predictions down with it.
            raise ValidationError(
                # Both at the same precision, so the length never reads as
                # below the limit (at worst equal, when within rounding).
                f"Audio is {seconds:.1f} s long; the limit is "
                f"{_MAX_REQUEST_AUDIO_SECONDS:.1f} s."
            )
        key = self._queue_key(
            model, shifts, split_overlap, isolate_stem, None if seed < 0 else seed
        )
        request = _Request(audio_path=PathlibPath(str(audio)), seconds=seconds)
        self._enqueue_request(key, request)

        # Separate and encode are timed apart so a slow first capture can be
        # told from compile or batching not engaging at all.
        started = time.perf_counter()
        try:
            separated = await request.future
        except asyncio.CancelledError:
            # Caller disconnected: mark the future done so the coalescer
            # discards the result when it lands.
            if not request.future.done():
                request.future.cancel()
            raise

        separate_s = time.perf_counter() - started

        if isolate_stem != "none":
            separated = separated.isolate_stem(isolate_stem)

        _prune_stale_outputs()
        output_dir = _OUTPUT_ROOT / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        output_data: dict[str, Path] = {}
        clip = None if clip_mode == "none" else clip_mode
        for stem in separated.sources:
            audio_bytes = separated.export_stem(stem, format=format, clip=clip)
            stem_path = output_dir / f"{stem}.{format}"
            stem_path.write_bytes(audio_bytes)
            output_data[stem] = Path(stem_path)

        encode_s = time.perf_counter() - started - separate_s
        sep = self.separators.get(model)
        audio_s = None
        if sep is not None and getattr(separated, "sources", None):
            first = next(iter(separated.sources.values()))
            audio_s = first.shape[-1] / sep.sample_rate
        # Repeated from setup() because Replicate's prediction API does not
        # return setup logs.
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
