"""Network-free lifecycle tests for the Cog request coalescer."""

import asyncio
import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import IO


class _BaseModel:
    """Small Cog ``BaseModel`` stand-in used only during module import."""

    def __init__(self, **values: object) -> None:
        """Store arbitrary output fields like Cog/Pydantic would."""
        self.__dict__.update(values)


class _BasePredictor:
    """Cog ``BasePredictor`` stand-in."""


def _input(**kwargs: object) -> object:
    """Return the declared default for a Cog input placeholder."""
    return kwargs.get("default")


_COG = ModuleType("cog")
_COG.BaseModel = _BaseModel
_COG.BasePredictor = _BasePredictor
_COG.Input = _input
_COG.Path = Path
# predictor.Output declares its stems as ``File``; only the annotation is
# needed at import time, so any type object stands in.
_COG.File = IO[bytes]
sys.modules.setdefault("cog", _COG)

_SPEC = importlib.util.spec_from_file_location(
    "_unblend_predictor_test_module",
    Path(__file__).resolve().parent.parent / "predictor.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_PREDICTOR_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREDICTOR_MODULE
_SPEC.loader.exec_module(_PREDICTOR_MODULE)


class _FakeSeparator:
    """Pointwise separator stand-in recording batch calls."""

    chunk_batch_size = 4
    # Mirrors the real Separator's attribute surface, which setup()'s
    # diagnostic log line reads. Keep this in sync with unblend.api.Separator:
    # stems live on ``model.sources``, NOT on the separator itself. An earlier
    # version of this fake invented ``sources`` here, which let a real
    # AttributeError reach a deployed image because the tests passed.
    device = "cpu"
    dtype = None
    sample_rate = 44100
    model = SimpleNamespace(
        sources=["drums", "bass", "other", "vocals"], samplerate=44100
    )
    _compile_enabled = False
    _chunk_batch_size_auto = False
    _calibration_attempts: list[int] = []
    _per_chunk_steady_bytes = None

    def __init__(self) -> None:
        """Initialize the call log."""
        self.calls: list[object] = []

    def separate(self, audio: object, **_kwargs: object) -> object:
        """Return one marker per list input, or one marker for a scalar."""
        self.calls.append(audio)
        if isinstance(audio, list):
            return [f"result:{path.name}" for path in audio]
        # setup()'s warmup passes the real API's ``(samples, sample_rate)``
        # form rather than a path; see unblend/api.py's input validation.
        if isinstance(audio, tuple):
            return "result:warmup"
        return f"result:{audio.name}"


def _predictor(separator: _FakeSeparator, window: float = 0.0):
    """Build a predictor with setup's coalescer state but no model loading."""
    predictor = object.__new__(_PREDICTOR_MODULE.Predictor)
    predictor.separators = {"htdemucs": separator}
    predictor._queues = {}
    predictor._coalescers = {}
    predictor._batch_window_s = window
    return predictor


def _request(name: str):
    """Create a request while an asyncio loop is running."""
    return _PREDICTOR_MODULE._Request(
        audio_path=Path(f"/{name}"),
        model_name="htdemucs",
        isolate_stem="none",
        format="wav",
        clip_mode="rescale",
    )


def test_same_key_requests_batch_and_worker_retires() -> None:
    """Compatible requests share one call and leave no permanent registry."""

    async def scenario() -> None:
        separator = _FakeSeparator()
        predictor = _predictor(separator, window=0.01)
        key = ("htdemucs", 1, 0.25, "none")
        requests = [_request(f"track-{index}.wav") for index in range(3)]
        for request in requests:
            predictor._enqueue_request(key, request)

        results = await asyncio.gather(*(request.future for request in requests))
        await asyncio.sleep(0)

        assert results == [
            "result:track-0.wav",
            "result:track-1.wav",
            "result:track-2.wav",
        ]
        assert len(separator.calls) == 1
        assert predictor._queues == {}
        assert predictor._coalescers == {}

    asyncio.run(scenario())


def test_many_distinct_keys_do_not_accumulate_workers() -> None:
    """Sequential client-controlled parameter keys are retired after use."""

    async def scenario() -> None:
        separator = _FakeSeparator()
        predictor = _predictor(separator)
        for index in range(200):
            key = ("htdemucs", 1, round(index / 1000, 3), "none")
            request = _request(f"track-{index}.wav")
            predictor._enqueue_request(key, request)
            assert await request.future == f"result:track-{index}.wav"
            await asyncio.sleep(0)

        assert predictor._queues == {}
        assert predictor._coalescers == {}

    asyncio.run(scenario())


def test_worker_initialization_failure_resolves_request_and_retires() -> None:
    """A missing separator cannot strand a future before guarded processing."""

    async def scenario() -> None:
        predictor = _predictor(_FakeSeparator())
        predictor.separators = {}
        key = ("missing", 1, 0.25, "none")
        request = _request("track.wav")
        predictor._enqueue_request(key, request)

        try:
            await request.future
        except KeyError:
            pass
        else:
            raise AssertionError("missing separator unexpectedly succeeded")
        await asyncio.sleep(0)
        assert predictor._queues == {}
        assert predictor._coalescers == {}

    asyncio.run(scenario())


def test_completed_batch_is_released_before_next_inference() -> None:
    """The worker frame does not retain prior result tensors across batches."""

    class Payload:
        """Weak-referenceable stand-in for a separated tensor bundle."""

    class LifetimeSeparator(_FakeSeparator):
        """Assert the first result is collectible before the second call."""

        chunk_batch_size = 1

        def __init__(self) -> None:
            """Initialize result-lifetime tracking."""
            super().__init__()
            self.first_ref: weakref.ReferenceType[Payload] | None = None
            self.call_count = 0

        def separate(self, audio: object, **_kwargs: object) -> object:
            """Return payloads and check collection on the second batch."""
            if not isinstance(audio, list):
                return Payload()
            self.call_count += 1
            if self.call_count == 1:
                payload = Payload()
                self.first_ref = weakref.ref(payload)
                return [payload]
            gc.collect()
            assert self.first_ref is not None and self.first_ref() is None
            return [Payload()]

    async def scenario() -> None:
        separator = LifetimeSeparator()
        predictor = _predictor(separator)
        key = ("htdemucs", 1, 0.25, "none")

        first = _request("first.wav")
        first_future = first.future
        predictor._enqueue_request(key, first)

        async def consume_first(future: asyncio.Future) -> None:
            """Consume and release the first future's result."""
            result = await future
            assert isinstance(result, Payload)

        consumer = asyncio.create_task(consume_first(first_future))
        del first, first_future

        second = _request("second.wav")
        predictor._enqueue_request(key, second)
        assert isinstance(await second.future, Payload)
        await consumer
        await asyncio.sleep(0)
        assert separator.call_count == 2
        assert predictor._queues == {}

    asyncio.run(scenario())


def test_failed_batch_falls_back_per_request_and_retires() -> None:
    """One invalid input does not strand neighbors or retain its worker."""

    class PartiallyFailingSeparator(_FakeSeparator):
        """Fail lists and one named scalar request."""

        def separate(self, audio: object, **kwargs: object) -> object:
            """Force fallback, then fail only ``bad.wav``."""
            if isinstance(audio, list):
                raise RuntimeError("batch failed")
            if audio.name == "bad.wav":
                raise ValueError("bad input")
            return super().separate(audio, **kwargs)

    async def scenario() -> None:
        predictor = _predictor(PartiallyFailingSeparator(), window=0.01)
        key = ("htdemucs", 1, 0.25, "none")
        good = _request("good.wav")
        bad = _request("bad.wav")
        predictor._enqueue_request(key, good)
        predictor._enqueue_request(key, bad)

        assert await good.future == "result:good.wav"
        try:
            await bad.future
        except ValueError as exc:
            assert str(exc) == "bad input"
        else:
            raise AssertionError("bad request unexpectedly succeeded")
        await asyncio.sleep(0)

        assert predictor._queues == {}
        assert predictor._coalescers == {}

    asyncio.run(scenario())


def test_setup_rejects_cpu_fallback(monkeypatch) -> None:
    """
    ``setup()`` refuses to boot on CPU because cog.yaml declares ``gpu: true``.

    A host driver too old for the installed CUDA wheel makes
    ``torch.cuda.is_available()`` return False with only a UserWarning, so
    without this guard the image boots clean and serves correct audio at ~35x
    the latency, with compile and batching silently disabled. The failure has
    to be loud to be diagnosable.
    """
    monkeypatch.delenv("UNBLEND_ALLOW_CPU", raising=False)
    monkeypatch.setattr(_PREDICTOR_MODULE.torch.cuda, "is_available", lambda: False)
    # Stubbed so that deleting the guard fails this test on the assertion
    # below rather than by silently downloading a model in CI.
    monkeypatch.setattr(
        _PREDICTOR_MODULE, "Separator", lambda **_kwargs: _FakeSeparator()
    )

    predictor = _PREDICTOR_MODULE.Predictor()
    try:
        asyncio.run(predictor.setup())
    except RuntimeError as exc:
        assert "CUDA is unavailable" in str(exc)
    else:
        raise AssertionError("setup() booted on CPU without an opt-in")


def test_setup_allows_cpu_when_opted_in(monkeypatch) -> None:
    """``UNBLEND_ALLOW_CPU=1`` keeps the image runnable on a GPU-less host."""
    monkeypatch.setenv("UNBLEND_ALLOW_CPU", "1")
    monkeypatch.setattr(_PREDICTOR_MODULE.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        _PREDICTOR_MODULE, "Separator", lambda **_kwargs: _FakeSeparator()
    )

    predictor = _PREDICTOR_MODULE.Predictor()
    asyncio.run(predictor.setup())

    assert "htdemucs" in predictor.separators


def test_output_stems_upload_rather_than_inline() -> None:
    """
    Output stems must be ``Path`` (uploaded), never ``File`` (inlined base64).

    Cog serializes a ``File`` into the prediction response as a base64 data
    URI. That is survivable for a 3s clip and fatal for real audio: a 225s song
    as wav is ~160MB of stems, so ~212MB of base64 in one JSON body. Measured
    on an H100 the separation took 0.65s and the prediction then sat in
    ``processing`` for over 22 minutes without ever returning, while the same
    song as mp3 (~19MB inlined) completed with predict_time 11.08s. A ``Path``
    is uploaded to object storage instead, so payload size stops mattering.
    """
    hints = _PREDICTOR_MODULE.Output.__annotations__
    assert hints, "Output must declare stems for Cog's static schema generator"
    for stem, annotation in hints.items():
        assert annotation is _PREDICTOR_MODULE.Path, (
            f"Output.{stem} must be Path so Cog uploads it rather than "
            f"inlining base64; got {annotation!r}"
        )


def test_setup_warms_up_on_cuda(monkeypatch) -> None:
    """
    ``setup()`` forces the compiled path's CUDAGraph capture at boot.

    Capture happens on the first forward rather than in ``Separator()``, so
    without a warmup the first *prediction* pays it -- 16.4s measured on an
    H200 against 0.29s warm, which is most of the ~20s predict_time seen in
    production on a 3s clip. Replicate does not bill setup per prediction, so
    the cost belongs here.
    """
    monkeypatch.setenv("UNBLEND_ALLOW_CPU", "1")
    monkeypatch.setattr(_PREDICTOR_MODULE.torch.cuda, "is_available", lambda: True)
    separator = _FakeSeparator()
    monkeypatch.setattr(_PREDICTOR_MODULE, "Separator", lambda **_kwargs: separator)

    predictor = _PREDICTOR_MODULE.Predictor()
    asyncio.run(predictor.setup())

    assert len(separator.calls) == 1, "expected exactly one warmup separation"
    warmup = separator.calls[0]
    assert isinstance(warmup, tuple), "warmup must use the (samples, rate) form"
    samples, rate = warmup
    assert rate == separator.model.samplerate
    assert samples.shape[0] == 2, "warmup mix must be stereo"
    assert float(samples.std()) > 0.0, (
        "warmup must not be silence: separation normalizes by the mix's "
        "standard deviation, which is zero for zeros"
    )


def test_setup_survives_a_failing_warmup(monkeypatch) -> None:
    """
    A warmup that raises degrades to a slow first prediction, never a dead boot.

    An earlier diagnostics block read an attribute the real Separator lacked and
    Replicate permanently disabled that version, so anything setup() runs purely
    for performance has to be non-fatal.
    """
    monkeypatch.setenv("UNBLEND_ALLOW_CPU", "1")
    monkeypatch.setattr(_PREDICTOR_MODULE.torch.cuda, "is_available", lambda: True)

    class _ExplodingSeparator(_FakeSeparator):
        def separate(self, audio: object, **kwargs: object) -> object:
            """Fail the way a shape/OOM error would during capture."""
            raise RuntimeError("CUDA error during graph capture")

    monkeypatch.setattr(
        _PREDICTOR_MODULE, "Separator", lambda **_kwargs: _ExplodingSeparator()
    )

    predictor = _PREDICTOR_MODULE.Predictor()
    asyncio.run(predictor.setup())

    assert "htdemucs" in predictor.separators


def test_diagnostic_attributes_exist_on_separator() -> None:
    """
    setup()'s diagnostic line reads private Separator attributes; keep them real.

    The line is wrapped in a try/except so a rename cannot break the boot, which
    means a typo would silently reduce it to "diagnostics unavailable" -- exactly
    when the numbers are needed to explain a slow GPU. This asserts against the
    real class rather than the fake above, because a fake that invented
    ``sources`` is what let an AttributeError reach a deployed image.
    """
    import inspect
    import re

    from unblend import Separator

    source = inspect.getsource(Separator)
    assigned = set(
        re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=\n]+)?=", source)
    )

    for attr in (
        "device",
        "dtype",
        "chunk_batch_size",
        "sample_rate",
        "_compile_enabled",
        "_chunk_batch_size_auto",
        "_calibration_attempts",
        "_per_chunk_steady_bytes",
    ):
        assert attr in assigned, f"Separator no longer assigns {attr}"

    # Stems come from sep.model.sources; Separator itself has no .sources.
    assert "sources" not in assigned
