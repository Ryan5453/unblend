"""Network-free lifecycle tests for the Cog request coalescer."""

import asyncio
import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType


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

    def __init__(self) -> None:
        """Initialize the call log."""
        self.calls: list[object] = []

    def separate(self, audio: object, **_kwargs: object) -> object:
        """Return one marker per list input, or one marker for a scalar."""
        self.calls.append(audio)
        if isinstance(audio, list):
            return [f"result:{path.name}" for path in audio]
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
