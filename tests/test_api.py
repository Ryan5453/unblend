from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch

from unblend import (
    SeparatedSources,
    __version__,
    get_version,
    select_model,
)
from unblend.api import Separator
from unblend.exceptions import LoadAudioError, ValidationError
from unblend.roformer import BSRoformer, RotaryEmbedding


def _stub_separator(
    sources: tuple[str, ...] = ("drums", "bass", "other", "vocals"),
) -> Separator:
    """
    Build a bare ``Separator`` that skips ``__init__``. Only the attributes the
    early-validation block in ``separate()`` touches are populated, so the
    validation paths that raise *before* any model access can be exercised
    offline. Anything past those raises a different (irrelevant) error.

    :param sources: Stem names to expose on the fake model.
    :return: A Separator-shaped object usable only for validation tests.
    """
    sep = object.__new__(Separator)
    sep.chunk_batch_size = 4
    sep._compile_enabled = False
    sep._chunk_batch_size_auto = True
    sep.model = SimpleNamespace(sources=list(sources))
    return sep


class _ProgressModel(torch.nn.Module):
    """Tiny pointwise model for exercising ``Separator`` progress wiring."""

    sources = ["one", "two"]
    samplerate = 100
    audio_channels = 1
    max_allowed_segment = 1.0
    external_normalization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the input and its double as two sources.

        :param x: Input ``[batch, channels, samples]``.
        :return: Output ``[batch, sources, channels, samples]``.
        """
        return torch.stack([x, 2 * x], dim=1)


def _progress_separator() -> Separator:
    """
    Build a CPU ``Separator`` around :class:`_ProgressModel` without loading.

    :return: Configured test separator.
    """
    separator = object.__new__(Separator)
    separator.model = _ProgressModel()
    separator.device = "cpu"
    separator.dtype = None
    separator.sample_rate = 100
    separator.audio_channels = 1
    separator.chunk_batch_size = 2
    return separator


def _make_sources() -> SeparatedSources:
    """
    Build a four-stem ``SeparatedSources`` with constant-valued tensors so the
    complement sum produced by ``isolate_stem`` is trivial to assert against.

    :return: A ``SeparatedSources`` with drums/bass/other/vocals stems filled
        with 1.0/2.0/3.0/4.0 respectively.
    """
    sources = {
        "drums": torch.full((2, 100), 1.0),
        "bass": torch.full((2, 100), 2.0),
        "other": torch.full((2, 100), 3.0),
        "vocals": torch.full((2, 100), 4.0),
    }
    return SeparatedSources(sources, sample_rate=44100, original=torch.zeros(2, 100))


def test_get_version_matches_dunder() -> None:
    """
    ``get_version`` reports the package ``__version__``.
    """
    assert get_version() == __version__


@pytest.mark.parametrize(
    "isolate_stem, expected",
    [
        (None, ("htdemucs", None)),
        ("drums", ("htdemucs", None)),
        ("guitar", ("htdemucs_6s", None)),
        ("piano", ("htdemucs_6s", None)),
        ("vocals", ("htdemucs_ft", "vocals")),
        ("bass", ("htdemucs_ft", "bass")),
        ("other", ("htdemucs_ft", "other")),
    ],
)
def test_select_model(
    isolate_stem: str | None, expected: tuple[str, str | None]
) -> None:
    """
    ``select_model`` maps each stem to its recommended (model, only_load) pair.

    :param isolate_stem: Stem name to isolate, or None
    :param expected: Expected (model, only_load) pair
    """
    assert select_model(isolate_stem=isolate_stem) == expected


def test_isolate_stem_builds_complement() -> None:
    """
    ``isolate_stem`` returns the chosen stem plus a ``no_{stem}`` complement
    equal to the sum of every other stem, carrying metadata through unchanged.
    """
    isolated = _make_sources().isolate_stem("vocals")

    assert set(isolated.sources) == {"vocals", "no_vocals"}
    assert torch.equal(isolated.sources["vocals"], torch.full((2, 100), 4.0))
    # no_vocals == drums + bass + other == 1 + 2 + 3 == 6.
    assert torch.equal(isolated.sources["no_vocals"], torch.full((2, 100), 6.0))
    assert isolated.sample_rate == 44100


def test_isolate_stem_unknown_name_raises() -> None:
    """
    ``isolate_stem`` rejects a stem name absent from the sources.
    """
    with pytest.raises(ValidationError):
        _make_sources().isolate_stem("nope")


def test_export_stem_unknown_name_raises() -> None:
    """
    ``export_stem`` validates the stem name before any encoding, so an unknown
    stem raises without needing FFmpeg/torchcodec.
    """
    with pytest.raises(ValidationError):
        _make_sources().export_stem("nope")


def test_normalize_denormalize_roundtrip() -> None:
    """
    ``_normalize``'s documented inverse (``out * (1e-5 + std) + mean``)
    reconstructs the input exactly — the two must stay symmetric or every
    separated stem carries a systematic gain error.
    """
    wav = torch.randn(2, 1000) * 3.0 + 0.5
    normed, mean, std = Separator._normalize(wav)
    restored = normed * (1e-5 + std) + mean
    assert torch.allclose(restored, wav, atol=1e-6)


def test_normalize_one_sample_is_finite_and_reversible() -> None:
    """One-sample input avoids undefined sample-standard-deviation NaNs."""
    wav = torch.tensor([[1.0], [3.0]])
    normed, mean, std = Separator._normalize(wav)
    restored = normed * (1e-5 + std) + mean

    assert torch.isfinite(normed).all()
    assert torch.isfinite(mean)
    assert std.item() == 0.0
    assert torch.equal(restored, wav)


def test_separate_releases_mps_cache_when_dispatch_fails(monkeypatch) -> None:
    """MPS cache cleanup runs once even when separation raises."""
    separator = _stub_separator()
    separator.device = "mps"
    released = []

    def fail_dispatch(*_args: object, **_kwargs: object) -> None:
        """Raise a representative inference failure."""
        raise RuntimeError("inference failed")

    monkeypatch.setattr(separator, "_run_with_oom_backoff", fail_dispatch)
    monkeypatch.setattr(separator, "_release_mps_cache", lambda: released.append(True))

    with pytest.raises(RuntimeError, match="inference failed"):
        separator.separate(b"audio")
    assert released == [True]


def test_separate_rejects_chunk_batch_size_too_large() -> None:
    """
    ``chunk_batch_size`` over the sanity cap (1024) is rejected before any
    model work — typo guard against four-figure batch sizes.
    """
    with pytest.raises(ValidationError, match="chunk_batch_size"):
        _stub_separator().separate(audio=b"", chunk_batch_size=10_000)


def test_separate_rejects_chunk_batch_size_below_one() -> None:
    """
    ``chunk_batch_size`` must be a positive integer; zero and negatives are
    rejected at the validation gate.
    """
    sep = _stub_separator()
    with pytest.raises(ValidationError, match="chunk_batch_size"):
        sep.separate(audio=b"", chunk_batch_size=0)
    with pytest.raises(ValidationError, match="chunk_batch_size"):
        sep.separate(audio=b"", chunk_batch_size=-1)


def test_separate_rejects_chunk_batch_size_non_integer() -> None:
    """
    Non-integer ``chunk_batch_size`` is rejected.
    """
    with pytest.raises(ValidationError, match="chunk_batch_size"):
        _stub_separator().separate(audio=b"", chunk_batch_size=2.5)


def test_separate_list_input_forwards_progress_callback() -> None:
    """
    Public list-input separation forwards aggregate and per-input progress
    instead of rejecting callbacks.
    """
    events: list[tuple[str, dict]] = []
    results = _progress_separator().separate(
        [(torch.randn(1, 250), 100), (torch.randn(1, 170), 100)],
        shifts=1,
        progress_callback=lambda event, data: events.append((event, dict(data))),
    )

    assert len(results) == 2
    assert events[0][0] == "processing_start"
    assert events[-1][0] == "processing_complete"
    assert events[0][1]["total_inputs"] == 2
    assert {
        data["input_index"] for event, data in events if event == "chunk_complete"
    } == {0, 1}


def test_separate_rejects_unknown_use_only_stem() -> None:
    """
    ``use_only_stem`` must name one of the loaded model's sources; an unknown
    stem fails loudly rather than silently running the full model.
    """
    with pytest.raises(ValidationError, match="not a source"):
        _stub_separator().separate(audio=b"", use_only_stem="banjo")


def test_unnormalized_cpu_sources_clones_per_stem() -> None:
    """
    Per-stem clone in ``_unnormalized_cpu_sources``: mutating one stem buffer
    must not bleed into siblings (regression guard for buffer aliasing).
    """
    sep = _stub_separator(sources=("drums", "bass"))
    src = torch.zeros(2, 2, 4)
    out = sep._unnormalized_cpu_sources(src, torch.tensor(0.0), torch.tensor(1.0))
    assert set(out) == {"drums", "bass"}
    out["drums"].fill_(7.0)
    assert torch.all(out["bass"] == 0)
    assert not out["drums"].data_ptr() == out["bass"].data_ptr()


def test_separator_rejects_unknown_device_string() -> None:
    """
    An unknown ``device`` value short-circuits before any model load with a
    ``ValidationError`` listing the supported devices.
    """
    with pytest.raises(ValidationError, match="device"):
        Separator(model="htdemucs", device="tpu")


def test_separator_rejects_invalid_dtype_string() -> None:
    """
    A string ``dtype`` other than ``"auto"`` raises immediately, before any
    model load.
    """
    with pytest.raises(ValidationError, match="dtype"):
        Separator(model="htdemucs", device="cpu", dtype="float8")


def test_separator_rejects_unsupported_torch_dtype() -> None:
    """
    Only float16 and bfloat16 are accepted as reduced-precision torch dtypes.
    """
    with pytest.raises(ValidationError, match="dtype"):
        Separator(model="htdemucs", device="cpu", dtype=torch.int8)


def test_separator_rejects_lowprec_on_cpu() -> None:
    """
    Float16/bfloat16 inference is rejected on CPU — no fast path exists in
    PyTorch and silently running at fp32 would be surprising.
    """
    with pytest.raises(ValidationError, match="not supported on CPU"):
        Separator(model="htdemucs", device="cpu", dtype=torch.float16)


def test_warmup_rejects_cpu_separator() -> None:
    """
    ``warmup()`` is CUDA-only — it raises ``ValidationError`` on a CPU
    separator before touching any model state.
    """
    sep = _stub_separator()
    sep.device = "cpu"
    with pytest.raises(ValidationError, match="CUDA"):
        sep.warmup()


def test_warmup_rejects_mps_separator() -> None:
    """
    ``warmup()`` is CUDA-only — same rejection path for MPS.
    """
    sep = _stub_separator()
    sep.device = "mps"
    with pytest.raises(ValidationError, match="CUDA"):
        sep.warmup()


def test_warmup_rejects_unsupported_model() -> None:
    """
    ``warmup()`` rejects objects outside the HTDemucs/RoFormer compile targets.
    """
    sep = _stub_separator()
    sep.device = "cuda"
    with pytest.raises(ValidationError, match="HTDemucs and RoFormer"):
        sep.warmup()


def test_compiled_roformer_batch_estimate_snaps_to_power_of_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    RoFormer CUDAGraph batches snap down from the memory ceiling to reduce
    fixed-shape tail padding without using GPU-specific lookup tables.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    separator = object.__new__(Separator)
    separator.device = "cuda"
    separator.model = BSRoformer(
        dim=32,
        depth=1,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=16,
        heads=2,
    )
    separator._compile_enabled = True
    separator._measure_per_chunk_steady_bytes = lambda: 100
    available = 6 * separator._CUDAGRAPH_RESERVATION_FACTOR * 100
    free = separator._CUDA_VRAM_SAFETY_BYTES + int(available)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free, free))

    assert separator._initial_chunk_batch_size_estimate() == 4


def test_cuda_constructor_restores_process_global_backend_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful CUDA setup cannot leak cuDNN/matmul policy to the host."""

    class FakeCudaModel(torch.nn.Module):
        sources = ["one"]
        samplerate = 100
        audio_channels = 1
        max_allowed_segment = 1.0
        external_normalization = False

        def to(self, *_args: object, **_kwargs: object):
            return self

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, None]

    monkeypatch.setattr("unblend.api._require_cuda_available", lambda: None)
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        Separator(
            model=FakeCudaModel(),
            device="cuda",
            dtype=None,
            compile=False,
            chunk_batch_size=1,
        )
        assert torch.backends.cudnn.benchmark is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.set_float32_matmul_precision(previous_precision)


def test_enable_compile_promotes_eager_cuda_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Workload-aware callers can switch an initialized RoFormer from eager to
    the normal compile/calibration path without reconstructing the model.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    separator = object.__new__(Separator)
    separator.model = BSRoformer(
        dim=32,
        depth=1,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=16,
        heads=2,
    )
    separator.device = "cuda"
    separator.chunk_batch_size = 2
    separator._compile_enabled = False
    separator._calibration_attempts = []
    monkeypatch.setattr(separator, "_initial_chunk_batch_size_estimate", lambda: 4)
    observed_during_setup: list[bool] = []

    def fake_calibrate(*, initial_guess: int, compile_enabled: bool) -> int:
        observed_during_setup.append(torch.backends.cudnn.benchmark)
        assert compile_enabled
        return initial_guess

    monkeypatch.setattr(separator, "_calibrate_chunk_batch_size", fake_calibrate)
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        separator.enable_compile()
        assert separator._compile_enabled is True
        assert separator.chunk_batch_size == 4
        assert observed_during_setup == [True]
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark


def test_compile_roformer_targets_transformer_core_without_state_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    RoFormer compilation wraps only ``_run_transformers``, pre-fills rotary
    caches, fixes the batch shape, and leaves checkpoint keys/output unchanged.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    compile_modes: list[str] = []

    def fake_compile(function: Callable[..., Any], *, mode: str) -> Callable[..., Any]:
        """
        Record the compile mode and return the eager callable.

        :param function: Callable passed to ``torch.compile``.
        :param mode: Requested compile mode.
        :return: The original callable.
        """
        compile_modes.append(mode)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = BSRoformer(
        dim=32,
        depth=1,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=16,
        heads=2,
    ).eval()
    model.configure_inference(
        sources=["vocals", "other"], samplerate=44100, segment_samples=44100
    )
    audio = torch.randn(1, 2, 44100)
    with torch.no_grad():
        expected = model(audio)
    state_keys = set(model.state_dict())
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module._phase_cache.clear()

    Separator._compile_roformer_transformer_core(model)
    with torch.no_grad():
        actual = model(audio)

    assert compile_modes == ["reduce-overhead"]
    assert hasattr(model, "_uncompiled_run_transformers")
    assert model._fixed_batch_shape is True
    assert set(model.state_dict()) == state_keys
    assert torch.equal(actual, expected)
    rotary_modules = [
        module for module in model.modules() if isinstance(module, RotaryEmbedding)
    ]
    assert rotary_modules
    assert all(module._phase_cache for module in rotary_modules)
    assert all(module._rotation_cache for module in rotary_modules)

    separator = object.__new__(Separator)
    separator.model = model
    separator.device = "cpu"
    monkeypatch.setattr(torch._dynamo, "reset", lambda: None)
    separator._teardown_compile_state()
    assert not hasattr(model, "_uncompiled_run_transformers")
    assert model._fixed_batch_shape is False


def test_read_pcm16_wav_rejects_non_pcm(tmp_path: object) -> None:
    """
    Non-PCM (``getcomptype() != "NONE"``) and >16-bit WAVs fall back to
    torchcodec; ``_read_pcm16_wav`` returns ``None`` rather than mis-decoding.

    :param tmp_path: pytest temporary directory fixture
    """
    import struct

    # Minimal RIFF/WAVE with PCM 16-bit. Will succeed.
    def _write_wav(path: object, sampwidth: int) -> None:
        """
        Write a 1-frame stereo RIFF/WAVE with the given sample width.

        :param path: Output path
        :param sampwidth: Sample width in bytes
        """
        ch, sr = 2, 44100
        byte_rate = sr * ch * sampwidth
        data = b"\x00" * (ch * sampwidth)
        body = (
            b"WAVEfmt \x10\x00\x00\x00"
            + struct.pack(
                "<HHIIHH", 1, ch, sr, byte_rate, ch * sampwidth, sampwidth * 8
            )
            + b"data"
            + struct.pack("<I", len(data))
            + data
        )
        path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + body)

    # 24-bit PCM: sampwidth=3 ≠ 2 → falls through to None.
    p24 = tmp_path / "wide.wav"
    _write_wav(p24, sampwidth=3)
    assert Separator._read_pcm16_wav(p24) is None


def test_read_pcm16_wav_rejects_zero_sample_rate(
    tmp_path: object, monkeypatch: object
) -> None:
    """
    A WAV header that advertises ``sample_rate == 0`` falls back to torchcodec
    instead of slipping through and triggering a divide-by-zero in
    downstream resampling.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    import wave

    class _FakeWave:
        """Pure stub for ``wave.open`` exercising the zero-rate edge."""

        def __enter__(self) -> "_FakeWave":
            """
            Open the context manager.

            :return: The same instance, mirroring ``wave.open``'s contract.
            """
            return self

        def __exit__(self, *_: object) -> bool:
            """
            Close the context manager.

            :param _: Unused exception info from the ``with`` block.
            :return: ``False`` so any exception inside the ``with`` propagates.
            """
            return False

        def getsampwidth(self) -> int:
            """
            :return: Sample width in bytes (2 = 16-bit PCM).
            """
            return 2

        def getcomptype(self) -> str:
            """
            :return: Compression type — ``"NONE"`` for raw PCM.
            """
            return "NONE"

        def getnframes(self) -> int:
            """
            :return: Frame count for the fake file.
            """
            return 0

        def getnchannels(self) -> int:
            """
            :return: Stereo channel count.
            """
            return 2

        def getframerate(self) -> int:
            """
            :return: The bogus zero sample rate under test.
            """
            return 0

        def readframes(self, n: int) -> bytes:
            """
            Return a stub frame payload.

            :param n: Frame count requested (ignored).
            :return: Four zero bytes (one stereo frame).
            """
            return b"\x00\x00\x00\x00"

    monkeypatch.setattr(wave, "open", lambda *_a, **_k: _FakeWave())
    assert Separator._read_pcm16_wav(tmp_path / "zero.wav") is None


def test_to_tensor_rejects_wrong_tuple_arity() -> None:
    """
    Tuple input must be exactly ``(Tensor, sample_rate)``.
    """
    with pytest.raises(ValidationError, match="tuple"):
        _stub_separator()._to_tensor((torch.zeros(1, 10), 44100, "extra"))


def test_to_tensor_rejects_non_tensor_waveform() -> None:
    """
    A non-Tensor waveform in the tuple raises ``ValidationError`` instead of
    an ``AttributeError`` deep in shape handling.
    """
    with pytest.raises(ValidationError, match="Tensor"):
        _stub_separator()._to_tensor(([0.0] * 10, 44100))


@pytest.mark.parametrize(
    "dtype", [torch.int16, torch.uint8, torch.bool, torch.complex64]
)
def test_to_tensor_rejects_non_floating_tuple_waveforms(dtype: torch.dtype) -> None:
    """Tuple tensors must already be normalized real floating-point audio."""
    sep = _stub_separator()
    sep.sample_rate = 44100
    sep.audio_channels = 1
    with pytest.raises(ValidationError, match="floating-point"):
        sep._to_tensor((torch.zeros(1, 10, dtype=dtype), 44100))


def test_to_tensor_rejects_bad_sample_rate() -> None:
    """
    Non-integer and non-positive sample rates are rejected up front rather
    than surfacing as cryptic resample errors.
    """
    for bad in (0, -1, 44100.5, "44100", True):
        with pytest.raises(ValidationError, match="[Ss]ample rate"):
            _stub_separator()._to_tensor((torch.zeros(1, 10), bad))


def test_to_tensor_accepts_int_like_sample_rates() -> None:
    """
    NumPy integers and whole floats — common outputs of numpy-based audio
    loaders — are accepted like plain ints (regression: strict
    ``isinstance(int)`` used to reject them).
    """
    sep = _stub_separator()
    sep.sample_rate = 44100
    sep.audio_channels = 1
    for sr in (np.int64(44100), np.int32(44100), 44100.0):
        wav = sep._to_tensor((torch.zeros(1, 10), sr))
        assert wav.shape == (1, 10)


def test_to_tensor_url_input_reaches_decoder() -> None:
    """
    URL strings must reach torchcodec (which supports them) instead of being
    rejected by the local-file existence pre-check.
    """
    with pytest.raises(LoadAudioError) as excinfo:
        _stub_separator()._to_tensor("notaproto://example.com/track.mp3")
    assert "File not found" not in str(excinfo.value)


def test_to_tensor_missing_file_message() -> None:
    """
    A nonexistent path reports "File not found", not a misleading hint about
    unsupported formats.
    """
    with pytest.raises(LoadAudioError, match="File not found"):
        _stub_separator()._to_tensor("/definitely/not/here.wav")


def test_is_url_classification(tmp_path: object) -> None:
    """
    Table-driven contract for ``_is_url``: substring match (chained FFmpeg
    protocols route as URLs), existing local files always win, ``Path``
    inputs are never URLs.
    """
    from pathlib import Path as _P

    from unblend.api import _is_url

    existing = tmp_path / "dir:"  # type: ignore[operator]
    existing.mkdir()
    (existing / "take.wav").write_bytes(b"x")

    cases = [
        ("https://example.com/song.mp3", True),
        ("cache:https://example.com/song.mp3", True),  # chained protocol
        ("notaproto://example.com/a.mp3", True),
        ("downloads/https://x.mp3", True),  # nonexistent, contains ://
        (f"{existing}//take.wav", False),  # existing local file wins
        ("plain/local/file.wav", False),
        (_P("https://example.com/song.mp3"), False),  # Path is never a URL
    ]
    for audio, expected in cases:
        assert _is_url(audio) is expected, audio


def test_separate_rejects_bool_numeric_params() -> None:
    """
    bool is an int subclass — every numeric ``separate()`` parameter must
    reject it explicitly rather than silently coercing.
    """
    sep = _stub_separator()
    for kwargs in (
        {"shifts": True},
        {"shifts": False},
        {"chunk_batch_size": True},
        {"seed": True},
        {"split_overlap": False},
    ):
        with pytest.raises(ValidationError):
            sep.separate(audio=b"", **kwargs)


@pytest.mark.parametrize("stem", ["", "bass"])
def test_init_rejects_named_only_load_before_model_load(
    monkeypatch: pytest.MonkeyPatch, stem: str
) -> None:
    """Named-model stem typos fail as ValidationError before get_model."""

    class FakeRepository:
        """Registry stub whose downloader must never be reached."""

        def list_models(self) -> dict[str, dict[str, object]]:
            return {"named": {"sources": ["vocals"]}}

        def get_model(self, **_kwargs: object) -> object:
            pytest.fail("invalid only_load reached get_model")

    monkeypatch.setattr("unblend.api.ModelRepository", FakeRepository)
    with pytest.raises(ValidationError, match="not found"):
        Separator(model="named", only_load=stem, device="cpu")


def test_init_rejects_direct_model_empty_only_load() -> None:
    """Direct model instances apply the same empty-stem validation."""
    with pytest.raises(ValidationError, match="not found"):
        Separator(
            model=_ProgressModel(),
            only_load="",
            device="cpu",
            chunk_batch_size=1,
        )


def test_init_rejects_invalid_chunk_batch_size_before_model_load() -> None:
    """
    A bad explicit ``chunk_batch_size`` fails fast at init (before any model
    download), for zero, negative, bool, and over-cap values.
    """
    for bad in (0, -3, True, 4096):
        with pytest.raises(ValidationError, match="chunk_batch_size"):
            Separator(device="cpu", chunk_batch_size=bad)


def test_separate_rejects_per_call_cbs_when_compiled() -> None:
    """
    A compiled separator's captured batch shape is fixed; per-call overrides
    are rejected with a pointer to the init param.
    """
    sep = _stub_separator()
    sep._compile_enabled = True
    with pytest.raises(ValidationError, match="per-call overrides"):
        sep.separate(audio=b"", chunk_batch_size=8)


def test_separate_allows_matching_cbs_when_compiled() -> None:
    """
    Passing the already-captured value is a harmless no-op: the call gets
    past the compile guard (and fails later on the unrelated bogus stem).
    """
    sep = _stub_separator()
    sep._compile_enabled = True
    with pytest.raises(ValidationError, match="not a source"):
        sep.separate(audio=b"", chunk_batch_size=4, use_only_stem="banjo")


def test_run_with_oom_backoff_compiled_recaptures_at_half() -> None:
    """
    Compiled + auto: an OOM mid-run tears down, recalibrates at half, and
    retries the request with the new size.
    """
    sep = _stub_separator()
    sep._compile_enabled = True
    events: list = []

    def fake_teardown() -> None:
        """
        Record the teardown call.
        """
        events.append("teardown")

    def fake_calibrate(initial_guess: int, compile_enabled: bool) -> int:
        """
        Record and accept the halved guess.

        :param initial_guess: Starting candidate.
        :param compile_enabled: Ignored.
        :return: The accepted chunk_batch_size.
        """
        events.append(initial_guess)
        sep.chunk_batch_size = initial_guess
        return initial_guess

    sep._teardown_compile_state = fake_teardown
    sep._calibrate_chunk_batch_size = fake_calibrate
    calls = {"n": 0}

    def dispatch(cbs: int, state: dict | None):
        """
        OOM once, then succeed.

        :param cbs: Batch size for this attempt.
        :param state: Backoff state dict.
        :return: Sentinel with the winning batch size.
        """
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory (fake)")
        return ("ok", cbs)

    result = sep._run_with_oom_backoff(dispatch, chunk_batch_size=4, allow=True)
    assert result == ("ok", 2)
    assert events == ["teardown", 2]


def test_run_with_oom_backoff_disallowed_or_eager_raises() -> None:
    """
    Explicit sizes (allow=False) and eager separators (whose in-apply
    backoff already exhausted) both propagate the OOM.
    """

    def dispatch(cbs: int, state: dict | None):
        """
        Always raise fake OOM.

        :param cbs: Ignored.
        :param state: Ignored.
        :return: Never returns.
        """
        raise RuntimeError("CUDA out of memory (fake)")

    compiled = _stub_separator()
    compiled._compile_enabled = True
    with pytest.raises(RuntimeError, match="out of memory"):
        compiled._run_with_oom_backoff(dispatch, chunk_batch_size=4, allow=False)

    eager = _stub_separator()
    with pytest.raises(RuntimeError, match="out of memory"):
        eager._run_with_oom_backoff(dispatch, chunk_batch_size=4, allow=True)


def test_run_with_oom_backoff_sticky_eager_downgrade() -> None:
    """
    An eager in-apply halving (reflected in the state dict) sticks to the
    separator for subsequent calls.
    """
    sep = _stub_separator()

    def dispatch(cbs: int, state: dict | None):
        """
        Simulate apply-level halving via the state dict.

        :param cbs: Batch size for this attempt.
        :param state: Backoff state dict (mutated).
        :return: Sentinel.
        """
        assert state is not None
        state["chunk_batch_size"] = 2
        return "done"

    assert sep._run_with_oom_backoff(dispatch, chunk_batch_size=4, allow=True) == "done"
    assert sep.chunk_batch_size == 2
