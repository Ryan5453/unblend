from types import SimpleNamespace

import pytest
import torch

from demucs import (
    SeparatedSources,
    __version__,
    get_version,
    select_model,
)
from demucs.api import Separator
from demucs.exceptions import ValidationError


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
    sep.model = SimpleNamespace(sources=list(sources))
    return sep


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


def test_warmup_rejects_non_htdemucs_model() -> None:
    """
    ``warmup()`` only makes sense for HTDemucs-based models (the only ones
    the CUDAGraphs compile path supports). It raises before invocation.
    """
    sep = _stub_separator()
    sep.device = "cuda"
    # ``model`` is a plain SimpleNamespace, neither HTDemucs nor ModelEnsemble.
    with pytest.raises(ValidationError, match="HTDemucs"):
        sep.warmup()


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
            + struct.pack("<HHIIHH", 1, ch, sr, byte_rate, ch * sampwidth, sampwidth * 8)
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
