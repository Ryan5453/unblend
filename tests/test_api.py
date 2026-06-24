import pytest
import torch

from demucs import (
    SeparatedSources,
    __version__,
    get_version,
    select_model,
)
from demucs.exceptions import ValidationError


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
    from demucs.api import Separator

    wav = torch.randn(2, 1000) * 3.0 + 0.5
    normed, mean, std = Separator._normalize(wav)
    restored = normed * (1e-5 + std) + mean
    assert torch.allclose(restored, wav, atol=1e-6)
