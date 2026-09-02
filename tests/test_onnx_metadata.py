"""
Guards the metadata every ONNX export embeds.

The graph is the network alone, so anything a consumer needs but cannot
recover by inspecting it has to travel in ``metadata_props``. The riskiest
of those is ``external_normalization``: it is a property of every family
(``ASSModel.external_normalization``), and HTDemucs is the one that says
``true``, so a consumer that never sees the key silently skips a step the
Python path performs in ``Separator._normalize``.
"""

import pytest
import torch

from unblend.backends import ASSModel
from unblend.htdemucs import HTDemucs
from unblend.onnx import _export_metadata
from unblend.roformer import BSRoformer, MelBandRoformer
from unblend.scnet import SCNet

STFT = {"n_fft": 2048, "hop_length": 512, "win_length": 2048, "normalized": True}


class _Stub(torch.nn.Module):
    """A model with only the attributes ``_export_metadata`` reads."""

    def __init__(self, external_normalization: object) -> None:
        """
        :param external_normalization: Value for the attribute, or ``None``
            to leave it unset and exercise the fallback.
        """
        super().__init__()
        self.sources = ["vocals", "other"]
        self.samplerate = 44100
        self.audio_channels = 2
        if external_normalization is not None:
            self.external_normalization = external_normalization


def _metadata(model: torch.nn.Module, family: str) -> dict[str, str]:
    """
    Build export metadata for ``model`` with everything else held fixed.

    :param model: Model to read the shared attributes from.
    :param family: Loader family name.
    :return: The embedded key/value pairs.
    """
    return _export_metadata(
        model,
        family=family,
        architecture="test",
        segment_samples=1,
        stft=STFT,
        stft_window="hann",
        storage=torch.float32,
        static_batch=False,
        license_label=None,
    )


@pytest.mark.parametrize(
    ("family", "external"),
    [("demucs", True), ("roformer", False), ("scnet", False)],
)
def test_external_normalization_is_embedded_for_every_family(
    family: str, external: bool
) -> None:
    """
    Every family carries the key, not just the one whose value varies.

    It used to be written only in the SCNet branch, which left HTDemucs —
    the sole family that needs the caller to normalize — without it.
    """
    metadata = _metadata(_Stub(external), family)
    assert metadata["external_normalization"] == ("true" if external else "false")


def test_external_normalization_defaults_to_true_when_unset() -> None:
    """
    An unset attribute reports ``true``, matching the base-class default.

    Under-normalizing is the silent failure; the fallback matches
    ``ASSModel`` so a new architecture cannot opt out by omission.
    """
    assert _metadata(_Stub(None), "demucs")["external_normalization"] == "true"


def test_family_classes_agree_with_the_exported_values() -> None:
    """
    The values above track the model classes rather than restating them.
    """
    assert ASSModel.external_normalization is True
    assert HTDemucs.external_normalization is True
    assert BSRoformer.external_normalization is False
    assert MelBandRoformer.external_normalization is False
    assert SCNet.external_normalization is False
