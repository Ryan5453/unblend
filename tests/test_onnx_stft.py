"""
Guards the ONNX-export STFT against drifting from the model's own preprocessing.

``compute_stft_for_export`` reimplements ``HTDemucs._spec`` for the traced graph;
if the two ever diverge, exported models would silently produce wrong spectra.
``_spec`` only reads ``self.hop_length``/``self.nfft`` (no submodules), so we can
exercise the real method via a lightweight stand-in instead of building a model.
"""

from types import SimpleNamespace

import torch

from demucs.htdemucs import HTDemucs
from demucs.onnx import compute_stft_for_export


def test_export_stft_matches_model_spec() -> None:
    """The exported STFT equals ``HTDemucs._spec`` for the same nfft/hop."""
    nfft = 4096
    hop_length = nfft // 4
    audio = torch.randn(1, 2, 44100)

    real, imag = compute_stft_for_export(audio, nfft, hop_length)

    stand_in = SimpleNamespace(hop_length=hop_length, nfft=nfft)
    z = HTDemucs._spec(stand_in, audio)

    assert z.shape == real.shape
    assert torch.allclose(z.real, real, atol=1e-5)
    assert torch.allclose(z.imag, imag, atol=1e-5)
