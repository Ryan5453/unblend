"""
Guards the ONNX-export path against drifting from the model's own processing.

``compute_stft_for_export`` reimplements ``HTDemucs._spec`` for the traced graph,
and ``HTDemucsONNXWrapper`` reimplements the CaC packing / normalization /
de-interleave around ``forward_core``; if either ever diverges, exported models
would silently produce wrong output. ``_spec`` only reads
``self.hop_length``/``self.nfft`` (no submodules), so the STFT test exercises
the real method via a lightweight stand-in; the end-to-end parity test builds a
tiny random-init ``HTDemucs`` instead.
"""

from types import SimpleNamespace

import torch

from demucs.htdemucs import HTDemucs
from demucs.onnx import HTDemucsONNXWrapper, compute_stft_for_export


def test_export_stft_matches_model_spec() -> None:
    """
    The exported STFT equals ``HTDemucs._spec`` for the same nfft/hop.
    """
    nfft = 4096
    hop_length = nfft // 4
    audio = torch.randn(1, 2, 44100)

    real, imag = compute_stft_for_export(audio, nfft, hop_length)

    stand_in = SimpleNamespace(hop_length=hop_length, nfft=nfft)
    z = HTDemucs._spec(stand_in, audio)

    assert z.shape == real.shape
    assert torch.allclose(z.real, real, atol=1e-5)
    assert torch.allclose(z.imag, imag, atol=1e-5)


def test_onnx_wrapper_matches_reference_forward() -> None:
    """
    ``HTDemucsONNXWrapper`` plus runtime pre/post-processing reproduces
    ``HTDemucs.forward``.

    The wrapper hand-reimplements the CaC packing, input normalization, and
    real/imag de-interleave that ``HTDemucs.forward`` does around
    ``forward_core``; this checks the whole pipeline end to end on a tiny
    random-init model (CPU, FP32 — no export needed).
    """
    torch.manual_seed(0)
    model = HTDemucs(
        sources=["drums", "bass"],
        audio_channels=1,
        channels=8,  # smallest that keeps the DConv hidden dim (C/8) nonzero
        # Smallest nfft whose frequency dim survives all 4 stride-4 layers
        # without collapsing (1024 -> 256 -> 64 -> 16), matching the real
        # htdemucs geometry (no time/freq branch merge).
        nfft=2048,
        t_layers=2,
        t_heads=2,
        samplerate=8000,
        segment=1,
    ).eval()
    wrapper = HTDemucsONNXWrapper(model).eval()

    # Exactly the training length, like the traced graph requires.
    samples = int(model.max_allowed_segment * model.samplerate)
    mix = torch.randn(1, model.audio_channels, samples)

    with torch.no_grad():
        ref = model(mix)
        spec_real, spec_imag = compute_stft_for_export(
            mix, model.nfft, model.hop_length
        )
        out_real, out_imag, out_wave = wrapper(spec_real, spec_imag, mix)

    # Runtime postprocessing (see onnx.md): recombine real/imag, iSTFT, and
    # sum the two branches. ``_ispec`` is the model's own inverse transform.
    zout = torch.complex(out_real, out_imag)
    freq_audio = model._ispec(zout, samples)
    out = freq_audio + out_wave

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)
