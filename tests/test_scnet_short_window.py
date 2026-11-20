"""Short Hann windows use torch's centered zero-padding semantics."""

import pytest
import torch
import torch.nn.functional as F

from unblend.onnx import (
    _export_scnet_to_onnx,
    compute_scnet_stft_for_export,
)
from unblend.scnet import SCNetMasked, stft_padding


@pytest.mark.parametrize("win_size", [384, 512])
def test_short_hann_window(win_size, tmp_path) -> None:
    """Eager, external STFT and ONNX synthesis agree without pretrained weights."""
    torch.manual_seed(0)
    sources = ["a", "b"]
    model = SCNetMasked(
        sources=sources,
        audio_channels=2,
        dims=[4, 8, 16, 32],
        nfft=512,
        hop_size=128,
        win_size=win_size,
        band_stride=[1, 2, 4],
        band_kernel=[3, 4, 4],
        conv_depths=[1, 1, 1],
        num_dplayer=2,
    ).eval()
    model.configure_inference(sources=sources, samplerate=44100, segment_samples=4096)
    audio = torch.randn(1, 2, 4096)
    padded = F.pad(audio, (0, stft_padding(audio.shape[-1], model.hop_length)))
    with torch.inference_mode():
        expected = model(audio)
        reference = torch.stft(
            padded.flatten(0, 1),
            n_fft=512,
            hop_length=128,
            win_length=win_size,
            window=torch.hann_window(win_size),
            normalized=bool(model.stft_config["normalized"]),
            return_complex=True,
        ).unflatten(0, (1, 2))
        real, imag = compute_scnet_stft_for_export(
            padded,
            512,
            128,
            win_size,
            bool(model.stft_config["normalized"]),
            window="hann",
        )
    stft_diff = (torch.complex(real, imag) - reference).abs().max().item()
    print(f"win_size={win_size} eager/export STFT max abs diff={stft_diff:.9g}")
    torch.testing.assert_close(torch.complex(real, imag), reference)
    assert expected.shape == (1, 2, 2, 4096)
    assert torch.isfinite(expected).all()

    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    path = str(tmp_path / "scnet.onnx")
    _export_scnet_to_onnx(
        model,
        path,
        opset_version=18,
        storage=torch.float32,
        license_label="unlicensed",
        static_batch=True,
    )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out_real, out_imag = session.run(
        None, {"spec_real": real.numpy(), "spec_imag": imag.numpy()}
    )
    spec = torch.complex(torch.from_numpy(out_real), torch.from_numpy(out_imag))
    with torch.inference_mode():
        wave = torch.istft(
            spec.flatten(0, 2),
            **model._stft_kwargs(spec.device),
            length=padded.shape[-1],
        ).reshape(1, 2, 2, -1)[..., : audio.shape[-1]]
    diff = (wave - expected).abs().max().item()
    print(f"win_size={win_size} eager/ONNX waveform max abs diff={diff:.9g}")
    torch.testing.assert_close(wave, expected, atol=1e-4, rtol=1e-4)
