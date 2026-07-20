"""Focused regression tests for low-level spectrogram helpers."""

import pytest
import torch

from unblend.blocks import _istft_fold


@pytest.mark.parametrize("win_length", [16, 12])
def test_istft_fold_matches_torch_and_has_finite_backward(win_length: int) -> None:
    """Custom iSTFT matches PyTorch while keeping input gradients finite."""
    n_fft = 16
    hop_length = 4
    length = 36
    signal = torch.randn(2, length)
    window = torch.hann_window(win_length)
    z = (
        torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            normalized=True,
            return_complex=True,
        )
        .detach()
        .requires_grad_(True)
    )

    expected = torch.istft(
        z.detach(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        normalized=True,
        length=length,
    )
    actual = _istft_fold(
        z,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_istft_fold_right_pads_extended_length() -> None:
    """An explicit overlong reconstruction is padded to the requested size."""
    n_fft = 16
    hop_length = 4
    available_signal = torch.randn(1, 32)
    window = torch.hann_window(n_fft)
    z = torch.stft(
        available_signal,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        normalized=True,
        return_complex=True,
    )
    requested = 48

    actual = _istft_fold(
        z,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=requested,
    )

    assert actual.shape == (1, requested)
    assert torch.count_nonzero(actual[..., 40:]) == 0
    assert torch.isfinite(actual).all()
