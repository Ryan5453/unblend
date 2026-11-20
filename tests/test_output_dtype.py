"""End-to-end FP32 output and export under a nonstandard default dtype."""

from pathlib import Path

import pytest
import torch

from unblend.api import Separator
from unblend.htdemucs import HTDemucs
from unblend.roformer import BSRoformer
from unblend.scnet import build_scnet


@pytest.fixture
def default_dtype():
    """Restore the process default even if separation or export fails."""
    previous = torch.get_default_dtype()
    yield torch.set_default_dtype
    torch.set_default_dtype(previous)


def _model(family: str):
    """Build small real backends without downloading checkpoints."""
    if family == "htdemucs":
        return HTDemucs(
            sources=["vocals", "other"],
            samplerate=8000,
            segment=1.0,
            nfft=512,
            depth=2,
            channels=16,
            t_layers=1,
        )
    if family == "roformer":
        model = BSRoformer(
            dim=32,
            depth=1,
            stereo=True,
            num_stems=1,
            time_transformer_depth=1,
            freq_transformer_depth=1,
            dim_head=16,
            heads=2,
        )
        model.configure_inference(
            sources=["vocals", "other"], samplerate=8000, segment_samples=8192
        )
        return model
    return build_scnet(
        "scnet",
        dict(
            sources=["vocals", "other"],
            audio_channels=2,
            dims=[4, 8, 16, 32],
            nfft=512,
            hop_size=128,
            win_size=512,
            band_stride=[1, 2, 4],
            band_kernel=[3, 4, 4],
            conv_depths=[1, 1, 1],
            num_dplayer=2,
        ),
        sources=["vocals", "other"],
        samplerate=8000,
        segment_samples=8192,
    )


@pytest.mark.parametrize("family", ["htdemucs", "roformer", "scnet"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
@pytest.mark.parametrize("shifts", [0, 1])
def test_fp32_separation_exports_with_nonstandard_default(
    family: str, dtype: torch.dtype, shifts: int, default_dtype, tmp_path: Path
) -> None:
    """FP32 inference keeps FP32 stems through overlap-add and shifts."""
    separator = Separator(model=_model(family), device="cpu", dtype=torch.float32)
    t = torch.arange(9000, dtype=torch.float32) / 8000
    tone = torch.sin(2 * torch.pi * 220 * t).repeat(2, 1)
    default_dtype(dtype)
    result = separator.separate((tone, 8000), shifts=shifts, seed=0)
    path = tmp_path / "vocals.wav"
    result.export_stem("vocals", path)
    assert path.stat().st_size > 0
    for stem in result.sources.values():
        assert stem.dtype == torch.float32
        assert stem.shape == tone.shape
        assert torch.isfinite(stem).all()
