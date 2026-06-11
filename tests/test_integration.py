"""
End-to-end separation test.

This is opt-in (marked ``slow``): it downloads the ``htdemucs`` weights and runs
a real CPU inference pass, so it is excluded from the default fast test run.
Run it explicitly with ``pytest -m slow``.
"""

import pytest
import torch

from demucs.api import Separator


@pytest.mark.slow
def test_separate_short_clip_cpu() -> None:
    """Separating a short synthetic clip yields the four named stems at full length."""
    sample_rate = 44100
    num_samples = sample_rate  # 1 second
    t = torch.linspace(0, 1, num_samples)
    tone = torch.stack([torch.sin(2 * torch.pi * 220 * t),
                        torch.sin(2 * torch.pi * 440 * t)])

    separator = Separator(model="htdemucs", device="cpu")
    try:
        result = separator.separate((tone, sample_rate))
    finally:
        # Release the model regardless of outcome.
        separator.model = None

    assert set(result.sources) == {"drums", "bass", "other", "vocals"}
    for stem in result.sources.values():
        assert stem.shape[0] == 2
        assert stem.shape[-1] == num_samples
    assert result.sample_rate == sample_rate
