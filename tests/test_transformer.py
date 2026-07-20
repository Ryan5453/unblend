"""Regression tests for documented transformer construction options."""

import pytest
import torch

from unblend.transformer import (
    CrossTransformerEncoder,
    CrossTransformerEncoderLayer,
    MyGroupNorm,
    MyTransformerEncoderLayer,
)


@pytest.mark.parametrize(
    "norm_in,norm_in_group,expected",
    [
        (False, False, torch.nn.Identity),
        (False, True, torch.nn.Identity),
        (True, False, torch.nn.LayerNorm),
        (True, True, MyGroupNorm),
    ],
)
def test_transformer_input_norm_truth_table(
    norm_in: bool, norm_in_group: bool, expected: type[torch.nn.Module]
) -> None:
    """Group input normalization only selects the norm used when enabled."""
    model = CrossTransformerEncoder(
        dim=16,
        num_heads=4,
        num_layers=0,
        norm_in=norm_in,
        norm_in_group=norm_in_group,
    )
    assert isinstance(model.norm_in, expected)
    assert isinstance(model.norm_in_t, expected)


@pytest.mark.parametrize(
    "layer_type",
    [MyTransformerEncoderLayer, CrossTransformerEncoderLayer],
)
def test_transformer_layer_propagates_dtype(layer_type) -> None:
    """Attention, normalization, and layer-scale parameters honor dtype."""
    layer = layer_type(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        norm_first=True,
        norm_out=True,
        layer_scale=True,
        batch_first=True,
        dtype=torch.float64,
    )
    assert {parameter.dtype for parameter in layer.parameters()} == {torch.float64}

    query = torch.randn(2, 5, 16, dtype=torch.float64)
    if layer_type is CrossTransformerEncoderLayer:
        output = layer(query, torch.randn(2, 7, 16, dtype=torch.float64))
    else:
        output = layer(query)
    assert output.dtype == torch.float64
    assert torch.isfinite(output).all()
