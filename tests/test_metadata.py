"""
Integrity checks for the bundled model registry (``demucs/metadata.json``).

These run fully offline: ``ModelRepository`` only reads the local metadata file
and builds download URLs as strings, so no network access is required.
"""

from demucs.repo import ModelRepository

EXPECTED_MODELS = {"htdemucs", "htdemucs_ft", "htdemucs_6s"}


def test_repository_lists_expected_models() -> None:
    """The shipped registry exposes the documented model names."""
    models = ModelRepository().list_models()
    assert EXPECTED_MODELS.issubset(models.keys())


def test_every_layer_has_remote_and_checksum() -> None:
    """Each model lists at least one layer with a remote path and a checksum."""
    for name, info in ModelRepository().list_models().items():
        layers = info.get("models")
        assert layers, f"{name} has no layers"
        for layer in layers:
            assert layer.get("remote"), f"{name} layer missing remote"
            assert layer.get("checksum"), f"{name} layer missing checksum"


def test_ensemble_weights_are_consistent() -> None:
    """Where present, ``weights`` has one row per layer and uniform width."""
    for name, info in ModelRepository().list_models().items():
        weights = info.get("weights")
        if weights is None:
            continue
        assert len(weights) == len(info["models"]), (
            f"{name}: weight rows must match layer count"
        )
        widths = {len(row) for row in weights}
        assert len(widths) == 1, f"{name}: ragged weight rows {widths}"
