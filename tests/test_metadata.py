"""
Integrity checks for the bundled model registry (``unblend/metadata.json``).

These run fully offline: ``ModelRepository`` only reads the local metadata file
and builds download URLs as strings, so no network access is required.
"""

from unblend.repo import ModelRepository

EXPECTED_DEMUCS_MODELS = {"htdemucs", "htdemucs_ft", "htdemucs_6s"}
EXPECTED_ROFORMER_MODELS = {"bs_roformer_sw", "melband_roformer_kim"}


def test_repository_lists_expected_models() -> None:
    """
    The shipped registry exposes the documented Demucs and RoFormer models.
    """
    models = ModelRepository().list_models()
    assert EXPECTED_DEMUCS_MODELS.issubset(models.keys())
    assert EXPECTED_ROFORMER_MODELS.issubset(models.keys())


def test_every_demucs_layer_has_safe_artifact_and_config() -> None:
    """Demucs entries construct allowlisted models from Safetensors only."""
    for name, info in ModelRepository().list_models().items():
        if info.get("backend") == "roformer":
            continue
        assert info["architecture"] == "htdemucs"
        assert info["config"]["sources"] == info["sources"]
        layers = info.get("models")
        assert layers, f"{name} has no layers"
        for layer in layers:
            assert layer["format"] == "safetensors"
            assert layer["remote"].endswith(".safetensors")
            assert layer["sha256"].startswith(layer["checksum"])
            assert layer["size_bytes"] > 0


def test_ensemble_weights_are_consistent() -> None:
    """
    Where present, ``weights`` has one row per layer and uniform width.
    """
    for name, info in ModelRepository().list_models().items():
        weights = info.get("weights")
        if weights is None:
            continue
        assert len(weights) == len(info["models"]), (
            f"{name}: weight rows must match layer count"
        )
        widths = {len(row) for row in weights}
        assert len(widths) == 1, f"{name}: ragged weight rows {widths}"


def test_every_model_is_licence_labelled() -> None:
    """
    Every model carries an explicit ``license`` label — the registry must be
    honest about weight licensing (Demucs weights are ``unlicensed``; the
    RoFormer checkpoints are non-commercial), surfaced in the CLI/API.
    """
    for name, info in ModelRepository().list_models().items():
        assert info.get("license"), f"{name} has no license label"


def test_roformer_entries_are_well_formed() -> None:
    """
    Each RoFormer entry carries the fields ``build_roformer`` needs: a known
    architecture, inline config, sources, sample rate, segment length, and a
    Safetensors checkpoint with an https URL, exact size, and full sha256.
    """
    for name, info in ModelRepository().list_models().items():
        if info.get("backend") != "roformer":
            continue
        assert info["architecture"] in {"bs_roformer", "mel_band_roformer"}
        assert isinstance(info["config"], dict) and info["config"]
        assert info["sources"], f"{name} has no sources"
        assert isinstance(info["samplerate"], int)
        assert isinstance(info["segment_samples"], int)
        checkpoint = info["checkpoint"]
        assert checkpoint["format"] == "safetensors"
        assert checkpoint["url"].startswith("https://")
        assert checkpoint["url"].endswith(".safetensors")
        assert len(checkpoint["sha256"]) == 64
        assert checkpoint["size_bytes"] > 0
