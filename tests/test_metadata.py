"""
Integrity checks for the bundled model registry (``unblend/metadata.yaml``).

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
    """
    Demucs entries construct allowlisted models from Safetensors only.
    """
    for name, info in ModelRepository().list_models().items():
        if info.get("backend") != "demucs":
            continue
        assert info["architecture"] == "htdemucs"
        assert info["config"]["sources"] == info["sources"]
        layers = info.get("models")
        assert layers, f"{name} has no layers"
        for layer in layers:
            assert layer["format"] == "safetensors"
            assert layer["remote"].endswith(".safetensors")
            assert len(layer["sha256"]) == 64
            assert layer["size_bytes"] > 0


def test_shipped_ensembles_reference_registered_members() -> None:
    """
    Every ``members`` entry resolves to registered models that agree on the
    inference contract — same stems in the same order, same sample rate — since
    that is what ``ModelEnsemble`` requires at build time.
    """

    def samplerate_of(info: dict) -> int:
        """
        A model's sample rate, wherever its backend keeps it.

        :param info: One model's registry metadata.
        :return: The sample rate in Hz.
        """
        return info.get("samplerate") or info["config"]["samplerate"]

    repo = ModelRepository()
    models = repo.list_models()
    ensembles = {name: info for name, info in models.items() if info.get("members")}
    assert ensembles, "the registry should ship at least one ensemble to try"

    for name, info in ensembles.items():
        assert len(repo._members[name]) > 1, f"{name} should combine several members"
        rates = set()
        for spec in info["members"]:
            referenced = spec.get("model")
            if referenced is None:
                continue
            assert referenced in models, f"{name} references unknown {referenced}"
            member = models[referenced]
            assert member["sources"] == info["sources"], (
                f"{name}: {referenced} emits {member['sources']}, "
                f"entry declares {info['sources']}"
            )
            assert not member.get("members"), (
                f"{name}: {referenced} is itself an ensemble"
            )
            rates.add(samplerate_of(member))
        assert len(rates) <= 1, f"{name}: members disagree on sample rate {rates}"


def test_ensemble_weights_are_consistent() -> None:
    """
    Where present, ``weights`` has one row per layer and uniform width.
    """
    for name, info in ModelRepository().list_models().items():
        weights = info.get("weights")
        if weights is None:
            continue
        member_count = len(info.get("members") or info.get("models") or [1])
        assert len(weights) == member_count, (
            f"{name}: weight rows must match member count"
        )
        widths = {len(row) for row in weights}
        assert len(widths) == 1, f"{name}: ragged weight rows {widths}"


def test_every_model_states_its_terms() -> None:
    """
    Every model carries a ``license`` label, and anything not under a plain
    grant explains itself in a ``license_note``.

    The registry is the only place these terms are stated — ``unblend models
    list`` and ``list_models`` read them from here — so an entry without them
    would leave users with no way to know what they are bound by.
    """
    plain_grants = {"MIT", "GPL-3.0"}
    for name, info in ModelRepository().list_models().items():
        label = info.get("license")
        assert label, f"{name} has no license label"
        if label not in plain_grants:
            assert info.get("license_note"), (
                f"{name} is labelled {label!r}, which needs a license_note "
                "saying what that means"
            )


def test_roformer_entries_are_well_formed() -> None:
    """
    Each RoFormer entry carries the fields ``build_roformer`` needs: a known
    architecture, inline config, sources, sample rate, segment length, and a
    Safetensors checkpoint with an https URL, exact size, and full sha256.
    """
    for name, info in ModelRepository().list_models().items():
        # Ensemble entries carry members rather than a checkpoint of their own.
        if info.get("backend") != "roformer" or "checkpoint" not in info:
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


def test_scnet_entries_are_well_formed() -> None:
    """
    Each SCNet entry carries the fields ``build_scnet`` needs, and names the
    architecture that matches its checkpoint: the masked variants carry
    ``mask_layer``/``pos_embed_f`` weights that plain SCNet has no slot for, so
    a mislabelled entry would fail to strict-load rather than degrade quietly.
    """
    for name, info in ModelRepository().list_models().items():
        if info.get("backend") != "scnet" or "checkpoint" not in info:
            continue
        assert info["architecture"] in {"scnet", "scnet_masked"}
        assert isinstance(info["config"], dict) and info["config"]
        assert len(info["sources"]) == 4, f"{name} should emit four stems"
        assert isinstance(info["samplerate"], int)
        assert isinstance(info["segment_samples"], int)
        checkpoint = info["checkpoint"]
        assert checkpoint["format"] == "safetensors"
        assert checkpoint["url"].startswith("https://")
        assert checkpoint["url"].endswith(".safetensors")
        assert len(checkpoint["sha256"]) == 64
        assert checkpoint["size_bytes"] > 0
