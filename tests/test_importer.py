"""
Tests for ``unblend models import``: repackaging a checkpoint from elsewhere.

The import is only worth anything if it refuses to write an entry that does not
work, so most of these check that a mismatch is caught rather than recorded.
"""

import json
from pathlib import Path

import pytest
import torch

from unblend.exceptions import ValidationError
from unblend.importer import (
    candidate_architectures,
    fields_from_config,
    import_checkpoint,
    read_config,
    read_embedded_fields,
    read_tensors,
    register_entry,
    strip_wrapper_prefix,
)
from unblend.repo import ModelRepository

_STEMS = ["drums", "bass", "other", "vocals"]


class _Pickled:
    """
    A plain object, to make a checkpoint that needs real unpickling.
    """


def _scnet_config() -> dict:
    """
    Constructor kwargs for a fast, structurally faithful SCNet.

    :return: Kwargs suitable for ``SCNet``/``SCNetMasked``.
    """
    return dict(
        audio_channels=2,
        dims=[4, 8, 16, 32],
        nfft=512,
        hop_size=128,
        win_size=512,
        band_stride=[1, 2, 4],
        band_kernel=[3, 4, 4],
        conv_depths=[1, 1, 1],
        num_dplayer=1,
    )


def _community_checkpoint(tmp_path: Path, masked: bool = True) -> tuple[Path, Path]:
    """
    Write a checkpoint shaped the way community weights actually ship: a
    training-framework container, a ``model.`` prefix on every key, and a
    separate Music-Source-Separation-Training config.

    :param tmp_path: pytest temporary directory fixture
    :param masked: Whether to save the masked SCNet variant
    :return: ``(checkpoint path, config path)``
    """
    from unblend.scnet import SCNet, SCNetMasked

    config = _scnet_config()
    klass = SCNetMasked if masked else SCNet
    model = klass(sources=_STEMS, **config)
    checkpoint = tmp_path / "community.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            "epoch": 156,
        },
        checkpoint,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "audio": {"chunk_size": 4096, "sample_rate": 8000},
                "model": config,
                "training": {"instruments": _STEMS, "target_instrument": None},
            }
        )
    )
    return checkpoint, config_path


def test_read_tensors_unwraps_a_training_container(tmp_path: Path) -> None:
    """
    A Lightning-style checkpoint yields just the model's parameters.
    """
    checkpoint, _ = _community_checkpoint(tmp_path)

    state = read_tensors(checkpoint)

    assert state, "expected tensors"
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert not any(key.startswith("model.") for key in state)
    assert "epoch" not in state


def test_read_tensors_refuses_a_checkpoint_that_needs_unpickling(
    tmp_path: Path,
) -> None:
    """
    A checkpoint holding pickled objects is rejected, not executed — importing
    must not be a way to run someone else's code.
    """
    checkpoint = tmp_path / "unsafe.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(2)}, "trainer": _Pickled()}, checkpoint)

    with pytest.raises(ValidationError, match="pickled objects"):
        read_tensors(checkpoint)


def test_strip_wrapper_prefix_needs_every_key_to_agree() -> None:
    """
    A prefix only some keys carry is part of the model, not a wrapper.
    """
    mixed = {"model.a": torch.zeros(1), "encoder.b": torch.zeros(1)}
    assert strip_wrapper_prefix(mixed) == mixed

    wrapped = {"module.a": torch.zeros(1), "module.b": torch.zeros(1)}
    assert set(strip_wrapper_prefix(wrapped)) == {"a", "b"}


def test_fields_from_config_translates_the_training_layout() -> None:
    """
    The MSST layout maps onto registry fields mechanically.
    """
    fields = fields_from_config(
        {
            "audio": {"chunk_size": 485100, "sample_rate": 44100},
            "model": {"dim": 384},
            "training": {"instruments": _STEMS, "target_instrument": None},
        }
    )

    assert fields == {
        "config": {"dim": 384},
        "samplerate": 44100,
        "segment_samples": 485100,
        "sources": _STEMS,
    }


@pytest.mark.parametrize(
    "target, expected",
    [("vocals", ["vocals", "other"]), ("instrumental", ["instrumental", "vocals"])],
)
def test_single_head_configs_get_a_complement_stem(
    target: str, expected: list[str]
) -> None:
    """
    A model trained on one target emits its complement as a second stem, and
    the order decides which is which.
    """
    fields = fields_from_config(
        {
            "audio": {"chunk_size": 100, "sample_rate": 44100},
            "model": {"dim": 1},
            "training": {"instruments": ["vocals"], "target_instrument": target},
        }
    )
    assert fields["sources"] == expected


def test_candidate_architectures_reads_the_parameter_names(tmp_path: Path) -> None:
    """
    The family is unmistakable from the keys, and the masked SCNet is too — it
    carries weights plain SCNet has no slot for.
    """
    from unblend.htdemucs import HTDemucs
    from unblend.scnet import SCNet, SCNetMasked

    masked = SCNetMasked(sources=_STEMS, **_scnet_config()).state_dict()
    plain = SCNet(sources=_STEMS, **_scnet_config()).state_dict()
    demucs = HTDemucs(
        sources=["a", "b"],
        samplerate=8000,
        segment=1.0,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    ).state_dict()

    assert candidate_architectures(masked, {}) == ["scnet_masked"]
    assert candidate_architectures(plain, {}) == ["scnet"]
    assert candidate_architectures(demucs, {}) == ["htdemucs"]
    assert candidate_architectures({"unrelated.weight": torch.zeros(1)}, {}) == []


def test_roformer_variants_come_from_the_config_or_are_both_tried() -> None:
    """
    BS- and Mel-Band RoFormer share parameter names, so the config decides —
    and when it cannot, both are tried rather than one being guessed.
    """
    band_split = {"band_split.to_features.0.0.gamma": torch.zeros(1)}

    assert candidate_architectures(band_split, {"num_bands": 60}) == [
        "mel_band_roformer"
    ]
    assert candidate_architectures(band_split, {"freqs_per_bands": [2, 2]}) == [
        "bs_roformer"
    ]
    assert candidate_architectures(band_split, {}) == [
        "bs_roformer",
        "mel_band_roformer",
    ]


def test_import_infers_verifies_and_registers(tmp_path: Path) -> None:
    """
    The whole path: a community checkpoint becomes a Safetensors artifact that
    describes itself and an entry the registry accepts — with the architecture
    inferred, never stated.
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    artifact = tmp_path / "imported.safetensors"

    entry, summary = import_checkpoint(
        checkpoint,
        artifact,
        config_path=config_path,
        license_label="see upstream model card",
    )

    assert summary["architecture"] == "scnet_masked", "masked variant inferred"
    assert summary["tensors"] > 0
    assert entry["architecture"] == "scnet_masked"
    assert entry["sources"] == _STEMS
    assert entry["samplerate"] == 8000
    assert entry["segment_samples"] == 4096
    assert entry["license"] == "see upstream model card"
    assert entry["checkpoint"] == {"format": "safetensors", "path": str(artifact)}

    # The artifact carries its own description, covered by its own hash.
    embedded = read_embedded_fields(artifact)
    assert embedded["architecture"] == "scnet_masked"
    assert embedded["sources"] == _STEMS
    assert embedded["config"] == _scnet_config()

    models_file = tmp_path / "models.json"
    register_entry(models_file, "community_scnet", entry)
    repo = ModelRepository(extra_models=models_file)
    assert "community_scnet" in repo.list_models()
    model = repo.get_model("community_scnet")
    assert model.sources == _STEMS


def test_a_registered_import_needs_no_entry_fields_beyond_its_path(
    tmp_path: Path,
) -> None:
    """
    Because the artifact describes itself, a hand-written entry only has to say
    where the file is.
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    artifact = tmp_path / "imported.safetensors"
    import_checkpoint(checkpoint, artifact, config_path=config_path)

    models_file = tmp_path / "models.json"
    models_file.write_text(
        json.dumps(
            {
                "models": {
                    "minimal": {
                        "sources": _STEMS,
                        "checkpoint": {
                            "format": "safetensors",
                            "path": str(artifact),
                        },
                    }
                }
            }
        )
    )

    repo = ModelRepository(extra_models=models_file)
    assert repo.list_models()["minimal"]["backend"] == "scnet"
    assert repo._members["minimal"][0]["architecture"] == "scnet_masked"


def test_a_mislabelled_architecture_is_caught_by_loading(tmp_path: Path) -> None:
    """
    An explicit architecture is verified, not trusted: the masked and plain
    SCNets differ by weights that would otherwise be silently dropped.
    """
    checkpoint, config_path = _community_checkpoint(tmp_path, masked=True)

    with pytest.raises(ValidationError, match="does not load as scnet"):
        import_checkpoint(
            checkpoint,
            tmp_path / "imported.safetensors",
            config_path=config_path,
            architecture="scnet",
        )
    assert not (tmp_path / "imported.safetensors").exists(), (
        "nothing should be written when verification fails"
    )


def test_missing_fields_are_reported_together(tmp_path: Path) -> None:
    """
    Without a config, the import says exactly what it still needs.
    """
    checkpoint, _ = _community_checkpoint(tmp_path)

    with pytest.raises(ValidationError, match="Missing sources, samplerate"):
        import_checkpoint(checkpoint, tmp_path / "imported.safetensors")


def test_unrecognisable_weights_say_so(tmp_path: Path) -> None:
    """
    Weights from an architecture Unblend does not implement fail with the
    reason, not a load error from a random candidate.
    """
    checkpoint = tmp_path / "mdx.ckpt"
    torch.save(
        {"stft.window": torch.zeros(4), "conv.weight": torch.zeros(4)}, checkpoint
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "audio": {"chunk_size": 44100, "sample_rate": 44100},
                "model": {"dim": 16},
                "training": {"instruments": ["vocals", "other"]},
            }
        )
    )

    with pytest.raises(ValidationError, match="do not resemble any architecture"):
        import_checkpoint(
            checkpoint, tmp_path / "imported.safetensors", config_path=config_path
        )


def test_register_entry_refuses_to_overwrite(tmp_path: Path) -> None:
    """
    Re-importing under a name already in the file is refused, not merged.
    """
    models_file = tmp_path / "models.json"
    register_entry(models_file, "a", {"architecture": "scnet"})
    assert json.loads(models_file.read_text())["models"]["a"]

    with pytest.raises(ValidationError, match="already defines"):
        register_entry(models_file, "a", {"architecture": "scnet"})


def test_a_real_yaml_config_reads(tmp_path: Path) -> None:
    """
    Community configs are YAML, so that is what a config file is parsed as —
    block scalars, comments and all.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
# Trained on MUSDB18-HQ.
audio:
  chunk_size: 485100
  sample_rate: 44100
model:
  dims: [4, 32, 64, 128]
  nfft: 4096
training:
  instruments:
    - drums
    - bass
    - other
    - vocals
  target_instrument: null
"""
    )

    fields = fields_from_config(read_config(config_path))

    assert fields["sources"] == _STEMS
    assert fields["samplerate"] == 44100
    assert fields["segment_samples"] == 485100
    assert fields["config"] == {"dims": [4, 32, 64, 128], "nfft": 4096}


def test_a_yaml_models_file_registers_and_loads(tmp_path: Path) -> None:
    """
    An import lands in a YAML models file the registry can read back — the
    format users hand-edit and the format Unblend writes are the same one.
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    artifact = tmp_path / "imported.safetensors"
    entry, _ = import_checkpoint(checkpoint, artifact, config_path=config_path)

    models_file = tmp_path / "models.yaml"
    register_entry(models_file, "community_scnet", entry)
    assert "architecture: scnet_masked" in models_file.read_text()

    repo = ModelRepository(extra_models=models_file)
    assert repo.get_model("community_scnet").sources == _STEMS


def test_config_files_use_the_parser_their_name_promises(tmp_path: Path) -> None:
    """
    A ``.json`` file is read as JSON, not as YAML.
    """

    import yaml

    from unblend.importer import _dump_mapping, _load_mapping

    payload = {"models": {"a": {"sources": ["vocals", "other"], "note": "x" * 200}}}

    as_yaml = tmp_path / "models.yaml"
    as_yaml.write_text(_dump_mapping(payload, as_yaml))
    as_json = tmp_path / "models.json"
    as_json.write_text(_dump_mapping(payload, as_json))

    assert _load_mapping(as_yaml) == payload
    assert _load_mapping(as_json) == payload
    assert as_json.read_text().lstrip().startswith("{")

    broken = tmp_path / "broken.yaml"
    broken.write_text("models: [unclosed\n")
    with pytest.raises((ValueError, yaml.YAMLError)):
        _load_mapping(broken)
