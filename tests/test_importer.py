"""
Tests for ``unblend models import``: repackaging a checkpoint from elsewhere.

The import is only worth anything if it refuses to write an entry that does not
work, so most of these check that a mismatch is caught rather than recorded.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from unblend.exceptions import ModelLoadingError, ValidationError
from unblend.importer import (
    candidate_architectures,
    check_registrable,
    fields_from_config,
    import_checkpoint,
    read_config,
    read_embedded_fields,
    read_tensors,
    register_entry,
    strip_wrapper_prefix,
    unregister_entry,
)
from unblend.repo import ModelRepository

_STEMS = ["drums", "bass", "other", "vocals"]

# The smallest valid entry: two registered models, no weights of its own.
_ENSEMBLE_ENTRY = {
    "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
    "sources": _STEMS,
}


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
        num_dplayer=2,
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

    with pytest.raises(ValidationError, match="pickles Python objects"):
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
    assert entry["checkpoint"]["format"] == "safetensors"
    assert entry["checkpoint"]["path"] == str(artifact.resolve())
    assert entry["checkpoint"]["size_bytes"] == artifact.stat().st_size
    assert len(entry["checkpoint"]["sha256"]) == 64

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


def test_import_records_an_absolute_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A relative ``-o`` is resolved at import time, not against whatever
    directory the model is later loaded from.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    monkeypatch.chdir(tmp_path)
    entry, _ = import_checkpoint(
        checkpoint, Path("rel.safetensors"), config_path=config_path
    )
    assert entry["checkpoint"]["path"] == str((tmp_path / "rel.safetensors").resolve())


def test_registering_refuses_built_in_and_existing_names(tmp_path: Path) -> None:
    """
    Names are checked before anything is written; a built-in name would
    otherwise break every later ``ModelRepository()``.

    :param tmp_path: pytest temporary directory fixture
    """
    models_file = tmp_path / "models.yaml"
    with pytest.raises(ValidationError, match="built-in"):
        check_registrable(models_file, "htdemucs")
    register_entry(models_file, "mine", _ENSEMBLE_ENTRY)
    with pytest.raises(ValidationError, match="already defines"):
        check_registrable(models_file, "mine")


def test_safetensors_input_is_unprefixed_like_other_containers(tmp_path: Path) -> None:
    """
    ``model.``-prefixed keys are stripped whatever the container.

    :param tmp_path: pytest temporary directory fixture
    """
    from safetensors.torch import save_file

    path = tmp_path / "w.safetensors"
    save_file({"model.a": torch.zeros(1), "model.b": torch.zeros(1)}, str(path))
    assert sorted(read_tensors(path)) == ["a", "b"]


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
    register_entry(models_file, "a", _ENSEMBLE_ENTRY)
    assert json.loads(models_file.read_text())["models"]["a"]

    with pytest.raises(ValidationError, match="already defines"):
        register_entry(models_file, "a", _ENSEMBLE_ENTRY)


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


def test_registering_refuses_names_from_any_loaded_models_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A name already defined in an ``UNBLEND_EXTRA_MODELS`` file is refused even
    when registering into a different file.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    env_file = tmp_path / "env.json"
    env_file.write_text(
        json.dumps(
            {
                "models": {
                    "dup": {
                        "members": [{"model": "htdemucs"}, {"model": "scnet_small"}],
                        "sources": _STEMS,
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(env_file))
    with pytest.raises(ValidationError, match="already registered"):
        check_registrable(tmp_path / "other.yaml", "dup")


@pytest.mark.parametrize("name", ["../../evil", "a/b", ".hidden", "", "auto", "x..y"])
def test_model_names_that_are_paths_or_reserved_are_refused(
    tmp_path: Path, name: str
) -> None:
    """
    The name becomes a filename and a ``--model`` value, so path characters and
    ``auto`` are rejected before anything is written.

    :param tmp_path: pytest temporary directory fixture
    :param name: Rejected name.
    """
    with pytest.raises(ValidationError):
        check_registrable(tmp_path / "models.yaml", name)


def test_an_entry_that_fails_validation_is_never_written(tmp_path: Path) -> None:
    """
    The updated file is validated before it replaces the old one, which is
    kept as ``.bak`` because the rewrite drops comments.

    :param tmp_path: pytest temporary directory fixture
    """
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "# my notes\nversion: 1\nmodels:\n  a:\n"
        "    members: [{model: htdemucs}, {model: scnet_small}]\n"
        "    sources: [drums, bass, other, vocals]\n"
    )
    before = models_file.read_text()
    with pytest.raises(ValidationError, match="does not load"):
        register_entry(models_file, "broken", {"sources": _STEMS})
    assert models_file.read_text() == before
    assert not list(tmp_path.glob(".models.staged*"))

    register_entry(models_file, "b", _ENSEMBLE_ENTRY)
    assert (tmp_path / "models.yaml.bak").read_text() == before
    assert "b" in ModelRepository(extra_models=models_file).list_models()


def test_names_differing_only_by_case_are_refused(tmp_path: Path) -> None:
    """
    ``MINE`` after ``mine`` would share a weights file on case-insensitive
    filesystems, so it's refused like an exact duplicate.

    :param tmp_path: pytest temporary directory fixture
    """
    models_file = tmp_path / "models.yaml"
    register_entry(models_file, "mine", _ENSEMBLE_ENTRY)
    with pytest.raises(ValidationError):
        check_registrable(models_file, "MINE")
    with pytest.raises(ValidationError, match="built-in"):
        check_registrable(models_file, "HTDemucs")


def test_htdemucs_import_takes_stems_from_the_entry(tmp_path: Path) -> None:
    """
    An HTDemucs config without ``sources`` imports when the stems are given
    separately (``--stem``), instead of failing on a missing argument.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.htdemucs import HTDemucs

    config = dict(
        samplerate=8000, segment=1.0, nfft=512, depth=2, channels=16, t_layers=1
    )
    checkpoint = tmp_path / "demucs.th"
    torch.save(HTDemucs(sources=["a", "b"], **config).state_dict(), checkpoint)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n" + "".join(f"  {k}: {v}\n" for k, v in config.items())
    )
    entry, summary = import_checkpoint(
        checkpoint,
        tmp_path / "out.safetensors",
        config_path=config_path,
        architecture="htdemucs",
        sources=["a", "b"],
        samplerate=8000,
        segment_samples=8000,
    )
    assert summary["architecture"] == "htdemucs"
    assert entry["config"]["sources"] == ["a", "b"]


def test_msst_htdemucs_config_imports_and_is_self_describing(tmp_path: Path) -> None:
    """
    MSST's HTDemucs layout (``model: htdemucs`` plus an ``htdemucs:``
    section, geometry under ``training``) imports without extra flags, and
    the artifact alone rebuilds the model.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.htdemucs import HTDemucs

    arch = dict(nfft=512, depth=2, channels=16, t_layers=1)
    checkpoint = tmp_path / "demucs.th"
    torch.save(
        HTDemucs(sources=["a", "b"], samplerate=8000, segment=2, **arch).state_dict(),
        checkpoint,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model: htdemucs\n"
        "training:\n  instruments: [a, b]\n  samplerate: 8000\n  segment: 2\n"
        "htdemucs:\n" + "".join(f"  {k}: {v}\n" for k, v in arch.items())
    )
    artifact = tmp_path / "out.safetensors"
    entry, summary = import_checkpoint(checkpoint, artifact, config_path=config_path)
    assert summary["architecture"] == "htdemucs"
    assert entry["segment_samples"] == 16000
    assert entry["config"]["samplerate"] == 8000
    assert entry["config"]["segment"] == 2

    rebuilt = tmp_path / "again.safetensors"
    again, _ = import_checkpoint(artifact, rebuilt)
    assert again["config"] == entry["config"]


def test_explicit_geometry_reaches_the_htdemucs_config(tmp_path: Path) -> None:
    """
    ``--samplerate``/``--segment-samples`` override the config's own
    HTDemucs ``samplerate``/``segment``.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.htdemucs import HTDemucs

    config = dict(
        samplerate=8000, segment=1, nfft=512, depth=2, channels=16, t_layers=1
    )
    checkpoint = tmp_path / "demucs.th"
    torch.save(HTDemucs(sources=["a", "b"], **config).state_dict(), checkpoint)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n" + "".join(f"  {k}: {v}\n" for k, v in config.items())
    )
    entry, _ = import_checkpoint(
        checkpoint,
        tmp_path / "out.safetensors",
        config_path=config_path,
        sources=["a", "b"],
        samplerate=16000,
        segment_samples=24000,
    )
    assert entry["config"]["samplerate"] == 16000
    assert entry["config"]["segment"] == 1.5


def test_msst_configs_with_python_tuples_and_training_keys(tmp_path: Path) -> None:
    """
    Music-Source-Separation-Training configs use ``!!python/tuple`` and carry
    training-only keys; both are handled rather than rejected.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.importer import constructor_config

    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  dim: 8\n  flash_attn: true\n  bands: !!python/tuple [1, 2]\n"
    )
    config = read_config(path)["model"]
    assert config["bands"] == [1, 2]
    kept, dropped = constructor_config("mel_band_roformer", config)
    assert "flash_attn" in dropped and kept["dim"] == 8


def test_tensors_sharing_storage_are_written(tmp_path: Path) -> None:
    """
    RoFormer checkpoints alias tensors (the rotary ``freqs``); safetensors
    refuses shared storage, so the artifact writer copies repeats.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.importer import write_artifact

    shared = torch.arange(4.0)
    state = {"a.freqs": shared, "b.freqs": shared}
    fields = {
        "architecture": "x",
        "sources": ["a"],
        "samplerate": 1,
        "segment_samples": 1,
        "config": {},
    }
    write_artifact(state, tmp_path / "w.safetensors", fields)
    assert sorted(read_tensors(tmp_path / "w.safetensors")) == ["a.freqs", "b.freqs"]


def test_write_artifact_never_replaces_an_existing_file(tmp_path: Path) -> None:
    """
    A second import racing for the same output path fails without touching
    the first import's weights, and leaves no partial file behind.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.exceptions import ValidationError
    from unblend.importer import write_artifact

    path = tmp_path / "mine.safetensors"
    write_artifact({"w": torch.ones(2)}, path, {})
    before = path.read_bytes()
    with pytest.raises(ValidationError, match="already exists"):
        write_artifact({"w": torch.zeros(2)}, path, {})
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["mine.safetensors"]


def test_a_broken_default_models_file_does_not_block_other_files(
    tmp_path: Path, _isolate_default_models_file: Path
) -> None:
    """
    The registry skips a broken default models file, so registering into and
    unregistering from another file must not trip over it either.

    :param tmp_path: pytest temporary directory fixture
    :param _isolate_default_models_file: the substituted default file path
    """
    _isolate_default_models_file.write_text("version: 1\nmodels: [not, a, mapping]\n")
    models_file = tmp_path / "models.yaml"
    with pytest.warns(UserWarning, match="Ignoring"):
        register_entry(models_file, "mine", _ENSEMBLE_ENTRY)
        unregister_entry(models_file, "mine")


def test_import_rejects_non_positive_geometry(tmp_path: Path) -> None:
    """
    A negative ``--samplerate`` is refused at import, for HTDemucs too,
    instead of registering an entry that fails at separation time.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.exceptions import ValidationError
    from unblend.htdemucs import HTDemucs

    config = dict(
        samplerate=8000, segment=1, nfft=512, depth=2, channels=16, t_layers=1
    )
    checkpoint = tmp_path / "demucs.th"
    torch.save(HTDemucs(sources=["a", "b"], **config).state_dict(), checkpoint)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n" + "".join(f"  {k}: {v}\n" for k, v in config.items())
    )
    with pytest.raises(ValidationError, match="samplerate"):
        import_checkpoint(
            checkpoint,
            tmp_path / "out.safetensors",
            config_path=config_path,
            sources=["a", "b"],
            samplerate=-44100,
            segment_samples=8000,
        )


def test_htdemucs_entry_geometry_must_match_its_config(tmp_path: Path) -> None:
    """
    An HTDemucs entry whose top-level samplerate contradicts its config is
    refused: the config is what builds the model.

    :param tmp_path: pytest temporary directory fixture
    """
    entry = dict(ModelRepository(extra_models=[]).list_models()["htdemucs"])
    entry.pop("backend", None)
    entry["samplerate"] = 48000
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"models": {"mine": entry}}))
    with pytest.raises(ModelLoadingError, match="config gives"):
        ModelRepository(extra_models=path)


def test_import_refuses_an_onnx_file(tmp_path: Path) -> None:
    """
    An ``.onnx`` graph is refused with a clear message instead of torch.load's
    advice to retry with ``weights_only=False``.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.exceptions import ValidationError

    path = tmp_path / "UVR-MDX-NET.onnx"
    path.write_bytes(b"\x08\x07")
    with pytest.raises(ValidationError, match="ONNX graph"):
        read_tensors(path)


def test_import_runs_the_model_before_registering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A config that keeps every weight shape but breaks the forward fails at
    import, not at the first separation.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    import unblend.importer as importer
    from unblend.exceptions import ValidationError

    def broken(*_args: object, **_kwargs: object) -> None:
        """
        Stand in for a forward that shape-mismatches.
        """
        raise RuntimeError("The size of tensor a (18) must match (34)")

    monkeypatch.setattr("unblend.apply.apply_model", broken)
    checkpoint, config_path = _community_checkpoint(tmp_path)
    with pytest.raises(ValidationError, match="doesn't run"):
        importer.import_checkpoint(
            checkpoint, tmp_path / "out.safetensors", config_path=config_path
        )
    assert not (tmp_path / "out.safetensors").exists()


def test_rewrites_and_imports_keep_normal_file_modes(tmp_path: Path) -> None:
    """
    Registering keeps a shared models file's 0644 (not mkstemp's 0600), and
    imported weights get the umask's mode rather than safetensors' 0600.

    :param tmp_path: pytest temporary directory fixture
    """
    import stat

    from unblend.importer import write_artifact

    models_file = tmp_path / "models.yaml"
    models_file.write_text("version: 1\nmodels: {}\n")
    os.chmod(models_file, 0o644)
    register_entry(models_file, "mine", _ENSEMBLE_ENTRY)
    assert stat.S_IMODE(models_file.stat().st_mode) == 0o644

    umask = os.umask(0o022)
    try:
        artifact = tmp_path / "w.safetensors"
        write_artifact({"w": torch.ones(2)}, artifact, {})
    finally:
        os.umask(umask)
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o644


def test_a_symlinked_models_file_is_updated_through_the_link(tmp_path: Path) -> None:
    """
    Registering into and unregistering from a symlinked models file edits the
    file it points at and keeps the link, instead of replacing the link.

    :param tmp_path: pytest temporary directory fixture
    """
    real = tmp_path / "dotfiles" / "models.yaml"
    real.parent.mkdir()
    real.write_text("version: 1\nmodels: {}\n")
    link = tmp_path / "models.yaml"
    link.symlink_to(real)
    register_entry(link, "mine", _ENSEMBLE_ENTRY)
    assert link.is_symlink() and "mine" in real.read_text()
    unregister_entry(link, "mine")
    assert link.is_symlink() and "mine" not in real.read_text()


def _scnet_entry() -> dict:
    """
    A registrable single-checkpoint entry reusing ``scnet_small``'s weights.

    :return: A registry entry.
    """
    base = ModelRepository(extra_models=[]).list_models()["scnet_small"]
    return {k: v for k, v in base.items() if k != "backend"}


def test_unregister_refuses_to_break_an_ensemble_in_the_default_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    A model in a listed file that an ensemble in the default file uses can't
    be unregistered: the default file loads today, so it must still load
    afterwards (it used to be dropped silently from then on).

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param _isolate_default_models_file: the substituted default file path
    """
    from unblend.exceptions import ValidationError

    listed = tmp_path / "x.json"
    listed.write_text(
        json.dumps({"models": {"sc2": _scnet_entry(), "sc3": _scnet_entry()}})
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    _isolate_default_models_file.write_text(
        json.dumps(
            {
                "models": {
                    "ens": {
                        "sources": _STEMS,
                        "members": [{"model": "sc2"}, {"model": "sc3"}],
                    }
                }
            }
        )
    )
    assert "ens" in ModelRepository().list_models()
    with pytest.raises(ValidationError, match="don't load"):
        unregister_entry(listed, "sc2")
    assert "sc2" in json.loads(listed.read_text())["models"]


def test_a_listed_ensemble_may_use_models_from_the_default_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_default_models_file: Path,
) -> None:
    """
    A listed file whose ensemble references models defined in the default
    file loads, as api.md promises for any registered model.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param _isolate_default_models_file: the substituted default file path
    """
    _isolate_default_models_file.write_text(
        json.dumps({"models": {"sc2": _scnet_entry(), "sc3": _scnet_entry()}})
    )
    listed = tmp_path / "y.json"
    listed.write_text(
        json.dumps(
            {
                "models": {
                    "ens2": {
                        "sources": _STEMS,
                        "members": [{"model": "sc2"}, {"model": "sc3"}],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("UNBLEND_EXTRA_MODELS", str(listed))
    models = ModelRepository().list_models()
    assert {"sc2", "sc3", "ens2"} <= set(models)


def test_relative_paths_in_a_symlinked_models_file_resolve_consistently(
    tmp_path: Path,
) -> None:
    """
    Relative ``path:`` entries in a symlinked models file resolve against the
    real file's folder both at runtime and when register/unregister validate
    an edit, so editing the file doesn't falsely refuse an untouched entry.

    :param tmp_path: pytest temporary directory fixture
    """
    from safetensors.torch import save_file

    from unblend.htdemucs import HTDemucs

    real = tmp_path / "dotfiles" / "models.yaml"
    (real.parent / "w").mkdir(parents=True)
    config = dict(
        sources=["a", "b"],
        samplerate=8000,
        segment=1,
        nfft=512,
        depth=2,
        channels=16,
        t_layers=1,
    )
    save_file(HTDemucs(**config).state_dict(), str(real.parent / "w" / "t.safetensors"))
    real.write_text(
        json.dumps(
            {
                "models": {
                    "tiny": {
                        "architecture": "htdemucs",
                        "sources": ["a", "b"],
                        "config": config,
                        "checkpoint": {
                            "format": "safetensors",
                            "path": "w/t.safetensors",
                        },
                    }
                }
            }
        )
    )
    link = tmp_path / "home" / "models.yaml"
    link.parent.mkdir()
    link.symlink_to(real)
    ModelRepository(extra_models=link).get_model("tiny")  # loads through the link
    register_entry(link, "ens", _ENSEMBLE_ENTRY)
    unregister_entry(link, "ens")
    assert link.is_symlink()


def test_an_unwritable_models_folder_is_a_clear_error(tmp_path: Path) -> None:
    """
    Registering into a models file whose folder can't be written raises
    ``ValidationError`` (which the CLI reports) instead of a raw OSError.

    :param tmp_path: pytest temporary directory fixture
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores permission bits")

    from unblend.exceptions import ValidationError

    folder = tmp_path / "ro"
    folder.mkdir()
    os.chmod(folder, 0o555)
    try:
        with pytest.raises(ValidationError, match="Could not update"):
            register_entry(folder / "models.yaml", "mine", _ENSEMBLE_ENTRY)
    finally:
        os.chmod(folder, 0o755)


def test_importing_into_a_broken_models_file_fails_before_conversion(
    tmp_path: Path,
) -> None:
    """
    A models file that already doesn't load is reported up front, before an
    import spends time converting weights, instead of blaming the new entry.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.exceptions import ValidationError

    models_file = tmp_path / "models.json"
    models_file.write_text(
        json.dumps(
            {
                "models": {
                    "bad": {
                        "sources": ["a"],
                        "members": [{"model": "nope"}, {"model": "htdemucs"}],
                    }
                }
            }
        )
    )
    with pytest.raises(ValidationError, match="doesn't load as it is"):
        check_registrable(models_file, "mine")


def test_a_folder_at_the_models_file_path_is_refused_up_front(tmp_path: Path) -> None:
    """
    A folder where the models file should be is refused before any
    conversion, not after it.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.exceptions import ValidationError

    (tmp_path / "dirmf").mkdir()
    with pytest.raises(ValidationError, match="not a file"):
        check_registrable(tmp_path / "dirmf", "mine")


def test_a_name_in_a_skipped_default_file_is_refused(
    tmp_path: Path, _isolate_default_models_file: Path
) -> None:
    """
    A name the (currently skipped, broken) default file defines can't be
    imported elsewhere: once the default is fixed, the clash would make the
    registry drop the whole file.

    :param tmp_path: pytest temporary directory fixture
    :param _isolate_default_models_file: the substituted default file path
    """
    from unblend.exceptions import ValidationError

    _isolate_default_models_file.write_text(
        json.dumps({"models": {"tinya": {"sources": ["a"], "architecture": "nope"}}})
    )
    with pytest.raises(ValidationError, match="already defines a model named"):
        check_registrable(tmp_path / "other.yaml", "TinyA")


def test_check_registrable_refuses_a_models_file_name_too_long_for_its_staging_copy(
    tmp_path: Path,
) -> None:
    """
    The models file is rewritten through a sibling ``.STEM.XXXXXXXX`` copy;
    a name with no room for it is refused before any conversion.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend._paths import NAME_MAX
    from unblend.importer import check_registrable

    models_file = tmp_path / ("m" * (NAME_MAX - 5 - 5) + ".yaml")
    with pytest.raises(ValidationError, match="too long"):
        check_registrable(models_file, "fresh")


def test_register_entry_refuses_a_models_file_that_is_a_symlink_loop(
    tmp_path: Path,
) -> None:
    """
    ``realpath`` stops at a loop and returns the link; writing there would
    replace the link, so it's refused.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.importer import register_entry

    (tmp_path / "a.yaml").symlink_to(tmp_path / "b.yaml")
    (tmp_path / "b.yaml").symlink_to(tmp_path / "a.yaml")
    with pytest.raises(ValidationError, match="symlink loop"):
        register_entry(tmp_path / "a.yaml", "x", {"sources": ["a", "b"]})
    assert (tmp_path / "a.yaml").is_symlink()


def test_check_registrable_refuses_a_symlink_loop_before_conversion(
    tmp_path: Path,
) -> None:
    """
    The pre-conversion check refuses a models file that is a symlink loop,
    so an import doesn't convert first and fail at registration.

    :param tmp_path: pytest temporary directory fixture
    """
    from unblend.importer import check_registrable

    (tmp_path / "a.yaml").symlink_to(tmp_path / "b.yaml")
    (tmp_path / "b.yaml").symlink_to(tmp_path / "a.yaml")
    with pytest.raises(ValidationError, match="symlink loop"):
        check_registrable(tmp_path / "a.yaml", "fresh")


def test_a_load_failure_message_is_kept_short(tmp_path: Path) -> None:
    """
    A strict load that doesn't fit lists every mismatched key; the import
    error keeps the start of each category's line and says how much was cut.
    """
    checkpoint, config_path = _community_checkpoint(tmp_path, masked=True)
    with pytest.raises(ValidationError) as info:
        import_checkpoint(
            checkpoint,
            tmp_path / "imported.safetensors",
            config_path=config_path,
            architecture="htdemucs",
        )
    message = str(info.value)
    assert "does not load as htdemucs" in message
    # Every category of torch's report survives the cut, each shortened.
    assert "Missing key(s)" in message and "Unexpected key(s)" in message
    assert "more characters" in message
    assert len(message) < 3000


def test_a_zero_samplerate_is_invalid_not_missing(tmp_path: Path) -> None:
    """
    ``--samplerate 0`` was given, so it is reported as invalid rather than
    "Missing samplerate".
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    with pytest.raises(ValidationError, match="samplerate must be a positive integer"):
        import_checkpoint(
            checkpoint,
            tmp_path / "imported.safetensors",
            config_path=config_path,
            samplerate=0,
        )


def _rewrite_config(config_path: Path, **changes: dict) -> None:
    """
    Update sections of a training config written by ``_community_checkpoint``.

    :param config_path: The JSON config.
    :param changes: Section name to the keys to set in it.
    """
    config = json.loads(config_path.read_text())
    for section, values in changes.items():
        config[section].update(values)
    config_path.write_text(json.dumps(config))


def test_a_wrong_typed_config_value_is_invalid_not_missing(tmp_path: Path) -> None:
    """
    ``sample_rate: 44100.0`` was given, so it is reported as invalid rather
    than "Missing samplerate"; an explicit null falls through to the next
    place a rate can be given.
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    _rewrite_config(config_path, audio={"sample_rate": 8000.0})
    with pytest.raises(ValidationError, match="samplerate must be a positive integer"):
        import_checkpoint(
            checkpoint, tmp_path / "a.safetensors", config_path=config_path
        )
    _rewrite_config(
        config_path, audio={"sample_rate": None}, training={"samplerate": 8000}
    )
    import_checkpoint(checkpoint, tmp_path / "b.safetensors", config_path=config_path)


def test_duplicate_stems_from_a_config_are_refused_before_converting(
    tmp_path: Path,
) -> None:
    """
    Repeated instruments in a config are refused before anything is written.

    :param tmp_path: pytest temporary directory fixture
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    stems = json.loads(config_path.read_text())["training"]["instruments"]
    _rewrite_config(config_path, training={"instruments": [stems[0]] * len(stems)})
    with pytest.raises(ValidationError, match="unique"):
        import_checkpoint(
            checkpoint, tmp_path / "c.safetensors", config_path=config_path
        )
    assert not (tmp_path / "c.safetensors").exists()


def test_an_embedded_header_of_the_wrong_shape_is_ignored(tmp_path: Path) -> None:
    """
    A hand-edited header whose ``config`` isn't a mapping (or ``sources`` a
    list) yields no fields rather than crashing later.

    :param tmp_path: pytest temporary directory fixture
    """
    from safetensors.torch import save_file

    from unblend.importer import EMBEDDED_FORMAT, read_embedded_fields

    for key, value in (("config", "[1, 2]"), ("sources", '"dbov"')):
        path = tmp_path / f"{key}.safetensors"
        save_file(
            {"x": torch.zeros(1)},
            str(path),
            metadata={"unblend_format": EMBEDDED_FORMAT, key: value},
        )
        assert read_embedded_fields(path) == {}


@pytest.mark.parametrize(
    "stems, problem",
    [
        (["a/b", "c", "d", "e"], "safe as filenames"),
        (["", "c", "d", "e"], "non-empty"),
        (["Drums", "drums", "d", "e"], "ignoring case"),
        (["\u00e9", "e\u0301", "d", "e"], "ignoring case"),
        (["\u0390", "\u0399\u0308\u0301", "d", "e"], "ignoring case"),
    ],
)
def test_stems_the_registry_would_refuse_are_caught_before_converting(
    tmp_path: Path, stems: list[str], problem: str
) -> None:
    """
    The registry's stem-name rules are checked before anything is written.

    :param tmp_path: pytest temporary directory fixture
    :param stems: Instruments to put in the config.
    :param problem: Expected part of the error.
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    count = len(json.loads(config_path.read_text())["training"]["instruments"])
    _rewrite_config(config_path, training={"instruments": stems[:count]})
    with pytest.raises(ValidationError, match=problem):
        import_checkpoint(
            checkpoint, tmp_path / "s.safetensors", config_path=config_path
        )
    assert not (tmp_path / "s.safetensors").exists()


def test_a_header_with_non_string_sources_is_ignored(tmp_path: Path) -> None:
    """
    ``sources`` must be a list of strings; anything else in a hand-edited
    header yields no fields, not a crash in later checks.

    :param tmp_path: pytest temporary directory fixture
    """
    from safetensors.torch import save_file

    from unblend.importer import EMBEDDED_FORMAT, read_embedded_fields

    for value in ('[["a"], ["b"]]', "[1, 1]"):
        path = tmp_path / "h.safetensors"
        save_file(
            {"x": torch.zeros(1)},
            str(path),
            metadata={"unblend_format": EMBEDDED_FORMAT, "sources": value},
        )
        assert read_embedded_fields(path) == {}


def test_a_segment_in_seconds_with_a_bad_rate_is_invalid_not_missing(
    tmp_path: Path,
) -> None:
    """
    ``training.segment`` seconds with a float samplerate reports the bad rate,
    not "Missing segment_samples".

    :param tmp_path: pytest temporary directory fixture
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    config["audio"].pop("chunk_size")
    config["audio"].pop("sample_rate")
    config["training"].update({"samplerate": 8000.0, "segment": 1})
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValidationError, match="samplerate must be a positive integer"):
        import_checkpoint(
            checkpoint, tmp_path / "t.safetensors", config_path=config_path
        )


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), 1e308])
def test_a_non_finite_segment_in_seconds_is_invalid_not_a_crash(
    tmp_path: Path, seconds: float
) -> None:
    """
    ``training.segment: .nan`` (or ``.inf``, or a length whose sample count
    overflows) is reported as invalid rather than raising from the
    seconds-to-samples conversion.

    :param tmp_path: pytest temporary directory fixture
    :param seconds: The non-finite segment.
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    config["audio"].pop("chunk_size")
    config["training"]["segment"] = seconds
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValidationError, match="segment_samples must be a positive"):
        import_checkpoint(
            checkpoint, tmp_path / "n.safetensors", config_path=config_path
        )


def test_a_segment_in_seconds_uses_a_rate_given_outside_the_config(
    tmp_path: Path,
) -> None:
    """
    With no rate in the config, ``training.segment`` seconds are converted at
    the explicitly given rate, not stored as a sample count.

    :param tmp_path: pytest temporary directory fixture
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    config["audio"].pop("chunk_size")
    config["audio"].pop("sample_rate")
    config["training"]["segment"] = 0.512
    config_path.write_text(json.dumps(config))
    entry, _ = import_checkpoint(
        checkpoint,
        tmp_path / "r.safetensors",
        config_path=config_path,
        samplerate=8000,
    )
    assert entry["segment_samples"] == 4096


def test_empty_instruments_fall_through_to_top_level_sources() -> None:
    """
    An empty ``training.instruments`` counts as not given, like null.
    """
    fields = fields_from_config(
        {"model": {"dim": 1}, "training": {"instruments": []}, "sources": _STEMS}
    )
    assert fields["sources"] == _STEMS


def test_non_string_instruments_are_refused_not_stringified(tmp_path: Path) -> None:
    """
    ``instruments: [null, x]`` is refused rather than becoming a stem named
    ``'None'``.

    :param tmp_path: pytest temporary directory fixture
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    count = len(json.loads(config_path.read_text())["training"]["instruments"])
    _rewrite_config(config_path, training={"instruments": [None] + ["x"] * (count - 1)})
    with pytest.raises(ValidationError, match="non-empty sources"):
        import_checkpoint(
            checkpoint, tmp_path / "u.safetensors", config_path=config_path
        )


@pytest.mark.parametrize("seconds, rate", [(10**400, 8000), (1.5, 10**400)])
def test_seconds_too_large_for_a_float_are_kept_not_a_crash(
    seconds: object, rate: int
) -> None:
    """
    Integers too large to multiply as floats are kept for validation to
    report, like a float overflow.

    :param seconds: The segment in seconds.
    :param rate: The sample rate.
    """
    from unblend.importer import _seconds_to_samples

    assert _seconds_to_samples(seconds, rate) == seconds


def test_a_capitalised_target_instrument_gets_the_right_complement() -> None:
    """
    ``target_instrument: Other`` pairs with ``vocals``, not a second "other".
    """
    fields = fields_from_config(
        {"model": {"dim": 1}, "training": {"target_instrument": "Other"}}
    )
    assert fields["sources"] == ["Other", "vocals"]


def test_a_config_nested_too_deeply_is_a_clean_error(tmp_path: Path) -> None:
    """
    A config the parser can't recurse through is reported, not a traceback.

    :param tmp_path: pytest temporary directory fixture
    """
    path = tmp_path / "deep.json"
    path.write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(ValidationError, match="nested too deeply"):
        read_config(path)


def test_config_seconds_override_a_header_segment(tmp_path: Path) -> None:
    """
    Re-importing an artifact with a config whose ``training.segment`` is in
    seconds converts it at the header's rate, replacing the header's length.

    :param tmp_path: pytest temporary directory fixture
    """
    checkpoint, config_path = _community_checkpoint(tmp_path)
    artifact = tmp_path / "first.safetensors"
    import_checkpoint(checkpoint, artifact, config_path=config_path)
    config = json.loads(config_path.read_text())
    config["audio"].pop("chunk_size")
    config["audio"].pop("sample_rate")
    config["training"]["segment"] = 0.256
    config_path.write_text(json.dumps(config))
    entry, _ = import_checkpoint(
        artifact, tmp_path / "second.safetensors", config_path=config_path
    )
    assert entry["segment_samples"] == 2048


def test_an_empty_checkpoint_says_so(tmp_path: Path) -> None:
    """
    A zero-byte ``.ckpt`` is reported as empty or truncated, not with a bare
    ``EOFError`` and a hint about pickled objects.

    :param tmp_path: pytest temporary directory fixture
    """
    empty = tmp_path / "zero.ckpt"
    empty.touch()
    with pytest.raises(ValidationError, match="empty or truncated"):
        read_tensors(empty)


def test_a_truncated_checkpoint_is_reported_as_possibly_truncated(
    tmp_path: Path,
) -> None:
    """
    A partial download is reported as possibly truncated, not only as a
    checkpoint that pickles objects.

    :param tmp_path: pytest temporary directory fixture
    """
    whole = tmp_path / "whole.ckpt"
    torch.save({"w": torch.zeros(64, 64)}, whole)
    partial = tmp_path / "partial.ckpt"
    partial.write_bytes(whole.read_bytes()[: whole.stat().st_size // 2])
    with pytest.raises(ValidationError, match="truncated or corrupt"):
        read_tensors(partial)


def test_load_failures_keep_only_the_informative_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    torch's advice to retry with ``weights_only=False`` isn't passed on; the
    refused global is named, and an official Demucs checkpoint gets a pointer
    to the built-in models (only HTDemucs; Unblend ships no HDemucs).

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    import sys
    import types

    module = types.ModuleType("demucs.htdemucs")

    class HTDemucs:
        pass

    HTDemucs.__module__ = "demucs.htdemucs"
    HTDemucs.__qualname__ = "HTDemucs"
    module.HTDemucs = HTDemucs
    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(sys.modules, "demucs.htdemucs", module)
    official = tmp_path / "955717e8.th"
    torch.save({"klass": HTDemucs}, official)
    with pytest.raises(ValidationError) as info:
        read_tensors(official)
    message = str(info.value)
    assert "GLOBAL demucs.htdemucs.HTDemucs" in message
    assert "built-in models" in message
    assert "weights_only" not in message

    other = types.ModuleType("demucs.hdemucs")

    class HDemucs:
        pass

    HDemucs.__module__ = "demucs.hdemucs"
    HDemucs.__qualname__ = "HDemucs"
    other.HDemucs = HDemucs
    monkeypatch.setitem(sys.modules, "demucs.hdemucs", other)
    unshipped = tmp_path / "75fc33f5.th"
    torch.save({"klass": HDemucs}, unshipped)
    with pytest.raises(ValidationError) as info:
        read_tensors(unshipped)
    assert "GLOBAL demucs.hdemucs.HDemucs" in str(info.value)
    assert "built-in models" not in str(info.value)

    page = tmp_path / "page.ckpt"
    page.write_text("<html></html>")
    with pytest.raises(ValidationError, match=r"\(Unsupported operand 60\)") as info:
        read_tensors(page)
    assert "weights_only" not in str(info.value)


def _chained(outer: str, inner: Exception | None) -> Exception:
    """
    An exception raised the way ``torch.load`` rewraps a weights-only
    refusal: ``outer``, with ``inner`` as its context.

    :param outer: The rewrapped message.
    :param inner: The unpickler's own error, or None.
    :return: The exception.
    """
    import pickle

    try:
        try:
            if inner is not None:
                raise inner
            raise RuntimeError(outer)
        except Exception:
            if inner is None:
                raise
            raise pickle.UnpicklingError(outer) from None
    except Exception as exc:
        return exc


_WO = "Weights only load failed. Re-running with weights_only=False will succeed."


@pytest.mark.parametrize(
    "exc, reason",
    [
        (
            _chained(
                _WO,
                RuntimeError(
                    "Unsupported global: GLOBAL a.B was not an allowed global by "
                    "default. Please use `torch.serialization.add_safe_globals`."
                ),
            ),
            "Unsupported global: GLOBAL a.B",
        ),
        (
            _chained(
                _WO,
                RuntimeError(
                    "Trying to load unsupported GLOBAL posix.system whose module "
                    "posix is blocked."
                ),
            ),
            "Trying to load unsupported GLOBAL posix.system whose module posix "
            "is blocked",
        ),
        (
            _chained(
                "PytorchStreamReader failed reading zip archive: failed finding "
                "central directory. This is an internal miniz error.",
                None,
            ),
            "PytorchStreamReader failed reading zip archive: failed finding "
            "central directory",
        ),
        (
            _chained(
                "[enforce fail at inline_container.cc:180] . file in archive is "
                "not in a subdirectory: readme.txt",
                None,
            ),
            "file in archive is not in a subdirectory: readme.txt",
        ),
        (
            _chained(
                _WO,
                RuntimeError(
                    "Unsupported global: GLOBAL x\x1bc\x07y.Z was not an allowed "
                    "global by default."
                ),
            ),
            "Unsupported global: GLOBAL xcy.Z",
        ),
        (_chained("", None), "unreadable"),
    ],
)
def test_load_failure_reason_keeps_the_informative_line(
    exc: Exception, reason: str
) -> None:
    """
    The unpickler's own error (not torch's retry advice), or the first
    sentence of any other failure.

    :param exc: The exception ``torch.load`` raised.
    :param reason: The line kept.
    """
    from unblend.importer import _load_failure_reason

    assert _load_failure_reason(exc) == reason


def test_a_pickle_that_calls_a_blocked_module_is_named(tmp_path: Path) -> None:
    """
    A checkpoint whose pickle calls ``os.system`` is refused without running
    it, and the message names the blocked call.

    :param tmp_path: pytest temporary directory fixture
    """

    class _Payload:
        def __reduce__(self) -> tuple:
            return (os.system, ("exit 7",))

    path = tmp_path / "evil.ckpt"
    torch.save({"w": _Payload()}, path)
    with pytest.raises(ValidationError, match="is blocked"):
        read_tensors(path)
