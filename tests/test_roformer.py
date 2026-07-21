"""
Structural and behavioural checks for the RoFormer backend
(``unblend.roformer``): BS-RoFormer and Mel-Band RoFormer.

These run fully offline with tiny models. Numerical parity against the
reference lucidrains/ZFTurbo implementations is verified out-of-band (it
requires their einops/librosa/rotary-embedding-torch/beartype deps); what we
guard here is the checkpoint-compat contract (strict ``state_dict`` round-trip
with zero missing/unexpected keys), the forward output shape, the
mixture-complement convention, and that both flow through the shared
``apply_model`` engine exactly like Demucs.
"""

import pytest
import torch

from unblend.apply import apply_model, apply_model_multi
from unblend.exceptions import ValidationError
from unblend.roformer import BSRoformer, MelBandRoformer, build_roformer

SR = 44100


def _bs(**overrides):
    """
    Build a tiny BS-RoFormer for tests.

    :param overrides: Constructor kwargs overriding the small defaults.
    :return: A ``BSRoformer`` instance.
    """
    kwargs = dict(
        dim=32,
        depth=2,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=16,
        heads=2,
    )
    kwargs.update(overrides)
    return BSRoformer(**kwargs)


def _mel(**overrides):
    """
    Build a tiny Mel-Band RoFormer for tests.

    :param overrides: Constructor kwargs overriding the small defaults.
    :return: A ``MelBandRoformer`` instance.
    """
    kwargs = dict(
        dim=32,
        depth=2,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        num_bands=60,
        dim_head=16,
        heads=2,
    )
    kwargs.update(overrides)
    return MelBandRoformer(**kwargs)


@pytest.mark.parametrize(
    "samplerate,segment_samples",
    [(0, SR), (SR, 0), (True, SR), (SR, 1.5)],
)
def test_configure_inference_rejects_invalid_geometry(
    samplerate: object, segment_samples: object
) -> None:
    """Sample rate and training segment must be positive integer counts."""
    model = _bs()
    with pytest.raises(ValidationError):
        model.configure_inference(
            sources=["vocals", "other"],
            samplerate=samplerate,  # type: ignore[arg-type]
            segment_samples=segment_samples,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("builder", [_bs, _mel], ids=["bs", "mel"])
def test_state_dict_roundtrip_is_strict(builder) -> None:
    """
    A checkpoint saved from one instance loads into a freshly-built instance
    with ``strict=True`` and zero key mismatches — the invariant that lets
    real community checkpoints load. The reloaded model must reproduce the
    original forward output bit-for-bit.
    """
    torch.manual_seed(0)
    model = builder().eval()
    audio = torch.randn(1, 2, SR)
    with torch.no_grad():
        expected = model(audio)

    state = model.state_dict()
    fresh = builder().eval()
    # strict=True must not raise: identical architecture => identical keys.
    fresh.load_state_dict(state, strict=True)
    with torch.no_grad():
        got = fresh(audio)

    assert torch.equal(expected, got)


@pytest.mark.parametrize("builder", [_bs, _mel], ids=["bs", "mel"])
def test_forward_output_shape(builder) -> None:
    """
    A single-stem model with the two-name (complement) convention returns
    ``[batch, 2, channels, samples]`` and finite values.
    """
    model = builder().eval()
    model.configure_inference(
        sources=["vocals", "other"], samplerate=SR, segment_samples=SR * 4
    )
    audio = torch.randn(1, 2, SR)
    with torch.no_grad():
        out = model(audio)
    assert out.shape == (1, 2, 2, SR)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("builder", [_bs, _mel], ids=["bs", "mel"])
def test_complement_stem_reconstructs_mixture(builder) -> None:
    """
    For single-mask checkpoints the second stem is ``mixture - prediction``,
    so the stems must sum back to the input mixture.
    """
    model = builder().eval()
    model.configure_inference(
        sources=["vocals", "other"], samplerate=SR, segment_samples=SR * 4
    )
    audio = torch.randn(1, 2, SR)
    with torch.no_grad():
        out = model(audio)
    reconstructed = out[:, 0] + out[:, 1]
    assert torch.allclose(reconstructed, audio, atol=1e-4)


@pytest.mark.parametrize("builder", [_bs, _mel], ids=["bs", "mel"])
def test_flows_through_apply_model(builder) -> None:
    """
    Both architectures run through the shared chunked-inference engine
    (single-input tiling and batched cross-mix pooling), producing
    input-length, finite output — the "shared internals" contract.
    """
    model = builder().eval()
    model.configure_inference(
        sources=["vocals", "other"], samplerate=SR, segment_samples=SR * 2
    )
    # Longer than one segment so tiling + overlap-add engage.
    mix = torch.randn(1, 2, int(SR * 5.0))
    out = apply_model(
        model, mix, device="cpu", shifts=1, overlap=0.25, chunk_batch_size=2
    )
    assert out.shape == (1, 2, 2, mix.shape[-1])
    assert torch.isfinite(out).all()

    shorter = torch.randn(1, 2, int(SR * 2.7))
    outs = apply_model_multi(
        model, [mix, shorter], device="cpu", shifts=1, overlap=0.25, chunk_batch_size=2
    )
    assert [o.shape[-1] for o in outs] == [mix.shape[-1], shorter.shape[-1]]


def test_multi_stem_has_no_complement() -> None:
    """
    When source names match the head count, every source is a real mask head
    and no complement stem is appended.
    """
    model = _bs(num_stems=6).eval()
    model.configure_inference(
        sources=["bass", "drums", "other", "vocals", "guitar", "piano"],
        samplerate=SR,
        segment_samples=SR * 2,
    )
    assert model.output_complement is False
    audio = torch.randn(1, 2, SR)
    with torch.no_grad():
        out = model(audio)
    assert out.shape == (1, 6, 2, SR)


def test_roformer_skips_external_normalization() -> None:
    """
    RoFormer checkpoints train on raw audio, so the backend must advertise
    that the Separator's Demucs-style mean/std normalization is skipped.
    """
    assert _bs().external_normalization is False
    assert _mel().external_normalization is False


def test_source_count_mismatch_rejected() -> None:
    """
    ``configure_inference`` rejects a source list that matches neither the
    head count nor the single-stem complement convention.
    """
    model = _bs(num_stems=2)
    with pytest.raises(ValidationError):
        model.configure_inference(
            sources=["vocals", "drums", "bass"], samplerate=SR, segment_samples=SR
        )


def test_bandsplit_must_cover_stft_bins() -> None:
    """
    BS-RoFormer rejects a band layout that doesn't sum to the STFT bin count.
    """
    with pytest.raises(ValidationError):
        _bs(freqs_per_bands=(2, 2, 2))  # nowhere near 1025


def test_linear_transformer_depth_unsupported() -> None:
    """
    The linear-attention path isn't carried over (no shipped checkpoint uses
    it); requesting it is rejected rather than silently mis-loading.
    """
    with pytest.raises(ValidationError):
        _bs(linear_transformer_depth=1)


def test_build_roformer_unknown_architecture() -> None:
    """
    ``build_roformer`` rejects an unknown architecture name.
    """
    with pytest.raises(ValidationError):
        build_roformer(
            "conformer",
            {"dim": 32, "depth": 1},
            sources=["vocals", "other"],
            samplerate=SR,
            segment_samples=SR,
        )


def test_repository_loads_roformer_from_cache(tmp_path, monkeypatch) -> None:
    """
    End to end through ``ModelRepository``: a RoFormer entry whose verified
    checkpoint is already cached builds via the config + strict state load,
    without touching the network.
    """
    import hashlib
    import json

    from unblend import repo as repo_module

    torch.manual_seed(2)
    config = dict(
        dim=32,
        depth=2,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=16,
        heads=2,
    )
    model = BSRoformer(**config)
    from safetensors.torch import save_file

    ckpt_bytes_path = tmp_path / "weights.safetensors"
    save_file(
        {key: value.clone() for key, value in model.state_dict().items()},
        ckpt_bytes_path,
    )
    raw = ckpt_bytes_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Content-addressed cache filename the loader looks up (sha256[:16]).
    (cache_dir / f"{digest[:16]}.safetensors").write_bytes(raw)
    monkeypatch.setattr(repo_module, "get_cache_dir", lambda: cache_dir)

    metadata = {
        "models": {
            "test_bs": {
                "backend": "roformer",
                "architecture": "bs_roformer",
                "license": "MIT",
                "sources": ["vocals", "other"],
                "samplerate": SR,
                "segment_samples": SR * 4,
                "config": config,
                "checkpoint": {
                    "format": "safetensors",
                    "url": "https://example.invalid/weights.safetensors",
                    "sha256": digest,
                    "size_bytes": len(raw),
                },
            }
        }
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    loaded = repo_module.ModelRepository(metadata_path).get_model("test_bs")
    assert loaded.sources == ["vocals", "other"]
    assert loaded.samplerate == SR

    audio = torch.randn(1, 2, SR)
    with torch.no_grad():
        original = model.eval()(audio)
        got = loaded(audio)
    # Loaded weights reproduce the original mask head (stem 0).
    assert torch.equal(original, got[:, :1])

    # Cache accounting sees the checkpoint as a complete download.
    info = repo_module.ModelRepository(metadata_path).get_cache_info()
    assert info["test_bs"]["complete"] is True


@pytest.mark.slow
def test_registry_roformer_checkpoints_strict_load() -> None:
    """
    Every shipped RoFormer registry entry downloads and strict-loads its real
    checkpoint — guarding the inline configs against drifting from the actual
    weights (a wrong config fails ``load_state_dict(strict=True)`` here
    instead of on a user's machine).
    """
    from unblend.repo import ModelRepository

    repo = ModelRepository()
    for name, info in repo.list_models().items():
        if info.get("backend") != "roformer":
            continue
        model = repo.get_model(name)
        assert model.sources == info["sources"], name
        assert model.samplerate == info["samplerate"], name


def test_build_roformer_loads_state() -> None:
    """
    ``build_roformer`` constructs, configures, and strict-loads a state dict
    end to end.
    """
    torch.manual_seed(1)
    reference = _bs().eval()
    state = reference.state_dict()
    built = build_roformer(
        "bs_roformer",
        dict(
            dim=32,
            depth=2,
            stereo=True,
            num_stems=1,
            time_transformer_depth=1,
            freq_transformer_depth=1,
            dim_head=16,
            heads=2,
        ),
        sources=["vocals", "other"],
        samplerate=SR,
        segment_samples=SR * 4,
        state=state,
    )
    audio = torch.randn(1, 2, SR)
    with torch.no_grad():
        assert torch.equal(reference(audio), built(audio)[:, :1])


def test_rotary_cache_is_invalidated_by_dtype_transform() -> None:
    """A warmed module converted to half matches a freshly converted module."""
    from unblend.roformer import RotaryEmbedding

    warmed = RotaryEmbedding(dim=16)
    sample = torch.randn(1, 2, 8, 16)
    warmed.rotate_queries_or_keys(sample)
    assert warmed._phase_cache

    warmed.half()
    assert not warmed._phase_cache
    assert not warmed._rotation_cache

    half_sample = sample.half()
    fresh = RotaryEmbedding(dim=16).half()
    torch.testing.assert_close(
        warmed.rotate_queries_or_keys(half_sample),
        fresh.rotate_queries_or_keys(half_sample),
        atol=0,
        rtol=0,
    )


def test_rotary_rotation_accepts_half_inputs() -> None:
    """
    Rotation must work on fp16 tensors from a model cast to half: the phase
    table is built in float32 regardless of the (cast) ``freqs`` dtype —
    ``torch.polar`` has no half kernel, so an fp16-following phase path
    crashes on CUDA/MPS. Regression test for the fp16 GPU path.
    """
    from unblend.roformer import RotaryEmbedding

    rotary = RotaryEmbedding(dim=16).to(torch.float16)
    t = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    out = rotary.rotate_queries_or_keys(t)
    assert out.dtype == torch.float16
    assert out.shape == t.shape
    assert torch.isfinite(out).all()
