"""
Regression tests for the SCNet backend.
"""

import pytest
import torch

from unblend import backends
from unblend.exceptions import ValidationError
from unblend.scnet import FeatureConversion, SCNet, build_scnet


def _tiny(**overrides) -> dict:
    """
    Constructor kwargs for a fast, structurally faithful SCNet.

    :param overrides: Values replacing the defaults below.
    :return: Kwargs suitable for :class:`SCNet`.
    """
    config = dict(
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
    config.update(overrides)
    return config


def test_scnet_registers_itself_as_a_backend() -> None:
    """
    Importing the module is what makes ``scnet`` loadable by name.
    """
    assert "scnet" in backends._BUILDERS


def test_scnet_does_not_apply_demucs_track_normalization() -> None:
    """
    SCNet defaults to raw input unless a checkpoint opts into normalization.
    """
    model = SCNet(sources=["a", "b", "c", "d"], **_tiny())
    assert model.external_normalization is False


def test_build_scnet_rejects_unknown_architecture() -> None:
    """
    An unknown architecture fails loudly rather than silently defaulting.
    """
    with pytest.raises(ValidationError, match="Unknown SCNet architecture"):
        build_scnet(
            "not_scnet",
            _tiny(),
            sources=["drums", "bass", "other", "vocals"],
            samplerate=44100,
            segment_samples=4096,
        )


def test_configure_inference_rejects_mismatched_source_count() -> None:
    """
    Source names must match the decoder's head count.
    """
    model = SCNet(sources=["drums", "bass", "other", "vocals"], **_tiny())
    with pytest.raises(ValidationError, match="emits 4 stems"):
        model.configure_inference(
            sources=["vocals"], samplerate=44100, segment_samples=4096
        )


def test_build_scnet_sets_the_inference_contract() -> None:
    """
    The four members apply_model relies on are populated.
    """
    sources = ["drums", "bass", "other", "vocals"]
    model = build_scnet(
        "scnet", _tiny(), sources=sources, samplerate=44100, segment_samples=88200
    )
    assert model.sources == sources
    assert model.samplerate == 44100
    assert model.max_allowed_segment == pytest.approx(2.0)
    assert callable(model.forward)


def test_forward_returns_one_waveform_per_stem() -> None:
    """
    Output is ``(batch, stems, channels, samples)`` at the input length.
    """
    sources = ["drums", "bass", "other", "vocals"]
    model = build_scnet(
        "scnet", _tiny(), sources=sources, samplerate=44100, segment_samples=4096
    )
    audio = torch.randn(1, 2, 4096)
    with torch.inference_mode():
        out = model(audio)
    assert out.shape == (1, len(sources), 2, 4096)


def test_conv_module_requires_an_odd_kernel() -> None:
    """
    An even kernel cannot be centre-padded, so it is rejected.
    """
    with pytest.raises(ValidationError, match="must be odd"):
        SCNet(sources=["a", "b", "c", "d"], **_tiny(conv_kernel=4))


@pytest.mark.parametrize("frames", [32, 64])
@pytest.mark.parametrize("inverse", [False, True])
def test_onnx_safe_dft_matches_torch_fft(frames: int, inverse: bool) -> None:
    """
    The explicit DFT used for ONNX export is numerically the same transform.

    ``torch.fft.irfft`` lowers to an ONNX ``DFT`` node with both ``inverse``
    and ``onesided`` set, which onnxruntime rejects, so exports take this path
    instead — it has to agree with the native one.
    """
    module = FeatureConversion(8, inverse=inverse)
    shape = (2, 8, 5, frames // 2 + 1) if inverse else (2, 4, 5, frames)
    x = torch.randn(*shape)

    native = module(x)
    module.onnx_safe = True
    explicit = module(x)

    assert native.shape == explicit.shape
    assert torch.allclose(native, explicit, atol=1e-4)


def test_compiled_core_hooks_round_trip() -> None:
    """
    Enabling then disabling the compiled core restores the eager callable.
    """
    model = build_scnet(
        "scnet",
        _tiny(),
        sources=["drums", "bass", "other", "vocals"],
        samplerate=44100,
        segment_samples=4096,
    )
    # Compare the underlying function, not the bound method: attribute access
    # builds a fresh bound method each time, so ``is`` would never hold.
    original = type(model).forward_core

    model.enable_compiled_core()
    assert getattr(model.forward_core, "__func__", None) is not original
    assert model._fixed_batch_shape is True

    model.disable_compiled_core()
    assert model.forward_core.__func__ is original
    assert model._fixed_batch_shape is False
    assert not hasattr(model, "_eager_core")


def test_masked_variant_is_registered_and_distinct() -> None:
    """
    ``scnet_masked`` is a separate architecture, not a flag on ``scnet``.

    The published ``scnet_small`` checkpoint carries ``pos_embed_f`` and
    ``mask_layer`` parameters that plain SCNet has no slot for (whereas
    ``scnet_xl_wide_v5`` does not), so loading one into the wrong class must
    fail rather than silently drop weights.
    """
    from unblend.scnet import _ARCHITECTURES, SCNetMasked

    assert _ARCHITECTURES["scnet_masked"] is SCNetMasked
    assert _ARCHITECTURES["scnet"] is SCNet

    plain = SCNet(sources=["a", "b", "c", "d"], **_tiny())
    masked = SCNetMasked(sources=["a", "b", "c", "d"], **_tiny())
    extra = set(masked.state_dict()) - set(plain.state_dict())
    assert any(k.startswith("mask_layer") for k in extra)
    assert "pos_embed_f" in extra


def test_masked_variant_forward_shape() -> None:
    """
    The masking head still yields one waveform per stem.
    """
    from unblend.scnet import SCNetMasked

    sources = ["drums", "bass", "other", "vocals"]
    model = SCNetMasked(sources=sources, **_tiny())
    model.configure_inference(sources=sources, samplerate=44100, segment_samples=4096)
    model.eval()
    with torch.inference_mode():
        out = model(torch.randn(1, 2, 4096))
    assert out.shape == (1, len(sources), 2, 4096)


def _write_local_model(tmp_path, name: str = "my_scnet") -> tuple:
    """
    Serialise a tiny SCNet and describe it in an extra-models file.

    :param tmp_path: pytest temporary directory.
    :param name: Model name to register.
    :return: ``(models_file, weights_file)`` paths.
    """
    import json

    from safetensors.torch import save_file

    model = SCNet(sources=["drums", "bass", "other", "vocals"], **_tiny())
    weights = tmp_path / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(weights))
    config = _tiny()
    models_file = tmp_path / "extra.json"
    models_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    name: {
                        "backend": "scnet",
                        "architecture": "scnet",
                        "license": "unknown",
                        "sources": ["drums", "bass", "other", "vocals"],
                        "samplerate": 44100,
                        "segment_samples": 4096,
                        "config": config,
                        "checkpoint": {
                            "format": "safetensors",
                            "path": str(weights),
                        },
                    }
                },
            }
        )
    )
    return models_file, weights


def test_extra_models_file_adds_a_local_checkpoint(tmp_path) -> None:
    """
    A user-supplied file makes a local checkpoint loadable by name.
    """
    from unblend.repo import ModelRepository

    models_file, _ = _write_local_model(tmp_path)
    repo = ModelRepository(extra_models=models_file)

    assert "my_scnet" in repo.list_models()
    # Built-ins are still present: the file overlays, it does not replace.
    assert "htdemucs" in repo.list_models()

    model = repo.get_model("my_scnet")
    with torch.inference_mode():
        out = model(torch.randn(1, 2, 4096))
    assert out.shape == (1, 4, 2, 4096)


def test_local_checkpoint_is_not_treated_as_managed_cache(tmp_path) -> None:
    """
    Cache inspection/removal must never delete a user-owned model file.
    """
    from unblend.repo import ModelRepository

    models_file, weights = _write_local_model(tmp_path)
    repo = ModelRepository(extra_models=models_file)

    assert "my_scnet" not in repo.get_cache_info()
    assert repo.remove_model("my_scnet") is False
    assert weights.is_file()


def test_extra_models_file_cannot_shadow_a_builtin(tmp_path) -> None:
    """
    Redefining a shipped name is rejected.

    Otherwise a dropped-in file could silently swap the weights behind
    ``htdemucs`` for existing callers.
    """
    from unblend.exceptions import ModelLoadingError
    from unblend.repo import ModelRepository

    models_file, _ = _write_local_model(tmp_path, name="htdemucs")
    with pytest.raises(ModelLoadingError, match="redefines built-in model"):
        ModelRepository(extra_models=models_file)


def test_local_checkpoint_digest_is_verified_when_supplied(tmp_path) -> None:
    """
    A stated sha256 is enforced, so a swapped file fails loudly.
    """
    import json

    from unblend.exceptions import ModelLoadingError
    from unblend.repo import ModelRepository

    models_file, weights = _write_local_model(tmp_path)
    spec = json.loads(models_file.read_text())
    spec["models"]["my_scnet"]["checkpoint"]["sha256"] = "0" * 64
    models_file.write_text(json.dumps(spec))

    repo = ModelRepository(extra_models=models_file)
    with pytest.raises(ModelLoadingError):
        repo.get_model("my_scnet")


def test_missing_local_checkpoint_is_reported_clearly(tmp_path) -> None:
    """
    A path that does not exist names the file rather than failing obscurely.
    """

    from unblend.exceptions import ModelLoadingError
    from unblend.repo import ModelRepository

    models_file, weights = _write_local_model(tmp_path)
    weights.unlink()
    repo = ModelRepository(extra_models=models_file)
    with pytest.raises(ModelLoadingError, match="does not exist"):
        repo.get_model("my_scnet")


def test_onnx_wrapper_reproduces_the_masked_forward() -> None:
    """
    The export wrapper must apply the masking head, not just the trunk.

    ``SCNetONNXWrapper`` originally traced ``forward_core`` alone, which is
    correct for plain SCNet but silently dropped ``pos_embed_f``, ``mask_layer``
    and the mask/mixture product for the masked variants -- producing a graph
    that exported and ran but returned the raw trunk output.
    """
    import torch.nn.functional as F

    from unblend.onnx import SCNetONNXWrapper, compute_scnet_stft_for_export
    from unblend.scnet import SCNetMasked, stft_padding

    sources = ["drums", "bass", "other", "vocals"]
    model = SCNetMasked(sources=sources, **_tiny())
    model.configure_inference(sources=sources, samplerate=44100, segment_samples=4096)
    model.eval()

    audio = torch.randn(1, 2, 4096)
    with torch.inference_mode():
        expected = model(audio)

        stft = model.stft_config
        padding = stft_padding(audio.shape[-1], model.hop_length)
        padded = F.pad(audio, (0, padding))
        real, imag = compute_scnet_stft_for_export(
            padded,
            int(stft["n_fft"]),
            int(stft["hop_length"]),
            int(stft["win_length"]),
            bool(stft["normalized"]),
            window="hann",
        )
        out_real, out_imag = SCNetONNXWrapper(model)(real, imag)

        spec = torch.complex(out_real, out_imag)
        batch, stems, channels, freq, frames = spec.shape
        wave = torch.istft(
            spec.reshape(-1, freq, frames),
            **model._stft_kwargs(spec.device),
            length=padded.shape[-1],
        )
        actual = wave.reshape(batch, stems, channels, -1)[..., : audio.shape[-1]]

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-4)


def test_scnet_export_uses_browser_io_and_records_both_segment_lengths(
    tmp_path,
) -> None:
    """
    The exported graph advertises the exact contract consumed in JS.
    """
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    from unblend.onnx import _export_scnet_to_onnx
    from unblend.scnet import SCNetMasked, stft_padding

    sources = ["drums", "bass", "other", "vocals"]
    segment = 4096
    model = SCNetMasked(sources=sources, **_tiny())
    model.configure_inference(
        sources=sources,
        samplerate=44100,
        segment_samples=segment,
    )
    path = str(tmp_path / "scnet.onnx")
    _export_scnet_to_onnx(
        model,
        path,
        opset_version=18,
        fp16=False,
        license_label="unlicensed",
        static_batch=True,
    )

    exported = onnx.load(path)
    assert [value.name for value in exported.graph.input] == [
        "spec_real",
        "spec_imag",
    ]
    assert [value.name for value in exported.graph.output] == [
        "out_spec_real",
        "out_spec_imag",
    ]
    metadata = {item.key: item.value for item in exported.metadata_props}
    assert metadata["unblend.logical_segment_samples"] == str(segment)
    assert metadata["unblend.segment_samples"] == str(
        segment + stft_padding(segment, model.hop_length)
    )
    assert metadata["unblend.stft_window"] == "hann"
