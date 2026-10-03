"""
Guards the SCNet ONNX export against drifting numerically from PyTorch.

onnxruntime reduces the stock ``GroupNorm(1, C)`` export
(InstanceNormalization) in a single fp32 pass. Over SCNet's multi-million
element groups that alone cost ~0.5% relative error (~46 dB SNR) on
``scnet_small``. The export therefore switches SCNet's norms to a two-level
reduction. These tests pin that and the end-to-end parity. The onnxruntime
tests skip unless the ``onnx`` extra and ``onnxruntime`` are installed.
"""

import pytest
import torch

from unblend.onnx import SCNetONNXWrapper, _export_scnet_to_onnx
from unblend.scnet import GroupNorm, SCNet, SCNetMasked

SOURCES = ["drums", "bass", "other", "vocals"]


def _tiny_config() -> dict:
    """
    Constructor kwargs for a fast, structurally faithful SCNet.

    :return: Kwargs suitable for :class:`SCNet`.
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


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """
    Relative L2 error of ``actual`` against ``expected``.

    :param actual: Values under test.
    :param expected: Reference values.
    :return: ``||actual - expected|| / ||expected||``.
    """
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm()).item()


def _random_norm(channels: int) -> GroupNorm:
    """
    A single-group norm with non-trivial affine parameters.

    :param channels: Channel count.
    :return: The norm, in eval mode.
    """
    norm = GroupNorm(1, channels).eval()
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()
    return norm


@pytest.mark.parametrize("shape", [(2, 8, 5, 7), (3, 6, 11)])
def test_onnx_safe_group_norm_matches_eager(shape) -> None:
    """
    The export formulation computes the same normalization as the stock op.

    :param shape: Input shape, 4-D as in the trunk or 3-D as in the conv stack.
    """
    torch.manual_seed(0)
    norm = _random_norm(shape[1])
    x = torch.randn(shape) * 3 + 2
    with torch.no_grad():
        expected = norm(x)
        norm.onnx_safe = True
        actual = norm(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_export_wrapper_switches_every_norm_to_the_onnx_safe_form() -> None:
    """
    Every SCNet norm takes the export path once the model is wrapped.
    """
    model = SCNetMasked(sources=SOURCES, **_tiny_config())
    norms = [m for m in model.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert norms
    assert all(isinstance(m, GroupNorm) and not m.onnx_safe for m in norms)
    SCNetONNXWrapper(model)
    assert all(m.onnx_safe for m in norms)


def test_exported_group_norm_stays_accurate_over_large_groups(tmp_path) -> None:
    """
    onnxruntime keeps a million-element group accurate to fp32 rounding.

    The stock export lands near 7e-6 here, and the error grows with the
    group size. The two-level form stays around 1e-7.

    :param tmp_path: pytest temporary directory fixture
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    torch.manual_seed(0)
    norm = _random_norm(64)
    norm.onnx_safe = True
    x = torch.randn(1, 64, 128, 128) * 2
    path = str(tmp_path / "group_norm.onnx")
    torch.onnx.export(norm, (x,), path, dynamo=True, opset_version=18, verbose=False)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
    expected = torch.nn.functional.group_norm(
        x.double(), 1, norm.weight.double(), norm.bias.double(), norm.eps
    )
    assert _relative_error(torch.from_numpy(actual), expected) < 1e-6


@pytest.mark.parametrize("cls", [SCNet, SCNetMasked])
def test_export_and_onnxruntime_parity(cls, tmp_path) -> None:
    """
    The exported graph reproduces the PyTorch wrapper in onnxruntime.

    Norm affines are randomized so the export's affine path is exercised,
    not just the identity it initializes to.

    :param cls: SCNet variant to export.
    :param tmp_path: pytest temporary directory fixture
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    torch.manual_seed(0)
    model = cls(sources=SOURCES, **_tiny_config())
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GroupNorm):
                module.weight.normal_(1.0, 0.2)
                module.bias.normal_(0.0, 0.2)
    model.configure_inference(sources=SOURCES, samplerate=44100, segment_samples=4096)
    model.eval()

    path = str(tmp_path / "scnet.onnx")
    _export_scnet_to_onnx(
        model,
        path,
        opset_version=18,
        storage=torch.float32,
        license_label="unlicensed",
        static_batch=True,
    )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    spec_real, spec_imag = torch.randn(shape), torch.randn(shape)

    with torch.inference_mode():
        expected = SCNetONNXWrapper(model)(spec_real, spec_imag)
    actual = session.run(
        None, {"spec_real": spec_real.numpy(), "spec_imag": spec_imag.numpy()}
    )

    for got, want in zip(actual, expected):
        assert got.shape == tuple(want.shape)
        assert _relative_error(torch.from_numpy(got), want) < 1e-5


def test_fp16_storage_narrows_only_learned_weights(tmp_path) -> None:
    """
    Weight-only fp16 storage narrows the model's parameters and leaves the
    constants the exporter folded into initializers (SCNet's DFT basis) at
    fp32, as the "arithmetic stays fp32" contract promises.

    :param tmp_path: pytest temporary directory fixture
    """
    import hashlib

    import numpy as np

    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from onnx import TensorProto, numpy_helper

    torch.manual_seed(0)
    model = SCNetMasked(sources=SOURCES, **_tiny_config())
    model.configure_inference(sources=SOURCES, samplerate=44100, segment_samples=4096)
    model.eval()
    path = str(tmp_path / "scnet_fp16.onnx")
    _export_scnet_to_onnx(
        model,
        path,
        opset_version=18,
        storage=torch.float16,
        license_label="unlicensed",
        static_batch=True,
    )
    exported = onnx.load(path)

    def digest(array: np.ndarray) -> str:
        """
        :param array: Values to fingerprint, flattened.
        :return: SHA-1 of the flattened bytes.
        """
        return hashlib.sha1(np.ascontiguousarray(array).ravel().tobytes()).hexdigest()

    params = [p.detach().float().numpy() for p in model.parameters()]
    as_fp16 = {digest(p.astype(np.float16)) for p in params}
    as_fp16 |= {digest(p.T.astype(np.float16)) for p in params if p.ndim == 2}
    as_fp32 = {digest(p) for p in params} | {digest(p.T) for p in params if p.ndim == 2}

    narrowed = [
        i for i in exported.graph.initializer if i.data_type == TensorProto.FLOAT16
    ]
    assert narrowed
    for init in narrowed:
        assert digest(numpy_helper.to_array(init)) in as_fp16, init.name

    folded = [
        i
        for i in exported.graph.initializer
        if i.data_type == TensorProto.FLOAT
        and int(np.prod(i.dims)) >= 1000
        and digest(numpy_helper.to_array(i)) not in as_fp32
    ]
    assert folded, "expected the DFT basis to stay a wide fp32 constant"


def test_export_carries_no_exporter_debug_metadata(tmp_path) -> None:
    """
    The dynamo exporter's per-node stack traces (the exporting machine's file
    paths) and ``pkg.torch.*`` properties are stripped, so two exports of the
    same model are byte-identical and leak no local paths.

    :param tmp_path: pytest temporary directory.
    """
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    torch.manual_seed(0)
    model = SCNet(sources=SOURCES, **_tiny_config())
    model.configure_inference(sources=SOURCES, samplerate=44100, segment_samples=4096)
    model.eval()
    paths = [str(tmp_path / f"m{i}.onnx") for i in range(2)]
    for path in paths:
        _export_scnet_to_onnx(
            model,
            path,
            opset_version=18,
            storage=torch.float32,
            license_label="unlicensed",
            static_batch=True,
        )
    exported = onnx.load(paths[0])
    assert not any(node.metadata_props for node in exported.graph.node)
    graph = exported.graph
    assert not any(
        getattr(value, "metadata_props", None)
        for value in (*graph.value_info, *graph.input, *graph.output)
    )
    assert not any(p.key.startswith("pkg.torch") for p in exported.metadata_props)
    assert {"model_family", "license"} <= {p.key for p in exported.metadata_props}
    with open(paths[0], "rb") as first, open(paths[1], "rb") as second:
        assert first.read() == second.read()
