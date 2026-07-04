# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math

import torch
import torch.nn as nn

from .blocks import pad1d, spectro
from .htdemucs import HTDemucs
from .repo import ModelRepository


class HTDemucsONNXWrapper(nn.Module):
    """
    Wrapper that makes HTDemucs compatible with ONNX export.
    """

    def __init__(self, model: HTDemucs) -> None:
        """
        Initialize the ONNX wrapper.

        :param model: The HTDemucs model to wrap for ONNX export
        """
        super().__init__()
        self.model = model
        self.sources = model.sources
        self.samplerate = model.samplerate
        self.audio_channels = model.audio_channels
        self.nfft = model.nfft
        self.hop_length = model.hop_length

    def forward(
        self, spec_real: torch.Tensor, spec_imag: torch.Tensor, mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.

        :param spec_real: Real part of spectrogram [B, C, Fq, T]
        :param spec_imag: Imaginary part of spectrogram [B, C, Fq, T]
        :param mix: Raw audio waveform [B, C, samples]
        :return: Tuple of (out_spec_real, out_spec_imag, out_wave) separated spectrograms and waveforms
        """
        B, C, Fq, T = spec_real.shape
        samples = mix.shape[-1]

        # Convert real/imag to CaC format: [ch0_real, ch0_imag, ch1_real, ch1_imag, ...]
        x = torch.stack([spec_real, spec_imag], dim=2).reshape(B, C * 2, Fq, T)

        # Normalize inputs
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        meant = mix.mean(dim=(1, 2), keepdim=True)
        stdt = mix.std(dim=(1, 2), keepdim=True)
        xt = (mix - meant) / (1e-5 + stdt)

        # Core encoder-transformer-decoder processing
        x, xt = self.model.forward_core(x, xt)

        # Denormalize and reshape frequency branch output
        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # Split CaC back into real/imag
        out_spec_real = x[:, :, 0::2, :, :]
        out_spec_imag = x[:, :, 1::2, :, :]

        # Denormalize and reshape time branch output
        xt = xt.view(B, S, -1, samples)
        xt = xt * stdt[:, None] + meant[:, None]

        return out_spec_real, out_spec_imag, xt


def compute_stft_for_export(
    audio: torch.Tensor, nfft: int, hop_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute STFT for model input, matching HTDemucs preprocessing.

    :param audio: Input audio [B, C, samples]
    :param nfft: FFT size
    :param hop_length: Hop length
    :return: Tuple of (real, imag) spectrograms [B, C, Fq, T]
    """
    # Padding to match HTDemucs._spec
    le = int(math.ceil(audio.shape[-1] / hop_length))
    pad = hop_length // 2 * 3

    # Pad the audio. Use pad1d (same as HTDemucs._spec) so the reflect-pad
    # handling stays identical even if this helper is ever reused on a clip
    # shorter than the pad amount.
    padded = pad1d(
        audio, (pad, pad + le * hop_length - audio.shape[-1]), mode="reflect"
    )

    # Compute STFT
    z = spectro(padded, nfft, hop_length)

    # Trim to expected size
    z = z[..., :-1, :]  # Remove last frequency bin
    # Same alignment sanity check HTDemucs._spec makes — guards against the
    # padding math here silently drifting out of sync with the model's STFT.
    # Not an ``assert`` so it survives ``python -O``.
    if z.shape[-1] != le + 4:
        raise RuntimeError(
            f"STFT frame count {z.shape[-1]} does not match expected {le + 4}"
        )
    z = z[..., 2 : 2 + le]  # Trim time dimension

    # Split into real and imaginary
    real = z.real
    imag = z.imag

    return real, imag


def export_to_onnx(
    model_name: str = "htdemucs",
    output_path: str | None = None,
    opset_version: int = 17,
    fp16: bool = False,
) -> str:
    """
    Export a Demucs model to ONNX. Traced at the model's training length
    so runtime callers must feed exactly that segment size.

    :param model_name: Name of the model to export.
    :param output_path: Path to save the ONNX model (defaults to ``{model_name}.onnx``).
    :param opset_version: ONNX opset version.
    :param fp16: If True, store weights as float16 (with a Cast(fp16->fp32)
        node inserted before each consumer) after export. This is weight-only:
        it roughly halves the on-disk file size while compute and IO stay in
        float32, so output is near-identical to the fp32 model (not bit-exact,
        since weights are rounded to fp16). Avoids the audible quantization
        noise that pure-fp16 accumulation produces on ORT-WASM.
    :return: Path to the exported ONNX model.
    :raises ImportError: If the ``onnx`` package is not installed.
    :raises ValueError: If the resolved model is not an ``HTDemucs`` instance.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with: uv pip install demucs-next[onnx]"
        )

    if fp16:
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

    if output_path is None:
        output_path = f"{model_name}.onnx"

    repo = ModelRepository()
    model = repo.get_model(model_name)

    if not isinstance(model, HTDemucs):
        raise ValueError(
            f"Model {model_name} is not a supported model type. "
            f"Expected HTDemucs, got {type(model).__name__}"
        )
    wrapper = HTDemucsONNXWrapper(model)

    model.eval()
    wrapper.eval()

    sample_rate = model.samplerate
    segment_samples = int(model.max_allowed_segment * sample_rate)
    nfft = model.nfft
    hop_length = model.hop_length

    batch_size = 1
    audio_channels = model.audio_channels

    dummy_audio = torch.randn(batch_size, audio_channels, segment_samples)
    dummy_spec_real, dummy_spec_imag = compute_stft_for_export(
        dummy_audio, nfft, hop_length
    )

    torch.onnx.export(
        wrapper,
        (dummy_spec_real, dummy_spec_imag, dummy_audio),
        output_path,
        input_names=["spec_real", "spec_imag", "audio"],
        output_names=["out_spec_real", "out_spec_imag", "out_wave"],
        # Only ``batch`` is dynamic. The model always runs at exactly the
        # trained segment length (HTDemucs.forward pads shorter inputs up and
        # valid_length() rejects longer ones), so the time/sample axes are
        # fixed -- advertising them as dynamic would be a false promise.
        dynamic_axes={
            "spec_real": {0: "batch"},
            "spec_imag": {0: "batch"},
            "audio": {0: "batch"},
            "out_spec_real": {0: "batch"},
            "out_spec_imag": {0: "batch"},
            "out_wave": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)

    if fp16:
        # Weight-only fp16: store every Conv/MatMul/Gemm/ConvTranspose
        # weight as float16, then insert a Cast(fp16->float32) node right
        # after the initializer so compute runs in full fp32. This halves
        # the on-disk model size without any fp16 compute precision loss —
        # ORT-WASM's pure-fp16 accumulation in Conv/MatMul kernels produces
        # audible 8-bit-style quantization noise that CUDA/MPS hide via
        # fp32 accumulation; we sidestep the issue entirely by computing in
        # fp32 regardless of EP. ORT's graph optimizer typically folds the
        # constant Cast at load time so there's no runtime overhead.
        weight_op_inputs = {
            "Conv": (1, 2),
            "ConvTranspose": (1, 2),
            "MatMul": (0, 1),
            "Gemm": (0, 1, 2),
        }
        weight_init_names: set[str] = set()
        for node in onnx_model.graph.node:
            for idx in weight_op_inputs.get(node.op_type, ()):
                if idx < len(node.input) and node.input[idx]:
                    weight_init_names.add(node.input[idx])

        existing_outputs = {n.output[0] for n in onnx_model.graph.node if n.output}
        existing_inputs = {i.name for i in onnx_model.graph.input}

        new_inits = []
        new_cast_nodes = []
        for init in onnx_model.graph.initializer:
            if (
                init.name in weight_init_names
                and init.data_type == TensorProto.FLOAT
                # Only rewrite tensors that are bona-fide weight initializers,
                # not ones already produced by some upstream node.
                and init.name not in existing_outputs
                and init.name not in existing_inputs
            ):
                arr = numpy_helper.to_array(init).astype(np.float16)
                fp16_name = init.name + "_fp16"
                new_inits.append(numpy_helper.from_array(arr, name=fp16_name))
                new_cast_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[fp16_name],
                        outputs=[init.name],
                        to=TensorProto.FLOAT,
                        name=init.name + "_cast_to_fp32",
                    )
                )
            else:
                new_inits.append(init)

        # If nothing matched, the exporter's op/initializer layout has changed
        # (e.g. a torch/onnx upgrade fused ops or renamed inputs) and we'd
        # otherwise write a byte-for-byte fp32 model stamped ``precision=fp16``.
        # Fail loudly instead of shipping a mislabeled, full-size artifact.
        if not new_cast_nodes:
            raise RuntimeError(
                "fp16 export requested but no fp32 weight initializers were "
                "converted — the exporter's op/initializer layout likely "
                "changed. Refusing to write a model mislabeled as fp16."
            )

        onnx_model.graph.ClearField("initializer")
        onnx_model.graph.initializer.extend(new_inits)
        # Prepend the weight-Cast nodes so ORT sees them before any consumer.
        original_nodes = list(onnx_model.graph.node)
        onnx_model.graph.ClearField("node")
        onnx_model.graph.node.extend(new_cast_nodes + original_nodes)

    sources_meta = onnx_model.metadata_props.add()
    sources_meta.key = "sources"
    sources_meta.value = json.dumps(model.sources)

    sample_rate_meta = onnx_model.metadata_props.add()
    sample_rate_meta.key = "sample_rate"
    sample_rate_meta.value = str(model.samplerate)

    channels_meta = onnx_model.metadata_props.add()
    channels_meta.key = "audio_channels"
    channels_meta.value = str(model.audio_channels)

    precision_meta = onnx_model.metadata_props.add()
    precision_meta.key = "precision"
    precision_meta.value = "fp16" if fp16 else "fp32"

    # Validate the graph before saving — catches any malformed nodes,
    # especially after the fp16 weight/Cast surgery above.
    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, output_path)

    return output_path
