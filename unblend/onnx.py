# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import onnx

from .blocks import pad1d, spectro
from .htdemucs import HTDemucs
from .repo import ModelRepository
from .roformer import MelBandRoformer, _RoformerBase


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


class RoformerONNXWrapper(nn.Module):
    """
    Wrapper that makes BS-RoFormer / Mel-Band RoFormer compatible with ONNX
    export: everything between the STFT and the iSTFT (both stay client-side,
    like the HTDemucs pipeline — but with no time-domain branch, since
    RoFormers are pure spectrogram maskers).

    The complex mask multiply is expressed in real arithmetic (ONNX has no
    complex dtype), DC-zeroing as slice+concat, and Mel-Band's overlapping
    scatter-average as one MatMul with a constant averaging matrix — the
    numerically-identical linear form of the scatter, and the only version
    that runs fast on every ORT execution provider (``ScatterElements`` with
    ``reduction=add`` has patchy WebGPU support).
    """

    def __init__(self, model: _RoformerBase) -> None:
        """
        Initialize the ONNX wrapper.

        :param model: The RoFormer model to wrap for ONNX export.
        """
        super().__init__()
        self.model = model
        self.sources = model.sources
        self.samplerate = model.samplerate
        self.audio_channels = model.audio_channels
        self.num_stems = model.num_stems

        if isinstance(model, MelBandRoformer):
            # A[g, j] = (freq_indices[j] == g) / bands_covering(g): applying A
            # to the per-band masks sums each bin's overlapping band estimates
            # and divides by the cover count — exactly the model's
            # scatter-add + divide, as a single constant MatMul.
            n_selected = int(model.freq_indices.numel())
            denom = model.num_bands_per_freq.repeat_interleave(
                model.audio_channels
            ).clamp(min=1e-8)
            averaging = torch.zeros(int(denom.numel()), n_selected)
            averaging[model.freq_indices, torch.arange(n_selected)] = 1.0
            averaging = averaging / denom[:, None]
            self.register_buffer("mel_averaging_matrix", averaging, persistent=False)
        else:
            self.mel_averaging_matrix = None

    def forward(
        self, spec_real: torch.Tensor, spec_imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.

        :param spec_real: Real part of the mixture STFT ``[B, C, F, T]``.
        :param spec_imag: Imaginary part of the mixture STFT ``[B, C, F, T]``.
        :return: ``(out_spec_real, out_spec_imag)`` masked per-stem
            spectrograms, each ``[B, num_stems, C, F, T]``.
        """
        m = self.model
        B, C, F, T = spec_real.shape

        # Channel-interleave into frequency: 'b s f t c -> b (f s) t c'.
        st = torch.stack([spec_real, spec_imag], dim=-1)
        st = st.permute(0, 2, 1, 3, 4).reshape(B, F * C, T, 2)

        if self.mel_averaging_matrix is not None:
            x = st.index_select(1, m.freq_indices)
        else:
            x = st
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)  # 'b f t c -> b t (f c)'

        x = m.band_split(x)
        x = m._run_transformers(x)
        if not isinstance(m, MelBandRoformer):
            x = m.final_norm(x)

        masks = torch.stack([head(x) for head in m.mask_estimators], dim=1)
        masks = masks.view(B, self.num_stems, T, -1, 2).permute(0, 1, 3, 2, 4)

        mask_real = masks[..., 0]
        mask_imag = masks[..., 1]
        if self.mel_averaging_matrix is not None:
            mask_real = torch.matmul(self.mel_averaging_matrix, mask_real)
            mask_imag = torch.matmul(self.mel_averaging_matrix, mask_imag)

        spec_r = st[..., 0].unsqueeze(1)
        spec_i = st[..., 1].unsqueeze(1)
        out_r = spec_r * mask_real - spec_i * mask_imag
        out_i = spec_r * mask_imag + spec_i * mask_real

        # De-interleave '(f s)' back to [B, stems, C, F, T].
        out_r = out_r.view(B, self.num_stems, F, C, T).permute(0, 1, 3, 2, 4)
        out_i = out_i.view(B, self.num_stems, F, C, T).permute(0, 1, 3, 2, 4)

        if m.zero_dc:
            zeros = torch.zeros_like(out_r[..., :1, :])
            out_r = torch.cat([zeros, out_r[..., 1:, :]], dim=-2)
            out_i = torch.cat([zeros, out_i[..., 1:, :]], dim=-2)
        return out_r, out_i


def compute_roformer_stft_for_export(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    normalized: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the STFT a RoFormer export expects, matching the models' internal
    ``torch.stft`` call (centered, Hann window — no Demucs-style pre-padding).

    :param audio: Input audio ``[B, C, samples]``.
    :param n_fft: FFT size.
    :param hop_length: Hop length.
    :param win_length: Window length.
    :param normalized: Whether the STFT is normalised (models' config value).
    :return: ``(real, imag)`` spectrograms ``[B, C, F, T]``.
    """
    B, C, samples = audio.shape
    window = torch.hann_window(win_length, device=audio.device)
    z = torch.stft(
        audio.reshape(B * C, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=normalized,
        return_complex=True,
    )
    z = z.view(B, C, z.shape[-2], z.shape[-1])
    return z.real.contiguous(), z.imag.contiguous()


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


def _convert_weights_to_fp16(onnx_model: "onnx.ModelProto") -> None:
    """
    Rewrite an ONNX graph's weights as float16 (weight-only precision).

    Stores every Conv/MatMul/Gemm/ConvTranspose weight as float16, then
    inserts a Cast(fp16->float32) node right after the initializer so compute
    runs in full fp32. This halves the on-disk model size without any fp16
    compute precision loss — ORT-WASM's pure-fp16 accumulation in Conv/MatMul
    kernels produces audible 8-bit-style quantization noise that CUDA/MPS hide
    via fp32 accumulation; we sidestep the issue entirely by computing in fp32
    regardless of EP. ORT's graph optimizer typically folds the constant Cast
    at load time so there's no runtime overhead.

    :param onnx_model: Loaded ONNX ``ModelProto``; modified in place.
    :raises RuntimeError: If no fp32 weight initializers were found (the
        exporter's layout changed) — refusing to write a mislabeled model.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

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


@contextmanager
def _atomic_onnx_path(output_path: str) -> Iterator[str]:
    """
    Yield a sibling staging path and atomically publish it on success.

    :param output_path: Caller-requested final ONNX path.
    :return: Context manager yielding a temporary sibling filename.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.onnx", dir=destination.parent
    )
    os.close(fd)
    staging = Path(raw_path)

    def sidecars() -> list[Path]:
        """
        Find external-data paths private to this random staging prefix.

        :return: Sibling sidecar files/directories created by the exporter.
        """
        # Dynamo/ONNX exporters commonly use either ``model.onnx.data`` or a
        # suffix-replaced ``model.data``. The staging basename is random, so
        # this prefix is private to the current export and safe to clean.
        # Do not feed the caller-derived destination name to glob(): legal
        # filenames can contain ``[]``, ``?``, or ``*`` and would turn into a
        # pattern, allowing sidecars to evade detection. Compare names
        # literally instead.
        return [
            candidate
            for candidate in staging.parent.iterdir()
            if candidate != staging and candidate.name.startswith(staging.stem)
        ]

    try:
        yield str(staging)
        external_files = sidecars()
        if external_files:
            names = ", ".join(path.name for path in external_files)
            raise RuntimeError(
                "External-data ONNX exports are not supported by this "
                f"single-file publisher; exporter created: {names}"
            )
        # Windows requires a write-capable descriptor for fsync/FlushFileBuffers.
        with open(staging, "rb+") as file:
            file.flush()
            os.fsync(file.fileno())
        # Replaces a destination symlink entry rather than following it.
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
        for candidate in sidecars():
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                candidate.unlink(missing_ok=True)


def _add_metadata(onnx_model: "onnx.ModelProto", metadata: dict[str, str]) -> None:
    """
    Attach key/value pairs to an ONNX model's ``metadata_props``.

    :param onnx_model: Loaded ONNX ``ModelProto``; modified in place.
    :param metadata: String key/value pairs to embed.
    """
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value


def _export_roformer_to_onnx(
    model: _RoformerBase,
    output_path: str,
    *,
    opset_version: int,
    fp16: bool,
    license_label: str | None = None,
) -> str:
    """
    Export a RoFormer model (BS or Mel-Band) to ONNX via the dynamo exporter.

    The legacy TorchScript exporter emits inconsistent shape metadata for the
    per-band mask heads (they have differing widths), which onnxruntime
    rejects at load — so this path requires the FX-based exporter and
    therefore opset >= 18 (``opset_version`` is raised if lower).

    :param model: The RoFormer model instance to export.
    :param output_path: Path to save the ONNX model.
    :param opset_version: Requested opset; clamped up to 18 (dynamo minimum).
    :param fp16: Store weights as float16 (weight-only; compute stays fp32).
    :param license_label: Weights license to embed in the model metadata.
    :return: Path to the exported ONNX model.
    :raises ImportError: If ``onnx`` or ``onnxscript`` is not installed.
    """
    try:
        import onnx
        import onnxscript  # noqa: F401  (required by the dynamo exporter)
    except ImportError:
        raise ImportError(
            "The 'onnx' and 'onnxscript' packages are required for RoFormer "
            "ONNX export. Install them with: uv pip install unblend[onnx]"
        )

    model.eval()
    wrapper = RoformerONNXWrapper(model).eval()

    segment_samples = int(round(model.max_allowed_segment * model.samplerate))
    stft = model.stft_kwargs
    # Batch=2 example inputs: with batch=1, torch.export 0/1-specializes the
    # batch axis (a size-1 dim is indistinguishable from broadcasting) and the
    # exported model silently rejects batched inputs.
    dummy_audio = torch.randn(2, model.audio_channels, segment_samples)
    dummy_real, dummy_imag = compute_roformer_stft_for_export(
        dummy_audio,
        n_fft=stft["n_fft"],
        hop_length=stft["hop_length"],
        win_length=stft["win_length"],
        normalized=stft["normalized"],
    )

    with _atomic_onnx_path(output_path) as staging_path:
        batch = torch.export.Dim("batch")
        program = torch.onnx.export(
            wrapper,
            (dummy_real, dummy_imag),
            input_names=["spec_real", "spec_imag"],
            output_names=["out_spec_real", "out_spec_imag"],
            # Like HTDemucs: only batch is dynamic. RoFormer checkpoints are
            # trained at a fixed chunk length, so the frequency/time axes are
            # fixed at the traced shape.
            dynamic_shapes={"spec_real": {0: batch}, "spec_imag": {0: batch}},
            opset_version=max(opset_version, 18),
            dynamo=True,
        )
        program.save(staging_path)

        onnx_model = onnx.load(staging_path)
        if fp16:
            _convert_weights_to_fp16(onnx_model)

        architecture = (
            "mel_band_roformer" if isinstance(model, MelBandRoformer) else "bs_roformer"
        )
        metadata = {
            "sources": json.dumps(model.sources),
            "sample_rate": str(model.samplerate),
            "audio_channels": str(model.audio_channels),
            "precision": "fp16" if fp16 else "fp32",
            "model_family": "roformer",
            "architecture": architecture,
            "num_stems": str(model.num_stems),
            "output_complement": "true" if model.output_complement else "false",
            "segment_samples": str(segment_samples),
            "stft_n_fft": str(stft["n_fft"]),
            "stft_hop_length": str(stft["hop_length"]),
            "stft_win_length": str(stft["win_length"]),
            "stft_normalized": "true" if stft["normalized"] else "false",
        }
        if license_label:
            metadata["license"] = license_label
        _add_metadata(onnx_model, metadata)

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, staging_path)
        # Validate the bytes that will actually be promoted, not only the
        # in-memory proto before serialization.
        onnx.checker.check_model(onnx.load(staging_path))
    return output_path


def export_to_onnx(
    model_name: str = "htdemucs",
    output_path: str | None = None,
    opset_version: int = 17,
    fp16: bool = False,
) -> str:
    """
    Export a model (HTDemucs or RoFormer) to ONNX. Traced at the model's
    training length so runtime callers must feed exactly that segment size.

    :param model_name: Name of the model to export.
    :param output_path: Path to save the ONNX model (defaults to ``{model_name}.onnx``).
    :param opset_version: ONNX opset version (raised to 18 for RoFormer models,
        the dynamo-exporter minimum).
    :param fp16: If True, store weights as float16 (with a Cast(fp16->fp32)
        node inserted before each consumer) after export. This is weight-only:
        it roughly halves the on-disk file size while compute and IO stay in
        float32, so output is near-identical to the fp32 model (not bit-exact,
        since weights are rounded to fp16). Avoids the audible quantization
        noise that pure-fp16 accumulation produces on ORT-WASM.
    :return: Path to the exported ONNX model.
    :raises ImportError: If the ``onnx`` package is not installed.
    :raises ValueError: If the resolved model is not a supported type, or an
        HTDemucs without complex-as-channels (``cac=False``).
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with: uv pip install unblend[onnx]"
        )

    if output_path is None:
        output_path = f"{model_name}.onnx"

    repo = ModelRepository()
    model = repo.get_model(model_name)

    if isinstance(model, _RoformerBase):
        model_info = repo.list_models().get(model_name, {})
        return _export_roformer_to_onnx(
            model,
            output_path,
            opset_version=opset_version,
            fp16=fp16,
            license_label=model_info.get("license"),
        )

    if not isinstance(model, HTDemucs):
        raise ValueError(
            f"Model {model_name} is not a supported model type. "
            f"Expected HTDemucs or a RoFormer, got {type(model).__name__}"
        )
    if not model.cac:
        raise ValueError(
            f"Model {model_name} does not use complex-as-channels (cac=False); "
            "the ONNX wrapper hardcodes CaC spectrogram packing."
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

    with _atomic_onnx_path(output_path) as staging_path:
        torch.onnx.export(
            wrapper,
            (dummy_spec_real, dummy_spec_imag, dummy_audio),
            staging_path,
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

        onnx_model = onnx.load(staging_path)

        if fp16:
            _convert_weights_to_fp16(onnx_model)

        _add_metadata(
            onnx_model,
            {
                "sources": json.dumps(model.sources),
                "sample_rate": str(model.samplerate),
                "audio_channels": str(model.audio_channels),
                "precision": "fp16" if fp16 else "fp32",
            },
        )

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, staging_path)
        onnx.checker.check_model(onnx.load(staging_path))

    return output_path
